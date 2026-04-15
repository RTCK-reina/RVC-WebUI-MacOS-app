"""RVC training loop — macOS single-device edition.

This file has been rewritten for macOS-only operation. Key differences from the
upstream (CUDA/XPU) training loop:

* No ``torch.distributed`` / DDP. Training is always single-process, single
  device (MPS if available, CPU otherwise). The ``rank`` / ``n_gpus`` arguments
  are kept at 0/1 for compatibility with the helpers in ``infer.lib.train``.
* No Intel Extension for PyTorch and no DirectML. The ``--dml`` flag and
  ``torch.xpu`` branches were removed.
* fp16 / AMP is disabled. MPS fp16 support is not reliable enough for RVC, and
  ``configs.Config.use_fp32_config()`` already flips ``fp16_run`` off for us.
  ``torch.cuda.amp.autocast`` / ``GradScaler`` are imported only for their API
  shape — they behave as no-ops when ``enabled=False`` on non-CUDA machines.
* Tensors move via ``.to(device)`` against the selected backend, not
  ``.cuda(rank)``.

Training on MPS is *much* slower than on CUDA (5-20x depending on op coverage),
but it completes without any other infrastructure.
"""

import datetime
import logging
import os
import sys
from random import shuffle
from time import sleep
from time import time as ttime
from typing import Tuple

import torch
import torch.multiprocessing as mp
from torch.cuda.amp import GradScaler, autocast  # no-op when enabled=False
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)
logging.getLogger("numba").setLevel(logging.WARNING)

now_dir = os.getcwd()
sys.path.append(os.path.join(now_dir))

from infer.lib.train import utils

hps = utils.get_hparams()
# CUDA_VISIBLE_DEVICES is a no-op on macOS, but keep the env var consistent with
# the value the caller selected so any downstream code inspecting it agrees.
os.environ["CUDA_VISIBLE_DEVICES"] = hps.gpus.replace("-", ",")

from infer.lib.train.data_utils import (
    DistributedBucketSampler,
    TextAudioCollate,
    TextAudioCollateMultiNSFsid,
    TextAudioLoader,
    TextAudioLoaderMultiNSFsid,
)
from rvc.layers.discriminators import MultiPeriodDiscriminator

if hps.version == "v1":
    from rvc.layers.synthesizers import SynthesizerTrnMs256NSFsid as RVC_Model_f0
    from rvc.layers.synthesizers import (
        SynthesizerTrnMs256NSFsid_nono as RVC_Model_nof0,
    )
else:
    from rvc.layers.synthesizers import (
        SynthesizerTrnMs768NSFsid as RVC_Model_f0,
        SynthesizerTrnMs768NSFsid_nono as RVC_Model_nof0,
    )

from infer.lib.train.losses import (
    discriminator_loss,
    feature_loss,
    generator_loss,
    kl_loss,
)
from infer.lib.train.mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from infer.lib.train.process_ckpt import save_small_model
from rvc.layers.utils import (
    slice_on_last_dim,
    total_grad_norm,
)

global_step = 0


def _select_device() -> torch.device:
    """Pick the best available torch device on macOS (MPS, else CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _detect_perf_cores() -> int:
    """Return the number of Apple Silicon performance cores (fallback: cpu_count/2).

    On Apple Silicon, spawning a DataLoader worker per P-core keeps the pipeline
    fed without saturating E-cores (which are useful for OS/background work).
    """
    import subprocess

    try:
        out = subprocess.check_output(
            ["sysctl", "-n", "hw.perflevel0.physicalcpu"],
            stderr=subprocess.DEVNULL,
            timeout=1,
        )
        n = int(out.strip())
        if n > 0:
            return n
    except Exception:
        pass
    # Intel Mac or non-macOS: half of logical cores, capped.
    return max(1, (os.cpu_count() or 4) // 2)


def _optimal_num_workers() -> int:
    """Conservative DataLoader worker count. Capped at 6 to avoid thrash."""
    return min(_detect_perf_cores(), 6)


def _maybe_compile(module: torch.nn.Module, name: str) -> torch.nn.Module:
    """Optionally wrap a module in ``torch.compile``.

    Enabled only when ``RVC_USE_TORCH_COMPILE=1`` is set. MPS compile support
    lands in PyTorch 2.3+ and is still experimental — we swallow failures and
    return the original module so training never breaks.
    """
    if os.environ.get("RVC_USE_TORCH_COMPILE", "0") != "1":
        return module
    if not hasattr(torch, "compile"):
        logger.warning("torch.compile unavailable (PyTorch < 2.0); skipping for %s", name)
        return module
    try:
        compiled = torch.compile(module, mode="reduce-overhead", dynamic=True)
        logger.info("torch.compile applied to %s", name)
        return compiled
    except Exception as e:
        logger.warning("torch.compile failed for %s (%s); falling back to eager", name, e)
        return module


class EpochRecorder:
    def __init__(self):
        self.last_time = ttime()

    def record(self):
        now_time = ttime()
        elapsed_time = now_time - self.last_time
        self.last_time = now_time
        elapsed_time_str = str(datetime.timedelta(seconds=elapsed_time))
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"[{current_time}] | ({elapsed_time_str})"


def main():
    """Entry point — single process, single device on macOS."""
    device = _select_device()
    logger.info(
        "macOS training entrypoint: device=%s (MPS=%s)",
        device,
        torch.backends.mps.is_available(),
    )
    # Unified Memory default: if the caller did not explicitly set this flag,
    # cache the dataset tensors on-device. It's a ~10-30% win for MPS.
    if not hasattr(hps, "if_cache_data_in_gpu") or hps.if_cache_data_in_gpu is None:
        hps.if_cache_data_in_gpu = True
        logger.info("if_cache_data_in_gpu defaulted to True (macOS Unified Memory)")
    run_logger = utils.get_logger(hps.model_dir)
    run(rank=0, n_gpus=1, hps=hps, logger=run_logger, device=device)


def run(
    rank: int,
    n_gpus: int,
    hps: utils.HParams,
    logger: logging.Logger,
    device: torch.device = None,
):
    global global_step
    if device is None:
        device = _select_device()

    if rank == 0:
        logger.info(hps)
        writer = SummaryWriter(log_dir=hps.model_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))
    else:  # pragma: no cover — macOS always uses rank 0
        writer = writer_eval = None

    torch.manual_seed(hps.train.seed)

    if hps.if_f0 == 1:
        train_dataset = TextAudioLoaderMultiNSFsid(hps.data.training_files, hps.data)
    else:
        train_dataset = TextAudioLoader(hps.data.training_files, hps.data)

    train_sampler = DistributedBucketSampler(
        train_dataset,
        hps.train.batch_size * n_gpus,
        [100, 200, 300, 400, 500, 600, 700, 800, 900],  # 16s
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True,
    )

    if hps.if_f0 == 1:
        collate_fn = TextAudioCollateMultiNSFsid()
    else:
        collate_fn = TextAudioCollate()

    # pin_memory only helps when CUDA is present; disable on macOS.
    # num_workers tracks Apple Silicon P-cores so we don't starve the E-cores.
    num_workers = _optimal_num_workers()
    logger.info("DataLoader num_workers=%d (Apple Silicon P-cores)", num_workers)
    train_loader = DataLoader(
        train_dataset,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=False,
        collate_fn=collate_fn,
        batch_sampler=train_sampler,
        persistent_workers=True,
        prefetch_factor=8,
    )

    mdl = hps.copy().model
    del mdl.use_spectral_norm
    if hps.if_f0 == 1:
        net_g = RVC_Model_f0(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            **mdl,
            sr=hps.sample_rate,
        )
    else:
        net_g = RVC_Model_nof0(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            **mdl,
        )
    net_g = net_g.to(device)
    net_g = _maybe_compile(net_g, "net_g")

    # has_xpu is kept in the Discriminator signature for checkpoint
    # compatibility with upstream, but on macOS it's always False.
    net_d = MultiPeriodDiscriminator(
        hps.version,
        use_spectral_norm=hps.model.use_spectral_norm,
        has_xpu=False,
    )
    net_d = net_d.to(device)
    net_d = _maybe_compile(net_d, "net_d")

    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )

    # No DDP on macOS — operate on the raw modules.

    try:  # 如果能加载自动resume
        _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d, optim_d
        )
        if rank == 0:
            logger.info("loaded D")
        _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g
        )
        global_step = (epoch_str - 1) * len(train_loader)
    except Exception:  # first run or missing checkpoint → load pretrain
        epoch_str = 1
        global_step = 0
        if hps.pretrainG != "":
            if rank == 0:
                logger.info("loaded pretrained %s" % (hps.pretrainG))
            target = net_g.module if hasattr(net_g, "module") else net_g
            logger.info(
                target.load_state_dict(
                    torch.load(hps.pretrainG, map_location="cpu", weights_only=True)[
                        "model"
                    ]
                )
            )
        if hps.pretrainD != "":
            if rank == 0:
                logger.info("loaded pretrained %s" % (hps.pretrainD))
            target = net_d.module if hasattr(net_d, "module") else net_d
            logger.info(
                target.load_state_dict(
                    torch.load(hps.pretrainD, map_location="cpu", weights_only=True)[
                        "model"
                    ]
                )
            )

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )

    # fp16_run is forced off on macOS by Config.use_fp32_config(); GradScaler
    # with enabled=False is a no-op regardless of backend.
    scaler = GradScaler(enabled=bool(hps.train.fp16_run))

    cache = []
    for epoch in range(epoch_str, hps.train.epochs + 1):
        train_and_evaluate(
            rank,
            epoch,
            hps,
            [net_g, net_d],
            [optim_g, optim_d],
            [scheduler_g, scheduler_d],
            scaler,
            [train_loader, None],
            logger if rank == 0 else None,
            [writer, writer_eval] if rank == 0 else None,
            cache,
            device,
        )
        scheduler_g.step()
        scheduler_d.step()


def _to_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Move a tensor to device. non_blocking is only useful with pinned CUDA."""
    return tensor.to(device, non_blocking=False)


def train_and_evaluate(
    rank,
    epoch,
    hps,
    nets: Tuple[RVC_Model_f0, MultiPeriodDiscriminator],
    optims,
    schedulers,
    scaler,
    loaders,
    logger,
    writers,
    cache,
    device: torch.device,
):
    net_g, net_d = nets
    optim_g, optim_d = optims
    train_loader, eval_loader = loaders
    if writers is not None:
        writer, writer_eval = writers

    train_loader.batch_sampler.set_epoch(epoch)
    global global_step

    net_g.train()
    net_d.train()

    # Prepare data iterator
    if hps.if_cache_data_in_gpu == True:
        data_iterator = cache
        if cache == []:
            for batch_idx, info in enumerate(train_loader):
                if hps.if_f0 == 1:
                    (
                        phone,
                        phone_lengths,
                        pitch,
                        pitchf,
                        spec,
                        spec_lengths,
                        wave,
                        wave_lengths,
                        sid,
                    ) = info
                else:
                    (
                        phone,
                        phone_lengths,
                        spec,
                        spec_lengths,
                        wave,
                        wave_lengths,
                        sid,
                    ) = info
                # Move to device (MPS or CPU)
                phone = _to_device(phone, device)
                phone_lengths = _to_device(phone_lengths, device)
                if hps.if_f0 == 1:
                    pitch = _to_device(pitch, device)
                    pitchf = _to_device(pitchf, device)
                sid = _to_device(sid, device)
                spec = _to_device(spec, device)
                spec_lengths = _to_device(spec_lengths, device)
                wave = _to_device(wave, device)
                wave_lengths = _to_device(wave_lengths, device)
                if hps.if_f0 == 1:
                    cache.append(
                        (
                            batch_idx,
                            (
                                phone,
                                phone_lengths,
                                pitch,
                                pitchf,
                                spec,
                                spec_lengths,
                                wave,
                                wave_lengths,
                                sid,
                            ),
                        )
                    )
                else:
                    cache.append(
                        (
                            batch_idx,
                            (
                                phone,
                                phone_lengths,
                                spec,
                                spec_lengths,
                                wave,
                                wave_lengths,
                                sid,
                            ),
                        )
                    )
        else:
            shuffle(cache)
    else:
        data_iterator = enumerate(train_loader)

    # Run steps
    epoch_recorder = EpochRecorder()
    for batch_idx, info in data_iterator:
        pitch = pitchf = None
        if hps.if_f0 == 1:
            (
                phone,
                phone_lengths,
                pitch,
                pitchf,
                spec,
                spec_lengths,
                wave,
                wave_lengths,
                sid,
            ) = info
        else:
            phone, phone_lengths, spec, spec_lengths, wave, wave_lengths, sid = info

        # Only move to device when not already cached there.
        if hps.if_cache_data_in_gpu == False:
            phone = _to_device(phone, device)
            phone_lengths = _to_device(phone_lengths, device)
            if hps.if_f0 == 1:
                pitch = _to_device(pitch, device)
                pitchf = _to_device(pitchf, device)
            sid = _to_device(sid, device)
            spec = _to_device(spec, device)
            spec_lengths = _to_device(spec_lengths, device)
            wave = _to_device(wave, device)

        with autocast(enabled=bool(hps.train.fp16_run)):
            (
                y_hat,
                ids_slice,
                x_mask,
                z_mask,
                (z, z_p, m_p, logs_p, m_q, logs_q),
            ) = net_g(phone, phone_lengths, spec, spec_lengths, sid, pitch, pitchf)
            mel = spec_to_mel_torch(
                spec,
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )
            y_mel = slice_on_last_dim(
                mel, ids_slice, hps.train.segment_size // hps.data.hop_length
            )
            with autocast(enabled=False):
                y_hat_mel = mel_spectrogram_torch(
                    y_hat.float().squeeze(1),
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.hop_length,
                    hps.data.win_length,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax,
                )
            if hps.train.fp16_run == True:
                y_hat_mel = y_hat_mel.half()
            wave = slice_on_last_dim(
                wave, ids_slice * hps.data.hop_length, hps.train.segment_size
            )

            # Discriminator
            y_d_hat_r, y_d_hat_g, _, _ = net_d(wave, y_hat.detach())
            with autocast(enabled=False):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                    y_d_hat_r, y_d_hat_g
                )
        optim_d.zero_grad()
        scaler.scale(loss_disc).backward()
        scaler.unscale_(optim_d)
        grad_norm_d = total_grad_norm(net_d.parameters())
        scaler.step(optim_d)

        with autocast(enabled=bool(hps.train.fp16_run)):
            # Generator
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(wave, y_hat)
            with autocast(enabled=False):
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, losses_gen = generator_loss(y_d_hat_g)
                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl
        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        grad_norm_g = total_grad_norm(net_g.parameters())
        scaler.step(optim_g)
        scaler.update()

        if rank == 0:
            if global_step % hps.train.log_interval == 0:
                lr = optim_g.param_groups[0]["lr"]
                logger.info(
                    "Train Epoch: {} [{:.0f}%]".format(
                        epoch, 100.0 * batch_idx / len(train_loader)
                    )
                )
                if loss_mel > 75:
                    loss_mel = 75
                if loss_kl > 9:
                    loss_kl = 9

                logger.info([global_step, lr])
                logger.info(
                    f"loss_disc={loss_disc:.3f}, loss_gen={loss_gen:.3f}, loss_fm={loss_fm:.3f},loss_mel={loss_mel:.3f}, loss_kl={loss_kl:.3f}"
                )
                scalar_dict = {
                    "loss/g/total": loss_gen_all,
                    "loss/d/total": loss_disc,
                    "learning_rate": lr,
                    "grad_norm_d": grad_norm_d,
                    "grad_norm_g": grad_norm_g,
                }
                scalar_dict.update(
                    {
                        "loss/g/fm": loss_fm,
                        "loss/g/mel": loss_mel,
                        "loss/g/kl": loss_kl,
                    }
                )
                scalar_dict.update(
                    {"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)}
                )
                scalar_dict.update(
                    {"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)}
                )
                scalar_dict.update(
                    {"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)}
                )
                image_dict = {
                    "slice/mel_org": utils.plot_spectrogram_to_numpy(
                        y_mel[0].data.cpu().numpy()
                    ),
                    "slice/mel_gen": utils.plot_spectrogram_to_numpy(
                        y_hat_mel[0].data.cpu().numpy()
                    ),
                    "all/mel": utils.plot_spectrogram_to_numpy(
                        mel[0].data.cpu().numpy()
                    ),
                }
                utils.summarize(
                    writer=writer,
                    global_step=global_step,
                    images=image_dict,
                    scalars=scalar_dict,
                )
        global_step += 1

    if epoch % hps.save_every_epoch == 0 and rank == 0:
        if hps.if_latest == 0:
            utils.save_checkpoint(
                net_g,
                optim_g,
                hps.train.learning_rate,
                epoch,
                os.path.join(hps.model_dir, "G_{}.pth".format(global_step)),
            )
            utils.save_checkpoint(
                net_d,
                optim_d,
                hps.train.learning_rate,
                epoch,
                os.path.join(hps.model_dir, "D_{}.pth".format(global_step)),
            )
        else:
            utils.save_checkpoint(
                net_g,
                optim_g,
                hps.train.learning_rate,
                epoch,
                os.path.join(hps.model_dir, "G_latest.pth"),
            )
            utils.save_checkpoint(
                net_d,
                optim_d,
                hps.train.learning_rate,
                epoch,
                os.path.join(hps.model_dir, "D_latest.pth"),
            )
        if rank == 0 and hps.save_every_weights == "1":
            ckpt_src = net_g.module if hasattr(net_g, "module") else net_g
            ckpt = ckpt_src.state_dict()
            logger.info(
                "saving ckpt %s_e%s:%s"
                % (
                    hps.name,
                    epoch,
                    save_small_model(
                        ckpt,
                        hps.sample_rate,
                        hps.if_f0,
                        hps.name + "_e%s_s%s" % (epoch, global_step),
                        epoch,
                        hps.version,
                        hps,
                    ),
                )
            )

    if rank == 0:
        logger.info("====> Epoch: {} {}".format(epoch, epoch_recorder.record()))
    if epoch >= hps.total_epoch and rank == 0:
        logger.info("Training is done. The program is closed.")

        ckpt_src = net_g.module if hasattr(net_g, "module") else net_g
        ckpt = ckpt_src.state_dict()
        logger.info(
            "saving final ckpt:%s"
            % (
                save_small_model(
                    ckpt, hps.sample_rate, hps.if_f0, hps.name, epoch, hps.version, hps
                )
            )
        )
        sleep(1)
        os._exit(0)


if __name__ == "__main__":
    # macOS requires 'spawn' to avoid fork-safety issues with PyTorch + DataLoader
    # workers. Call it defensively even though we no longer spawn child processes
    # for DDP — the DataLoader workers still benefit.
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
