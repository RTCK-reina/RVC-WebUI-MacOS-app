import os
import sys
import logging
from typing import Tuple

logger = logging.getLogger(__name__)
logging.getLogger("numba").setLevel(logging.WARNING)

now_dir = os.getcwd()
sys.path.append(os.path.join(now_dir))
# train.py lives at infer/modules/train/train.py; walk up 3 dirs to reach the
# package root so `from infer.lib…` resolves when rpc_training launches us
# with cwd=user_dir (the writable logs dir).
_pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

import datetime

from infer.lib.train import utils

hps = utils.get_hparams()
from random import randint, shuffle

import torch
from contextlib import nullcontext


# macOS (MPS/CPU) has no AMP/GradScaler — fp16 training is disabled everywhere
# in this port. These stubs preserve the call-site shape used by the training
# loop without pulling in any CUDA-specific code paths.
def autocast(enabled=False):
    return nullcontext()


class GradScaler:
    def __init__(self, enabled=False):
        self.enabled = enabled

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        pass

    def step(self, optimizer):
        return optimizer.step()

    def update(self):
        pass


from time import sleep
from time import time as ttime

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
# tensorboard is intentionally not shipped in the macOS .app bundle
# (requirements/app.txt drops tensorboard* to keep the env lean). We still
# want the upstream training loop to run unmodified, so provide a no-op
# SummaryWriter stub that swallows every call used by the training loop.
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    class SummaryWriter:  # type: ignore[no-redef]
        """No-op tensorboard SummaryWriter for bundles without tensorboard."""

        def __init__(self, *args, **kwargs):
            pass

        def add_scalar(self, *args, **kwargs):
            pass

        def add_scalars(self, *args, **kwargs):
            pass

        def add_image(self, *args, **kwargs):
            pass

        def add_images(self, *args, **kwargs):
            pass

        def add_audio(self, *args, **kwargs):
            pass

        def add_histogram(self, *args, **kwargs):
            pass

        def add_text(self, *args, **kwargs):
            pass

        def add_figure(self, *args, **kwargs):
            pass

        def add_graph(self, *args, **kwargs):
            pass

        def flush(self):
            pass

        def close(self):
            pass

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


def _select_device():
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


_TRAIN_DEVICE = _select_device()
_IS_MPS = _TRAIN_DEVICE == "mps"


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
    # macOS: always a single training process. MPS if available, CPU otherwise.
    if not _IS_MPS:
        print("MPS not available: falling back to CPU - this will be slow.")
    n_gpus = 1
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(randint(20000, 55555))
    children = []
    logger = utils.get_logger(hps.model_dir)
    for i in range(n_gpus):
        subproc = mp.Process(
            target=run,
            args=(i, n_gpus, hps, logger),
        )
        children.append(subproc)
        subproc.start()

    for i in range(n_gpus):
        children[i].join()


def run(rank, n_gpus, hps: utils.HParams, logger: logging.Logger):
    global global_step
    if rank == 0:
        # logger = utils.get_logger(hps.model_dir)
        logger.info(hps)
        # utils.check_git_hash(hps.model_dir)
        writer = SummaryWriter(log_dir=hps.model_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

    try:
        dist.init_process_group(
            backend="gloo",
            init_method="env://",
            world_size=n_gpus,
            rank=rank,
        )
    except:
        dist.init_process_group(
            backend="gloo",
            init_method="env://?use_libuv=False",
            world_size=n_gpus,
            rank=rank,
        )
    torch.manual_seed(hps.train.seed)
    # MPS has unreliable fp16 support; CPU has none. Force fp32 so
    # GradScaler/autocast stay no-ops regardless of the config default.
    hps.train.fp16_run = False

    if hps.if_f0 == 1:
        train_dataset = TextAudioLoaderMultiNSFsid(hps.data.training_files, hps.data)
    else:
        train_dataset = TextAudioLoader(hps.data.training_files, hps.data)
    train_sampler = DistributedBucketSampler(
        train_dataset,
        hps.train.batch_size * n_gpus,
        # [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200,1400],  # 16s
        [100, 200, 300, 400, 500, 600, 700, 800, 900],  # 16s
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True,
    )
    # It is possible that dataloader's workers are out of shared memory. Please try to raise your shared memory limit.
    # num_workers=8 -> num_workers=4
    if hps.if_f0 == 1:
        collate_fn = TextAudioCollateMultiNSFsid()
    else:
        collate_fn = TextAudioCollate()
    train_loader = DataLoader(
        train_dataset,
        num_workers=4,
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
    if _IS_MPS:
        net_g = net_g.to("mps")
    has_xpu = bool(hasattr(torch, "xpu") and torch.xpu.is_available())
    net_d = MultiPeriodDiscriminator(
        hps.version,
        use_spectral_norm=hps.model.use_spectral_norm,
        has_xpu=has_xpu,
    )
    if _IS_MPS:
        net_d = net_d.to("mps")
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
    # DDP wrapping removed for single-process macOS training: world_size=1
    # has no peers to all-gather with, and MPS does not implement
    # c10d::allgather_ (pytorch#141287), so `DDP(net_g)` crashed during
    # _verify_param_shape_across_processes with
    # "NotImplementedError: The distributed operator 'c10d::allgather_' is
    # not currently implemented for the MPS device" — distributed ops cannot
    # fall back to CPU. Operate on the raw modules instead; the
    # `hasattr(net_g, "module")` guards below fall through to the non-DDP
    # branch (plain `net_g.load_state_dict`), which is the desired behavior.

    try:  # 如果能加载自动resume
        _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d, optim_d
        )  # D多半加载没事
        if rank == 0:
            logger.info("loaded D")
        # _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g,load_opt=0)
        _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g
        )
        global_step = (epoch_str - 1) * len(train_loader)
        # epoch_str = 1
        # global_step = 0
    except:  # 如果首次不能加载，加载pretrain
        # traceback.print_exc()
        epoch_str = 1
        global_step = 0
        if hps.pretrainG != "":
            if rank == 0:
                logger.info("loaded pretrained %s" % (hps.pretrainG))
            if hasattr(net_g, "module"):
                logger.info(
                    net_g.module.load_state_dict(
                        torch.load(
                            hps.pretrainG, map_location="cpu", weights_only=True
                        )["model"]
                    )
                )  ##测试不加载优化器
            else:
                logger.info(
                    net_g.load_state_dict(
                        torch.load(
                            hps.pretrainG, map_location="cpu", weights_only=True
                        )["model"]
                    )
                )  ##测试不加载优化器
        if hps.pretrainD != "":
            if rank == 0:
                logger.info("loaded pretrained %s" % (hps.pretrainD))
            if hasattr(net_d, "module"):
                logger.info(
                    net_d.module.load_state_dict(
                        torch.load(
                            hps.pretrainD, map_location="cpu", weights_only=True
                        )["model"]
                    )
                )
            else:
                logger.info(
                    net_d.load_state_dict(
                        torch.load(
                            hps.pretrainD, map_location="cpu", weights_only=True
                        )["model"]
                    )
                )

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )

    scaler = GradScaler(enabled=hps.train.fp16_run)

    cache = []
    for epoch in range(epoch_str, hps.train.epochs + 1):
        if rank == 0:
            train_and_evaluate(
                rank,
                epoch,
                hps,
                [net_g, net_d],
                [optim_g, optim_d],
                [scheduler_g, scheduler_d],
                scaler,
                [train_loader, None],
                logger,
                [writer, writer_eval],
                cache,
            )
        else:
            train_and_evaluate(
                rank,
                epoch,
                hps,
                [net_g, net_d],
                [optim_g, optim_d],
                [scheduler_g, scheduler_d],
                scaler,
                [train_loader, None],
                None,
                None,
                cache,
            )
        scheduler_g.step()
        scheduler_d.step()


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
        # Use Cache
        data_iterator = cache
        if cache == []:
            # Make new cache
            for batch_idx, info in enumerate(train_loader):
                # Unpack
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
                # Load on MPS (or stay on CPU)
                if _IS_MPS:
                    phone = phone.to("mps", non_blocking=True)
                    phone_lengths = phone_lengths.to("mps", non_blocking=True)
                    if hps.if_f0 == 1:
                        pitch = pitch.to("mps", non_blocking=True)
                        pitchf = pitchf.to("mps", non_blocking=True)
                    sid = sid.to("mps", non_blocking=True)
                    spec = spec.to("mps", non_blocking=True)
                    spec_lengths = spec_lengths.to("mps", non_blocking=True)
                    wave = wave.to("mps", non_blocking=True)
                    wave_lengths = wave_lengths.to("mps", non_blocking=True)
                # Cache on list
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
            # Load shuffled cache
            shuffle(cache)
    else:
        # Loader
        data_iterator = enumerate(train_loader)

    # Run steps
    epoch_recorder = EpochRecorder()
    for batch_idx, info in data_iterator:
        # Data
        ## Unpack
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
        ## Load on MPS (or stay on CPU)
        if (hps.if_cache_data_in_gpu == False) and _IS_MPS:
            phone = phone.to("mps", non_blocking=True)
            phone_lengths = phone_lengths.to("mps", non_blocking=True)
            if hps.if_f0 == 1:
                pitch = pitch.to("mps", non_blocking=True)
                pitchf = pitchf.to("mps", non_blocking=True)
            sid = sid.to("mps", non_blocking=True)
            spec = spec.to("mps", non_blocking=True)
            spec_lengths = spec_lengths.to("mps", non_blocking=True)
            wave = wave.to("mps", non_blocking=True)

        # Calculate
        with autocast(enabled=hps.train.fp16_run):
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
            )  # slice

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

        with autocast(enabled=hps.train.fp16_run):
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
                # Amor For Tensorboard display
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
    # /Run steps

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
            if hasattr(net_g, "module"):
                ckpt = net_g.module.state_dict()
            else:
                ckpt = net_g.state_dict()
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

        if hasattr(net_g, "module"):
            ckpt = net_g.module.state_dict()
        else:
            ckpt = net_g.state_dict()
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
    mp.set_start_method("spawn", force=True)
    main()
