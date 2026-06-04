import os
import traceback
import logging
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

from infer.lib.audio import resample_audio, get_audio_properties
import torch

from configs import Config
from infer.modules.uvr5.mdxnet import MDXNetDereverb
from infer.modules.uvr5.vr import AudioPre

config = Config()

# File extensions to accept as audio input.
_AUDIO_EXTS = {
    ".wav",
    ".mp3",
    ".flac",
    ".m4a",
    ".ogg",
    ".opus",
    ".aac",
    ".aiff",
    ".aif",
    ".wma",
    ".mp4",
    ".mov",
}


def _is_audio_file(path: str, *, strict: bool = False) -> bool:
    """Return True if `path` looks like a processable audio file.

    strict=True  -- directory scanning. Also skips our own .reformatted.wav
                    temp files to avoid infinite-loop reprocessing when the
                    user points the input dir at TEMP.
    strict=False -- user-selected file picker paths. Only rejects obviously
                    non-audio entries (.DS_Store, wrong extension). Files
                    named "vocal_*" / "instrument_*" are ALLOWED because
                    chaining (extract → dereverb → extract again) is a
                    legitimate workflow the UI explicitly recommends.
    """
    name = os.path.basename(path)
    if not name or name.startswith("."):
        return False  # hidden files (.DS_Store, ._*, etc.)
    if strict and name.endswith(".reformatted.wav"):
        return False
    ext = os.path.splitext(name)[1].lower()
    return ext in _AUDIO_EXTS


def uvr(
    model_name,
    inp_root,
    save_root_vocal,
    paths,
    save_root_ins,
    agg,
    format0,
    cancel_event=None,
):
    infos = []
    try:
        inp_root = inp_root.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        save_root_vocal = (
            save_root_vocal.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        )
        save_root_ins = (
            save_root_ins.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        )
        if model_name == "onnx_dereverb_By_FoxJoy":
            pre_fun = MDXNetDereverb(15, config.device)
        else:
            pre_fun = AudioPre(
                agg=int(agg),
                model_path=os.path.join(
                    os.getenv("weight_uvr5_root"), model_name + ".pth"
                ),
                device=config.device,
                is_half=config.is_half,
            )
        if inp_root != "":
            # Directory scan: use strict filtering to avoid pulling our
            # own temp files back into the processing queue.
            paths = [
                os.path.join(inp_root, name)
                for name in os.listdir(inp_root)
                if _is_audio_file(name, strict=True)
            ]
        else:
            # Individual files selected by the user: lenient filtering.
            paths = [p.name if hasattr(p, "name") else p for p in paths]
            paths = [p for p in paths if _is_audio_file(p, strict=False)]
        if not paths:
            infos.append("No valid audio files to process.")
            yield "\n".join(infos)
            return

        # 絶対パスに揃える
        abs_paths = []
        for path in paths:
            if inp_root and not os.path.isabs(path):
                abs_paths.append(os.path.join(inp_root, path))
            else:
                abs_paths.append(path)

        def _preprocess_file(inp_path):
            """ファイルを 44100Hz/ステレオ に変換する。

            変換不要なら元パスをそのまま返す。変換した場合は一時パスを返す。
            戻り値: (処理済みパス, 一時ファイルであるか)
            resample_audio は PyAV ベースの I/O バウンド処理なので
            ThreadPoolExecutor で安全に並列化できる。
            """
            try:
                channels, rate = get_audio_properties(inp_path)
                if channels == 2 and rate == 44100:
                    return inp_path, False  # 変換不要
            except Exception as e:
                logger.warning(f"Exception {e} occurred. Will reformat {inp_path}")

            tmp_path = "%s/%s.reformatted.wav" % (
                os.path.join(os.environ["TEMP"]),
                os.path.basename(inp_path),
            )
            resample_audio(inp_path, tmp_path, "pcm_s16le", "s16", 44100, "stereo")
            # NOTE: do NOT remove inp_path here — it may point at a user-owned
            # file outside the app's temp space. We keep the reformatted copy in
            # TEMP and leave the original untouched.
            return tmp_path, True

        # I/O バウンドなフォーマット変換を ThreadPoolExecutor で並列化。
        # モデル推論（GPU/MPS）はシリアルで安全に実行する。
        n_workers = min(len(abs_paths), 4)
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            preproc_results = list(pool.map(_preprocess_file, abs_paths))

        # モデル推論（シリアル）
        for inp_path, was_reformatted in preproc_results:
            if cancel_event is not None and cancel_event.is_set():
                infos.append("Cancelled.")
                yield "\n".join(infos)
                return
            try:
                pre_fun._path_audio_(inp_path, save_root_ins, save_root_vocal, format0)
                infos.append("%s->Success" % (os.path.basename(inp_path)))
                yield "\n".join(infos)
            except:
                infos.append(
                    "%s->%s" % (os.path.basename(inp_path), traceback.format_exc())
                )
                yield "\n".join(infos)
            finally:
                if was_reformatted and os.path.exists(inp_path):
                    try:
                        os.remove(inp_path)
                    except Exception as e:
                        logger.warning(
                            "Failed to remove temporary file %s: %s", inp_path, e
                        )
    except:
        infos.append(traceback.format_exc())
        yield "\n".join(infos)
    finally:
        try:
            if model_name == "onnx_dereverb_By_FoxJoy":
                del pre_fun.pred.model
                del pre_fun.pred.model_
            else:
                del pre_fun.model
                del pre_fun
        except:
            traceback.print_exc()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Executed torch.cuda.empty_cache()")
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
            logger.info("Executed torch.mps.empty_cache()")
    yield "\n".join(infos)
