from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import importlib
import os
import platform
from typing import Optional

import numpy as np


@dataclass
class SpeakerSimConfig:
    model_id: str = "speechbrain/spkrec-ecapa-voxceleb"
    sample_rate: int = 16000
    device: str = "cpu"


_SB_MODEL = None


def _load_model(cfg: SpeakerSimConfig):
    global _SB_MODEL
    if _SB_MODEL is not None:
        return _SB_MODEL

    # On Windows, HuggingFace Hub and some downstream libraries may attempt to
    # create symlinks/hardlinks when materializing a model directory.
    # That can fail with WinError 1314 when the current user lacks privileges.
    # Force a safe copy-based strategy and write to a user-writable cache.
    if platform.system().lower() == "windows":
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")

    try:
        import torch

        # SpeechBrain moved import paths across versions. Use importlib to keep
        # this optional and robust.
        try:
            mod = importlib.import_module("speechbrain.inference.speaker")
            EncoderClassifier = getattr(mod, "EncoderClassifier")
        except Exception:
            mod = importlib.import_module("speechbrain.pretrained")
            EncoderClassifier = getattr(mod, "EncoderClassifier")
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "speechbrain is required for speaker similarity metrics. "
            "Install it with: pip install speechbrain"
        ) from e

    run_opts = {"device": cfg.device}

    savedir = os.path.join(
        os.path.expanduser("~"),
        ".cache",
        "videodubbing",
        "speechbrain",
        cfg.model_id.replace("/", "--"),
    )
    os.makedirs(savedir, exist_ok=True)

    _SB_MODEL = EncoderClassifier.from_hparams(
        source=cfg.model_id,
        savedir=savedir,
        run_opts=run_opts,
    )

    # Make sure we never accidentally keep gradients around.
    _SB_MODEL.eval()
    if hasattr(torch, "set_grad_enabled"):
        torch.set_grad_enabled(False)

    return _SB_MODEL


def _load_mono_waveform(path: str, target_sr: int):
    try:
        import torch
        import torchaudio
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "torchaudio is required for speaker similarity metrics. "
            "Install it with: pip install torchaudio"
        ) from e

    wav, sr = torchaudio.load(path)
    if wav.ndim == 2 and wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if wav.ndim == 2:
        wav = wav.squeeze(0)

    if int(sr) != int(target_sr):
        wav = torchaudio.functional.resample(wav, int(sr), int(target_sr))

    # SpeechBrain expects float tensor [batch, time]
    wav = wav.to(dtype=torch.float32)
    return wav.unsqueeze(0), int(target_sr)


@lru_cache(maxsize=256)
def speaker_embedding(
    path: str,
    *,
    model_id: str = "speechbrain/spkrec-ecapa-voxceleb",
    sample_rate: int = 16000,
    device: str = "cpu",
) -> np.ndarray:
    """Compute a speaker embedding for an audio file.

    Cached by file path to keep metrics fast.
    """

    cfg = SpeakerSimConfig(
        model_id=model_id,
        sample_rate=int(sample_rate),
        device=str(device),
    )
    model = _load_model(cfg)

    wav_bt, _sr = _load_mono_waveform(path, cfg.sample_rate)

    emb = model.encode_batch(wav_bt)
    try:
        emb_np = emb.squeeze(0).detach().cpu().numpy()
    except Exception:
        emb_np = np.asarray(emb).squeeze(0)

    # Normalize to unit length (cosine similarity stable)
    denom = float(np.linalg.norm(emb_np) + 1e-12)
    return (emb_np / denom).astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b) / denom)
