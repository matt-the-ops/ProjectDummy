"""
bmo_voice.py — BMO RVC v2 Voice Engine
Converts text → Edge-TTS audio → RVC BMO voice using BMO.pth model.
Requires: torch, torchaudio, edge-tts, pygame, transformers,
          librosa, soundfile, faiss-cpu, praat-parselmouth, scipy
"""
import os, sys, io, time, asyncio, threading, logging
import numpy as np
import torch
import torch.nn.functional as F
import librosa
import soundfile as sf
import parselmouth
from scipy import signal
from pathlib import Path

logger = logging.getLogger(__name__)

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent
MODELS_DIR  = BASE_DIR / "models"
RVC_ENGINE  = BASE_DIR / "rvc_engine"
BMO_PTH     = MODELS_DIR / "BMO Voice" / "BMO.pth"
BMO_INDEX   = MODELS_DIR / "BMO Voice" / "added_IVF548_Flat_nprobe_1_BMO_v2.index"
HF_HUBERT   = "facebook/hubert-base-ls960"   # auto-downloaded on first run

sys.path.insert(0, str(BASE_DIR))

# ─── Device ───────────────────────────────────────────────────────────────────
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
USE_HALF  = DEVICE == "cuda"
dtype     = torch.float16 if USE_HALF else torch.float32

# ─── Lazy singletons ──────────────────────────────────────────────────────────
_hubert_model     = None
_hubert_extractor = None
_net_g            = None
_faiss_index      = None
_bmo_loaded       = False
_load_lock        = threading.Lock()

# ─── RVC Synthesizer (SynthesizerTrnMs768NSFsid v2) ──────────────────────────
def _load_synthesizer(cpt):
    """Load the SynthesizerTrnMs768NSFsid model from rvc_engine."""
    from rvc_engine.module.models import SynthesizerTrnMs768NSFsid
    cfg = cpt["config"]
    net_g = SynthesizerTrnMs768NSFsid(*cfg, is_half=USE_HALF)
    net_g.eval()
    net_g.load_state_dict(cpt["weight"], strict=False)
    net_g = net_g.to(dtype).to(DEVICE)
    return net_g

# ─── HuBERT feature extractor (Transformers, auto-download ~360 MB once) ─────
def _load_hubert():
    global _hubert_model, _hubert_extractor
    if _hubert_model is not None:
        return
    from transformers import HubertModel, AutoFeatureExtractor
    logger.info("[BMO Voice] Loading HuBERT feature extractor...")
    _hubert_extractor = AutoFeatureExtractor.from_pretrained(HF_HUBERT)
    _hubert_model = HubertModel.from_pretrained(HF_HUBERT, torch_dtype=dtype)
    _hubert_model = _hubert_model.to(DEVICE).eval()
    logger.info("[BMO Voice] HuBERT ready.")

def _extract_features(audio_16k: np.ndarray) -> torch.Tensor:
    """Return (1, T, 768) HuBERT hidden states for RVC v2."""
    inputs = _hubert_extractor(audio_16k, sampling_rate=16000,
                               return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(dtype).to(DEVICE)
    with torch.no_grad():
        out = _hubert_model(input_values=input_values,
                            output_hidden_states=False,
                            return_dict=True)
    return out.last_hidden_state   # (1, T, 768)

# ─── F0 (pitch) extraction via Parselmouth ────────────────────────────────────
def _get_f0_pm(audio_16k: np.ndarray, sr=16000, f0_up_key=0,
               f0_min=50, f0_max=1100):
    """Crepe-free F0 via Parselmouth; fast, no extra deps."""
    sound = parselmouth.Sound(audio_16k, sampling_frequency=sr)
    pitch = sound.to_pitch_ac(time_step=0.01, voicing_threshold=0.6,
                               pitch_floor=float(f0_min),
                               pitch_ceiling=float(f0_max))
    f0 = pitch.selected_array["frequency"]
    f0[f0 == 0] = np.nan
    f0 = np.interp(np.arange(len(f0)),
                   np.where(~np.isnan(f0))[0],
                   f0[~np.isnan(f0)]) if np.any(~np.isnan(f0)) else f0
    f0 = np.nan_to_num(f0)
    f0 *= 2 ** (f0_up_key / 12.0)
    # Quantize to 256-bin scale used by RVC
    f0_mel = 1127 * np.log(1 + f0 / 700.0)
    f0_mel_min = 1127 * np.log(1 + f0_min / 700.0)
    f0_mel_max = 1127 * np.log(1 + f0_max / 700.0)
    f0_mel_norm = np.where(
        f0_mel > 0,
        np.clip((f0_mel - f0_mel_min) / (f0_mel_max - f0_mel_min) * 254 + 1,
                1, 255),
        0,
    ).astype(np.int64)
    return f0.astype(np.float32), f0_mel_norm

# ─── Faiss index retrieval ─────────────────────────────────────────────────────
def _apply_index(feats: np.ndarray, ratio=0.75) -> np.ndarray:
    """Blend feats with nearest-neighbour vectors from BMO index."""
    if _faiss_index is None or ratio <= 0:
        return feats
    score, ix = _faiss_index.search(feats, k=8)
    weight = np.square(1 / (score + 1e-6))
    weight /= weight.sum(axis=1, keepdims=True)
    nv = np.einsum("ij,ijk->ik", weight,
                   _faiss_index.reconstruct_batch(ix.flatten()).reshape(*ix.shape, -1))
    return ratio * nv + (1 - ratio) * feats

# ─── Core voice conversion ─────────────────────────────────────────────────────
def _vc_audio(audio_src: np.ndarray, src_sr: int,
              f0_up_key: int = 0, index_ratio: float = 0.75) -> np.ndarray:
    """Convert source audio → BMO voice. Returns float32 array @ tgt_sr."""
    # 1. Resample source to 16 kHz for HuBERT
    audio16 = librosa.resample(audio_src, orig_sr=src_sr, target_sr=16000)
    audio16 = signal.filtfilt(*signal.butter(N=5, Wn=48, btype="high", fs=16000), audio16)
    audio16 = audio16.astype(np.float32)

    # 2. Extract HuBERT features
    feats = _extract_features(audio16)           # (1, T, 768)
    feats = feats[0].float().cpu().numpy()        # (T, 768)
    feats = _apply_index(feats, index_ratio)

    # 3. Upsample features to match synthesizer frame rate (2× interp)
    feats_t = torch.from_numpy(feats).to(dtype).to(DEVICE).unsqueeze(0).permute(0, 2, 1)
    feats_t = F.interpolate(feats_t, scale_factor=2, mode="nearest")
    phone   = feats_t.permute(0, 2, 1)           # (1, T', 768)

    # 4. F0
    tgt_sr = 40000
    f0_raw, f0_coarse = _get_f0_pm(audio16, 16000, f0_up_key)
    T = phone.shape[1]
    f0_raw    = np.interp(np.linspace(0, len(f0_raw)-1, T),
                          np.arange(len(f0_raw)), f0_raw)
    f0_coarse = np.interp(np.linspace(0, len(f0_coarse)-1, T),
                          np.arange(len(f0_coarse)), f0_coarse).astype(np.int64)
    f0_t  = torch.from_numpy(f0_raw).unsqueeze(0).to(dtype).to(DEVICE)
    f0c_t = torch.from_numpy(f0_coarse).unsqueeze(0).to(torch.long).to(DEVICE)
    lengths = torch.tensor([phone.shape[1]], device=DEVICE)

    # 5. Synthesizer inference
    with torch.no_grad():
        audio_out = _net_g.infer(phone, lengths, f0c_t, f0_t, None)[0][0, 0]
        audio_out = audio_out.float().cpu().numpy()

    return audio_out, tgt_sr

# ─── Public: load BMO models once ─────────────────────────────────────────────
def load_bmo_models():
    """Call once at startup to warm up GPU models."""
    global _net_g, _faiss_index, _bmo_loaded
    with _load_lock:
        if _bmo_loaded:
            return True
        if not BMO_PTH.exists():
            logger.error("[BMO Voice] BMO.pth not found at %s", BMO_PTH)
            return False
        try:
            _load_hubert()
            cpt = torch.load(str(BMO_PTH), map_location="cpu", weights_only=False)
            _net_g = _load_synthesizer(cpt)
            logger.info("[BMO Voice] BMO synthesizer loaded on %s.", DEVICE)

            if BMO_INDEX.exists():
                import faiss
                _faiss_index = faiss.read_index(str(BMO_INDEX))
                logger.info("[BMO Voice] FAISS index loaded (%d vectors).", _faiss_index.ntotal)
        except Exception as e:
            logger.error("[BMO Voice] Load failed: %s", e)
            return False
        _bmo_loaded = True
        return True

# ─── Public: text → BMO audio bytes (mp3 via edge-tts → RVC) ─────────────────
async def _synthesize_raw(text: str) -> bytes:
    """Generate base speech bytes (mp3) using edge-tts."""
    import edge_tts
    comm = edge_tts.Communicate(text, voice="en-US-AnaNeural",
                                pitch="+5Hz", rate="+8%")
    buf = bytearray()
    async for chunk in comm.stream():
        if chunk["type"] == "audio":
            buf.extend(chunk["data"])
    return bytes(buf)

def speak_bmo(text: str, f0_up_key: int = 0, index_ratio: float = 0.75) -> bool:
    """
    Full pipeline: text → edge-tts base voice → RVC BMO conversion → pygame playback.
    Returns True on success, False on failure.
    Should be called from a background thread (non-blocking caller).
    """
    try:
        import pygame
        # 1. Synthesize base TTS
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        mp3_bytes = loop.run_until_complete(_synthesize_raw(text))
        loop.close()

        # 2. Decode mp3 → float32 numpy
        audio_buf = io.BytesIO(mp3_bytes)
        src_audio, src_sr = librosa.load(audio_buf, sr=None, mono=True)

        # 3. RVC voice conversion to BMO
        out_audio, tgt_sr = _vc_audio(src_audio, src_sr, f0_up_key, index_ratio)

        # Normalise output
        peak = np.abs(out_audio).max()
        if peak > 0:
            out_audio /= peak
        out_audio = (out_audio * 32767).astype(np.int16)

        # 4. Playback via pygame
        if not pygame.get_init():
            pygame.mixer.init(frequency=tgt_sr, size=-16, channels=1, buffer=2048)

        sound_io = io.BytesIO()
        sf.write(sound_io, out_audio, tgt_sr, format="WAV", subtype="PCM_16")
        sound_io.seek(0)
        pygame.mixer.music.load(sound_io)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.04)
        return True

    except Exception as e:
        logger.error("[BMO Voice] speak_bmo error: %s", e, exc_info=True)
        return False
