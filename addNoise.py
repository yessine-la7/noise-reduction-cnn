#################################### add 2 noises
import numpy as np
import soundfile as sf
import librosa
import os

# --- Pfade ---
BASE_DIR = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()

INPUT_AUDIO_DIR = os.path.join(BASE_DIR, "audios")
INPUT_AUDIO_NAME = "p232_232_clean.wav"
INPUT_AUDIO_PATH = os.path.join(INPUT_AUDIO_DIR, INPUT_AUDIO_NAME)

INPUT_NOISE_DIR = os.path.join(BASE_DIR, "noise")
INPUT_NOISE_NAME_1 = "airplane_edited.m4a"   # Noise 1
INPUT_NOISE_NAME_2 = "baby_cry.m4a"          # Noise 2
INPUT_NOISE_PATH_1 = os.path.join(INPUT_NOISE_DIR, INPUT_NOISE_NAME_1)
INPUT_NOISE_PATH_2 = os.path.join(INPUT_NOISE_DIR, INPUT_NOISE_NAME_2)

OUTPUT_AUDIO = os.path.join(
    INPUT_AUDIO_DIR,
    f"{os.path.splitext(INPUT_AUDIO_NAME)[0]}_addNoise2.wav"
)

# ---------------- Utils ----------------
def load_mono(path, sr=None):
    """
    Beliebige Audioformate (auch .m4a) als Mono-Float32 laden.
    Benötigt ein installiertes FFmpeg für Formate wie M4A.
    """
    y, file_sr = librosa.load(path, sr=sr, mono=True)
    y = np.asarray(y, dtype=np.float32)
    return y, (file_sr if sr is None else sr)

def save_wav(path, y, sr):
    y = np.clip(y, -1.0, 1.0)
    sf.write(path, y, sr)

def rms(x, eps=1e-12):
    return float(np.sqrt(np.mean(np.square(x), dtype=np.float64) + eps))

def db_to_lin(db):
    return float(10.0 ** (db / 20.0))

def hard_limiter(x, ceiling=0.98):
    peak = np.max(np.abs(x)) if x.size else 0.0
    return x * (ceiling / peak) if peak > ceiling and peak > 0 else x

# ---------------- Nicht-stationäre Hüllkurve ----------------
def random_level_envelope(n, sr, min_hold_ms=500, max_hold_ms=2500, min_lvl=0.3, max_lvl=1.0):
    """Stückweise konstante Zufallspegel, leicht geglättet -> nicht-stationärer Hintergrund."""
    env = np.zeros(n, dtype=np.float32)
    i = 0
    rng = np.random.default_rng()
    while i < n:
        hold = rng.integers(int(sr*min_hold_ms/1000), int(sr*max_hold_ms/1000)+1)
        lvl = rng.uniform(min_lvl, max_lvl)
        j = min(n, i + int(hold))
        env[i:j] = lvl
        i = j
    # weiche Glättung (einfaches Faltungsfenster ~20 ms)
    k = max(1, int(0.02 * sr))
    if k > 1:
        win = np.hanning(2*k+1).astype(np.float32)
        win = win / win.sum()
        env = np.convolve(env, win, mode="same").astype(np.float32)
    return env

def fade_io(env, sr, fade_ms=60):
    """Sanfte Ein- und Ausblendung für die Hüllkurve."""
    n = len(env)
    f = max(1, int(sr * fade_ms / 1000))
    ramp_in = np.linspace(0, 1, f, dtype=np.float32)
    ramp_out = np.linspace(1, 0, f, dtype=np.float32)
    env[:f] *= ramp_in
    env[-f:] *= ramp_out
    return env

# ---------------- Hilfsfunktion für einzelne Noise-Spur ----------------
def prepare_noise_track(noise_path: str,
                        sr: int,
                        speech_len: int,
                        speech_signal: np.ndarray,
                        snr_db: float,
                        start_sec: float,
                        make_nonstationary: bool,
                        fade_ms: int = 60) -> np.ndarray:
    """
    Lädt eine Noise-Datei, bringt sie auf Sprachlänge (mit Offset),
    macht sie optional nicht-stationär und skaliert auf Ziel-SNR gegen die Sprache.
    """
    noise, _ = load_mono(noise_path, sr=sr)

    # auf Länge bringen (kacheln/trimmen + Offset)
    need = speech_len + int(start_sec * sr)
    if len(noise) < need:
        reps = int(np.ceil(need / max(1, len(noise))))
        noise = np.tile(noise, reps)

    offset = max(0, int(start_sec * sr))
    noise = noise[offset:offset + speech_len].astype(np.float32)

    if make_nonstationary:
        env = random_level_envelope(speech_len, sr, min_hold_ms=500, max_hold_ms=2500, min_lvl=0.3, max_lvl=1.0)
        env = fade_io(env.copy(), sr, fade_ms=fade_ms)
        noise = noise * env

    # SNR setzen
    sig_rms = rms(speech_signal)
    noi_rms = rms(noise)
    if noi_rms > 0 and sig_rms > 0:
        target_noi_rms = sig_rms / db_to_lin(snr_db)
        noise *= (target_noi_rms / noi_rms)

    return noise

# ---------------- Kernfunktion ----------------
def mix_speech_with_two_noises(
    speech_path: str,
    noise_path_1: str,
    noise_path_2: str,
    out_wav: str,
    sr: int | None = 48000,
    # pro Noise eigener SNR und Startzeit
    snr_db_1: float = 15.0,
    snr_db_2: float = 15.0,
    noise1_start_sec: float = 0.0,
    noise2_start_sec: float = 0.0,
    make_nonstationary: bool = True,
    # optionales Ducking der GESAMTEN Noise-Summe (Sprache bleibt vorne)
    duck_db: float = 0.0,   # z.B. 6.0 für leichtes Ducking, 0.0 zum Deaktivieren
    seed: int | None = 42,
):
    if seed is not None:
        np.random.seed(seed)

    # 1) laden
    speech, sr = load_mono(speech_path, sr=sr)
    N = len(speech)

    # 2) beide Noise-Tracks vorbereiten
    noise1 = prepare_noise_track(
        noise_path=noise_path_1,
        sr=sr,
        speech_len=N,
        speech_signal=speech,
        snr_db=snr_db_1,
        start_sec=noise1_start_sec,
        make_nonstationary=make_nonstationary
    )

    noise2 = prepare_noise_track(
        noise_path=noise_path_2,
        sr=sr,
        speech_len=N,
        speech_signal=speech,
        snr_db=snr_db_2,
        start_sec=noise2_start_sec,
        make_nonstationary=make_nonstationary
    )

    # 3) Noise-Gesamtsumme (optional mit Ducking)
    noise_sum = noise1 + noise2

    if duck_db > 0.0:
        # einfache Onset-basierte Lautstärkehüllkurve der Sprache
        env_s = librosa.onset.onset_strength(y=speech, sr=sr, hop_length=256).astype(np.float32)
        env_s = np.repeat(env_s, 256)[:N] if env_s.size > 0 else np.zeros(N, dtype=np.float32)
        env_s = env_s / (np.max(env_s) + 1e-8)
        # mehr Sprache => weniger Noise
        noise_gain = 1.0 - (env_s * (1.0 - db_to_lin(-duck_db)))
        noise_sum *= noise_gain.astype(np.float32)

    # 4) Summe + Limiter
    out = speech + noise_sum
    out = hard_limiter(out, ceiling=0.98)

    save_wav(out_wav, out, sr)
    print(
        f"[OK] gespeichert: {out_wav} | sr={sr} | Dauer={N/sr:.2f}s | "
        f"SNRs≈({snr_db_1:.1f} dB, {snr_db_2:.1f} dB)"
    )

if __name__ == "__main__":
    mix_speech_with_two_noises(
        speech_path=INPUT_AUDIO_PATH,
        noise_path_1=INPUT_NOISE_PATH_1,
        noise_path_2=INPUT_NOISE_PATH_2,
        out_wav=OUTPUT_AUDIO,
        sr=48000,
        snr_db_1=15.0,            # 12–20 dB: Sprache klar vorn
        snr_db_2=18.0,            # du kannst die SNRs unabhängig setzen
        noise1_start_sec=0.0,     # Startversatz für Noise 1
        noise2_start_sec=1.5,     # Startversatz für Noise 2
        make_nonstationary=True,
        duck_db=0.0,              # 0 = aus, 6 = leichtes Ducking
        seed=42,
    )


















# ################################## add 1 noise
# import numpy as np
# import soundfile as sf
# import librosa
# import os


# # --- Pfade ---
# BASE_DIR = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()
# INPUT_AUDIO_DIR = os.path.join(os.path.dirname(__file__), "audios")
# INPUT_AUDIO_NAME = "p232_232_clean.wav"
# INPUT_AUDIO_PATH = os.path.join(INPUT_AUDIO_DIR, INPUT_AUDIO_NAME)

# INPUT_NOISE_DIR = os.path.join(os.path.dirname(__file__), "noise")
# INPUT_NOISE_NAME = "airplane_edited.m4a"
# INPUT_NOISE_PATH = os.path.join(INPUT_NOISE_DIR, INPUT_NOISE_NAME)

# OUTPUT_AUDIO = os.path.join(INPUT_AUDIO_DIR, f"{os.path.splitext(INPUT_AUDIO_NAME)[0]}_addNoise.wav")


# # ---------------- Utils ----------------
# def load_mono(path, sr=None):
#     y, file_sr = librosa.load(path, sr=sr, mono=True)
#     y = np.asarray(y, dtype=np.float32)
#     return y, (file_sr if sr is None else sr)

# def save_wav(path, y, sr):
#     y = np.clip(y, -1.0, 1.0)
#     sf.write(path, y, sr)

# def rms(x, eps=1e-12):
#     return float(np.sqrt(np.mean(np.square(x), dtype=np.float64) + eps))

# def db_to_lin(db):
#     return float(10.0 ** (db / 20.0))

# def hard_limiter(x, ceiling=0.98):
#     peak = np.max(np.abs(x)) if x.size else 0.0
#     return x * (ceiling / peak) if peak > ceiling and peak > 0 else x

# # ---------------- Nicht-stationäre Hüllkurve (einfach) ----------------
# def random_level_envelope(n, sr, min_hold_ms=500, max_hold_ms=2500, min_lvl=0.3, max_lvl=1.0):
#     """Stückweise konstante Zufallspegel, leicht geglättet -> nicht-stationärer Hintergrund."""
#     env = np.zeros(n, dtype=np.float32)
#     i = 0
#     rng = np.random.default_rng()
#     while i < n:
#         hold = rng.integers(int(sr*min_hold_ms/1000), int(sr*max_hold_ms/1000)+1)
#         lvl = rng.uniform(min_lvl, max_lvl)
#         j = min(n, i + int(hold))
#         env[i:j] = lvl
#         i = j
#     # weiche Glättung (einfaches Faltungsfenster)
#     k = max(1, int(0.02 * sr))  # ~20 ms
#     if k > 1:
#         win = np.hanning(2*k+1).astype(np.float32)
#         win = win / win.sum()
#         env = np.convolve(env, win, mode="same").astype(np.float32)
#     return env

# def fade_io(env, sr, fade_ms=60):
#     """Sanfte Ein- und Ausblendung für die Hüllkurve."""
#     n = len(env)
#     f = max(1, int(sr * fade_ms / 1000))
#     ramp_in = np.linspace(0, 1, f, dtype=np.float32)
#     ramp_out = np.linspace(1, 0, f, dtype=np.float32)
#     env[:f] *= ramp_in
#     env[-f:] *= ramp_out
#     return env

# # ---------------- Kernfunktion ----------------
# def mix_speech_with_noise(
#     speech: str,
#     noise: str,
#     out_wav: str,
#     sr: int | None = 48000,
#     snr_db: float = 15.0,          # Ziel-SNR: höher => leiseres Geräusch
#     make_nonstationary: bool = True,
#     duck_db: float = 0.0,          # z.B. 6.0 für leichtes Ducking, 0.0 zum Deaktivieren
#     noise_start_sec: float = 0.0,  # Startversatz des Geräusches relativ zur Sprache
#     seed: int | None = 42,
# ):
#     if seed is not None:
#         np.random.seed(seed)

#     # 1) laden
#     speech, sr = load_mono(speech, sr=sr)
#     noise, _   = load_mono(noise,  sr=sr)

#     N = len(speech)
#     # 2) Geräusch auf Sprachlänge bringen (kacheln/trimmen)
#     need = N + int(noise_start_sec * sr)
#     if len(noise) < need:
#         reps = int(np.ceil(need / max(1, len(noise))))
#         noise = np.tile(noise, reps)
#     # zeitlichen Offset anwenden und dann auf Länge schneiden
#     offset = max(0, int(noise_start_sec * sr))
#     noise = noise[offset:offset + N].astype(np.float32)

#     # 3) optional: nicht-stationäre Hüllkurve
#     if make_nonstationary:
#         env = random_level_envelope(N, sr, min_hold_ms=500, max_hold_ms=2500, min_lvl=0.3, max_lvl=1.0)
#         env = fade_io(env.copy(), sr, fade_ms=60)
#         noise = noise * env

#     # 4) SNR setzen
#     sig_rms = rms(speech)
#     noi_rms = rms(noise)
#     if noi_rms > 0:
#         target_noi_rms = sig_rms / db_to_lin(snr_db)
#         noise *= (target_noi_rms / noi_rms)

#     # 5) optional: einfaches Ducking (Noise etwas absenken, wenn Sprache laut ist)
#     if duck_db > 0.0:
#         # grobe Lautstärkehüllkurve der Sprache
#         env_s = librosa.onset.onset_strength(y=speech, sr=sr, hop_length=256).astype(np.float32)
#         env_s = np.repeat(env_s, 256)[:N] if env_s.size > 0 else np.zeros(N, dtype=np.float32)
#         env_s = env_s / (np.max(env_s) + 1e-8)
#         gain = 1.0 - (env_s * (1.0 - db_to_lin(-duck_db)))  # mehr Sprache => etwas weniger Noise
#         noise *= gain.astype(np.float32)

#     # 6) Summe + Limiter
#     out = speech + noise
#     out = hard_limiter(out, ceiling=0.98)

#     save_wav(out_wav, out, sr)
#     print(f"[OK] gespeichert: {out_wav} | sr={sr} | Dauer={N/sr:.2f}s | SNR~{snr_db:.1f} dB")


# if __name__ == "__main__":
#     mix_speech_with_noise(
#         speech=INPUT_AUDIO_PATH,
#         noise=INPUT_NOISE_PATH,
#         out_wav=OUTPUT_AUDIO,
#         sr=48000,
#         snr_db=15.0,                    # 12–20 dB: Sprache klar vorn
#         make_nonstationary=True,
#         duck_db=0.0,                    # 0 = aus, 6 = leichtes Ducking
#         noise_start_sec=0.0,
#         seed=42,
#     )
