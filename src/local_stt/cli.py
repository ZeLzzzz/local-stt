"""Transcribe audio files locally with faster-whisper. Audio never leaves this machine."""

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path

QUIET = False


def log(*args, **kwargs) -> None:
    """Progress/status output, suppressed by -q."""
    if not QUIET:
        print(*args, **kwargs)


def fmt_time(seconds: float, srt: bool = False) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if srt:
        ms = int((seconds - int(seconds)) * 1000)
        return f"{h:02}:{m:02}:{s:02},{ms:03}"
    return f"{h:02}:{m:02}:{s:02}" if h else f"{m:02}:{s:02}"


def nvidia_lib_dirs() -> list[str]:
    """cuBLAS/cuDNN folders from the nvidia-* pip wheels, if installed."""
    spec = importlib.util.find_spec("nvidia")
    if not spec or not spec.submodule_search_locations:
        return []
    sub = "bin" if os.name == "nt" else "lib"
    return [
        str(Path(root) / pkg / sub)
        for root in spec.submodule_search_locations
        for pkg in ("cublas", "cudnn")
        if (Path(root) / pkg / sub).is_dir()
    ]


def ensure_cuda_libs() -> None:
    """Make the pip-installed CUDA libraries visible to CTranslate2."""
    dirs = nvidia_lib_dirs()
    if not dirs:
        return
    if os.name == "nt":
        for d in dirs:
            os.add_dll_directory(d)
        os.environ["PATH"] = os.pathsep.join([*dirs, os.environ.get("PATH", "")])
        return
    # The Linux loader reads LD_LIBRARY_PATH only at startup, so re-exec once with it set.
    current = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if p]
    if all(d in current for d in dirs):
        return
    os.environ["LD_LIBRARY_PATH"] = ":".join([*dirs, *current])
    os.execv(sys.executable, [sys.executable, "-m", "local_stt", *sys.argv[1:]])


def open_model(name: str, device: str, compute_type: str):
    from faster_whisper import WhisperModel

    # Use the local cache without contacting Hugging Face; only download when missing.
    try:
        return WhisperModel(name, device=device, compute_type=compute_type, local_files_only=True)
    except FileNotFoundError:
        log(f"Model {name} belum ada di cache, mengunduh sekali...")
        return WhisperModel(name, device=device, compute_type=compute_type)


def load_model(name: str, cpu: bool):
    if not cpu:
        try:
            return open_model(name, "cuda", "int8_float16"), "GPU"
        except Exception as e:  # noqa: BLE001 - any CUDA failure falls back to CPU
            print(f"GPU tidak bisa dipakai ({e}); pindah ke CPU.", file=sys.stderr)
    return open_model(name, "cpu", "int8"), "CPU"


def transcribe(model, path: Path, args) -> None:
    start = time.monotonic()
    segments, info = model.transcribe(
        str(path),
        language=None if args.lang == "auto" else args.lang,
        vad_filter=True,  # skip silences; reduces hallucinated text in long recordings
        beam_size=5,
    )
    duration = info.duration or 0
    if args.lang == "auto":
        log(f"  bahasa terdeteksi: {info.language} ({info.language_probability:.0%})")

    txt_lines, srt_blocks = [], []
    for i, seg in enumerate(segments, 1):
        text = seg.text.strip()
        txt_lines.append(text)
        srt_blocks.append(f"{i}\n{fmt_time(seg.start, True)} --> {fmt_time(seg.end, True)}\n{text}\n")
        pct = min(seg.end / duration, 1) if duration else 0
        log(f"\r  [{pct:6.1%}] {fmt_time(seg.end)} / {fmt_time(duration)}", end="", flush=True)
        if args.verbose:
            log(f"\n  {text}", end="")
    log(f"\r  [{1:6.1%}] {fmt_time(duration)} / {fmt_time(duration)}")

    out_dir = args.output or path.parent
    outputs = []
    if args.format in ("txt", "both"):
        out = out_dir / f"{path.stem}.txt"
        out.write_text("\n".join(txt_lines) + "\n", encoding="utf-8")
        outputs.append(out)
    if args.format in ("srt", "both"):
        out = out_dir / f"{path.stem}.srt"
        out.write_text("\n".join(srt_blocks), encoding="utf-8")
        outputs.append(out)

    elapsed = time.monotonic() - start
    speed = f", {duration / elapsed:.1f}x real-time" if elapsed else ""
    log(f"  selesai dalam {fmt_time(elapsed)}{speed}")
    for out in outputs:
        # In quiet mode the result paths are the only output, one per line.
        print(out.resolve() if QUIET else f"  → {out}")


def main() -> int:
    p = argparse.ArgumentParser(
        prog="stt",
        description="Transkripsi audio secara lokal dengan Whisper (faster-whisper).",
    )
    p.add_argument("files", nargs="+", type=Path, help="file audio/video (format apa pun yang dibaca ffmpeg)")
    p.add_argument("-l", "--lang", default="id", help="kode bahasa, atau 'auto' untuk deteksi (bawaan: id)")
    p.add_argument("-m", "--model", default="large-v3-turbo", help="model Whisper (bawaan: large-v3-turbo)")
    p.add_argument("-f", "--format", choices=["txt", "srt", "both"], default="txt", help="format hasil (bawaan: txt)")
    p.add_argument("--cpu", action="store_true", help="paksa pakai CPU")
    p.add_argument("-o", "--output", type=Path, help="folder hasil (bawaan: folder yang sama dengan audio)")
    p.add_argument("-q", "--quiet", action="store_true", help="senyap: hanya cetak path file hasil (untuk script/agent)")
    p.add_argument("-v", "--verbose", action="store_true", help="tampilkan teks saat diproses")
    args = p.parse_args()

    global QUIET
    QUIET = args.quiet
    if not args.cpu:
        ensure_cuda_libs()
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)

    missing = [f for f in args.files if not f.is_file()]
    for f in missing:
        print(f"stt: file tidak ditemukan: {f}", file=sys.stderr)
    files = [f for f in args.files if f.is_file()]
    if not files:
        return 1

    log(f"Memuat model {args.model}...")
    model, device = load_model(args.model, args.cpu)
    log(f"Model siap ({device}).")

    failed = 0
    for n, path in enumerate(files, 1):
        log(f"[{n}/{len(files)}] {path}")
        try:
            transcribe(model, path, args)
        except Exception as e:  # noqa: BLE001 - keep going with the remaining files
            failed += 1
            print(f"\nstt: gagal memproses {path}: {e}", file=sys.stderr)
    return 1 if failed or missing else 0


def run() -> None:
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nDibatalkan.", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    run()
