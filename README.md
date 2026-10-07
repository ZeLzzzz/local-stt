# local-stt

Transcribe audio and video files on your own machine with [Whisper](https://github.com/openai/whisper), via [faster-whisper](https://github.com/SYSTRAN/faster-whisper). Nothing is uploaded: after the model is downloaded once, transcription runs fully offline.

Built for long recordings such as lectures and meetings, where sending the audio to an online transcription service is not an option.

## Install

Requires [uv](https://docs.astral.sh/uv/).

```sh
uv tool install git+https://github.com/ZeLzzzz/local-stt
```

This installs the `stt` command. On Linux and Windows with an NVIDIA GPU, the CUDA libraries come from pip packages, so no system-wide CUDA install is needed. Without a usable GPU, `stt` falls back to the CPU (slower). You do not need to install ffmpeg separately: PyAV bundles it.

Update or remove:

```sh
uv tool upgrade local-stt
uv tool uninstall local-stt
```

The first run downloads the model (`large-v3-turbo`, about 1.6 GB) from Hugging Face into its cache. Later runs load it from the cache without contacting the network.

## Usage

```sh
stt lecture.m4a                  # -> lecture.txt next to the audio
stt *.m4a                        # several files with one model load
stt -f srt lecture.m4a           # subtitles with timestamps (.srt); -f both for .txt and .srt
stt -o transcripts lecture.m4a   # write results to another folder (created if missing)
stt -l en lecture.m4a            # language (default: id); -l auto to detect
stt -m small lecture.m4a         # another Whisper model (default: large-v3-turbo)
stt -v lecture.m4a               # show text while transcribing
stt --cpu lecture.m4a            # force CPU
stt -q lecture.m4a               # quiet: print only the result paths
```

Any format ffmpeg can read works, including video files. Existing result files are overwritten.

The default language is Indonesian (`id`) and the CLI messages are in Indonesian, because that is what the author uses. Pass `-l <code>` for other languages.

### Use from scripts and AI agents

`-q` prints only the absolute path of each result file, one per line, on stdout. Errors go to stderr. The exit code is `0` when every file succeeded and `1` otherwise, so a script or coding agent can run `stt -q <file>` and read the resulting `.txt`.

## How it works

- Model: `large-v3-turbo` by default, `int8_float16` on the GPU (fits in 4 GB of VRAM) and `int8` on the CPU.
- Voice activity detection skips silence, which reduces hallucinated text on long recordings.
- On Linux, `stt` re-executes itself once with `LD_LIBRARY_PATH` pointing to the pip-installed cuBLAS and cuDNN. On Windows it adds them to the DLL search path.

## Notes

- `av` is pinned below 19 because faster-whisper 1.2.1 passes `metadata_errors` to `av.open()`, which PyAV 19 removed.
- Tested on Arch Linux with a GTX 1650 (4 GB). Windows support follows the same code path but has not been tested yet.

## License

MIT
