#!/usr/bin/env python3
import argparse
import sys

from faster_whisper import WhisperModel


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument(
        "--language",
        default=None,
        help="BCP-47 language code (e.g. 'es', 'en'). Omit for auto-detection.",
    )
    args = parser.parse_args()

    # Normalize: treat empty string or "auto" as None (auto-detect).
    language = args.language if args.language and args.language.strip().lower() not in ("", "auto", "automatic") else None

    model = WhisperModel(
        args.model,
        device=args.device,
        compute_type=args.compute_type,
    )
    segments, _info = model.transcribe(
        args.audio,
        vad_filter=True,
        beam_size=5,
        language=language,
    )
    text = " ".join(segment.text.strip() for segment in segments).strip()
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
