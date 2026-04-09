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
    args = parser.parse_args()

    model = WhisperModel(
        args.model,
        device=args.device,
        compute_type=args.compute_type,
    )
    segments, _info = model.transcribe(
        args.audio,
        vad_filter=True,
        beam_size=5,
        language=None,
    )
    text = " ".join(segment.text.strip() for segment in segments).strip()
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
