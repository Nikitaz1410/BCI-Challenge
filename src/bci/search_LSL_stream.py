"""
Utility script to search for Lab Streaming Layer (LSL) streams on the local network.

Usage (from the repo root):
    uv run src/bci/search_LSL_stream.py
or:
    python -m bci.search_LSL_stream
"""

from __future__ import annotations

import argparse
import sys
from typing import List

try:
    from pylsl import resolve_streams, StreamInfo
except ImportError as exc:  # pragma: no cover
    print(
        "ERROR: `pylsl` is not installed.\n"
        "Install it via:\n"
        "  uv add pylsl\n"
        "or\n"
        "  pip install pylsl",
        file=sys.stderr,
    )
    raise


def format_stream(info: "StreamInfo", index: int) -> str:
    """Return a human‑readable description of a stream."""
    try:
        uid = info.source_id()
    except Exception:  # pragma: no cover - very unlikely
        uid = "N/A"

    try:
        channel_count = info.channel_count()
        nominal_srate = info.nominal_srate()
        channel_format = info.channel_format()
    except Exception:  # pragma: no cover
        channel_count = nominal_srate = channel_format = "N/A"

    return (
        f"[{index}] Name: {info.name()} | Type: {info.type()} | "
        f"UID: {uid}\n"
        f"    Channels: {channel_count} | "
        f"Sampling rate: {nominal_srate} | "
        f"Channel format: {channel_format}\n"
        f"    Host: {info.hostname()} | "
        f"Manufacturer: {info.desc().child_value('manufacturer') or 'N/A'}"
    )


def search_streams(
    stream_type: str | None = None,
    timeout: float = 2.0,
) -> List["StreamInfo"]:
    """
    Resolve all LSL streams, optionally filtering by type.

    - **stream_type**: LSL stream type to filter by (e.g. 'EEG', 'Markers').
    - **timeout**: Resolve timeout in seconds.
    """
    if stream_type:
        # Resolve only matching type
        query = f"type='{stream_type}'"
        streams = resolve_streams(wait_time=timeout)
        streams = [s for s in streams if s.type() == stream_type]
    else:
        # Resolve all streams
        streams = resolve_streams(wait_time=timeout)
    return streams


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Search for available LSL streams on the network."
    )
    parser.add_argument(
        "-t",
        "--type",
        dest="stream_type",
        help="Filter by LSL stream type (e.g. 'EEG', 'Markers').",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=2.0,
        help="Resolve timeout in seconds (default: 2.0).",
    )

    args = parser.parse_args(argv)

    print(
        f"Resolving LSL streams (timeout={args.timeout}s"
        f"{', type=' + args.stream_type if args.stream_type else ''})..."
    )

    try:
        streams = search_streams(
            stream_type=args.stream_type,
            timeout=args.timeout,
        )
    except Exception as exc:  # pragma: no cover
        print(f"Failed to resolve streams: {exc}", file=sys.stderr)
        return 1

    if not streams:
        print("No LSL streams found.")
        return 0

    print(f"Found {len(streams)} LSL stream(s):\n")
    for idx, info in enumerate(streams, start=1):
        print(format_stream(info, idx))
        print("-" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

