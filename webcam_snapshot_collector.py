"""
Capture webcam snapshots at a fixed interval for dataset collection.

Cross-platform support:
- Linux (tries V4L2 first)
- Windows (tries DirectShow first)

Usage examples:
    python webcam_snapshot_collector.py
    python webcam_snapshot_collector.py --output custom_dataset/images --interval 1.0
    python webcam_snapshot_collector.py --camera 1 --width 1280 --height 720

Controls:
    q  -> quit
"""

from __future__ import annotations

import argparse
import platform
import time
from datetime import datetime
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture webcam snapshots every N seconds for dataset creation"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="custom_dataset/images",
        help="Directory where captured images are saved (default: custom_dataset/images)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Seconds between snapshots (default: 1.0)",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam index to use (default: 0)",
    )
    parser.add_argument("--width", type=int, default=1280, help="Capture width")
    parser.add_argument("--height", type=int, default=720, help="Capture height")
    parser.add_argument(
        "--prefix",
        type=str,
        default="arrow",
        help="Filename prefix for saved images (default: arrow)",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Disable preview window (still captures images)",
    )
    return parser.parse_args()


def open_camera(camera_index: int) -> cv2.VideoCapture:
    system_name = platform.system().lower()

    # Try OS-preferred backend first, then fallback to default backend.
    if "windows" in system_name:
        preferred_backends = [cv2.CAP_DSHOW, cv2.CAP_ANY]
    elif "linux" in system_name:
        preferred_backends = [cv2.CAP_V4L2, cv2.CAP_ANY]
    else:
        preferred_backends = [cv2.CAP_ANY]

    for backend in preferred_backends:
        cap = cv2.VideoCapture(camera_index, backend)
        if cap.isOpened():
            return cap
        cap.release()

    raise RuntimeError(
        f"Could not open webcam index {camera_index}. Try another index with --camera 1"
    )


def main() -> None:
    args = parse_args()

    if args.interval <= 0:
        raise ValueError("--interval must be > 0")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = open_camera(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    print("Webcam snapshot collector started")
    print(f"Saving images to: {output_dir.resolve()}")
    print(f"Interval: {args.interval:.2f}s")
    print("Press 'q' in preview window to quit (or Ctrl+C in terminal).")

    last_capture_time = 0.0
    count = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame from webcam. Stopping.")
                break

            now = time.time()
            should_capture = (now - last_capture_time) >= args.interval

            if should_capture:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"{args.prefix}_{timestamp}.jpg"
                path = output_dir / filename
                cv2.imwrite(str(path), frame)
                count += 1
                last_capture_time = now
                print(f"[{count}] Saved: {path}")

            if not args.no_preview:
                preview = frame.copy()
                cv2.putText(
                    preview,
                    f"Captured: {count}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    preview,
                    "Press q to quit",
                    (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("Dataset Capture", preview)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
            else:
                # Keep CPU usage reasonable in headless mode.
                time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"Done. Total images captured: {count}")


if __name__ == "__main__":
    main()
