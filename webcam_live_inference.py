"""
Headless webcam inference for arrow detection using Ultralytics YOLO + PyAutoGUI.

Usage:
    python webcam_live_inference.py

Optional:
    python webcam_live_inference.py --weights /absolute/path/to/other_model.pt --camera 0 --conf 0.35

Behavior:
    - Captures webcam frames continuously
    - Detects arrows and sorts detections left-to-right
    - Converts detections into key sequence (left/right/up/down)
    - Re-checks the same sequence N times; sends only if stable
    - Sleeps between cycles (default 1 second)
    - Does NOT open a display window
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
import cv2
import pyautogui
from ultralytics import YOLO

DEFAULT_WEIGHTS = Path(r"C:\Users\user\Documents\arrow_clicker\best.pt")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Headless webcam live inference with YOLO + PyAutoGUI key presses"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=str(DEFAULT_WEIGHTS),
        help="Path to trained weights file (.pt)",
    )
    parser.add_argument("--camera", type=int, default=0, help="Webcam index")
    parser.add_argument("--conf", type=float, default=0.20, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--width", type=int, default=1280, help="Webcam capture width")
    parser.add_argument("--height", type=int, default=720, help="Webcam capture height")
    parser.add_argument(
        "--key-interval",
        type=float,
        default=0.25,
        help="Delay between key presses in a sent sequence (seconds)",
    )
    parser.add_argument(
        "--confirm-reads",
        type=int,
        default=2,
        help="How many additional inferences must match before sending keys",
    )
    parser.add_argument(
        "--cycle-sleep",
        type=float,
        default=1.0,
        help="Sleep time between inference cycles (seconds)",
    )
    return parser.parse_args()


def normalize_label_to_key(label: str) -> str | None:
    mapping = {
        "left": "left",
        "right": "right",
        "up": "up",
        "down": "down",
        "arrow_left": "left",
        "arrow_right": "right",
        "arrow_up": "up",
        "arrow_down": "down",
    }
    return mapping.get(label.strip().lower())


def extract_arrow_sequence(result, class_names: dict[int, str]) -> tuple[str, ...]:
    if result.boxes is None or len(result.boxes) == 0:
        return tuple()

    xyxy = result.boxes.xyxy.cpu().numpy()
    cls_ids = result.boxes.cls.cpu().numpy().astype(int)

    detections: list[tuple[float, str]] = []
    for i, cls_id in enumerate(cls_ids):
        label = class_names.get(int(cls_id), str(int(cls_id)))
        key_name = normalize_label_to_key(label)
        if key_name is None:
            continue
        x1, _, x2, _ = xyxy[i]
        center_x = float((x1 + x2) * 0.5)
        detections.append((center_x, key_name))

    detections.sort(key=lambda item: item[0])
    return tuple(key for _, key in detections)


def main() -> None:
    args = parse_args()

    # Move mouse to a corner to trigger fail-safe stop if needed.
    pyautogui.FAILSAFE = True

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Weights file not found: {weights_path}\n"
            "Tip: pass --weights C:/path/to/best.pt or update DEFAULT_WEIGHTS."
        )

    model = YOLO(str(weights_path))
    class_names = model.names if isinstance(model.names, dict) else {}

    print(f"Using weights: {args.weights}")
    if class_names:
        printable = ", ".join(f"{k}:{v}" for k, v in sorted(class_names.items()))
        print(f"Model classes ({len(class_names)}): {printable}")
    if len(class_names) != 4:
        print(
            f"[warn] Expected 4 labels, but model reports {len(class_names)} classes. "
            "Continuing with model-provided labels."
        )

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open webcam index {args.camera}. Try --camera 1 (or another index)."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    print("Live inference started (headless).")
    print("No OpenCV window is shown. Stop with Ctrl+C.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame from webcam.")
                break

            results = model.predict(
                source=frame,
                conf=float(args.conf),
                iou=args.iou,
                imgsz=args.imgsz,
                verbose=False,
            )
            result = results[0]
            sequence = extract_arrow_sequence(result, class_names)

            confirmed = bool(sequence)
            if confirmed:
                for _ in range(max(args.confirm_reads, 0)):
                    ok, verify_frame = cap.read()
                    if not ok:
                        confirmed = False
                        break

                    verify_results = model.predict(
                        source=verify_frame,
                        conf=float(args.conf),
                        iou=args.iou,
                        imgsz=args.imgsz,
                        verbose=False,
                    )
                    verify_sequence = extract_arrow_sequence(
                        verify_results[0],
                        class_names,
                    )
                    if verify_sequence != sequence:
                        confirmed = False
                        break

            if confirmed:
                print(f"SENT sequence: {sequence}")
                pyautogui.press(list(sequence), interval=args.key_interval)
            else:
                print("Sequence not stable; skipping send.")

            time.sleep(args.cycle_sleep)

    except KeyboardInterrupt:
        print("\nStopped by user (Ctrl+C).")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
