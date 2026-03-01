"""
Live webcam inference for arrow detection using Ultralytics YOLO.

Usage:
    python webcam_live_inference.py --weights /absolute/path/to/best.pt

Optional:
    python webcam_live_inference.py --weights yolov8n.pt --camera 0 --conf 0.35
"""

from __future__ import annotations

import argparse
import os
import time

# Fix common OpenCV(Qt) font warning on some Linux setups
if os.path.isdir("/usr/share/fonts/truetype/dejavu"):
    os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype/dejavu")

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Webcam live inference with YOLO")
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to trained weights file (.pt), e.g. runs/detect/train/weights/best.pt",
    )
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument("--conf", type=float, default=0.20, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--width", type=int, default=1280, help="Webcam capture width")
    parser.add_argument("--height", type=int, default=720, help="Webcam capture height")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    conf_thres = float(args.conf)

    # Load model
    model = YOLO(args.weights)

    # Open webcam
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open webcam index {args.camera}. Try --camera 1 (or another index)."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    prev_time = time.time()

    print("Live inference started.")
    print("Press 'q' to quit. Use '[' and ']' to decrease/increase confidence threshold.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame from webcam.")
                break

            # Run inference
            results = model.predict(
                source=frame,
                conf=conf_thres,
                iou=args.iou,
                imgsz=args.imgsz,
                verbose=False,
            )
            result = results[0]

            # Use Ultralytics built-in renderer for robust label+box drawing
            annotated = result.plot()

            # FPS overlay
            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now
            cv2.putText(
                annotated,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )

            n = 0 if result.boxes is None else len(result.boxes)
            max_conf = 0.0
            if result.boxes is not None and len(result.boxes):
                max_conf = float(result.boxes.conf.max().item())
            cv2.putText(
                annotated,
                f"Detections: {n} | conf={conf_thres:.2f} | max={max_conf:.2f} | imgsz={args.imgsz}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Arrow Detection - Live", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("["):
                conf_thres = max(0.01, conf_thres - 0.05)
                print(f"conf -> {conf_thres:.2f}")
            elif key == ord("]"):
                conf_thres = min(0.95, conf_thres + 0.05)
                print(f"conf -> {conf_thres:.2f}")
    except KeyboardInterrupt:
        print("\nStopped by user (Ctrl+C).")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
