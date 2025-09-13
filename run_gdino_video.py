import argparse
import json
import pathlib
from typing import List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import random

Detection = Tuple[int, int, int, int, str, float]  # x1, y1, x2, y2, label, score

def load_grounding_dino(checkpoint: str, device: str = "cuda"):
    model = AutoModelForZeroShotObjectDetection.from_pretrained(checkpoint).to(device).eval()
    processor = AutoProcessor.from_pretrained(checkpoint)
    return model, processor

def detect_objects(
    model,
    processor,
    bgr_image: np.ndarray,
    prompt: str,
    device: str = "cuda"
) -> List[Detection]:
    pil_image = Image.fromarray(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB))
    inputs = processor(images=pil_image, text=prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([pil_image.size[::-1]]).to(device)  # (height, width)
    results = processor.post_process_grounded_object_detection(
        outputs,
        target_sizes=target_sizes,
        threshold=0.0  # Filter manually later using `--conf`
    )[0]

    # Now extract detections
    detections: List[Detection] = []
    boxes = results["boxes"].tolist()
    scores = results["scores"].tolist()
    labels = results.get("text_labels", results.get("labels", []))

    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = map(int, box)
        detections.append((x1, y1, x2, y2, label, float(score)))

    return detections


def _draw_label(
    img: np.ndarray,
    text: str,
    origin: Tuple[int, int],
    scale=0.4,
    thickness=1,
    txt_color=(0, 0, 0),
    bg_color=(255, 255, 255),
):
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, y = origin
    cv2.rectangle(img, (x, y - h - base), (x + w, y + base), bg_color, -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, txt_color, thickness, cv2.LINE_AA)

def random_color(seed_text):
    random.seed(seed_text)
    return tuple(random.randint(0, 255) for _ in range(3))

def annotate_frame(frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
    out = frame.copy()
    label_colors = {}

    for x1, y1, x2, y2, label, score in detections:
        if label not in label_colors:
            label_colors[label] = random_color(label)

        color = label_colors[label]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        _draw_label(out, f"{label} {score:.2f}", (x1, y1 - 4), bg_color=color)

    return out


def process_video(
    video_path: pathlib.Path,
    text_prompt: str,
    out_path: pathlib.Path,
    frames_dir: pathlib.Path,
    threshold: float,
    model,
    processor,
    device: str,
    max_frames: int | None = None,
):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames_dir.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        detections = detect_objects(model, processor, frame, text_prompt, device)
        detections = [d for d in detections if d[5] >= threshold]
        # print(detections)
        annotated = annotate_frame(frame, detections)

        writer.write(annotated)

        idx += 1
        if max_frames and idx >= max_frames:
            break
        if idx % 200 == 0:
            # cv2.imwrite(str(frames_dir / f"{idx:06d}.jpg"), annotated)
            print(f"  ↳ {video_path.name} @ {threshold:.2f}: {idx} frames", end="\r")

    cap.release()
    writer.release()
    print(f"     [{video_path.name}] threshold {threshold:.2f} → done ({idx} frames)")

def process_json(
    input_json: pathlib.Path,
    out_dir: pathlib.Path,
    thresholds: List[float],
    model,
    processor,
    device: str,
    max_frames: int | None = None,
):
    with open(input_json, "r") as f:
        data = json.load(f)

    for threshold in thresholds:
        thr_dir = out_dir / f"{threshold:.2f}"
        thr_dir.mkdir(parents=True, exist_ok=True)

        for entry in data:
            video_path = pathlib.Path(entry["video_path"]).resolve()
            prompt = entry["text_prompt"]

            vid_dir = thr_dir / video_path.stem
            frames_dir = vid_dir / "frames"
            vid_dir.mkdir(parents=True, exist_ok=True)

            output_video = vid_dir / video_path.name
            process_video(
                video_path,
                prompt,
                output_video,
                frames_dir,
                threshold,
                model,
                processor,
                device,
                max_frames=max_frames,
            )

def parse_args():
    parser = argparse.ArgumentParser("Grounding DINO video object detection")
    parser.add_argument("--input_json", required=True, help="Path to JSON with video paths and prompts")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--conf", nargs="+", type=float, default=[0.3], help="Confidence threshold(s)")
    parser.add_argument("--device", default="cuda", help="Torch device")
    parser.add_argument("--max_frames", type=int, default=None, help="Process only N frames (debug)")
    return parser.parse_args()

def main():
    args = parse_args()
    out_dir = pathlib.Path(args.out_dir).expanduser().resolve()
    input_json = pathlib.Path(args.input_json).expanduser().resolve()

    model_path = "./grounding-dino-base"
    model, processor = load_grounding_dino(model_path, device=args.device)

    process_json(
        input_json,
        out_dir,
        args.conf,
        model,
        processor,
        args.device,
        max_frames=args.max_frames,
    )

if __name__ == "__main__":
    main()
