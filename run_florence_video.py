"""
Batch Florence‑2 Object Detection with Threshold‑Specific Outputs
================================================================
Usage
-----
```bash
python video_object_detection.py \
       --in_dir  /path/to/videos \
       --out_dir /path/to/outputs \
       --conf    0.3 0.5 0.7        # one or more thresholds
```
"""

from __future__ import annotations

import argparse
import pathlib
from typing import Iterable, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

################################################################################
# Constants                                                                    #
################################################################################

MODEL_MAP = {
    "large": "microsoft/Florence-2-large",
    "large-finetuned": "microsoft/Florence-2-large-finetuned",
}

VIDEO_SUFFIXES = {".mp4", ".mkv", ".avi", ".mov", ".webm"}
Detection = Tuple[int, int, int, int, str, float]  # x1, y1, x2, y2, label, score

################################################################################
# Florence‑2 helpers                                                           #
################################################################################

def load_florence(checkpoint: str, *, device: str = "cuda", half: bool = True):
    dtype = torch.float16 if half else torch.float32
    model = (
        AutoModelForCausalLM.from_pretrained(
            checkpoint, torch_dtype=dtype, trust_remote_code=True
        )
        .eval()
        .to(device)
    )
    processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
    return model, processor, dtype


def detect_objects(
    model,
    model_type,
    processor,
    dtype: torch.dtype,
    bgr_image: np.ndarray,
    *,
    device: str = "cuda",
) -> List[Detection]:
    prompt = "<OD>"
    pil = Image.fromarray(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB))
    inputs = processor(text=prompt, images=pil, return_tensors="pt").to(device, dtype)

    with torch.inference_mode():
        gen = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
        )

    

    if model_type == "large":
        transition = model.compute_transition_scores(
        sequences=gen.sequences, scores=gen.scores, beam_indices=gen.beam_indices
        )[0]

        parsed = processor.post_process_generation(
            sequence=gen.sequences[0],
            transition_beam_score=transition,
            task=prompt,
            image_size=(pil.width, pil.height),
        )

        dets: List[Detection] = []
        for bbox, label, score in zip(
            parsed[prompt]["bboxes"], parsed[prompt]["labels"], parsed[prompt]["scores"]
        ):
            x1, y1, x2, y2 = map(int, bbox)
            dets.append((x1, y1, x2, y2, label, float(score)))
        # return dets

        

    elif model_type == "large-finetuned":
        generated_text = processor.batch_decode(gen.sequences, skip_special_tokens=False)[0]
        parsed = processor.post_process_generation(
            generated_text,
            task=prompt,
            image_size=(pil.width, pil.height),
        )

        dets: List[Detection] = []
        for bbox, label in zip(
            parsed[prompt]["bboxes"], parsed[prompt]["labels"]
        ):
            x1, y1, x2, y2 = map(int, bbox)
            dets.append((x1, y1, x2, y2, label))
    
    return dets

################################################################################
# Drawing utilities                                                            #
################################################################################

def _draw_label(
    img: np.ndarray,
    text: str,
    origin: Tuple[int, int],
    *,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    scale=0.4,
    thickness=1,
    txt_color=(0, 0, 0),
    bg_color=(255, 255, 255),
):
    (w, h), base = cv2.getTextSize(text, font, scale, thickness)
    x, y = origin
    cv2.rectangle(img, (x, y - h - base), (x + w, y + base), bg_color, -1)
    cv2.putText(img, text, (x, y), font, scale, txt_color, thickness, cv2.LINE_AA)


def annotate_frame(frame: np.ndarray, dets: List[Detection], model_type) -> np.ndarray:
    out = frame.copy()
    if model_type == "large":
        for x1, y1, x2, y2, label, score in dets:
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 1)
            _draw_label(out, f"{label} {score:.2f}", (x1, y1 - 4))
    elif model_type == "large-finetuned":
        for x1, y1, x2, y2, label in dets:
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 1)
            _draw_label(out, f"{label}", (x1, y1 - 4))
    return out

################################################################################
# Video processing                                                             #
################################################################################

def process_video(
    src: pathlib.Path,
    video_path: pathlib.Path,
    frames_dir: pathlib.Path,
    threshold: float,
    model_type,
    model,
    processor,
    dtype: torch.dtype,
    *,
    device: str,
    max_frames: int | None = None,
):
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {src}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames_dir.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        dets_all = detect_objects(model, model_type, processor, dtype, frame, device=device)
        if model_type == "large":
            dets = [d for d in dets_all if d[5] >= threshold]  # score filter
        else:
            dets = [d for d in dets_all]
        
        annotated = annotate_frame(frame, dets, model_type)
        writer.write(annotated)

        

        idx += 1
        if max_frames and idx >= max_frames:
            break
        if idx % 100 == 0:
            frame_file = frames_dir / f"{idx:06d}.jpg"
            cv2.imwrite(str(frame_file), annotated)
            print(f"  ↳ {src.name} @ {threshold:.2f}: {idx} frames", end="\r")

    cap.release()
    writer.release()
    print(f"     [{src.name}] threshold {threshold:.2f} → done ({idx} frames)")

################################################################################
# Directory traversal & orchestration                                          #
################################################################################

def iter_videos(folder: pathlib.Path) -> Iterable[pathlib.Path]:
    return (p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES)


def process_directory(
    in_dir: pathlib.Path,
    out_dir: pathlib.Path,
    thresholds: List[float],
    model_type,
    model,
    processor,
    dtype: torch.dtype,
    *,
    device: str,
    max_frames: int | None = None,
):
    vids = list(iter_videos(in_dir))
    if not vids:
        raise RuntimeError(f"No videos found in {in_dir}")

    for thr in thresholds:
        thr_dir = out_dir / f"{thr:.2f}"
        thr_dir.mkdir(parents=True, exist_ok=True)

        print(f"[INFO] Processing at threshold {thr:.2f} ({len(vids)} videos)…")
        for src in vids:
            vid_dir = thr_dir / src.stem  # directory named after video
            frames_dir = vid_dir / "frames"
            vid_dir.mkdir(parents=True, exist_ok=True)
            video_out_path = vid_dir / src.name  # annotated video file

            process_video(
                src,
                video_out_path,
                frames_dir,
                thr,
                model_type,
                model,
                processor,
                dtype,
                device=device,
                max_frames=max_frames,
            )

################################################################################
# CLI                                                                          #
################################################################################

def parse_args():
    p = argparse.ArgumentParser("Florence‑2 batch video object detection")
    p.add_argument("--in_dir", required=True, help="Folder with input videos")
    p.add_argument("--out_dir", required=True, help="Folder for outputs")
    p.add_argument("--conf", nargs="+", type=float, default=[0.3], help="Confidence threshold(s)")
    p.add_argument("--model_type", choices=list(MODEL_MAP.keys()), default="large")
    p.add_argument("--ckpt", default=None, help="Override checkpoint path")
    p.add_argument("--device", default="cuda", help="Torch device")
    p.add_argument("--full", action="store_true", help="Use FP32 instead of FP16")
    p.add_argument("--max_frames", type=int, default=None, help="Process only N frames (debug)")
    return p.parse_args()


def main():
    args = parse_args()
    in_dir = pathlib.Path(args.in_dir).expanduser().resolve()
    out_dir = pathlib.Path(args.out_dir).expanduser().resolve()

    ckpt = args.ckpt or MODEL_MAP[args.model]
    model, processor, dtype = load_florence(ckpt, device=args.device, half=not args.full)

    process_directory(
        in_dir,
        out_dir,
        args.conf,
        args.model_type,
        model,
        processor,
        dtype,
        device=args.device,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
