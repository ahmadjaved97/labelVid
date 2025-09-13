"""
CLI:
  python run_autodistill_video.py \
         --max_frames 1000 \
         --out_root /mnt/results

If you omit the flags:
  --max_frames defaults to 2000
  --out_root   defaults to ./annotated
"""

import cv2, json, pathlib, gc, torch, argparse
from tqdm import tqdm
import supervision as sv
from autodistill_grounded_sam   import GroundedSAM
from autodistill_grounding_dino import GroundingDINO
from autodistill_detic           import DETIC
from autodistill.detection      import CaptionOntology

# ────────────── CONFIGURABLE GRIDS ──────────────
CONF_THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70]
# NMS_THRESHOLDS  = [0.45, 0.50, 0.60]
NMS_THRESHOLDS  = [0]


AVAILABLE_MODELS = {
    "gsam" : GroundedSAM,
    "gdino": GroundingDINO,
    "detic": DETIC,
}

# ────────────── SHARED ANNOTATORS ──────────────
MASK_ANN  = sv.MaskAnnotator(opacity=0.40)
BOX_ANN   = sv.BoxAnnotator(thickness=1)
LABEL_ANN = sv.LabelAnnotator(text_scale=0.35, text_thickness=1, text_padding=5)


def build_variants(frame, dets, names):

    def id2name(cid):
        return names[cid] if 0 <= cid < len(names) else f"id_{cid}"

    # labels = [f"{names[c]} {conf:.2f}" for c, conf in zip(dets.class_id, dets.confidence)]
    labels = [f"{id2name(cid)} {conf:.2f}"
          for cid, conf in zip(dets.class_id, dets.confidence)]
    
    full  = LABEL_ANN.annotate(BOX_ANN.annotate(MASK_ANN.annotate(frame.copy(), dets), dets),
                               detections=dets, labels=labels)
    boxes = LABEL_ANN.annotate(BOX_ANN.annotate(frame.copy(), dets),
                               detections=dets, labels=labels)
    masks = LABEL_ANN.annotate(MASK_ANN.annotate(frame.copy(), dets),
                               detections=dets, labels=labels)
    return full, boxes, masks


def process_video(video_path, ontology, model_key, model,
                  conf_thr, nms_thr, max_frames, out_root):
    stem     = pathlib.Path(video_path).stem
    base_dir = pathlib.Path(out_root) / f"annotated_{model_key}" / f"c{conf_thr:.2f}_n{nms_thr:.2f}" / stem
    raw_dir  = base_dir / "raw_frames"
    ann_dir  = base_dir / "annotated_frames"
    raw_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    cap  = cv2.VideoCapture(video_path)
    fps  = cap.get(cv2.CAP_PROP_FPS)
    W,H  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    four = cv2.VideoWriter_fourcc(*"mp4v")

    full_v = cv2.VideoWriter(str(base_dir / f"{stem}.annotated.mp4"), four, fps, (W, H))
    box_v  = cv2.VideoWriter(str(base_dir / f"{stem}.boxes.mp4"),     four, fps, (W, H))
    mask_v = cv2.VideoWriter(str(base_dir / f"{stem}.masks.mp4"),     four, fps, (W, H))

    # class_names, json_frames, idx = list(ontology.values()), [], 0
    class_names, json_frames, idx  = model.ontology.classes(), [], 0

    while idx < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        dets = model.predict(frame) #.with_nms(threshold=nms_thr)
        keep = dets.confidence > conf_thr
        dets = dets[keep] if keep.any() else sv.Detections.empty()

        full, boxes, masks = build_variants(frame, dets, class_names)
        if idx % 10 == 0:
            cv2.imwrite(str(raw_dir / f"frame{idx:05d}.jpg"), frame)
            cv2.imwrite(str(ann_dir / f"frame{idx:05d}.jpg"), full)

        full_v.write(full); box_v.write(boxes); mask_v.write(masks)

        json_frames.append({
            "frame_index": idx,
            "detections": [
                {"class": class_names[cid],
                 "confidence": float(conf),
                 "bbox": [float(x) for x in box]}
                for box, conf, cid in zip(dets.xyxy, dets.confidence, dets.class_id)
            ]
        })
        idx += 1

    cap.release(); full_v.release(); box_v.release(); mask_v.release()
    with open(base_dir / f"{stem}.annotated.json", "w") as f:
        json.dump(json_frames, f, indent=2)
    print(f"[{model_key}] {stem}: {idx}/{max_frames} frames → {base_dir}")


def run_for_model_conf_iou(model_key, jobs, conf_thr, nms_thr, max_frames, out_root):
    dummy_ont = CaptionOntology(jobs[0]["ontology"])
    model = AVAILABLE_MODELS[model_key](dummy_ont)

    for job in tqdm(jobs, desc=f"{model_key} c{conf_thr:.2f} n{nms_thr:.2f}", leave=False):
        model.ontology = CaptionOntology(job["ontology"])
        process_video(job["video_path"], job["ontology"],
                      model_key, model, conf_thr, nms_thr, max_frames, out_root)

    del model
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_frames", type=int, default=2000,
                        help="Frames to process per video (default 2000)")
    parser.add_argument("--out_root", type=str, default="annotated",
                        help="Root folder for all outputs (default ./annotated)")
    args = parser.parse_args()

    with open("./video_info3.json") as f:
        VIDEO_JOBS = json.load(f)

    for model_key in AVAILABLE_MODELS:
        for conf_thr in CONF_THRESHOLDS:
            for nms_thr in NMS_THRESHOLDS:
                run_for_model_conf_iou(model_key,
                                       VIDEO_JOBS,
                                       conf_thr,
                                       nms_thr,
                                       args.max_frames,
                                       args.out_root)
