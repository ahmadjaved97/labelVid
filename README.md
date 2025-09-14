# labelVid - Video Annotation Toolkit

This repo helps you **annotate videos automatically** using three different AI models:

- **AutoDistill** → runs GroundedSAM, GroundingDINO, and DETIC for object detection with an ontology (a map of labels you care about).
- **Grounding DINO** → lets you write text prompts like "a person, a car" and finds them in videos.
- **Florence-2** → an open-vocabulary model that can detect many objects without training.

You can use it to get:
- Annotated videos with boxes, masks, or overlays.
- JSON files with object detections for each frame.
- Sampled frames saved as images.


## How the repo looks

- run_autodistill_video8.py # AutoDistill pipeline
- run_gdino_video.py # Grounding DINO pipeline
- run_florence_video4.py # Florence-2 pipeline
- video_info3.json # Example jobs file
- outputs/ # Annotated results go here


## Installation
You need Python 3.10+ and PyTorch (with CUDA if you have a GPU).

```bash
# Install PyTorch first from https://pytorch.org/get-started/locally/

pip install --upgrade pip
pip install transformers opencv-python pillow numpy tqdm supervision
pip install autodistill autodistill-grounding-dino autodistill-grounded-sam autodistill-detic
```

Some models (like Florence-2 or GroundingDINO) may need you to log into Hugging Face.

## How to run

### 1) AutoDistill (ontology based)

Edit `video_info.json` to list your videos and the labels you want:

```json
[
  {
    "video_path": "./sample_videos/people_walking.mp4",
    "ontology": {
      "a person": "person",
      "a bag": "bag"
    }
  }
]
```
Run

```bash

python run_autodistill_video.py --max_frames 1000 --out_root outputs
```

### 2) Grounding DINO (prompt based)

```json
[
  { "video_path": "./sample_videos/people.mp4", "text_prompt": "a person, a bag" }
]
```

Run
```bash
python run_gdino_video.py --input_json jobs.json --out_dir outputs/gdino --conf 0.3 0.5 0.7 --device cuda
```


### 3) Florence-2 (folder based)

Put your videos in a folder and run:

```bash
python run_florence_video.py --in_dir sample_videos --out_dir outputs/florence --conf 0.3 0.5 0.7 --model_type large --device cuda
```

## Example Output

[▶️ Watch a sample annotated video](assets/race_highlights.annotated.mp4)