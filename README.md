# Body Score Detector

Anonymous research prototype for single-photo appearance scoring. Outputs abdomen, relative shoulder width, arm size appearance, and their valid-component mean. Scores are 0–100; unassessable components return `NA`.

## Run locally

Use Python 3.10 or later in a dedicated environment.

```bash
pip install -r requirements.txt
python app.py --open-browser
```

Open http://127.0.0.1:7868. Upload an image, select the abdomen rubric, and run scoring. JSON and PNG reports can be downloaded. Images are processed locally; outputs are stored under `results/`. On Windows, `start.bat` also launches the app and `stop.bat` stops the recorded app process.

CLI inference:

```bash
python predict.py --checkpoint models/model_v2.pt --image photo.jpg --abdomen-mode abs
```

Use `--abdomen-mode lines` for the waist-line rubric. This is an explicit user selection; no gender classifier is used.

## Model and training

Frozen ImageNet ResNet-18, a shared 64-dimensional head, four rubric outputs (two abdomen rubrics, shoulders, arms), and three assessability logits. Only labelled rubric targets contribute to regression loss. Assessability is learned from missing-label indicators. The total is the equal mean of assessable component scores.

```bash
python train_multitask.py --manifest data/manifest.json --pretrained resnet18.pth --output runs/new_run
```

The manifest has a `records` list. Each record contains `image` (relative to the manifest directory), image `sha256`, `scores` with numeric or null `abdomen`, `shoulder`, `arm`, `abdomen_references` with numeric or null `abs` and `lines`, and `abdomen_mode` (`腹肌` or `马甲线`). Image paths must remain inside the manifest directory. Supply your own appropriately labelled data and ImageNet ResNet-18 weights for training.

## Important limits

The included checkpoint was fitted to only eight unique images with AI visual initial annotations. Only two images had shoulder labels and four had arm labels. This is an overfitting/pipeline demonstration, with no independent validation and no established generalization accuracy. Scores and NA predictions can be wrong. Arm scores are visual appearance scores, not measured circumferences or muscle mass. Lighting, pose, clothing and framing affect predictions. The model is not a medical or body-composition measurement tool.

Source photos, labels, datasets, logs, credentials and server connection scripts are not distributed. File-origin metadata and training-image hashes were removed from the released checkpoint; all model tensors are unchanged. See `SHA256SUMS.txt` for the released weight checksum.
