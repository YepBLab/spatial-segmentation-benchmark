# CPSAM v2 fine-tuning workflow

This directory provides a manifest-driven workflow for fine-tuning CPSAM v2
with Cellpose 4.2 or newer.

## Input manifest

Create a private CSV based on
[`config/training_manifest.example.csv`](../config/training_manifest.example.csv):

```text
region,split,image_path,label_path
region_001,train,/private/path/region_001_img.tif,/private/path/region_001_masks.tif
region_002,validation,/private/path/region_002_img.tif,/private/path/region_002_masks.tif
```

`split` accepts `train` or `validation`. The column is optional; if omitted, all
rows are assigned to training. Paths may be absolute or relative to the manifest
location. Keep the populated manifest outside the Git repository.

Each image must be a 3D TIFF with one channel axis and two spatial axes. The
channel axis is configurable and defaults to axis 0. Each label must be a
registered 2D integer instance mask with `0` as background and one positive ID
per object.

The preparation step copies the image, reindexes a derived label consecutively,
records SHA-256 hashes, and never modifies the source label.

## Validation design

Use independent regions or specimens for validation whenever the dataset allows
it. Define the split in the manifest rather than randomly dividing objects from
the same image. If the manifest contains no validation rows, training proceeds
without validation and records that policy in its metadata.

Validation loss from `cellpose.train.train_seg` is preserved when the installed
Cellpose version returns it. Final performance claims should use the separate
instance-segmentation evaluation workflow and an appropriate independent test
set.

## Execution

Run the workflow on a compute node with a compatible Cellpose environment. Point
`PYTHON_BIN` at that environment. Set `N_EPOCHS`, `LEARNING_RATE`, and
`WEIGHT_DECAY` explicitly to the values selected for the experiment, then run:

```bash
export PYTHON_BIN=/path/to/cellpose/bin/python

bash training/run_training_pipeline.sh \
  /private/path/training_project \
  /private/path/training_manifest.csv
```

The runner uses CUDA by default. Other configurable variables include
`PRETRAINED_MODEL`, `MODEL_NAME`, `DEVICE`, `BATCH_SIZE`, `BSIZE`,
`CHANNEL_AXIS`, and `SMOOTH_WINDOW`.

Set `OVERWRITE_PREPARED_DATA=1` only after reviewing an existing frozen training
dataset. The default refuses to overwrite derived pairs.

## Stages

1. `prepare_training_data.py`
   - validates image/label registration and manifest splits;
   - creates derived training/validation pairs and provenance hashes.
2. `preflight_training.py`
   - checks Cellpose, the requested model, device availability, data pairs, and
     one pilot inference.
3. `train_cpsam_v2.py`
   - loads the prepared training and optional validation regions;
   - saves the model, available loss histories, and hyperparameter metadata.
4. `validate_trained_model.py`
   - reloads the final model and runs one in-memory smoke test.
5. `plot_training_history.py`
   - renders a dark-theme training-history figure.

## Outputs

All outputs are written under `PROJECT_ROOT`:

```text
training_data/train/*_img.tif
training_data/train/*_masks.tif
training_data/validation/*_img.tif
training_data/validation/*_masks.tif
training_data/label_maps/*.csv
training_data/training_manifest.csv
training_data/training_data_summary.json
provenance/training_preflight.json
training/final/models/<model_name>
training/final/losses.csv
training/final/training_metadata.json
training/final/final_model_smoke_test.json
training/final/training_history_dark.png
logs/*.log
```

Inspect `git status` before every push to ensure generated data and model files
remain outside version control.
