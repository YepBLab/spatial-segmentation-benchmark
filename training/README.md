# CPSAM v2 fine-tuning workflow

This directory contains only the model-training workflow. It intentionally does
not contain full-slide inference, tiled-mask stitching, full-mask construction,
Xenium reimport, annotations, masks, source images, or model weights.

## Reproduced training design

The defaults reproduce the final all-region training configuration:

- base model: `cpsam_v2`;
- Cellpose: version 4.2 or newer;
- input tensor: `CYX` with exactly two channels;
- tensor channel 0: DAPI from physical morphology channel 0;
- tensor channel 1: boundary/18S signal from physical morphology channel 2;
- population in the reproduced run: every supplied manual region is used for training;
- validation split in the reproduced run: none, solely because the available
  manual-label set is small;
- epochs: 100;
- learning rate: `1e-5`;
- weight decay: `0.1`;
- batch size: 1;
- Cellpose patch size (`bsize`): 256;
- normalization: enabled;
- rescaling: disabled;
- random seed: 0.

**Validation remains the recommended design.** With sufficient labels, reserve
held-out regions or, preferably, held-out specimens for validation. Split at the
region/specimen level rather than randomly splitting cells from the same image.
Validation data can support model selection, early stopping, and overfitting
assessment; an independent external specimen is still preferable for the final
generalization test.

The reproduced run does not reserve validation data only because the manual-label
set is limited. Consequently, training loss and metrics calculated on those same
regions are optimization/in-sample diagnostics. They cannot estimate validation
performance or performance on unseen specimens.

## Input contract

Create a private CSV based on
[`config/training_manifest.example.csv`](../config/training_manifest.example.csv):

```text
region,image_path,label_path
region_001,/private/path/region_001_img.tif,/private/path/region_001_masks.tif
```

Paths may be absolute or relative to the manifest location. Do not put the
populated manifest in the Git repository.

Each image must be a two-channel `CYX` TIFF. Each label must be a registered 2D
integer instance mask with `0` as background and one positive ID per cell. The
preparation step copies the image, reindexes a derived label mask consecutively,
records SHA-256 hashes, and never modifies the source label.

## O2/GPU execution

Start an interactive or scheduled GPU job first; do not run training on a login
node. Load the compiler/CUDA modules required by the local Cellpose environment,
then point `PYTHON_BIN` at that environment:

```bash
module load gcc/14.2.0 cuda/12.8
export PYTHON_BIN=/path/to/cellpose42/bin/python

bash training/run_training_pipeline.sh \
  /private/path/training_project \
  /private/path/training_manifest.csv
```

The runner requires CUDA. Environment variables can override the defaults:

```bash
MODEL_NAME=my_cpsam_v2_model \
N_EPOCHS=100 \
LEARNING_RATE=1e-5 \
WEIGHT_DECAY=0.1 \
BATCH_SIZE=1 \
BSIZE=256 \
SMOOTH_WINDOW=9 \
bash training/run_training_pipeline.sh PROJECT_ROOT TRAINING_MANIFEST
```

Set `OVERWRITE_PREPARED_DATA=1` only after reviewing an existing frozen training
dataset. The default refuses to overwrite derived pairs.

## Stages

1. `prepare_training_data.py`
   - validates image/label registration and channel order;
   - requires at least five instances per region by default;
   - creates derived training pairs and provenance hashes.
2. `preflight_training.py`
   - checks Cellpose version, `cpsam_v2`, CUDA, all pairs, and one pilot inference.
3. `train_cpsam_v2.py`
   - loads every prepared region;
   - for this limited-label reproduction, calls `cellpose.train.train_seg` with
     `test_data=None` and `test_labels=None`;
   - saves the trained model, loss history, and full hyperparameter metadata.
4. `validate_trained_model.py`
   - reloads the final model;
   - runs one in-memory inference to confirm shape and non-empty output;
   - does not save an inference mask.
5. `plot_training_history.py`
   - creates a dark-theme raw/smoothed loss plot and per-epoch loss-change panel;
   - contains training-process diagnostics only, not accuracy results.

## Outputs

All outputs are written under the private `PROJECT_ROOT`:

```text
training_data/all/*_img.tif
training_data/all/*_masks.tif
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

These outputs, especially source-derived masks and model weights, must remain
outside the repository. The repository `.gitignore` includes defensive patterns,
but users must still inspect `git status` before every push.
