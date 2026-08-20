# Data contract

This repository contains code and aggregate example figures only. Reference
annotations, model outputs, images, expression matrices, and spatial data must
remain outside the Git checkout.

## Instance masks

- 2D TIFF with integer dtype.
- `0` is background.
- Every cell is represented by one positive integer ID.
- IDs do not need to be consecutive; the evaluator relabels them internally.
- Reference and prediction masks must have identical shape and registration.
- Masks must represent filled cell instances, not binary boundaries.
- Pixel size must be supplied in micrometers per pixel.

## Manual ROI manifest

The full multi-region pipeline expects a CSV with at least:

```text
region_id,x0,y0,x1,y1
```

Coordinates use zero-based pixel coordinates in the full-resolution image.
The interval is half-open: `[x0, x1)`, `[y0, y1)`.

For each `region_id`, the pipeline expects:

```text
manual_label_dir/<region_id>.tif
manual_image_dir/<region_id>_img.tif
```

## Raster model source

`kind: raster_mask` points to a full-resolution integer instance-label TIFF.

## Xenium vertex Parquet source

`kind: vector_parquet` expects these columns:

```text
cell_id, vertex_x, vertex_y, label_id
```

Vertex coordinates are interpreted in micrometers and converted to pixels with
`pixel_size_um`.

## ProSeg geometry collection

`kind: proseg_geometry_collection` expects the ProSeg geometry-collection JSON
layout with a top-level `geometries` array containing Polygon or MultiPolygon
objects in micrometers.

## ROI images and SNR

ROI images should be `CYX` or `YX`. For two-channel morphology, the first two
channels are used for the robust SNR summary. Display-enhanced images must not
be used for SNR computation.

## Data that must not be committed

- manual or consensus annotations;
- masks or cell-boundary files;
- morphology images;
- transcript tables;
- H5AD, Zarr, GeoPackage, Parquet, or Xenium output bundles;
- model weights;
- private absolute paths or access credentials.
