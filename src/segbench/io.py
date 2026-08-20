from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tifffile
import yaml
from shapely.geometry import Polygon, shape
from shapely.ops import unary_union
from skimage.draw import polygon as draw_polygon


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as handle:
        return yaml.safe_load(handle)


@dataclass(frozen=True)
class Roi:
    region: str
    x0: int
    y0: int
    x1: int
    y1: int
    label_path: Path
    image_path: Path
    medullary_overlap_fraction: float
    tissue: str

    @property
    def shape(self) -> tuple[int, int]:
        return self.y1 - self.y0, self.x1 - self.x0


def read_medullary_union(path: str | Path):
    document = json.loads(Path(path).read_text())
    geometries = [
        shape(feature["geometry"])
        for feature in document.get("features", [])
        if feature.get("geometry")
    ]
    if not geometries:
        raise ValueError(f"No geometries found in {path}")
    return unary_union(geometries)


def read_rois(config: dict[str, Any]) -> list[Roi]:
    label_dir = Path(config["manual_label_dir"])
    image_dir = Path(config["manual_image_dir"])
    medullary = read_medullary_union(config["medullary_geojson"])
    rows: list[Roi] = []
    with Path(config["manual_crop_manifest"]).open(newline="") as handle:
        for row in csv.DictReader(handle):
            region = row["region_id"]
            label_path = label_dir / f"{region}.tif"
            image_path = image_dir / f"{region}_img.tif"
            if not label_path.exists():
                continue
            x0, y0, x1, y1 = (int(row[k]) for k in ("x0", "y0", "x1", "y1"))
            roi_polygon = Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
            overlap = float(roi_polygon.intersection(medullary).area / roi_polygon.area)
            tissue = "medullary" if overlap >= 0.8 else ("mixed" if overlap >= 0.2 else "outside")
            rows.append(
                Roi(
                    region=region,
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    label_path=label_path,
                    image_path=image_path,
                    medullary_overlap_fraction=overlap,
                    tissue=tissue,
                )
            )
    return rows


class RasterMaskSource:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.array = tifffile.memmap(self.path)

    def read_roi(self, roi: Roi) -> tuple[np.ndarray, dict[str, Any]]:
        mask = np.asarray(self.array[roi.y0 : roi.y1, roi.x0 : roi.x1]).copy()
        return mask, {
            "source": str(self.path),
            "source_kind": "raster_mask",
            "raw_labels": int(np.unique(mask[mask > 0]).size),
            "raster_overlap_pixels": 0,
        }


def _draw_ring(
    output: np.ndarray,
    ring: list[list[float]],
    label: int,
    roi: Roi,
    pixel_size_um: float,
) -> int:
    coords = np.asarray(ring, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[0] < 3:
        return 0
    xs = coords[:, 0] / pixel_size_um - roi.x0
    ys = coords[:, 1] / pixel_size_um - roi.y0
    rr, cc = draw_polygon(ys, xs, shape=output.shape)
    overlap = int(np.count_nonzero(output[rr, cc]))
    output[rr, cc] = label
    return overlap


def _rasterize_polygon_coordinates(
    output: np.ndarray,
    coordinates: Any,
    label: int,
    roi: Roi,
    pixel_size_um: float,
) -> int:
    if not coordinates:
        return 0
    overlap = _draw_ring(output, coordinates[0], label, roi, pixel_size_um)
    for hole in coordinates[1:]:
        coords = np.asarray(hole, dtype=np.float64)
        if coords.ndim != 2 or coords.shape[0] < 3:
            continue
        xs = coords[:, 0] / pixel_size_um - roi.x0
        ys = coords[:, 1] / pixel_size_um - roi.y0
        rr, cc = draw_polygon(ys, xs, shape=output.shape)
        output[rr, cc] = 0
    return overlap


class VertexParquetSource:
    def __init__(self, path: str | Path, pixel_size_um: float):
        self.path = Path(path)
        self.pixel_size_um = float(pixel_size_um)
        self.vertices = pd.read_parquet(
            self.path,
            columns=["cell_id", "vertex_x", "vertex_y", "label_id"],
        )

    def read_roi(self, roi: Roi) -> tuple[np.ndarray, dict[str, Any]]:
        pad_um = 10.0
        x0_um = roi.x0 * self.pixel_size_um - pad_um
        x1_um = roi.x1 * self.pixel_size_um + pad_um
        y0_um = roi.y0 * self.pixel_size_um - pad_um
        y1_um = roi.y1 * self.pixel_size_um + pad_um
        vertices = self.vertices
        candidate_rows = vertices[
            vertices["vertex_x"].between(x0_um, x1_um)
            & vertices["vertex_y"].between(y0_um, y1_um)
        ]
        candidate_ids = candidate_rows["cell_id"].unique()
        selected = vertices[vertices["cell_id"].isin(candidate_ids)]
        output = np.zeros(roi.shape, dtype=np.uint32)
        overlap_pixels = 0
        written = 0
        for written, (_, group) in enumerate(selected.groupby("cell_id", sort=False), start=1):
            ring = group[["vertex_x", "vertex_y"]].to_numpy().tolist()
            overlap_pixels += _draw_ring(
                output,
                ring,
                written,
                roi,
                self.pixel_size_um,
            )
        return output, {
            "source": str(self.path),
            "source_kind": "vector_parquet",
            "raw_labels": written,
            "raster_overlap_pixels": overlap_pixels,
        }


class ProsegGeometryCollectionSource:
    def __init__(self, path: str | Path, pixel_size_um: float):
        self.path = Path(path)
        self.pixel_size_um = float(pixel_size_um)
        document = json.loads(self.path.read_text())
        self.geometries = document.get("geometries", [])
        bounds: list[tuple[float, float, float, float]] = []
        for geometry in self.geometries:
            points: list[list[float]] = []
            if geometry.get("type") == "Polygon":
                points = [p for ring in geometry.get("coordinates", []) for p in ring]
            elif geometry.get("type") == "MultiPolygon":
                points = [
                    p
                    for polygon in geometry.get("coordinates", [])
                    for ring in polygon
                    for p in ring
                ]
            if points:
                coords = np.asarray(points, dtype=np.float64)
                bounds.append(
                    (
                        float(coords[:, 0].min()),
                        float(coords[:, 1].min()),
                        float(coords[:, 0].max()),
                        float(coords[:, 1].max()),
                    )
                )
            else:
                bounds.append((np.inf, np.inf, -np.inf, -np.inf))
        self.bounds = np.asarray(bounds)

    def read_roi(self, roi: Roi) -> tuple[np.ndarray, dict[str, Any]]:
        x0_um = roi.x0 * self.pixel_size_um
        x1_um = roi.x1 * self.pixel_size_um
        y0_um = roi.y0 * self.pixel_size_um
        y1_um = roi.y1 * self.pixel_size_um
        b = self.bounds
        selected_indices = np.flatnonzero(
            (b[:, 2] >= x0_um)
            & (b[:, 0] <= x1_um)
            & (b[:, 3] >= y0_um)
            & (b[:, 1] <= y1_um)
        )
        output = np.zeros(roi.shape, dtype=np.uint32)
        overlap_pixels = 0
        written = 0
        for geometry_index in selected_indices.tolist():
            geometry = self.geometries[geometry_index]
            written += 1
            if geometry.get("type") == "Polygon":
                overlap_pixels += _rasterize_polygon_coordinates(
                    output,
                    geometry.get("coordinates", []),
                    written,
                    roi,
                    self.pixel_size_um,
                )
            elif geometry.get("type") == "MultiPolygon":
                for polygon in geometry.get("coordinates", []):
                    overlap_pixels += _rasterize_polygon_coordinates(
                        output,
                        polygon,
                        written,
                        roi,
                        self.pixel_size_um,
                    )
        return output, {
            "source": str(self.path),
            "source_kind": "proseg_geometry_collection",
            "raw_labels": written,
            "raster_overlap_pixels": overlap_pixels,
        }


def build_model_sources(
    registry: dict[str, Any],
    pixel_size_um: float,
) -> dict[str, Any]:
    sources: dict[str, Any] = {}
    for model_key, model in registry["models"].items():
        kind = model["kind"]
        path = model["accuracy_source"]
        if kind == "raster_mask":
            sources[model_key] = RasterMaskSource(path)
        elif kind == "vector_parquet":
            sources[model_key] = VertexParquetSource(path, pixel_size_um)
        elif kind == "proseg_geometry_collection":
            sources[model_key] = ProsegGeometryCollectionSource(path, pixel_size_um)
        else:
            raise ValueError(f"Unsupported source kind for {model_key}: {kind}")
    return sources
