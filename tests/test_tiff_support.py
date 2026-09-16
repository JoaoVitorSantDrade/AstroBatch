from pathlib import Path
import json
import hashlib
import threading

import numpy as np
import pytest
from PIL import Image
from astropy.io import fits

from batch_logic import ProcessingConfig, find_fits_files, prepare_fits_file
from astroflow_logic import load_fits_data
from astroalign_logic import _load_fits_data_and_optional_masks
from astroalign_logic import process_all_alignments
from batch_compaction import BatchAccumulator, combine_bundles
import batch_compaction as compact
from hdr_logic import run_hdr_pipeline
from image_io import ImageFormat, read_tiff, validate_single_format, write_sidecar_json, write_tiff
from stacking_logic import discover_aligned_frames, inspect_fits, load_source_header
from temporal_analysis import read_fits_timestamp
from app.engines.execution import science_frame_bytes


def _write_tiff(path: Path, *, timestamp: str | None = None) -> None:
    image = Image.fromarray(np.arange(24, dtype=np.uint16).reshape(4, 6))
    kwargs = {}
    if timestamp:
        # TIFF tag 306 (DateTime), as emitted by common capture software.
        kwargs["tiffinfo"] = {306: timestamp}
    image.save(path, format="TIFF", **kwargs)


def test_tiff_is_discovered_and_prepared(tmp_path: Path):
    path = tmp_path / "light_002.tif"
    _write_tiff(path)
    config = ProcessingConfig(
        input_dir=tmp_path, output_dir=tmp_path, threshold_factor=1.0,
        crop_size=4, dry_run=True, copy_files=True, overwrite=False,
        opt_method="Crop", downsample_method="Nearest", downsample_scale=1.0,
    )
    assert find_fits_files(tmp_path) == [path]
    assert science_frame_bytes(path) == 4 * 6 * 4
    source, prepared, error = prepare_fits_file(path, config)
    assert source == path and error is None
    assert prepared is not None and prepared.dtype == np.float32
    assert prepared.shape == (4, 4)


def test_tiff_signature_is_detected_when_suffix_is_wrong(tmp_path: Path):
    source = tmp_path / "capture.tif"
    renamed = tmp_path / "capture.data"
    _write_tiff(source)
    source.replace(renamed)
    assert science_frame_bytes(renamed) == 4 * 6 * 4


def test_singular_fit_extension_is_supported(tmp_path: Path):
    path = tmp_path / "capture.fit"
    fits.PrimaryHDU(np.zeros((4, 6), dtype=np.uint16)).writeto(path)
    assert find_fits_files(tmp_path) == [path]
    assert science_frame_bytes(path) == 4 * 6 * 4


def test_tiff_loaders_and_timestamp(tmp_path: Path):
    path = tmp_path / "capture.tiff"
    _write_tiff(path, timestamp="2026:09:16 21:34:56")
    flow_data, flow_header = load_fits_data(path)
    assert flow_data.dtype == np.float32 and flow_data.shape == (4, 6)
    assert flow_header["DATE-OBS"].startswith("2026-09-16T21:34:56")
    aligned, header, valid, rejected = _load_fits_data_and_optional_masks(path)
    assert aligned.shape == (4, 6) and header["DATE-OBS"]
    assert valid.all() and not rejected.any()
    timestamp = read_fits_timestamp(path)
    assert timestamp["timestamp_state"] == "valid"
    assert timestamp["timestamp_normalized"].startswith("2026-09-16T21:34:56")
    assert load_source_header(path)["DATE-OBS"]


def test_tiff_geometry_is_available_to_stack(tmp_path: Path):
    path = tmp_path / "aligned.tif"
    _write_tiff(path)
    found, _ = discover_aligned_frames(tmp_path)
    assert found == [path]
    geometry = inspect_fits(path)
    assert (geometry.height, geometry.width, geometry.image_kind) == (4, 6, "Mono")


def test_mixed_fits_tiff_session_is_rejected(tmp_path: Path):
    tiff_path = tmp_path / "light.tif"
    fits_path = tmp_path / "light.fits"
    _write_tiff(tiff_path)
    fits.PrimaryHDU(np.zeros((4, 6), dtype=np.uint16)).writeto(fits_path)
    with pytest.raises(ValueError, match="Sessão mista"):
        validate_single_format([tiff_path, fits_path])


def test_compact_tiff_batches_keep_frame_weight_and_output_family(tmp_path: Path):
    shape = (4, 5, 3)
    manifests = []
    for batch_index, values in enumerate(((10.0, 20.0), (30.0, 40.0))):
        output = tmp_path / f"batch_{batch_index:02d}"
        accumulator = BatchAccumulator(
            batch_name=output.name,
            output_dir=output,
            image_format=ImageFormat.TIFF,
            method="Mean",
            memory_budget_mb=512,
        )
        for sequence, value in enumerate(values):
            accumulator.submit(
                sequence,
                np.full(shape, value, dtype=np.float32),
                np.ones(shape[:2], dtype=bool),
                np.zeros(shape[:2], dtype=bool),
                {"source": f"{batch_index}-{sequence}.tif", "frame": f"{sequence}.tif", "weight": 1.0},
            )
        result = accumulator.finalize()
        manifests.append(result["manifest"])

    final = combine_bundles(
        manifests,
        method="Mean",
        output_path=tmp_path / "final.fits",
        rejection_method="None",
        output_format="auto",
    )
    assert final["status"] == "success"
    assert Path(final["output_path"]).suffix == ".tif"
    image, _header = read_tiff(Path(final["output_path"]))
    # Four equally weighted frames: 25 ADU in every channel.
    np.testing.assert_array_equal(image, np.full(shape, 25, dtype=np.uint16))
    assert final["n_frames"] == 4


def test_compact_fits_batches_keep_fits_output_family(tmp_path: Path):
    manifests = []
    for index, value in enumerate((100.0, 300.0)):
        accumulator = BatchAccumulator(
            batch_name=f"fits_{index}",
            output_dir=tmp_path / f"fits_{index}",
            image_format=ImageFormat.FITS,
            method="Mean",
            memory_budget_mb=512,
        )
        accumulator.submit(
            0,
            np.full((3, 4), value, np.float32),
            np.ones((3, 4), bool),
            np.zeros((3, 4), bool),
            {},
        )
        manifests.append(accumulator.finalize()["manifest"])
    final = combine_bundles(
        manifests,
        method="Mean",
        output_path=tmp_path / "final.tif",
        rejection_method="None",
    )
    output = Path(final["output_path"])
    assert output.suffix == ".fits"
    with fits.open(output, memmap=False) as hdul:
        np.testing.assert_array_equal(hdul[0].data, np.full((3, 4), 200, np.uint16))


def test_hdr_tiff_output_is_normalized_to_tiff(tmp_path: Path):
    paths = []
    for index, value in enumerate((100, 200)):
        path = tmp_path / f"aligned_{index}.tif"
        write_tiff(path, np.full((4, 5), value, dtype=np.uint16))
        write_sidecar_json(path.with_suffix(path.suffix + ".json"), {"header": {"EXPTIME": 1.0}})
        paths.append(path)
    result = run_hdr_pipeline(
        {"input_paths": paths, "output_path": tmp_path / "hdr.fits", "row_band": 2},
        lambda _message: None,
        lambda *_args: None,
        None,
    )
    assert result["status"] == "success"
    assert Path(result["output_path"]).suffix == ".tif"
    assert read_tiff(Path(result["output_path"]))[0].shape == (4, 5)


def test_compact_align_drains_batches_without_individual_outputs(tmp_path: Path):
    base = tmp_path / "session"
    output = tmp_path / "aligned"
    base.mkdir()
    identity = np.eye(3).tolist()
    batches = {}
    source_digests = {}
    for batch_index in range(2):
        batch = base / f"batch_{batch_index:02d}"
        batch.mkdir()
        frames = {}
        for frame_index in range(3):
            name = f"frame_{frame_index:02d}.tif"
            source_path = batch / name
            write_tiff(source_path, np.full((16, 16, 3), batch_index * 30 + frame_index, dtype=np.uint16))
            source_digests[source_path] = hashlib.sha256(source_path.read_bytes()).hexdigest()
            frames[name] = {"status": "accepted", "matrix": identity}
        (batch / "flow_local.json").write_text(
            json.dumps({"frames": frames, "batch_anchor": "frame_00.tif", "transform_revision": "r"}),
            encoding="utf-8",
        )
        batches[batch.name] = {"status": "accepted", "matrix": identity, "local_transform_revision": "r"}
    (base / "global_flow.json").write_text(
        json.dumps({"batches": batches, "transform_revision": "g"}), encoding="utf-8"
    )

    result = process_all_alignments(
        base,
        output,
        {
            "aligned_storage": "batch_compact",
            "batch_stack_method": "Mean",
            "batch_rejection_method": "None",
            "debayer_pattern": "Nenhum",
            "interpolation": "nearest",
            "rgb_registration": False,
            "workers": 2,
            "memory_budget_mb": 512,
            "overwrite": True,
        },
        lambda _message: None,
        lambda *_args: None,
        threading.Event(),
    )
    assert result == (6, 0)
    assert sorted(path.name for path in output.rglob("batch_stack.tif")) == ["batch_stack.tif", "batch_stack.tif"]
    assert not list(output.rglob("frame_*.tif"))
    assert {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source_digests
    } == source_digests


def test_compact_accumulator_refuses_state_above_safe_ram_budget(monkeypatch, tmp_path: Path):
    class _Memory:
        available = 64 * 1024 * 1024

    monkeypatch.setattr(compact.psutil, "virtual_memory", lambda: _Memory())
    accumulator = BatchAccumulator(
        batch_name="large",
        output_dir=tmp_path,
        image_format=ImageFormat.TIFF,
        method="Mean",
        memory_budget_mb=64,
    )
    image = np.zeros((1024, 1024, 3), dtype=np.float32)
    with pytest.raises(MemoryError, match="limite seguro"):
        accumulator.submit(
            0, image, np.ones(image.shape[:2], bool), np.zeros(image.shape[:2], bool), {}
        )


def test_compact_median_is_explicitly_hierarchical(tmp_path: Path):
    accumulator = BatchAccumulator(
        batch_name="median",
        output_dir=tmp_path,
        image_format=ImageFormat.TIFF,
        method="Median",
        expected_frames=3,
        memory_budget_mb=512,
    )
    mask = np.ones((2, 2), dtype=bool)
    for sequence, value in enumerate((1.0, 100.0, 3.0)):
        accumulator.submit(sequence, np.full((2, 2), value, np.float32), mask, ~mask, {})
    result = accumulator.finalize()
    manifest = json.loads(result["manifest"].read_text(encoding="utf-8"))
    assert manifest["hierarchical"] is True
    np.testing.assert_array_equal(read_tiff(result["image"])[0], np.full((2, 2), 3, np.uint16))


def test_batch_rejection_request_is_not_silently_associative(tmp_path: Path):
    manifests = []
    for index, value in enumerate((10.0, 20.0)):
        accumulator = BatchAccumulator(
            batch_name=f"reject_{index}",
            output_dir=tmp_path / f"reject_{index}",
            image_format=ImageFormat.TIFF,
            method="Mean",
            rejection_method="SigmaClip",
            memory_budget_mb=512,
        )
        accumulator.submit(
            0,
            np.full((2, 2), value, np.float32),
            np.ones((2, 2), bool),
            np.zeros((2, 2), bool),
            {},
        )
        manifests.append(accumulator.finalize()["manifest"])
    result = combine_bundles(
        manifests,
        method="Mean",
        output_path=tmp_path / "rejected.fits",
        rejection_method="None",
    )
    report = json.loads(Path(result["report"]).read_text(encoding="utf-8"))
    assert report["hierarchical"] is True
    assert report["warning"]


def test_compact_rgb_extrema_merge_scientific_state(tmp_path: Path):
    manifests = []
    for batch_index, value in enumerate((10.0, 30.0)):
        accumulator = BatchAccumulator(
            batch_name=f"max_{batch_index}",
            output_dir=tmp_path / f"max_{batch_index}",
            image_format=ImageFormat.TIFF,
            method="Maximum",
            memory_budget_mb=512,
        )
        accumulator.submit(
            0,
            np.full((3, 4, 3), value, np.float32),
            np.ones((3, 4), bool),
            np.zeros((3, 4), bool),
            {},
        )
        manifests.append(accumulator.finalize()["manifest"])
    merged = combine_bundles(
        manifests,
        method="Maximum",
        output_path=tmp_path / "max.fits",
        rejection_method="None",
    )
    assert merged["status"] == "success"
    np.testing.assert_array_equal(read_tiff(Path(merged["output_path"]))[0], np.full((3, 4, 3), 30, np.uint16))
