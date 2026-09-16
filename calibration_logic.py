from __future__ import annotations

import os
import threading
import tempfile
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path

import numpy as np
import psutil
from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning

from cpu_kernels import calibrate_inplace
from cpu_runtime import configure_worker_runtime, physical_core_count
from image_io import (
    IMAGE_SUFFIXES,
    ImageFormat,
    TIFF_SUFFIXES,
    detect_format,
    read_tiff,
    read_tiff_header,
    read_tiff_sidecar_arrays,
    tiff_shape,
    validate_single_format,
    write_npz_sidecar,
    write_sidecar_json,
    write_tiff,
)

# Suprime avisos de verificação de cabeçalho do Astropy.
warnings.simplefilter("ignore", category=AstropyWarning)


FITS_SUFFIXES = IMAGE_SUFFIXES
MAX_WORKERS = 16


def _sanitize_float_header(header: fits.Header | None) -> fits.Header:
    """
    Remove metadados de escala de inteiros que não são mais válidos
    quando os dados passam a ser float32.
    """
    result = header.copy() if header is not None else fits.Header()

    # Esses campos podem vir de FITS uint16/signed-int16 anteriores.
    result.remove("BITPIX", ignore_missing=True)
    result.remove("BZERO", ignore_missing=True)
    result.remove("BSCALE", ignore_missing=True)
    result.remove("BLANK", ignore_missing=True)

    return result


def load_fits_data(filepath: Path) -> tuple[np.ndarray, fits.Header]:
    """
    Carrega a primeira imagem mono/RGB encontrada como float32.

    Importante: nenhuma conversão para uint16 é feita aqui.
    """
    if filepath.suffix.casefold() in TIFF_SUFFIXES:
        data, header = read_tiff(filepath)
        return np.asarray(data, dtype=np.float32), _sanitize_float_header(header)
    with fits.open(
        filepath,
        memmap=False,
        ignore_missing_end=True,
    ) as hdul:
        for hdu in hdul:
            if (
                hdu.is_image
                and hdu.name not in {"VALID_MASK", "SAT_MASK", "DISAGREE", "HDR_META"}
                and hdu.data is not None
                and hdu.data.ndim in (2, 3)
            ):
                data = np.asarray(
                    hdu.data,
                    dtype=np.float32,
                ).copy()
                header = _sanitize_float_header(hdu.header)
                return data, header

    raise ValueError(f"Imagem mono/RGB não encontrada em {filepath.name}")


def _read_for_master(
    filepath: Path,
) -> tuple[np.ndarray, fits.Header]:
    """Leitura isolada para despacho paralelo."""
    return load_fits_data(filepath)


def _inspect_master_frame(filepath: Path) -> tuple[tuple[int, ...], fits.Header, int]:
    """Read master geometry without retaining the image in RAM."""
    if filepath.suffix.casefold() in TIFF_SUFFIXES:
        shape = tiff_shape(filepath)
        if len(shape) in (2, 3):
            return tuple(shape), _sanitize_float_header(read_tiff_header(filepath)), 0
        raise ValueError(f"Imagem mono/RGB não encontrada em {filepath.name}")
    with fits.open(filepath, memmap=True, ignore_missing_end=True) as hdul:
        for index, hdu in enumerate(hdul):
            if hdu.is_image and hdu.name not in {"VALID_MASK", "SAT_MASK", "DISAGREE", "HDR_META"} and hdu.shape is not None and len(hdu.shape) in (2, 3):
                return tuple(hdu.shape), _sanitize_float_header(hdu.header), index
    raise ValueError(f"Imagem mono/RGB não encontrada em {filepath.name}")


def _read_master_band(filepath: Path, hdu_index: int, y1: int, y2: int) -> np.ndarray:
    """Load one scaled FITS row band as a compact float32 array."""
    if filepath.suffix.casefold() in TIFF_SUFFIXES:
        try:
            import tifffile
            data = tifffile.memmap(filepath)
        except Exception:
            data, _header = read_tiff(filepath)
        source = np.asarray(data)
        if source.ndim == 2:
            band = source[y1:y2, :]
        elif source.ndim == 3 and source.shape[-1] in (3, 4):
            band = source[y1:y2, :, :]
        elif source.ndim == 3 and source.shape[0] in (3, 4):
            band = source[:, y1:y2, :]
        else:
            raise ValueError(f"Geometria RGB não reconhecida em {filepath.name}: {source.shape}")
        return np.ascontiguousarray(band, dtype=np.float32)
    with fits.open(filepath, memmap=False, ignore_missing_end=True) as hdul:
        hdu = hdul[hdu_index]
        if hdu.data is None or hdu.shape is None or len(hdu.shape) not in (2, 3):
            raise ValueError(f"Imagem mono/RGB não encontrada em {filepath.name}")
        if len(hdu.shape) == 2:
            section = hdu.section[y1:y2, :]
        elif hdu.shape[0] in (3, 4):
            section = hdu.section[:, y1:y2, :]
        else:
            section = hdu.section[y1:y2, :, :]
        return np.ascontiguousarray(section, dtype=np.float32)


def _master_band_rows(frame_count: int, width: int, height: int, channels: int = 1) -> int:
    """Choose a full-width band that consumes at most 25% of free RAM."""
    available = max(64 * 1024 * 1024, int(psutil.virtual_memory().available))
    budget = max(32 * 1024 * 1024, available // 4)
    bytes_per_row = max(1, frame_count * width * max(1, channels) * np.dtype(np.float32).itemsize)
    return max(1, min(height, budget // bytes_per_row))


def _finite_range(data: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(data, dtype=np.float32)[np.isfinite(data)]
    if finite.size == 0:
        return 0.0, 0.0
    return float(np.min(finite)), float(np.max(finite))


def normalize_to_uint16(
    data: np.ndarray, data_min: float, data_max: float
) -> np.ndarray:
    """Map data to a shared unsigned-16-bit range."""
    values = np.asarray(data, dtype=np.float32)
    if not np.isfinite(data_min) or not np.isfinite(data_max) or data_max <= data_min:
        return np.zeros(values.shape, dtype=np.uint16)
    clean = np.nan_to_num(values, nan=data_min, posinf=data_max, neginf=data_min)
    scale = np.float32(65535.0 / (data_max - data_min))
    return np.clip((clean - np.float32(data_min)) * scale, 0.0, 65535.0).astype(
        np.uint16
    )


def _restore_normalized_values(data: np.ndarray, header: fits.Header) -> np.ndarray:
    """Restore a normalized calibration master to its in-RAM value range."""
    if not bool(header.get("CALNORM", False)):
        return np.asarray(data, dtype=np.float32)
    data_min = float(header.get("CALMIN", 0.0))
    data_max = float(header.get("CALMAX", data_min))
    if not np.isfinite(data_min) or not np.isfinite(data_max) or data_max <= data_min:
        return np.full(np.asarray(data).shape, data_min, dtype=np.float32)
    scale = np.float32((data_max - data_min) / 65535.0)
    return np.asarray(data, dtype=np.float32) * scale + np.float32(data_min)


def load_calibration_master(filepath: Path) -> np.ndarray:
    """Load a uint16 calibration master and restore its recorded scale in RAM."""
    data, header = load_fits_data(filepath)
    return _restore_normalized_values(data, header)


def save_uint16_fits(
    data: np.ndarray,
    header: fits.Header | None,
    output_path: Path,
    data_min: float,
    data_max: float,
    valid_mask: np.ndarray | None = None,
    sat_mask: np.ndarray | None = None,
) -> None:
    """Persist a FITS 16-bit image using the provided shared value range."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_data = normalize_to_uint16(data, data_min, data_max)

    output_header = _sanitize_float_header(header)
    output_header["CALNORM"] = (True, "Shared calibration normalization")
    output_header["CALMIN"] = (float(data_min), "Calibration normalization minimum")
    output_header["CALMAX"] = (float(data_max), "Calibration normalization maximum")

    if output_path.suffix.casefold() in TIFF_SUFFIXES:
        write_tiff(
            output_path,
            output_data,
            metadata={
                "schema_version": 1,
                "format": "TIFF",
                "calnorm": True,
                "calmin": float(data_min),
                "calmax": float(data_max),
                "linear": True,
            },
        )
        arrays: dict[str, np.ndarray] = {}
        if valid_mask is not None:
            arrays["valid_mask"] = np.asarray(valid_mask, dtype=np.uint8)
        if sat_mask is not None:
            arrays["sat_mask"] = np.asarray(sat_mask, dtype=np.uint8)
        sidecar_name = output_path.name + ".astrobatch.npz"
        if arrays:
            write_npz_sidecar(output_path.with_suffix(output_path.suffix + ".astrobatch.npz"), **arrays)
        write_sidecar_json(
            output_path.with_suffix(output_path.suffix + ".json"),
            {
                "schema_version": 1,
                "format": "TIFF",
                "calnorm": True,
                "calmin": float(data_min),
                "calmax": float(data_max),
                "sidecar": sidecar_name if arrays else None,
                "header": {str(k): str(v) for k, v in output_header.items()},
            },
        )
        return

    # Astropy writes uint16 using the standard FITS unsigned scaling cards.
    hdu = fits.PrimaryHDU(
        data=output_data,
        header=output_header,
    )

    hdul = fits.HDUList([hdu])
    if valid_mask is not None:
        hdul.append(fits.ImageHDU(np.asarray(valid_mask, np.uint8), name="VALID_MASK"))
    if sat_mask is not None:
        hdul.append(fits.ImageHDU(np.asarray(sat_mask, np.uint8), name="SAT_MASK"))
    fd, temporary = tempfile.mkstemp(prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent)
    os.close(fd)
    try:
        hdul.writeto(temporary, overwrite=True, output_verify="ignore")
        os.replace(temporary, output_path)
    finally:
        hdul.close()
        Path(temporary).unlink(missing_ok=True)


def _safe_median(data: np.ndarray) -> float:
    """Mediana finita sem alterar o array original."""
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return 1.0
    return float(np.median(finite))


def make_master(
    folder_path: Path,
    output_path: Path,
    app_print,
    cancel_event: threading.Event,
) -> np.ndarray | None:
    """
    Cria Master Dark ou Master Flat e o persiste em FIT 16-bit.

    A mediana usa float32 somente em RAM; nenhum FITS float32 é criado.
    """
    files = sorted(
        [
            f
            for f in folder_path.iterdir()
            if (f.is_file() and f.suffix.lower() in FITS_SUFFIXES)
        ],
        key=lambda p: p.name.casefold(),
    )

    if not files:
        app_print(f"Nenhum frame encontrado em {folder_path.name} para criar Master.\n")
        return None

    # A single master cannot safely combine FITS and TIFF planes: their
    # metadata/mask conventions differ and the resulting output family would
    # be ambiguous. Keep the guard here as well as at the pipeline boundary
    # so direct callers receive the same behavior.
    validate_single_format(files)

    app_print(f"Inspecionando {len(files)} frames para gerar Master ({folder_path.name})...\n")
    reference_shape, base_header, hdu_index = _inspect_master_frame(files[0])
    incompatible = []
    for index, filepath in enumerate(files[1:], start=1):
        shape, _, candidate_hdu_index = _inspect_master_frame(filepath)
        if shape != reference_shape or candidate_hdu_index != hdu_index:
            incompatible.append(index)
    if incompatible:
        raise ValueError(
            "Frames incompatíveis encontrados durante a criação do Master. "
            f"Shape de referência={reference_shape}; índices incompatíveis={incompatible[:10]}"
        )

    channels_first = len(reference_shape) == 3 and reference_shape[0] in (3, 4) and reference_shape[-1] not in (3, 4)
    if len(reference_shape) == 2:
        height, width, channels = reference_shape[0], reference_shape[1], 1
    elif len(reference_shape) == 3 and not channels_first and reference_shape[-1] in (3, 4):
        height, width, channels = reference_shape[0], reference_shape[1], reference_shape[2]
    elif channels_first:
        channels, height, width = reference_shape
    else:
        raise ValueError(f"Geometria mono/RGB não reconhecida: {reference_shape}")
    band_rows = _master_band_rows(len(files), width, height, channels)
    app_print(
        f"Calculando Master em bandas de {band_rows} linhas; "
        f"buffer={len(files)}x{band_rows}x{width} float32.\n"
    )
    master_data = np.empty(reference_shape, dtype=np.float32)
    for y1 in range(0, height, band_rows):
        if cancel_event.is_set():
            return None
        y2 = min(height, y1 + band_rows)
        band_shape = (len(files), y2 - y1, width) if channels == 1 else (len(files), y2 - y1, width, channels)
        # FITS RGB uses CHW; TIFF RGB uses HWC.  Preserve the source layout in
        # the master so calibration remains shape-compatible with its lights.
        if channels_first:
            band_shape = (len(files), channels, y2 - y1, width)
        band_values = np.empty(band_shape, dtype=np.float32)
        for index, filepath in enumerate(files):
            band_values[index] = _read_master_band(filepath, hdu_index, y1, y2)
        median_band = np.median(band_values, axis=0)
        if channels_first:
            master_data[:, y1:y2, :] = median_band
        else:
            master_data[y1:y2] = median_band

    if cancel_event.is_set():
        return None

    master_min, master_max = _finite_range(master_data)
    save_uint16_fits(master_data, base_header, output_path, master_min, master_max)

    app_print(f"Master 16-bit normalizado salvo em: {output_path.name}\n")

    return master_data


def prepare_master_flat(
    master_flat: np.ndarray,
) -> np.ndarray:
    """
    Normaliza o Master Flat em RAM antes de sua gravação em 16-bit.

    Pixels não utilizáveis são neutralizados para 1.0, evitando divisão
    por zero sem modificar artificialmente pixels válidos.
    """
    flat = np.asarray(
        master_flat,
        dtype=np.float32,
    ).copy()

    median_value = _safe_median(flat)

    if not np.isfinite(median_value) or median_value <= 0:
        median_value = 1.0

    flat /= np.float32(median_value)

    # Valores não finitos ou próximos de zero não podem funcionar como
    # divisor. Eles são neutralizados para 1.0.
    invalid = ~np.isfinite(flat) | (flat <= np.float32(0.01))

    flat[invalid] = np.float32(1.0)

    return flat


def calibrate_single_frame(
    light_path: Path,
    output_dir: Path,
    master_dark: np.ndarray | None,
    master_flat: np.ndarray | None,
    overwrite: bool,
    normalization_range: tuple[float, float],
) -> str | None:
    """
    Calibra um LIGHT e o grava normalizado em FIT 16-bit.

    Fórmula:

        calibrated = (light - dark) / flat

    Sem clipping para uint16 e sem saturação artificial em zero.
    """
    out_path = output_dir / light_path.name

    if out_path.exists() and not overwrite:
        return f"Ignorado (já existe): {light_path.name}"

    try:
        data, header = load_fits_data(light_path)
        channels_first = data.ndim == 3 and data.shape[0] in (3, 4) and data.shape[-1] not in (3, 4)
        channel_axis = 0 if channels_first else (-1 if data.ndim == 3 else None)
        if light_path.suffix.casefold() in TIFF_SUFFIXES:
            spatial_shape = data.shape[:2] if data.ndim == 3 else data.shape
            side_arrays = read_tiff_sidecar_arrays(light_path)
            valid = np.asarray(side_arrays.get("valid_mask", np.ones(spatial_shape)), dtype=bool)
            saturated = np.asarray(side_arrays.get("sat_mask", np.zeros(spatial_shape)), dtype=bool)
            if valid.shape != spatial_shape:
                valid = np.ones(spatial_shape, dtype=bool)
            if saturated.shape != spatial_shape:
                saturated = np.zeros(spatial_shape, dtype=bool)
        else:
            from app.infrastructure.fits_masks import read_science_masks
            spatial_shape = data.shape[-2:] if data.ndim == 3 else data.shape
            valid, saturated = read_science_masks(light_path, spatial_shape)
        threshold = header.get("SATURATE", header.get("SATLEVEL"))
        if threshold is not None:
            clipped = data >= float(threshold)
            saturated |= np.any(clipped, axis=channel_axis) if data.ndim == 3 else clipped
            header["SATKNOWN"] = True
        # The calibrated plane will have a different encoding; preserve the
        # source classification in a mask, never reuse its numeric threshold.
        header.remove("SATURATE", ignore_missing=True)
        header.remove("SATLEVEL", ignore_missing=True)

        if master_dark is not None and master_dark.shape != data.shape:
            raise ValueError(f"Master Dark possui dimensão incompatível: {master_dark.shape} vs {data.shape}")
        if master_flat is not None and master_flat.shape != data.shape:
            raise ValueError(f"Master Flat possui dimensão incompatível: {master_flat.shape} vs {data.shape}")
        data = calibrate_inplace(
            np.ascontiguousarray(data, dtype=np.float32), master_dark, master_flat
        )

        # ----------------------------------------------------
        # Garantia final de dtype.
        # ----------------------------------------------------
        data = np.asarray(
            data,
            dtype=np.float32,
        )

        # Não sanitizamos NaN/Inf silenciosamente aqui.
        # Eles podem indicar problema real no Flat/Dark e não devem ser
        # escondidos nesta etapa do workflow.

        finite = np.isfinite(data)
        valid &= np.all(finite, axis=channel_axis) if data.ndim == 3 else finite
        save_uint16_fits(data, header, out_path, *normalization_range, valid_mask=valid, sat_mask=saturated)

        return None

    except Exception as exc:
        return f"Erro em {light_path.name}: {exc}"


def _calibration_range(
    lights: list[Path],
    master_dark: np.ndarray | None,
    master_flat: np.ndarray | None,
    app_print,
    cancel_event: threading.Event,
) -> tuple[float, float] | None:
    """Determine one finite post-calibration range for all LIGHT frames."""
    data_min = np.inf
    data_max = -np.inf
    for index, light_path in enumerate(lights, start=1):
        if cancel_event.is_set():
            return None
        data, _ = load_fits_data(light_path)
        if master_dark is not None and master_dark.shape != data.shape:
            raise ValueError(f"Master Dark incompatível com {light_path.name}")
        if master_flat is not None and master_flat.shape != data.shape:
            raise ValueError(f"Master Flat incompatível com {light_path.name}")
        calibrated = calibrate_inplace(
            np.ascontiguousarray(data, dtype=np.float32), master_dark, master_flat
        )
        current_min, current_max = _finite_range(calibrated)
        data_min = min(data_min, current_min)
        data_max = max(data_max, current_max)
        if index % 10 == 0 or index == len(lights):
            app_print(f"Normalização: analisando {index}/{len(lights)} frames...\n")
    return float(data_min), float(data_max)


def run_calibration_pipeline(
    config: dict,
    app_print,
    app_progress,
    cancel_event: threading.Event,
):
    """
    Executa a calibração completa.

    A matemática de calibração usa float32 apenas em RAM. Masters e LIGHTs
    calibrados são persistidos como FITS uint16; os LIGHTs compartilham uma
    única faixa global de normalização calculada antes da gravação.
    """

    input_dir = Path(config["input_dir"])
    lights = sorted(
        [
            filepath
            for filepath in input_dir.iterdir()
            if filepath.is_file() and filepath.suffix.casefold() in IMAGE_SUFFIXES
        ],
        key=lambda p: p.name.casefold(),
    )
    if not lights:
        app_print("Nenhum LIGHT encontrado na pasta de entrada.\n")
        return
    try:
        source_format = validate_single_format(lights)
    except ValueError as exc:
        app_print(f"ERRO: {exc}\n")
        return

    def _validate_master_format(label: str, value: str | os.PathLike) -> None:
        """Keep dark/flat masters in the same family as the LIGHT session."""
        candidate = Path(value)
        if candidate.is_file():
            master_format = validate_single_format([candidate])
        elif candidate.is_dir():
            master_files = sorted(
                (item for item in candidate.iterdir()
                 if item.is_file() and item.suffix.casefold() in IMAGE_SUFFIXES),
                key=lambda item: item.name.casefold(),
            )
            if not master_files:
                return
            master_format = validate_single_format(master_files)
        else:
            return
        if master_format is not source_format:
            raise ValueError(
                f"{label} usa {master_format.value.upper()}, mas os LIGHTs usam "
                f"{source_format.value.upper()}; misturar FIT/FITS e TIF/TIFF não é permitido."
            )

    for enabled, key in (("apply_dark", "dark_path"), ("apply_flat", "flat_path")):
        if config.get(enabled) and (not config.get(key) or not Path(config[key]).exists()):
            raise ValueError(f"{key}: caminho de calibração obrigatório e válido")
        if config.get(enabled) and config.get(key):
            _validate_master_format("Master Dark" if key == "dark_path" else "Master Flat", config[key])

    output_dir = Path(config["output_dir"])

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # 1. MASTERS
    # ========================================================

    master_dark: np.ndarray | None = None

    if config.get("apply_dark", False) and config.get("dark_path"):
        dark_path = Path(config["dark_path"])

        if dark_path.is_dir() and config.get(
            "create_master",
            False,
        ):
            master_dark_path = input_dir.parent / f"MasterDark{source_format.default_suffix}"

            master_dark = make_master(
                dark_path,
                master_dark_path,
                app_print,
                cancel_event,
            )

        elif dark_path.is_file():
            master_dark = load_calibration_master(dark_path)

            app_print("Master Dark 16-bit carregado para cálculo em RAM.\n")

    if cancel_event.is_set():
        app_print("\nCalibração cancelada durante o Master Dark.\n")
        return

    master_flat: np.ndarray | None = None

    if config.get("apply_flat", False) and config.get("flat_path"):
        flat_path = Path(config["flat_path"])

        if flat_path.is_dir() and config.get(
            "create_master",
            False,
        ):
            master_flat_path = input_dir.parent / f"MasterFlat{source_format.default_suffix}"

            raw_master_flat = make_master(
                flat_path,
                master_flat_path,
                app_print,
                cancel_event,
            )

            if raw_master_flat is not None:
                master_flat = prepare_master_flat(raw_master_flat)

                # Atualiza o arquivo do Master Flat com a forma
                # normalizada que será efetivamente usada.
                flat_min, flat_max = _finite_range(master_flat)
                save_uint16_fits(
                    master_flat,
                    None,
                    master_flat_path,
                    flat_min,
                    flat_max,
                )

                app_print("Master Flat normalizado e salvo em 16-bit.\n")

                del raw_master_flat

        elif flat_path.is_file():
            raw_master_flat = load_calibration_master(flat_path)

            master_flat = prepare_master_flat(raw_master_flat)

            del raw_master_flat

            app_print("Master Flat 16-bit carregado e normalizado em RAM.\n")

    if cancel_event.is_set():
        app_print("\nCalibração cancelada durante os Masters.\n")
        return

    # ========================================================
    # 2. LIGHTS
    # ========================================================

    total = len(lights)

    app_print("\nCalculando faixa global para normalização 16-bit...\n")
    normalization_range = _calibration_range(
        lights,
        master_dark,
        master_flat,
        app_print,
        cancel_event,
    )
    if normalization_range is None or cancel_event.is_set():
        app_print("\nCalibração cancelada durante a normalização.\n")
        return
    app_print(
        f"Faixa global calibrada: {normalization_range[0]:.6g} a "
        f"{normalization_range[1]:.6g}.\n"
    )

    app_print(f"\nIniciando calibração paralela de {total} frames...\n")

    processed = 0
    failed = 0
    skipped = 0

    workers = max(
        1,
        min(
            MAX_WORKERS,
            8,
            physical_core_count(),
            (os.cpu_count() or 1),
        ),
    )

    # Janela limitada de futures para evitar criar milhares de objetos
    # Future simultaneamente. Mantém o uso de memória previsível.
    max_inflight = max(
        workers,
        workers * 2,
    )

    with ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix="calibration",
        initializer=partial(configure_worker_runtime, 1),
    ) as executor:
        future_to_light: dict = {}
        next_index = 0

        def submit_more() -> None:
            nonlocal next_index

            while (
                next_index < total
                and len(future_to_light) < max_inflight
                and not cancel_event.is_set()
            ):
                light_path = lights[next_index]

                next_index += 1

                future = executor.submit(
                    calibrate_single_frame,
                    light_path,
                    output_dir,
                    master_dark,
                    master_flat,
                    bool(
                        config.get(
                            "overwrite",
                            True,
                        )
                    ),
                    normalization_range,
                )

                future_to_light[future] = light_path

        submit_more()

        while future_to_light:
            if cancel_event.is_set():
                app_print(
                    "\nCancelamento solicitado. Interrompendo novas calibrações...\n"
                )

                for future in future_to_light:
                    future.cancel()

                break

            done_future = next(as_completed(future_to_light))

            light_path = future_to_light.pop(done_future)

            processed += 1

            try:
                message = done_future.result()

            except Exception as exc:
                message = f"Erro inesperado em {light_path.name}: {exc}"

            if message:
                if message.startswith("Ignorado (já existe)"):
                    skipped += 1
                else:
                    failed += 1

                app_print(message + "\n")

            if processed % 10 == 0 or processed == total:
                app_progress(
                    processed,
                    total,
                    f"Calibrando ({processed}/{total})...",
                )

            submit_more()

    if cancel_event.is_set():
        app_print("\n>>> Calibração cancelada pelo usuário. <<<\n")

        return

    saved = processed - failed - skipped

    app_print(
        "\n>>> Calibração Finalizada: "
        f"{saved} arquivos salvos, "
        f"{skipped} ignorados, "
        f"{failed} erros. <<<\n"
    )
    return {"status": "partial" if failed and saved else "failed" if failed else "success",
            "saved": saved, "skipped": skipped, "failed": failed,
            "message": f"Calibration: {saved} salvos, {skipped} ignorados, {failed} erros."}
