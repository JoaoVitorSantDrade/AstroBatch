"""
stacking_logic.py

AstroStack - Empilhamento otimizado para o AstroProcessManager.

Objetivos principais desta versão:

- API pública compatível com main.py:
      process_all_stacking(...)
- processamento por blocos (streaming);
- não cria um array global [N, H, W];
- para Mean/Sum/Maximum/Minimum sem rejeição, combina os frames
  de forma incremental, usando memória O(bloco);
- para Median/rejeição, mantém apenas o bloco atual em RAM;
- leitura FITS por section/slicing sempre que possível;
- suporte a RGB em CHW/HWC;
- suporte à extensão VALID_MASK criada pelo AstroAlign;
- seleção de frames sem reabrir flow_local.json para cada frame;
- normalização opcional com amostragem determinística para reduzir IO;
- cancelamento cooperativo;
- overwrite da saída;
- preservação do header do primeiro frame selecionado;
- FITS comprimido na saída, quando configurado.

Observação importante sobre memória:

Para métodos que exigem rejeição/outlier rejection, ainda é necessário
manter todos os valores do bloco atual para decidir quais pixels serão
rejeitados. A diferença é que somente o bloco atual permanece em RAM,
e não a imagem inteira.
"""

from __future__ import annotations

import json
import math
import os
import threading
import warnings
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning

warnings.simplefilter("ignore", category=AstropyWarning)


# ============================================================
# CONSTANTES
# ============================================================

FITS_SUFFIXES = {".fits", ".fit", ".fts"}

DEFAULT_CHUNK_SIZE = 512
DEFAULT_MEMORY_BUDGET_MB = 4096 * 4
DEFAULT_NORMALIZATION_MAX_SAMPLES = 1_000_000


# ============================================================
# CONFIGURAÇÃO
# ============================================================


@dataclass
class StackingConfig:
    """Configuração completa do AstroStack."""

    # ---- Diretórios ----
    base_dir: Path = field(default_factory=Path)
    input_dir: Path = field(default_factory=Path)
    output_dir: Path = field(default_factory=Path)

    # ---- Seleção de Frames ----
    selection_mode: Literal["All", "BestPercentage"] = "BestPercentage"
    selection_percentage: float = 80.0
    selection_metric: Literal[
        "quality",
        "fwhm",
        "star_count",
        "snr",
    ] = "quality"

    # ---- Combinação ----
    method: Literal[
        "Median",
        "Mean",
        "Sum",
        "Maximum",
        "Minimum",
    ] = "Median"

    # ---- Rejeição de Outliers ----
    rejection_method: Literal[
        "None",
        "SigmaClip",
        "Winsorized",
        "MAD",
    ] = "SigmaClip"
    rejection_low: float = 3.0
    rejection_high: float = 3.0

    # ---- Normalização ----
    normalize: bool = True
    normalize_method: Literal["Median", "Mode"] = "Median"

    # ---- Pós-processamento ----
    apply_dither_correction: bool = False

    # ---- Saída ----
    output_name: str = "stacked_image.fits"
    output_bit_depth: Literal["16-bit", "32-bit"] = "32-bit"
    compress_output: bool = True

    # ---- Performance ----
    workers: int | None = None
    chunk_size: int = DEFAULT_CHUNK_SIZE
    memory_budget_mb: int = DEFAULT_MEMORY_BUDGET_MB
    normalization_max_samples: int = DEFAULT_NORMALIZATION_MAX_SAMPLES

    @property
    def worker_count(self) -> int:
        try:
            cpu_count = (
                getattr(
                    os,
                    "process_cpu_count",
                    os.cpu_count,
                )()
                or 1
            )
        except Exception:
            cpu_count = os.cpu_count() or 1

        if self.workers is not None:
            try:
                return max(
                    1,
                    min(16, int(self.workers)),
                )
            except (TypeError, ValueError):
                pass

        return max(
            1,
            min(16, cpu_count),
        )

    @property
    def effective_chunk_size(self) -> int:
        size = max(32, int(self.chunk_size))
        return min(1024, size)


# ============================================================
# ESTRUTURAS INTERNAS
# ============================================================


@dataclass(slots=True)
class FrameInfo:
    path: Path
    name: str
    batch: str
    metrics: dict[str, Any]
    quality: float
    star_count: float
    fwhm: float
    snr: float
    rms: float
    has_valid_mask: bool
    shape: tuple[int, ...]
    image_kind: str
    channels: int


@dataclass(slots=True)
class FrameGeometry:
    height: int
    width: int
    image_kind: str
    channels: int
    hdu_index: int
    mask_hdu_index: int | None


@dataclass(slots=True)
class BlockRead:
    data: np.ndarray
    mask: np.ndarray


# ============================================================
# CPU / PERFORMANCE
# ============================================================


def get_optimal_worker_count() -> int:
    try:
        cpu_count = (
            getattr(
                os,
                "process_cpu_count",
                os.cpu_count,
            )()
            or 1
        )
    except Exception:
        cpu_count = os.cpu_count() or 1

    return max(
        1,
        min(16, cpu_count),
    )


def estimate_block_memory_bytes(
    n_frames: int,
    chunk_size: int,
    channels: int,
    rejection: bool,
) -> int:
    """
    Estima aproximadamente a RAM do bloco de trabalho.

    O fator considera arrays temporários criados durante rejeição.
    """

    pixels = chunk_size * chunk_size

    # Base: float32 por valor.
    base = n_frames * pixels * channels * 4

    # Máscara/validação e temporários de rejeição.
    if rejection:
        # masked_block + valid + temporários estatísticos.
        return int(base * 3.2)

    return int(base * 1.35)


def adapt_chunk_size(
    requested: int,
    n_frames: int,
    channels: int,
    memory_budget_mb: int,
    rejection: bool,
) -> int:
    """
    Reduz o tamanho do bloco automaticamente para respeitar a meta
    aproximada de memória.

    Mantém a dimensão >= 32 quando possível.
    """

    chunk = max(32, min(1024, int(requested)))
    budget = max(64, int(memory_budget_mb)) * 1024 * 1024

    while chunk > 32:
        estimated = estimate_block_memory_bytes(
            n_frames,
            chunk,
            channels,
            rejection,
        )

        if estimated <= budget:
            break

        chunk //= 2

    return max(32, chunk)


# ============================================================
# DESCOBERTA
# ============================================================


def discover_aligned_frames(
    input_dir: Path,
) -> tuple[list[Path], dict[str, dict]]:
    """
    Descobre FITS recursivamente com uma única caminhada pelo diretório.

    Evita fazer três rglob independentes + set() + nova ordenação.
    """

    if not input_dir.exists():
        return [], {}

    found: list[Path] = []
    batch_metadata: dict[str, dict] = {}

    for root, _, filenames in os.walk(input_dir):
        root_path = Path(root)

        for filename in filenames:
            path = root_path / filename

            if path.suffix.casefold() not in FITS_SUFFIXES:
                continue

            found.append(path)

            batch_name = root_path.name
            meta = batch_metadata.setdefault(
                batch_name,
                {
                    "path": root_path,
                    "frame_count": 0,
                    "frames": [],
                },
            )

            meta["frame_count"] += 1
            meta["frames"].append(filename)

    found.sort(
        key=lambda p: (
            str(p.parent).casefold(),
            p.name.casefold(),
        )
    )

    return found, batch_metadata


# ============================================================
# JSON / FLOW
# ============================================================


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)

    return value if isinstance(value, dict) else {}


def load_flow_cache(
    base_dir: Path,
    batch_metadata: dict[str, dict],
) -> dict[str, dict]:
    """
    Carrega cada flow_local.json no máximo uma vez, lendo diretamente da
    pasta base original onde o AstroFlow calculou as métricas, e não
    da pasta de destino alinhada.
    """
    cache: dict[str, dict] = {}

    if not base_dir or not str(base_dir).strip() or str(base_dir) == ".":
        return cache

    for batch_name in batch_metadata:
        flow_path = base_dir / batch_name / "flow_local.json"

        if not flow_path.is_file():
            continue

        try:
            cache[batch_name] = load_json(flow_path)
        except Exception:
            cache[batch_name] = {}

    # Se a estrutura não tiver subpasta Batch, ainda permite root.
    root_flow = base_dir / "flow_local.json"
    if root_flow.is_file() and "root" not in cache:
        try:
            cache["root"] = load_json(root_flow)
        except Exception:
            cache["root"] = {}

    return cache


# ============================================================
# INSPEÇÃO FITS
# ============================================================


def _find_primary_image_hdu(
    hdul: fits.HDUList,
) -> tuple[int, fits.hdu.base.NonstandardHDU]:
    for index, hdu in enumerate(hdul):
        if hdu.name == "VALID_MASK":
            continue

        if not hdu.is_image:
            continue

        shape = getattr(hdu, "shape", None)
        if shape is None:
            continue

        ndim = len(shape)

        if ndim in (2, 3):
            return index, hdu

    raise ValueError("Nenhuma imagem FITS 2D/3D encontrada.")


def _classify_shape(
    shape: tuple[int, ...],
    header: fits.Header,
) -> tuple[str, int]:
    """
    Classifica a imagem sem tocar nos pixels.
    """

    if len(shape) == 2:
        return "Mono", 1

    if len(shape) != 3:
        raise ValueError(f"Dimensão FITS não suportada: {shape}")

    # AstroStack/AstroAlign normalmente usa CHW em FITS.
    if shape[0] in (3, 4):
        return "RGB", int(shape[0])

    if shape[-1] in (3, 4):
        return "RGB", int(shape[-1])

    raise ValueError(f"Formato RGB não reconhecido: {shape}")


def inspect_fits(
    filepath: Path,
) -> FrameGeometry:
    """
    Inspeção leve: não carrega a imagem inteira conscientemente.
    """

    with fits.open(
        filepath,
        memmap=False,
        lazy_load_hdus=True,
    ) as hdul:
        hdu_index, image_hdu = _find_primary_image_hdu(hdul)

        shape = tuple(image_hdu.shape)
        image_kind, channels = _classify_shape(
            shape,
            image_hdu.header,
        )

        mask_hdu_index = None

        for index, hdu in enumerate(hdul):
            if hdu.name == "VALID_MASK" and hdu.is_image:
                mask_hdu_index = index
                break

        if image_kind == "Mono":
            height, width = shape
        elif shape[0] in (3, 4):
            _, height, width = shape
        else:
            height, width, _ = shape

        return FrameGeometry(
            height=int(height),
            width=int(width),
            image_kind=image_kind,
            channels=channels,
            hdu_index=hdu_index,
            mask_hdu_index=mask_hdu_index,
        )


def load_frame_metrics(filepath: Path) -> dict:
    """
    Compatibilidade com versões antigas que usavam JSON adjacente.
    """

    metrics_path = filepath.parent / (filepath.stem + "_metrics.json")

    if not metrics_path.exists():
        return {}

    try:
        return load_json(metrics_path)
    except Exception:
        return {}


# ============================================================
# METADADOS DOS FRAMES / SELEÇÃO
# ============================================================


def build_frame_infos(
    fits_files: list[Path],
    batch_metadata: dict[str, dict],
    flow_cache: dict[str, dict],
    app_print: Callable[[str], None],
) -> list[FrameInfo]:
    """
    Cria os metadados sem carregar os pixels dos frames.
    """
    path_to_batch: dict[Path, str] = {
        Path(meta["path"]): name for name, meta in batch_metadata.items()
    }

    result: list[FrameInfo] = []

    for index, filepath in enumerate(fits_files, start=1):
        batch_path = filepath.parent
        batch_name = path_to_batch.get(
            batch_path,
            batch_path.name or "root",
        )

        metrics = load_frame_metrics(filepath)

        # O Flow já está cacheado e indexado pelo NOME da batch; não reabrimos JSON por frame.
        flow = flow_cache.get(batch_name, {})
        frame_flow = (
            flow.get("frames", {}).get(
                filepath.name,
                {},
            )
            if isinstance(flow, dict)
            else {}
        )

        if frame_flow:
            metrics = {
                **metrics,
                **frame_flow,
            }

        try:
            geometry = inspect_fits(filepath)
        except Exception as exc:
            app_print(f"⚠️ Ignorando {filepath.name}: {exc}")
            continue

        star_count = float(metrics.get("star_count", 0) or 0)
        fwhm = float(metrics.get("fwhm", 0) or 0)
        snr = float(metrics.get("snr", 0) or 0)
        rms = float(metrics.get("rms", 999.0) or 999.0)
        quality = star_count / max(fwhm, 0.1) if fwhm > 0 else star_count

        if rms < 999.0 and rms > 0:
            quality /= 1.0 + rms / 10.0

        has_valid_mask = bool(geometry.mask_hdu_index is not None)

        if has_valid_mask:
            quality *= 1.1

        metrics["quality"] = quality
        metrics["has_valid_mask"] = has_valid_mask

        result.append(
            FrameInfo(
                path=filepath,
                name=filepath.name,
                batch=batch_name,
                metrics=metrics,
                quality=quality,
                star_count=star_count,
                fwhm=fwhm,
                snr=snr,
                rms=rms,
                has_valid_mask=has_valid_mask,
                shape=tuple(
                    [
                        geometry.height,
                        geometry.width,
                    ]
                    + ([geometry.channels] if geometry.image_kind == "RGB" else [])
                ),
                image_kind=geometry.image_kind,
                channels=geometry.channels,
            )
        )

        if index % 100 == 0:
            app_print(f"🔎 Metadados: {index}/{len(fits_files)}")

    return result


def select_frames(
    all_frames: list[FrameInfo],
    config: StackingConfig,
) -> list[FrameInfo]:
    if not all_frames:
        return []

    if config.selection_mode == "All":
        return list(all_frames)

    metric_name = config.selection_metric

    if metric_name == "quality":
        key = lambda x: x.quality
        reverse = True
    elif metric_name == "fwhm":
        key = lambda x: x.fwhm if x.fwhm > 0 else float("inf")
        reverse = False
    elif metric_name == "star_count":
        key = lambda x: x.star_count
        reverse = True
    else:
        key = lambda x: x.snr
        reverse = True

    sorted_frames = sorted(
        all_frames,
        key=key,
        reverse=reverse,
    )

    valid = []

    for frame in sorted_frames:
        value = key(frame)

        if isinstance(value, (int, float)) and math.isfinite(value):
            if metric_name == "fwhm" and value <= 0:
                continue
            if metric_name != "fwhm" and value <= 0:
                continue
            valid.append(frame)

    if not valid:
        valid = list(all_frames)

    percentage = max(
        1.0,
        min(100.0, float(config.selection_percentage)),
    )

    n_select = max(
        3,
        int(len(valid) * percentage / 100.0),
    )

    n_select = min(
        n_select,
        len(valid),
    )

    return valid[:n_select]


# ============================================================
# FITS BLOCK READING
# ============================================================


def _read_hdu_section(
    hdu,
    slices: tuple[slice, ...],
) -> np.ndarray:
    """
    Tenta o caminho section, que é particularmente útil para imagens
    comprimidas, e cai para data[slices] quando necessário.
    """

    try:
        section = getattr(hdu, "section", None)

        if section is not None:
            value = section[slices]
            return np.asarray(
                value,
                dtype=np.float32,
            )
    except Exception:
        pass

    return np.asarray(
        hdu.data[slices],
        dtype=np.float32,
    )


def _read_mask_section(
    hdu,
    y1: int,
    y2: int,
    x1: int,
    x2: int,
) -> np.ndarray:
    try:
        section = getattr(hdu, "section", None)

        if section is not None:
            value = section[
                y1:y2,
                x1:x2,
            ]
        else:
            value = hdu.data[
                y1:y2,
                x1:x2,
            ]

        return np.asarray(value) > 0

    except Exception:
        return np.ones(
            (y2 - y1, x2 - x1),
            dtype=bool,
        )


def read_frame_block(
    frame: FrameInfo,
    geometry: FrameGeometry,
    y1: int,
    y2: int,
    x1: int,
    x2: int,
    channel: int | None,
    normalization_factor: float,
) -> BlockRead:
    """
    Lê somente o bloco solicitado.

    A função retorna sempre dados 2D para que a combinação por canal
    permaneça com memória previsível.
    """

    with fits.open(
        frame.path,
        memmap=False,
        lazy_load_hdus=True,
    ) as hdul:
        image_hdu = hdul[geometry.hdu_index]

        if geometry.image_kind == "Mono":
            data = _read_hdu_section(
                image_hdu,
                (
                    slice(y1, y2),
                    slice(x1, x2),
                ),
            )

        elif geometry.channels <= 4:
            if image_hdu.shape[0] in (3, 4):
                # CHW
                assert channel is not None
                data = _read_hdu_section(
                    image_hdu,
                    (
                        slice(channel, channel + 1),
                        slice(y1, y2),
                        slice(x1, x2),
                    ),
                )[0]
            else:
                # HWC
                assert channel is not None
                data = _read_hdu_section(
                    image_hdu,
                    (
                        slice(y1, y2),
                        slice(x1, x2),
                        slice(channel, channel + 1),
                    ),
                )[:, :, 0]
        else:
            raise ValueError(f"Número de canais inválido: {geometry.channels}")

        if normalization_factor != 1.0:
            data = data * np.float32(normalization_factor)

        # Valores não finitos são tratados como inválidos.
        finite = np.isfinite(data)

        if geometry.mask_hdu_index is not None:
            mask_hdu = hdul[geometry.mask_hdu_index]
            external_mask = _read_mask_section(
                mask_hdu,
                y1,
                y2,
                x1,
                x2,
            )
            mask = finite & external_mask
        else:
            mask = finite

        # Evita deixar NaN/Inf escaparem para as operações posteriores.
        if not finite.all():
            data = np.where(
                finite,
                data,
                0.0,
            ).astype(
                np.float32,
                copy=False,
            )

        return BlockRead(
            data=np.asarray(
                data,
                dtype=np.float32,
            ),
            mask=np.asarray(
                mask,
                dtype=bool,
            ),
        )


def load_normalization_sample(
    frame: FrameInfo,
    geometry: FrameGeometry,
    max_samples: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Lê apenas uma amostra espacial determinística do frame.

    Isso evita carregar uma imagem inteira apenas para descobrir o valor
    de fundo usado na normalização.
    """

    with fits.open(
        frame.path,
        memmap=False,
        lazy_load_hdus=True,
    ) as hdul:
        image_hdu = hdul[geometry.hdu_index]
        shape = tuple(image_hdu.shape)

        height = geometry.height
        width = geometry.width
        pixels = max(1, height * width)

        # Mantém aproximadamente max_samples pixels espaciais.
        spatial_step = max(1, int(math.sqrt(pixels / max(1, max_samples))))

        y_slice = slice(0, height, spatial_step)
        x_slice = slice(0, width, spatial_step)

        if geometry.image_kind == "Mono":
            sample = _read_hdu_section(
                image_hdu,
                (y_slice, x_slice),
            )
        elif shape[0] in (3, 4):
            sample = _read_hdu_section(
                image_hdu,
                (slice(None), y_slice, x_slice),
            )
        else:
            sample = _read_hdu_section(
                image_hdu,
                (y_slice, x_slice, slice(None)),
            )

        mask = None
        if geometry.mask_hdu_index is not None:
            mask_hdu = hdul[geometry.mask_hdu_index]
            mask = _read_mask_section(
                mask_hdu,
                0,
                height,
                0,
                width,
            )
            mask = mask[
                ::spatial_step,
                ::spatial_step,
            ]

        return (
            np.asarray(sample, dtype=np.float32),
            mask,
        )


def _sample_valid_values(
    values: np.ndarray,
    mask: np.ndarray | None,
    max_samples: int,
) -> np.ndarray:
    """
    Amostra de forma determinística para evitar armazenar milhões de
    pixels adicionais quando uma normalização é solicitada.
    """

    if values.ndim == 2:
        flat = values.ravel()

        if mask is not None:
            valid = mask.ravel()
            flat = flat[valid]
    else:
        flat = values.ravel()

    finite = np.isfinite(flat)
    flat = flat[finite]

    if flat.size <= max_samples:
        return flat

    step = max(
        1,
        flat.size // max_samples,
    )

    return flat[::step][:max_samples]


def estimate_normalization_value(
    frame: FrameInfo,
    geometry: FrameGeometry,
    method: str,
    max_samples: int,
) -> float:
    """
    Estima o valor de fundo de normalização.

    Median:
        mediana da amostra válida.

    Mode:
        aproximação robusta por centro do intervalo em torno da mediana.
        Mantém o custo baixo sem bibliotecas adicionais.
    """

    data, mask = load_normalization_sample(
        frame,
        geometry,
        max_samples,
    )

    # RGB: achata os canais; a máscara espacial é repetida para os canais.
    if data.ndim == 3:
        if data.shape[0] in (3, 4):
            if mask is not None:
                expanded_mask = np.broadcast_to(
                    mask,
                    data.shape,
                )
            else:
                expanded_mask = None
        else:
            if mask is not None:
                expanded_mask = np.broadcast_to(
                    mask[..., None],
                    data.shape,
                )
            else:
                expanded_mask = None
    else:
        expanded_mask = mask

    sample = _sample_valid_values(
        data,
        expanded_mask,
        max_samples,
    )

    if sample.size == 0:
        return 0.0

    median = float(np.median(sample))

    if method == "Mode":
        # Estimativa leve de modo usando histograma local em torno
        # da mediana. O objetivo aqui é normalização, não fotometria.
        p05, p95 = np.percentile(
            sample,
            [5.0, 95.0],
        )

        if p95 > p05:
            hist, edges = np.histogram(
                sample,
                bins=64,
                range=(p05, p95),
            )

            if hist.size and np.any(hist):
                idx = int(np.argmax(hist))
                return float((edges[idx] + edges[idx + 1]) * 0.5)

    return median


# ============================================================
# NORMALIZAÇÃO
# ============================================================


def calculate_normalization_factors(
    selected_frames: list[FrameInfo],
    geometries: dict[Path, FrameGeometry],
    config: StackingConfig,
    app_print: Callable[[str], None],
    cancel_event: threading.Event | None,
) -> list[float] | None:
    """
    Calcula fatores sem montar um stack global.

    Para controlar RAM, a normalização é feita serialmente.
    O custo da CPU é baixo comparado ao I/O e evita múltiplos frames
    gigantes simultaneamente.
    """

    if not config.normalize:
        return [1.0] * len(selected_frames)

    if not selected_frames:
        return []

    app_print(f"\n⚖️ Calculando normalização ({config.normalize_method})...")

    reference_value = estimate_normalization_value(
        selected_frames[0],
        geometries[selected_frames[0].path],
        config.normalize_method,
        config.normalization_max_samples,
    )

    if not math.isfinite(reference_value) or reference_value <= 0:
        reference_value = 1.0

    factors: list[float] = []

    for index, frame in enumerate(selected_frames):
        if cancel_event and cancel_event.is_set():
            return None

        current_value = estimate_normalization_value(
            frame,
            geometries[frame.path],
            config.normalize_method,
            config.normalization_max_samples,
        )

        if not math.isfinite(current_value) or current_value <= 0:
            factor = 1.0
        else:
            factor = reference_value / current_value

        factors.append(float(factor))

        if (
            index % max(1, len(selected_frames) // 20) == 0
            or index == len(selected_frames) - 1
        ):
            app_print(f"   Normalização: {index + 1}/{len(selected_frames)}")

    return factors


# ============================================================
# LEITURA PARA BLOCO EM PARALELO
# ============================================================


def read_block_worker(
    args: tuple[
        FrameInfo,
        FrameGeometry,
        int,
        int,
        int,
        int,
        int | None,
        float,
    ],
) -> BlockRead:
    return read_frame_block(*args)


def load_channel_block_parallel(
    selected_frames: list[FrameInfo],
    geometries: dict[Path, FrameGeometry],
    y1: int,
    y2: int,
    x1: int,
    x2: int,
    channel: int | None,
    normalization_factors: list[float],
    executor: ThreadPoolExecutor | None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Carrega um bloco de cada frame.

    O retorno possui:
        values [N, h, w]
        mask   [N, h, w]

    Para manter a RAM previsível, o limite é dado pelo tamanho do bloco,
    não pela resolução inteira da imagem.
    """

    args_list = [
        (
            frame,
            geometries[frame.path],
            y1,
            y2,
            x1,
            x2,
            channel,
            normalization_factors[index],
        )
        for index, frame in enumerate(selected_frames)
    ]

    count = len(args_list)

    if count == 0:
        raise ValueError("Nenhum frame selecionado.")

    first_args = args_list[0]
    first = read_block_worker(first_args)

    block_h, block_w = first.data.shape

    values = np.empty(
        (
            count,
            block_h,
            block_w,
        ),
        dtype=np.float32,
    )

    masks = np.empty(
        (
            count,
            block_h,
            block_w,
        ),
        dtype=bool,
    )

    values[0] = first.data
    masks[0] = first.mask

    if count == 1:
        return values, masks

    if executor is None:
        for index in range(1, count):
            item = read_block_worker(args_list[index])
            values[index] = item.data
            masks[index] = item.mask
        return values, masks

    futures = {
        executor.submit(
            read_block_worker,
            args_list[index],
        ): index
        for index in range(1, count)
    }

    for future in as_completed(futures):
        index = futures[future]
        item = future.result()
        values[index] = item.data
        masks[index] = item.mask

    return values, masks


# ============================================================
# COMBINAÇÃO SEM REJEIÇÃO
# ============================================================


def combine_block_streaming(
    selected_frames: list[FrameInfo],
    geometries: dict[Path, FrameGeometry],
    y1: int,
    y2: int,
    x1: int,
    x2: int,
    channel: int | None,
    normalization_factors: list[float],
    method: str,
    executor: ThreadPoolExecutor | None,
) -> np.ndarray:
    """
    Combina um bloco incrementalmente.

    Importante:
        Mean/Sum/Maximum/Minimum não criam [N,H,W].

    Isso reduz a memória para aproximadamente O(H,W).
    """

    accumulator: np.ndarray | None = None
    counts: np.ndarray | None = None

    for index, frame in enumerate(selected_frames):
        block = read_frame_block(
            frame,
            geometries[frame.path],
            y1,
            y2,
            x1,
            x2,
            channel,
            normalization_factors[index],
        )

        data = block.data
        mask = block.mask

        if accumulator is None:
            if method in {"Mean", "Sum"}:
                accumulator = np.zeros_like(
                    data,
                    dtype=np.float32,
                )
                counts = np.zeros(
                    data.shape,
                    dtype=np.uint32,
                )
            else:
                accumulator = np.full_like(
                    data,
                    -np.inf if method == "Maximum" else np.inf,
                    dtype=np.float32,
                )

        if method in {"Mean", "Sum"}:
            assert counts is not None
            accumulator[mask] += data[mask]
            counts[mask] += 1

        elif method == "Maximum":
            np.maximum(
                accumulator,
                data,
                out=accumulator,
                where=mask,
            )

        elif method == "Minimum":
            np.minimum(
                accumulator,
                data,
                out=accumulator,
                where=mask,
            )

        else:
            raise ValueError(f"Método streaming não suportado: {method}")

    if accumulator is None:
        raise ValueError("Bloco sem dados.")

    if method == "Mean":
        assert counts is not None

        result = np.zeros_like(
            accumulator,
            dtype=np.float32,
        )

        np.divide(
            accumulator,
            counts,
            out=result,
            where=counts > 0,
        )

        return result

    if method == "Sum":
        return accumulator

    invalid = ~np.isfinite(accumulator)

    if np.any(invalid):
        accumulator[invalid] = 0.0

    return accumulator


# ============================================================
# REJEIÇÃO + COMBINAÇÃO POR BLOCO
# ============================================================


def _masked_values(
    values: np.ndarray,
    masks: np.ndarray,
) -> np.ndarray:
    """
    Converte inválidos em NaN para uso com nan*.
    Trabalha somente no bloco atual.
    """

    masked = values.copy()
    masked[~masks] = np.nan
    return masked


def reject_and_combine_block(
    values: np.ndarray,
    masks: np.ndarray,
    combine_method: str,
    rejection_method: str,
    low: float,
    high: float,
) -> np.ndarray:
    """
    Executa rejeição e combinação exclusivamente no bloco atual.
    """

    masked = _masked_values(
        values,
        masks,
    )

    n_frames = masked.shape[0]

    if rejection_method != "None" and n_frames > 3:
        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore",
                category=RuntimeWarning,
            )

            if rejection_method == "SigmaClip":
                center = np.nanmedian(
                    masked,
                    axis=0,
                )

                std = np.nanstd(
                    masked,
                    axis=0,
                )

                lower = center - low * std
                upper = center + high * std

                valid = (masked >= lower) & (masked <= upper)

                stable = np.isfinite(std) & (std > 1e-10)

                valid |= ~stable

                masked[~valid] = np.nan

            elif rejection_method == "MAD":
                center = np.nanmedian(
                    masked,
                    axis=0,
                )

                mad = np.nanmedian(
                    np.abs(masked - center),
                    axis=0,
                )

                sigma = 1.4826 * mad

                lower = center - low * sigma
                upper = center + high * sigma

                valid = (masked >= lower) & (masked <= upper)

                stable = np.isfinite(mad) & (mad > 1e-10)

                valid |= ~stable

                masked[~valid] = np.nan

            elif rejection_method == "Winsorized":
                p_low = np.nanpercentile(
                    masked,
                    low * 10.0,
                    axis=0,
                )

                p_high = np.nanpercentile(
                    masked,
                    100.0 - high * 10.0,
                    axis=0,
                )

                valid = (masked >= p_low) & (masked <= p_high)

                stable = np.isfinite(p_low) & np.isfinite(p_high)

                valid |= ~stable

                masked[~valid] = np.nan

    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore",
            category=RuntimeWarning,
        )

        if combine_method == "Median":
            result = np.nanmedian(
                masked,
                axis=0,
            )

        elif combine_method == "Mean":
            result = np.nanmean(
                masked,
                axis=0,
            )

        elif combine_method == "Sum":
            result = np.nansum(
                masked,
                axis=0,
            )

        elif combine_method == "Maximum":
            result = np.nanmax(
                masked,
                axis=0,
            )

        elif combine_method == "Minimum":
            result = np.nanmin(
                masked,
                axis=0,
            )

        else:
            result = np.nanmedian(
                masked,
                axis=0,
            )

    return np.asarray(
        np.nan_to_num(
            result,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ),
        dtype=np.float32,
    )


# ============================================================
# STACKING BLOCO A BLOCO
# ============================================================


def combine_channel_blocks(
    selected_frames: list[FrameInfo],
    geometries: dict[Path, FrameGeometry],
    normalization_factors: list[float],
    height: int,
    width: int,
    channel: int | None,
    config: StackingConfig,
    executor: ThreadPoolExecutor | None,
    cancel_event: threading.Event | None,
    progress_callback: Callable[[int, int, str], None] | None,
    channel_label: str,
) -> np.ndarray | None:
    """
    Faz o stacking de uma imagem 2D/canal inteiro em blocos.
    """

    rejection_enabled = config.rejection_method != "None"

    use_streaming = not rejection_enabled and config.method in {
        "Mean",
        "Sum",
        "Maximum",
        "Minimum",
    }

    chunk = adapt_chunk_size(
        config.effective_chunk_size,
        len(selected_frames),
        1,
        config.memory_budget_mb,
        rejection_enabled,
    )

    result = np.zeros(
        (
            height,
            width,
        ),
        dtype=np.float32,
    )

    n_blocks_y = math.ceil(height / chunk)

    n_blocks_x = math.ceil(width / chunk)

    total_blocks = n_blocks_y * n_blocks_x
    completed_blocks = 0

    # Executor para leitura de blocos somente quando não estamos no
    # caminho streaming. No caminho streaming, abrir uma thread por frame
    # para cada bloco desperdiçaria bastante I/O e criaria muita pressão.
    block_executor = executor

    for y1 in range(0, height, chunk):
        if cancel_event and cancel_event.is_set():
            return None

        y2 = min(
            y1 + chunk,
            height,
        )

        for x1 in range(0, width, chunk):
            if cancel_event and cancel_event.is_set():
                return None

            x2 = min(
                x1 + chunk,
                width,
            )

            if use_streaming:
                block_result = combine_block_streaming(
                    selected_frames=selected_frames,
                    geometries=geometries,
                    y1=y1,
                    y2=y2,
                    x1=x1,
                    x2=x2,
                    channel=channel,
                    normalization_factors=normalization_factors,
                    method=config.method,
                    executor=None,
                )
            else:
                values, masks = load_channel_block_parallel(
                    selected_frames=selected_frames,
                    geometries=geometries,
                    y1=y1,
                    y2=y2,
                    x1=x1,
                    x2=x2,
                    channel=channel,
                    normalization_factors=normalization_factors,
                    executor=block_executor,
                )

                block_result = reject_and_combine_block(
                    values,
                    masks,
                    combine_method=config.method,
                    rejection_method=config.rejection_method,
                    low=config.rejection_low,
                    high=config.rejection_high,
                )

                # Libera referências imediatamente antes do próximo bloco.
                del values
                del masks

            result[
                y1:y2,
                x1:x2,
            ] = block_result

            del block_result

            completed_blocks += 1

            if progress_callback:
                progress_callback(
                    completed_blocks,
                    total_blocks,
                    f"{channel_label}: bloco {completed_blocks}/{total_blocks}",
                )

    return result


# ============================================================
# PÓS-PROCESSAMENTO / MÁSCARA
# ============================================================


def build_reference_mask(
    selected_frames: list[FrameInfo],
    geometries: dict[Path, FrameGeometry],
    normalization_factors: list[float],
    height: int,
    width: int,
    config: StackingConfig,
) -> np.ndarray:
    """
    Cria uma máscara final de cobertura.

    A máscara representa pixels que possuem pelo menos uma fonte válida.
    Ela é calculada em blocos para não consumir RAM adicional relevante.
    """

    chunk = adapt_chunk_size(
        config.effective_chunk_size,
        len(selected_frames),
        1,
        min(config.memory_budget_mb, 256),
        False,
    )

    mask_result = np.zeros(
        (height, width),
        dtype=bool,
    )

    for y1 in range(0, height, chunk):
        y2 = min(y1 + chunk, height)

        for x1 in range(0, width, chunk):
            x2 = min(x1 + chunk, width)

            block_mask = np.zeros(
                (y2 - y1, x2 - x1),
                dtype=bool,
            )

            for index, frame in enumerate(selected_frames):
                item = read_frame_block(
                    frame,
                    geometries[frame.path],
                    y1,
                    y2,
                    x1,
                    x2,
                    0 if frame.image_kind == "RGB" else None,
                    normalization_factors[index],
                )

                block_mask |= item.mask

                del item

                if block_mask.all():
                    break

            mask_result[
                y1:y2,
                x1:x2,
            ] = block_mask

    return mask_result


# ============================================================
# DITHER CORRECTION
# ============================================================


def apply_dither_correction_inplace(
    result: np.ndarray,
) -> None:
    """
    Mantém a compatibilidade da opção existente.

    Executa canal por canal para evitar uma cópia RGB inteira.
    """

    from scipy.ndimage import median_filter

    if result.ndim == 2:
        filtered = median_filter(
            result,
            size=5,
        )
        result[...] = filtered
        del filtered
        return

    if result.ndim == 3:
        for channel in range(result.shape[0]):
            filtered = median_filter(
                result[channel],
                size=5,
            )
            result[channel] = filtered
            del filtered


# ============================================================
# HEADER / SAÍDA
# ============================================================


def sanitize_ascii(text: str) -> str:
    replacements = {
        "á": "a",
        "à": "a",
        "ã": "a",
        "â": "a",
        "ä": "a",
        "é": "e",
        "è": "e",
        "ê": "e",
        "ë": "e",
        "í": "i",
        "ì": "i",
        "î": "i",
        "ï": "i",
        "ó": "o",
        "ò": "o",
        "õ": "o",
        "ô": "o",
        "ö": "o",
        "ú": "u",
        "ù": "u",
        "û": "u",
        "ü": "u",
        "ç": "c",
        "Ç": "C",
        "ñ": "n",
        "Ñ": "N",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return "".join(char for char in text if 32 <= ord(char) <= 126)


def prepare_output_header(
    source_header: fits.Header,
    config: StackingConfig,
    n_frames: int,
    n_frames_total: int,
    n_batches: int,
    avg_quality: float,
    avg_star_count: float,
    avg_fwhm: float,
) -> fits.Header:
    header = source_header.copy()

    header["STACKING"] = (
        config.method,
        "AstroStack combine method",
    )

    header["STACK_NFR"] = (
        n_frames,
        "Frames combined",
    )

    header["STACK_TOT"] = (
        n_frames_total,
        "Frames available",
    )

    header["STACK_SEL"] = (
        config.selection_mode,
        "Frame selection mode",
    )

    header["STACK_SELP"] = (
        config.selection_percentage
        if config.selection_mode == "BestPercentage"
        else 100.0,
        "Percentage selected",
    )

    header["STACK_METR"] = (
        config.selection_metric,
        "Selection metric",
    )

    header["STACK_REJ"] = (
        config.rejection_method if config.rejection_method != "None" else "NONE",
        "Outlier rejection",
    )

    header["STACK_REJ_L"] = (
        config.rejection_low,
        "Low rejection parameter",
    )

    header["STACK_REJ_H"] = (
        config.rejection_high,
        "High rejection parameter",
    )

    header["STACK_NORM"] = (
        bool(config.normalize),
        "Frame normalization",
    )

    header["STACK_NORMM"] = (
        config.normalize_method if config.normalize else "NONE",
        "Normalization method",
    )

    header["STACK_BITS"] = (
        config.output_bit_depth,
        "Output precision",
    )

    header["STACK_CHNK"] = (
        config.effective_chunk_size,
        "Requested processing block size",
    )

    header["STACK_MEM"] = (
        config.memory_budget_mb,
        "Memory budget MB",
    )

    header["STACK_STREAM"] = (
        True,
        "Block streaming enabled",
    )

    header["AVG_QUAL"] = round(
        avg_quality,
        3,
    )

    header["AVG_STARS"] = round(
        avg_star_count,
        1,
    )

    header["AVG_FWHM"] = (
        round(
            avg_fwhm,
            2,
        )
        if avg_fwhm > 0
        else 0.0
    )

    header["HISTORY"] = sanitize_ascii(f"Stacking by AstroStack ({config.method})")

    header["HISTORY"] = sanitize_ascii(
        f"{n_frames} frames combinados de {n_frames_total} disponiveis"
    )

    if config.selection_mode == "BestPercentage":
        header["HISTORY"] = sanitize_ascii(
            f"Selecionados {config.selection_percentage}% por {config.selection_metric}"
        )

    if config.normalize:
        header["HISTORY"] = sanitize_ascii(f"Normalizacao: {config.normalize_method}")

    if config.rejection_method != "None":
        header["HISTORY"] = sanitize_ascii(
            f"Rejeicao: {config.rejection_method} ({config.rejection_low}, {config.rejection_high})"
        )

    return header


def convert_output_dtype(
    result: np.ndarray,
    config: StackingConfig,
) -> np.ndarray:
    """
    Converte sem criar cópias desnecessárias sempre que possível.
    """

    result = np.nan_to_num(
        result,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    if config.output_bit_depth == "16-bit":
        data_min = float(np.min(result))

        data_max = float(np.max(result))

        if data_max > data_min:
            scale = np.float32(65535.0 / (data_max - data_min))
            result = (result - np.float32(data_min)) * scale
        else:
            result = np.zeros_like(
                result,
                dtype=np.float32,
            )

        return np.clip(
            result,
            0.0,
            65535.0,
        ).astype(np.uint16)

    return result.astype(
        np.float32,
        copy=False,
    )


def write_stack_output(
    output_path: Path,
    data_to_save: np.ndarray,
    mask: np.ndarray,
    header: fits.Header,
    config: StackingConfig,
) -> None:
    """
    Escreve em arquivo temporário e substitui o destino ao final.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = output_path.with_name(output_path.name + ".tmp")

    if temporary.exists():
        temporary.unlink()

    try:
        if config.compress_output:
            primary = fits.PrimaryHDU(
                data=data_to_save,
                header=header,
            )

            mask_hdu = fits.CompImageHDU(
                data=mask.astype(np.uint8),
                name="VALID_MASK",
                compression_type="RICE_1",
            )

            hdul = fits.HDUList([primary, mask_hdu])
        else:
            primary = fits.PrimaryHDU(
                data=data_to_save,
                header=header,
            )

            mask_hdu = fits.ImageHDU(
                data=mask.astype(np.uint8),
                name="VALID_MASK",
            )

            hdul = fits.HDUList([primary, mask_hdu])

        hdul.writeto(
            temporary,
            overwrite=True,
            output_verify="ignore",
        )

        hdul.close()

        os.replace(
            temporary,
            output_path,
        )

    finally:
        if temporary.exists():
            try:
                temporary.unlink()
            except OSError:
                pass


# ============================================================
# CARREGAMENTO DO HEADER
# ============================================================


def load_source_header(
    filepath: Path,
) -> fits.Header:
    with fits.open(
        filepath,
        memmap=False,
        lazy_load_hdus=True,
    ) as hdul:
        index, hdu = _find_primary_image_hdu(hdul)
        return hdu.header.copy()


# ============================================================
# PROCESSO PRINCIPAL
# ============================================================


def process_stacking(
    config: StackingConfig,
    progress_callback: Callable[[int, int, str], None] | None = None,
    status_callback: Callable[[str], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> dict:
    """Executa o AstroStack otimizado."""

    if cancel_event is None:
        cancel_event = threading.Event()

    def log(message: str) -> None:
        if status_callback:
            status_callback(message + "\n")

    def progress(
        current: int,
        total: int,
        message: str = "",
    ) -> None:
        if progress_callback:
            progress_callback(
                current,
                total,
                message,
            )

    # ========================================================
    # 1. DESCOBERTA
    # ========================================================

    log("🔍 Procurando frames alinhados...")

    if not config.input_dir.exists():
        log(f"❌ Pasta de entrada não encontrada: {config.input_dir}")
        return {
            "status": "error",
            "reason": "input_dir_not_found",
        }

    fits_files, batch_metadata = discover_aligned_frames(config.input_dir)

    if not fits_files:
        log(f"❌ Nenhum arquivo FITS encontrado em: {config.input_dir}")
        return {
            "status": "error",
            "reason": "no_fits_files",
        }

    log(f"📁 Total de {len(fits_files)} arquivos FITS encontrados.")

    # ========================================================
    # 2. CACHE DOS FLOWS
    # ========================================================

    flow_cache = load_flow_cache(
        config.base_dir,
        batch_metadata,
    )

    if flow_cache:
        log(f"🧭 Flows locais em cache: {len(flow_cache)}")
    else:
        log(
            "[Stack] Aviso: Nenhum arquivo de flow encontrado. "
            "O Stacking usará apenas os metadados disponíveis."
        )

    # ========================================================
    # 3. METADADOS / SELEÇÃO
    # ========================================================

    log("🔎 Analisando metadados dos frames...")

    all_frames = build_frame_infos(
        fits_files,
        batch_metadata,
        flow_cache,
        log,
    )

    if not all_frames:
        log("❌ Nenhum frame válido encontrado.")
        return {
            "status": "error",
            "reason": "no_valid_frames",
        }

    selected_frames = select_frames(
        all_frames,
        config,
    )

    log(
        f"✅ Selecionados {len(selected_frames)} frames "
        f"de {len(all_frames)} disponíveis."
    )

    if len(selected_frames) < 3:
        log("❌ Número insuficiente de frames selecionados (mínimo: 3).")
        return {
            "status": "error",
            "reason": "insufficient_frames",
        }

    if cancel_event.is_set():
        return {
            "status": "cancelled",
        }

    # ========================================================
    # 4. GEOMETRIA
    # ========================================================

    log("📐 Validando geometria dos frames...")

    geometries: dict[Path, FrameGeometry] = {}

    reference_frame = selected_frames[0]
    reference_geometry = inspect_fits(reference_frame.path)

    for frame in selected_frames:
        try:
            geometry = inspect_fits(frame.path)
        except Exception as exc:
            log(f"❌ Falha de geometria em {frame.name}: {exc}")
            return {
                "status": "error",
                "reason": "geometry_error",
                "file": frame.name,
            }

        if (
            geometry.height != reference_geometry.height
            or geometry.width != reference_geometry.width
            or geometry.image_kind != reference_geometry.image_kind
            or geometry.channels != reference_geometry.channels
        ):
            log(
                f"❌ Geometria incompatível em {frame.name}: "
                f"{geometry.height}x{geometry.width}, "
                f"{geometry.image_kind}/{geometry.channels}"
            )
            return {
                "status": "error",
                "reason": "incompatible_geometry",
                "file": frame.name,
            }

        geometries[frame.path] = geometry

    height = reference_geometry.height
    width = reference_geometry.width
    channels = reference_geometry.channels

    log(
        f"📐 Dimensão: {width}×{height} | "
        f"{reference_geometry.image_kind} | "
        f"canais={channels}"
    )

    # ========================================================
    # 5. ESTATÍSTICAS DE SELEÇÃO
    # ========================================================

    quality_values = [frame.quality for frame in selected_frames]

    star_counts = [frame.star_count for frame in selected_frames]

    fwhm_values = [frame.fwhm for frame in selected_frames if frame.fwhm > 0]

    avg_quality = float(np.mean(quality_values) if quality_values else 0.0)

    avg_star_count = float(np.mean(star_counts) if star_counts else 0.0)

    avg_fwhm = float(np.mean(fwhm_values) if fwhm_values else 0.0)

    # ========================================================
    # 6. NORMALIZAÇÃO
    # ========================================================

    normalization_factors = calculate_normalization_factors(
        selected_frames,
        geometries,
        config,
        log,
        cancel_event,
    )

    if normalization_factors is None:
        return {
            "status": "cancelled",
        }

    # ========================================================
    # 7. DIMENSIONAMENTO DE BLOCO
    # ========================================================

    effective_chunk = adapt_chunk_size(
        config.effective_chunk_size,
        len(selected_frames),
        1,
        config.memory_budget_mb,
        config.rejection_method != "None",
    )

    log(
        f"🚀 Streaming: workers={config.worker_count} | "
        f"bloco={effective_chunk}×{effective_chunk} | "
        f"orçamento≈{config.memory_budget_mb} MB"
    )

    if config.rejection_method == "None" and config.method in {
        "Mean",
        "Sum",
        "Maximum",
        "Minimum",
    }:
        log("⚡ Modo incremental ativo: não mantém stack de todos os frames na RAM.")
    else:
        log("🧮 Rejeição/Mediana: somente o bloco atual permanece na RAM.")

    # ========================================================
    # 8. STACKING
    # ========================================================

    output_result: np.ndarray

    if reference_geometry.image_kind == "Mono":
        result_2d = combine_channel_blocks(
            selected_frames=selected_frames,
            geometries=geometries,
            normalization_factors=normalization_factors,
            height=height,
            width=width,
            channel=None,
            config=config,
            executor=None,
            cancel_event=cancel_event,
            progress_callback=lambda c, t, m: progress(
                c,
                t,
                m,
            ),
            channel_label="Mono",
        )

        if result_2d is None:
            return {
                "status": "cancelled",
            }

        output_result = result_2d

    else:
        output_result = np.zeros(
            (
                channels,
                height,
                width,
            ),
            dtype=np.float32,
        )

        # Um canal por vez é intencional:
        # evita multiplicar por 3/4 a memória do bloco.
        for channel in range(channels):
            if cancel_event.is_set():
                return {
                    "status": "cancelled",
                }

            log(f"🎨 Processando canal {channel + 1}/{channels}...")

            channel_result = combine_channel_blocks(
                selected_frames=selected_frames,
                geometries=geometries,
                normalization_factors=normalization_factors,
                height=height,
                width=width,
                channel=channel,
                config=config,
                executor=None,
                cancel_event=cancel_event,
                progress_callback=lambda c, t, m: progress(
                    c,
                    t,
                    m,
                ),
                channel_label=f"Canal {channel + 1}/{channels}",
            )

            if channel_result is None:
                return {
                    "status": "cancelled",
                }

            output_result[channel] = channel_result
            del channel_result

    # ========================================================
    # 9. MÁSCARA DE VALIDADE
    # ========================================================

    log("🛡️ Construindo máscara de validade...")

    validity_mask = build_reference_mask(
        selected_frames,
        geometries,
        normalization_factors,
        height,
        width,
        config,
    )

    # ========================================================
    # 10. PÓS-PROCESSAMENTO
    # ========================================================

    if config.apply_dither_correction:
        log("🔧 Aplicando correção de dithering...")
        apply_dither_correction_inplace(output_result)

    output_result = np.nan_to_num(
        output_result,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    # ========================================================
    # 11. HEADER
    # ========================================================

    source_header = load_source_header(reference_frame.path)

    header = prepare_output_header(
        source_header=source_header,
        config=config,
        n_frames=len(selected_frames),
        n_frames_total=len(all_frames),
        n_batches=len(batch_metadata),
        avg_quality=avg_quality,
        avg_star_count=avg_star_count,
        avg_fwhm=avg_fwhm,
    )

    # ========================================================
    # 12. DTYPE
    # ========================================================

    log(f"💾 Preparando saída {config.output_bit_depth}...")

    data_to_save = convert_output_dtype(
        output_result,
        config,
    )

    # ========================================================
    # 13. SAÍDA
    # ========================================================

    config.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = config.output_dir / config.output_name

    log(f"💾 Salvando: {output_path}")

    write_stack_output(
        output_path=output_path,
        data_to_save=data_to_save,
        mask=validity_mask,
        header=header,
        config=config,
    )

    # ========================================================
    # 14. ESTATÍSTICAS
    # ========================================================

    stats = {
        "status": "success",
        "output_path": str(output_path),
        "n_frames": len(selected_frames),
        "n_frames_total": len(all_frames),
        "n_batches": len(batch_metadata),
        "method": config.method,
        "rejection": config.rejection_method,
        "selection_mode": config.selection_mode,
        "selection_percentage": config.selection_percentage,
        "selection_metric": config.selection_metric,
        "shape": tuple(output_result.shape),
        "dtype": str(data_to_save.dtype),
        "bit_depth": config.output_bit_depth,
        "min": float(np.min(output_result)),
        "max": float(np.max(output_result)),
        "mean": float(np.mean(output_result)),
        "std": float(np.std(output_result)),
        "avg_quality": avg_quality,
        "avg_star_count": avg_star_count,
        "avg_fwhm": avg_fwhm,
        "workers": config.worker_count,
        "chunk_size": effective_chunk,
        "memory_budget_mb": config.memory_budget_mb,
        "streaming": True,
        "batches": {name: meta["frame_count"] for name, meta in batch_metadata.items()},
    }

    log("\n✅ Stacking concluído!")
    log(f"📁 Resultado: {output_path}")
    log(
        f"📊 {len(selected_frames)} frames combinados "
        f"(de {len(all_frames)} disponíveis)"
    )
    log(f"📁 {len(batch_metadata)} batches processados")
    log(f"📈 Qualidade média: {avg_quality:.2f}")
    log(f"📈 FWHM médio: {avg_fwhm:.2f}px")
    log(f"💾 Profundidade: {config.output_bit_depth}")

    return stats


# ============================================================
# INTERFACE PÚBLICA
# ============================================================


def process_all_stacking(
    input_dir: Path,
    config_dict: dict,
    progress_callback: Callable | None = None,
    status_callback: Callable | None = None,
    cancel_event: threading.Event | None = None,
) -> dict:
    """
    Interface pública mantida compatível com main.py.
    """

    config_dict = config_dict if isinstance(config_dict, dict) else {}

    config = StackingConfig(
        base_dir=Path(config_dict.get("base_dir", "")),
        input_dir=Path(input_dir),
        output_dir=Path(
            config_dict.get(
                "output_dir",
                "",
            )
        ),
        selection_mode=config_dict.get(
            "selection_mode",
            "BestPercentage",
        ),
        selection_percentage=float(
            config_dict.get(
                "selection_percentage",
                80.0,
            )
        ),
        selection_metric=config_dict.get(
            "selection_metric",
            "quality",
        ),
        method=config_dict.get(
            "method",
            "Median",
        ),
        rejection_method=config_dict.get(
            "rejection_method",
            "SigmaClip",
        ),
        rejection_low=float(
            config_dict.get(
                "rejection_low",
                3.0,
            )
        ),
        rejection_high=float(
            config_dict.get(
                "rejection_high",
                3.0,
            )
        ),
        normalize=bool(
            config_dict.get(
                "normalize",
                True,
            )
        ),
        normalize_method=config_dict.get(
            "normalize_method",
            "Median",
        ),
        output_name=config_dict.get(
            "output_name",
            "stacked_image.fits",
        ),
        output_bit_depth=config_dict.get(
            "output_bit_depth",
            "32-bit",
        ),
        compress_output=bool(
            config_dict.get(
                "compress_output",
                True,
            )
        ),
        apply_dither_correction=bool(
            config_dict.get(
                "apply_dither_correction",
                False,
            )
        ),
        workers=(
            int(config_dict["workers"])
            if config_dict.get("workers") is not None
            else None
        ),
        chunk_size=int(
            config_dict.get(
                "chunk_size",
                DEFAULT_CHUNK_SIZE,
            )
        ),
        memory_budget_mb=int(
            config_dict.get(
                "memory_budget_mb",
                DEFAULT_MEMORY_BUDGET_MB,
            )
        ),
        normalization_max_samples=int(
            config_dict.get(
                "normalization_max_samples",
                DEFAULT_NORMALIZATION_MAX_SAMPLES,
            )
        ),
    )

    return process_stacking(
        config,
        progress_callback,
        status_callback,
        cancel_event,
    )
