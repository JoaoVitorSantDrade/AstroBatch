"""Validated application commands independent of widgets and storage."""
from dataclasses import dataclass
from pathlib import Path
import math


@dataclass(frozen=True)
class ResourceSettings:
    workers: int = 2
    memory_mb: int = 512

    @classmethod
    def from_values(cls, workers, memory_mb):
        parsed = []
        for name, value, minimum in (("workers",workers,1),("memory_mb",memory_mb,64)):
            number = float(value)
            if isinstance(value,bool) or not math.isfinite(number) or not number.is_integer() or number < minimum:
                raise ValueError(f"{name}: informe um inteiro maior ou igual a {minimum}.")
            parsed.append(int(number))
        return cls(*parsed)


@dataclass(frozen=True)
class CalibrationCommand:
    """Validated calibration input consumed by the pipeline adapter.

    The legacy calibration function still receives a dictionary through
    :meth:`to_legacy_config`; keeping that conversion here lets the UI migrate
    without changing the processing implementation or saved settings format.
    """

    input_dir: Path
    output_dir: Path
    apply_dark: bool = True
    dark_path: str = ""
    apply_flat: bool = True
    flat_path: str = ""
    create_master: bool = True
    overwrite: bool = False

    @classmethod
    def from_values(
        cls,
        input_dir,
        output_dir,
        apply_dark=True,
        dark_path="",
        apply_flat=True,
        flat_path="",
        create_master=True,
        overwrite=False,
    ):
        return cls(
            input_dir=Path(input_dir).expanduser().resolve(),
            output_dir=Path(output_dir).expanduser().resolve(),
            apply_dark=cls._as_bool("apply_dark", apply_dark),
            dark_path=str(dark_path or ""),
            apply_flat=cls._as_bool("apply_flat", apply_flat),
            flat_path=str(flat_path or ""),
            create_master=cls._as_bool("create_master", create_master),
            overwrite=cls._as_bool("overwrite", overwrite),
        )

    @staticmethod
    def _as_bool(name, value):
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)) and value in (0, 1):
            return bool(value)
        raise ValueError(f"{name}: informe um valor booleano.")

    def to_legacy_config(self) -> dict:
        return {
            "input_dir": str(self.input_dir),
            "output_dir": str(self.output_dir),
            "apply_dark": self.apply_dark,
            "dark_path": self.dark_path,
            "apply_flat": self.apply_flat,
            "flat_path": self.flat_path,
            "create_master": self.create_master,
            "overwrite": self.overwrite,
        }


@dataclass(frozen=True)
class BatchCommand:
    """Validated Batch settings with a legacy ``ProcessingConfig`` adapter."""

    input_dir: Path
    output_dir: Path
    threshold_factor: float
    crop_size: int
    dry_run: bool
    copy_files: bool
    overwrite: bool
    opt_method: str
    downsample_method: str
    downsample_scale: float

    @staticmethod
    def _as_bool(name, value):
        return CalibrationCommand._as_bool(name, value)

    @classmethod
    def from_values(
        cls,
        input_dir,
        output_dir,
        threshold_factor,
        crop_size,
        dry_run,
        copy_files,
        overwrite,
        opt_method,
        downsample_method,
        downsample_scale,
    ):
        threshold = float(threshold_factor)
        if not math.isfinite(threshold) or threshold <= 0:
            raise ValueError("threshold_factor: informe um número positivo e finito.")
        crop = float(crop_size)
        if isinstance(crop_size, bool) or not math.isfinite(crop) or not crop.is_integer() or crop < 1:
            raise ValueError("crop_size: informe um inteiro positivo.")
        scale = float(downsample_scale)
        if not math.isfinite(scale) or not 0 < scale <= 1:
            raise ValueError("downsample_scale: informe um valor entre 0 e 1.")
        if opt_method not in {"Crop", "Downsampling"}:
            raise ValueError("opt_method: escolha Crop ou Downsampling.")
        return cls(
            Path(input_dir).expanduser().resolve(),
            Path(output_dir).expanduser().resolve(),
            threshold,
            int(crop),
            cls._as_bool("dry_run", dry_run),
            cls._as_bool("copy_files", copy_files),
            cls._as_bool("overwrite", overwrite),
            str(opt_method),
            str(downsample_method),
            scale,
        )

    def to_legacy_config(self):
        from batch_logic import ProcessingConfig

        return ProcessingConfig(
            input_dir=self.input_dir,
            output_dir=self.output_dir,
            threshold_factor=self.threshold_factor,
            crop_size=self.crop_size,
            dry_run=self.dry_run,
            copy_files=self.copy_files,
            overwrite=self.overwrite,
            opt_method=self.opt_method,
            downsample_method=self.downsample_method,
            downsample_scale=self.downsample_scale,
        )


@dataclass(frozen=True)
class FlowCommand:
    batch_dir: Path
    config: dict

    @classmethod
    def from_values(cls, batch_dir, **values):
        required_floats = {
            "fwhm": (0, None), "sigma": (0, None), "ransac": (0, None),
            "min_ratio": (0, 1),
        }
        config = dict(values)
        for name, (minimum, maximum) in required_floats.items():
            value = float(config[name])
            if not math.isfinite(value) or value <= minimum or (maximum is not None and value > maximum):
                raise ValueError(f"{name}: valor numérico inválido.")
            config[name] = value
        for name, minimum in (("matching_radius", 1), ("min_stars", 1), ("min_inliers", 1), ("max_stars", 1), ("flow_workers", 1), ("memory_budget_mb", 64)):
            value = float(config[name])
            if isinstance(config[name], bool) or not math.isfinite(value) or not value.is_integer() or value < minimum:
                raise ValueError(f"{name}: informe um inteiro maior ou igual a {minimum}.")
            config[name] = int(value)
        if config.get("engine_profile", "Stable") not in {"Stable", "Fast"}:
            raise ValueError("engine_profile: escolha Stable ou Fast.")
        if config.get("registration_strategy", "neighbor_bfs") not in {"neighbor_bfs", "legacy", "incremental_chain"}:
            raise ValueError("registration_strategy: escolha neighbor_bfs ou legacy.")
        config["custom_anchors"] = dict(config.get("custom_anchors", {}))
        config["debug_images"] = cls._as_bool("debug_images", config.get("debug_images", False))
        return cls(Path(batch_dir).expanduser().resolve(), config)

    @staticmethod
    def _as_bool(name, value):
        return CalibrationCommand._as_bool(name, value)

    def to_legacy_config(self):
        return dict(self.config)


@dataclass(frozen=True)
class ReferenceChangeCommand:
    """Metadata-only rebase of an existing local/global Flow revision."""

    batch_dir: Path
    new_reference: str
    base_dir: Path | None = None

    @classmethod
    def from_values(cls, batch_dir, new_reference, base_dir=None):
        batch = Path(batch_dir).expanduser().resolve()
        reference = str(new_reference or "").strip()
        if not reference:
            raise ValueError("new_reference: selecione um frame válido.")
        base = Path(base_dir).expanduser().resolve() if base_dir else None
        return cls(batch, reference, base)

    def to_legacy_config(self):
        return {
            "batch_dir": self.batch_dir,
            "new_reference": self.new_reference,
            "base_dir": self.base_dir,
        }


@dataclass(frozen=True)
class AlignCommand:
    base_dir: Path
    output_dir: Path
    config: dict

    @classmethod
    def from_values(cls, base_dir, output_dir, **values):
        base = Path(base_dir).expanduser().resolve()
        output = Path(output_dir).expanduser().resolve()
        if base == output:
            raise ValueError("A pasta de destino deve ser diferente da pasta base.")
        shift = float(values.get("quality_max_shift", 1.5))
        if not math.isfinite(shift) or shift < 0:
            raise ValueError("Desvio residual deve ser finito e não negativo.")
        config = dict(values)
        config["quality_max_shift"] = shift
        for name, minimum in (("memory_budget_mb", 64), ("workers", 1)):
            value = float(config[name])
            if isinstance(config[name], bool) or not math.isfinite(value) or not value.is_integer() or value < minimum:
                raise ValueError(f"{name}: informe um inteiro maior ou igual a {minimum}.")
            config[name] = int(value)
        for name in ("overwrite", "dry_run", "keep_header", "delete_intermediates", "compress_output", "quality_gate"):
            config[name] = cls._as_bool(name, config[name])
        if config.get("engine_profile", "Stable") not in {"Stable", "Fast"}:
            raise ValueError("engine_profile: escolha Stable ou Fast.")
        return cls(base, output, config)

    @staticmethod
    def _as_bool(name, value):
        return CalibrationCommand._as_bool(name, value)

    def to_legacy_config(self):
        return dict(self.config)


@dataclass(frozen=True)
class StackCommand:
    """Validated Stack settings adapted to ``process_all_stacking``."""

    input_dir: Path
    config: dict

    @staticmethod
    def validate_options(min_roundness, min_shape_stars):
        roundness = float(min_roundness)
        if not math.isfinite(roundness) or not 0 <= roundness <= 1:
            raise ValueError("A roundness mínima b/a deve ser um número finito entre 0 e 1.")
        shape = float(min_shape_stars)
        if isinstance(min_shape_stars, bool) or not math.isfinite(shape) or not shape.is_integer() or not 1 <= shape <= 64:
            raise ValueError("O mínimo de estrelas medidas deve ser um inteiro entre 1 e 64.")
        return roundness, int(shape)

    @classmethod
    def from_values(cls, input_dir, output_dir, **values):
        input_path = Path(input_dir).expanduser().resolve()
        output_text = str(output_dir or "").strip()
        if not output_text:
            raise ValueError("output_dir: selecione uma pasta de saída.")
        output_path = Path(output_text).expanduser().resolve()
        if not input_path.is_dir():
            raise ValueError("input_dir: selecione uma pasta de FITS alinhados.")
        if input_path == output_path:
            raise ValueError("A pasta de saída deve ser diferente da entrada.")

        def finite(name, default, minimum=None, maximum=None):
            number = float(values.get(name, default))
            if not math.isfinite(number):
                raise ValueError(f"{name}: informe um número finito.")
            if minimum is not None and number < minimum:
                raise ValueError(f"{name}: valor abaixo do mínimo permitido.")
            if maximum is not None and number > maximum:
                raise ValueError(f"{name}: valor acima do máximo permitido.")
            return number

        roundness, shape = cls.validate_options(
            values.get("min_roundness", .65), values.get("min_shape_stars", 5)
        )
        config = dict(values)
        config.update({
            "input_dir": str(input_path),
            "output_dir": str(output_path),
            "min_roundness": roundness,
            "min_shape_stars": int(shape),
            "selection_percentage": finite("selection_percentage", 80, 0, 100),
            "rejection_low": finite("rejection_low", 3, 0),
            "rejection_high": finite("rejection_high", 3, 0),
            "output_bit_depth": "16-bit",
        })
        boolean_defaults = {
            "trail_filter_enabled": False,
            "normalize": True,
            "compress_output": True,
            "apply_dither_correction": False,
        }
        for name, default in boolean_defaults.items():
            config[name] = CalibrationCommand._as_bool(name, config.get(name, default))
        if config.get("engine_profile", "Stable") not in {"Stable", "Fast"}:
            raise ValueError("engine_profile: escolha Stable ou Fast.")
        config["base_dir"] = str(config.get("base_dir", input_path))
        return cls(input_path, config)

    def to_legacy_config(self):
        return dict(self.config)


@dataclass(frozen=True)
class HDRCommand:
    """Validated HDR configuration behind the legacy pipeline boundary."""

    config: object

    @classmethod
    def from_values(cls, values):
        from hdr_logic import build_hdr_config

        if not isinstance(values, dict):
            raise ValueError("HDR: informe uma configuração em dicionário.")
        return cls(build_hdr_config(dict(values)))

    def to_legacy_config(self):
        return self.config
