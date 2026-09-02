from __future__ import annotations

import os
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning

# Suprime avisos de verificação de cabeçalho do Astropy.
warnings.simplefilter("ignore", category=AstropyWarning)


FITS_SUFFIXES = {".fit", ".fits", ".fts"}
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
    Carrega a primeira imagem 2D encontrada como float32.

    Importante: nenhuma conversão para uint16 é feita aqui.
    """
    with fits.open(
        filepath,
        memmap=False,
        ignore_missing_end=True,
    ) as hdul:
        for hdu in hdul:
            if hdu.is_image and hdu.data is not None and hdu.data.ndim == 2:
                data = np.asarray(
                    hdu.data,
                    dtype=np.float32,
                ).copy()
                header = _sanitize_float_header(hdu.header)
                return data, header

    raise ValueError(f"Imagem 2D não encontrada em {filepath.name}")


def _read_for_master(
    filepath: Path,
) -> tuple[np.ndarray, fits.Header]:
    """Leitura isolada para despacho paralelo."""
    return load_fits_data(filepath)


def save_float32_fits(
    data: np.ndarray,
    header: fits.Header | None,
    output_path: Path,
) -> None:
    """
    Salva a imagem em FITS float32 SEM conversão para uint16.

    Não há:
        - clipping 0..65535;
        - cast uint16;
        - BZERO/BSCALE de unsigned integer;
        - quantização de float durante compressão.

    A saída é intencionalmente um FITS float32 não comprimido para
    preservar exatamente os valores float32 produzidos pela calibração.

    A conversão para uint16 anterior era inadequada para este estágio:
    subtração de Dark e divisão por Flat produzem valores fracionários
    e podem também produzir valores negativos legítimos do ponto de vista
    matemático do pipeline. Esses valores não devem ser descartados aqui.
    """
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    float_data = np.asarray(
        data,
        dtype=np.float32,
    )

    float_header = _sanitize_float_header(header)

    # Não definimos BITPIX manualmente; o Astropy o determina a partir
    # do dtype=float32 e evita inconsistência entre header e dados.
    hdu = fits.PrimaryHDU(
        data=float_data,
        header=float_header,
    )

    hdul = fits.HDUList([hdu])

    hdul.writeto(
        str(output_path),
        overwrite=True,
        output_verify="ignore",
    )

    hdul.close()


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
    Cria Master Dark ou Master Flat em float32.

    Todos os frames são carregados como float32 e a mediana é calculada
    nesse mesmo domínio. O arquivo resultante também permanece float32.
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

    app_print(
        f"Lendo {len(files)} frames em paralelo para gerar Master "
        f"({folder_path.name})...\n"
    )

    data_stack: list[np.ndarray] = []
    base_header: fits.Header | None = None

    workers = max(
        1,
        min(
            MAX_WORKERS,
            (os.cpu_count() or 1) + 2,
        ),
    )

    with ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix="master_read",
    ) as executor:
        futures = {
            executor.submit(
                _read_for_master,
                filepath,
            ): filepath
            for filepath in files
        }

        for future in as_completed(futures):
            if cancel_event.is_set():
                for pending in futures:
                    pending.cancel()
                return None

            filepath = futures[future]

            try:
                data, header = future.result()

                # Garante explicitamente float32 mesmo que o leitor
                # de uma instalação do Astropy retorne outro dtype.
                data = np.asarray(
                    data,
                    dtype=np.float32,
                )

                data_stack.append(data)

                if base_header is None:
                    base_header = header

            except Exception as exc:
                app_print(f"Erro ao ler {filepath.name}: {exc}\n")

    if cancel_event.is_set():
        return None

    if not data_stack:
        app_print(f"Nenhum frame válido para gerar Master em {folder_path.name}.\n")
        return None

    # Verificação de geometria antes da mediana.
    reference_shape = data_stack[0].shape

    incompatible = [
        index for index, data in enumerate(data_stack) if data.shape != reference_shape
    ]

    if incompatible:
        raise ValueError(
            "Frames incompatíveis encontrados durante a criação do Master. "
            f"Shape de referência={reference_shape}; "
            f"índices incompatíveis={incompatible[:10]}"
        )

    app_print(f"Calculando a mediana dos {len(data_stack)} frames em float32...\n")

    # np.median preserva a natureza float32 dos dados da entrada em
    # versões atuais do NumPy, mas fazemos cast explícito para garantir
    # o contrato do módulo.
    master_data = np.asarray(
        np.median(
            np.stack(data_stack, axis=0),
            axis=0,
        ),
        dtype=np.float32,
    )

    del data_stack

    if cancel_event.is_set():
        del master_data
        return None

    save_float32_fits(
        master_data,
        base_header,
        output_path,
    )

    app_print(f"Master float32 salvo em: {output_path.name}\n")

    return master_data


def prepare_master_flat(
    master_flat: np.ndarray,
) -> np.ndarray:
    """
    Normaliza o Master Flat preservando float32.

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
) -> str | None:
    """
    Calibra um LIGHT preservando float32.

    Fórmula:

        calibrated = (light - dark) / flat

    Sem clipping para uint16 e sem saturação artificial em zero.
    """
    out_path = output_dir / light_path.name

    if out_path.exists() and not overwrite:
        return f"Ignorado (já existe): {light_path.name}"

    try:
        data, header = load_fits_data(light_path)

        # ----------------------------------------------------
        # Dark subtraction
        # ----------------------------------------------------
        if master_dark is not None:
            if master_dark.shape != data.shape:
                raise ValueError(
                    "Master Dark possui dimensão incompatível: "
                    f"{master_dark.shape} vs {data.shape}"
                )

            data -= np.asarray(
                master_dark,
                dtype=np.float32,
            )

            # NÃO fazemos:
            #
            # data[data < 0] = 0.0
            #
            # Valores negativos não são convertidos artificialmente.

        # ----------------------------------------------------
        # Flat correction
        # ----------------------------------------------------
        if master_flat is not None:
            if master_flat.shape != data.shape:
                raise ValueError(
                    "Master Flat possui dimensão incompatível: "
                    f"{master_flat.shape} vs {data.shape}"
                )

            # np.divide evita uma exceção/warning desnecessário caso
            # um Flat inválido apareça inesperadamente.
            np.divide(
                data,
                master_flat,
                out=data,
                where=np.isfinite(master_flat) & (master_flat > np.float32(0.01)),
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

        save_float32_fits(
            data,
            header,
            out_path,
        )

        return None

    except Exception as exc:
        return f"Erro em {light_path.name}: {exc}"


def run_calibration_pipeline(
    config: dict,
    app_print,
    app_progress,
    cancel_event: threading.Event,
):
    """
    Executa a calibração completa.

    Todos os produtos permanecem em float32:

        RAW / Master Dark / Master Flat / LIGHT calibrado

    Nenhum produto de calibração é convertido para uint16.
    """

    input_dir = Path(config["input_dir"])

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
            master_dark_path = input_dir.parent / "MasterDark.fits"

            master_dark = make_master(
                dark_path,
                master_dark_path,
                app_print,
                cancel_event,
            )

        elif dark_path.is_file():
            master_dark, _ = load_fits_data(dark_path)

            master_dark = np.asarray(
                master_dark,
                dtype=np.float32,
            )

            app_print("Master Dark float32 carregado.\n")

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
            master_flat_path = input_dir.parent / "MasterFlat.fits"

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
                save_float32_fits(
                    master_flat,
                    None,
                    master_flat_path,
                )

                app_print("Master Flat normalizado e salvo em float32.\n")

                del raw_master_flat

        elif flat_path.is_file():
            raw_master_flat, _ = load_fits_data(flat_path)

            master_flat = prepare_master_flat(raw_master_flat)

            del raw_master_flat

            app_print("Master Flat float32 carregado e normalizado.\n")

    if cancel_event.is_set():
        app_print("\nCalibração cancelada durante os Masters.\n")
        return

    # ========================================================
    # 2. LIGHTS
    # ========================================================

    lights = sorted(
        [
            filepath
            for filepath in input_dir.iterdir()
            if (filepath.is_file() and filepath.suffix.lower() in FITS_SUFFIXES)
        ],
        key=lambda p: p.name.casefold(),
    )

    total = len(lights)

    if total == 0:
        app_print("Nenhum LIGHT encontrado na pasta de entrada.\n")
        return

    app_print(f"\nIniciando calibração paralela de {total} frames...\n")

    processed = 0
    failed = 0
    skipped = 0

    workers = max(
        1,
        min(
            MAX_WORKERS,
            (os.cpu_count() or 1) + 2,
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
