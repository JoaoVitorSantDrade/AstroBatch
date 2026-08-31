# astroalign_logic.py

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from astropy.io import fits

FITS_SUFFIXES = {".fit", ".fits", ".fts"}

INTERPOLATION_MODES = {
    "Nearest": cv2.INTER_NEAREST,
    "Bilinear": cv2.INTER_LINEAR,
    "Bicubic": cv2.INTER_CUBIC,
    "Lanczos": cv2.INTER_LANCZOS4,
}


@dataclass(frozen=True)
class AlignConfig:
    base_dir: Path
    output_dir: Path
    interpolation: str
    overwrite: bool
    dry_run: bool
    keep_header: bool


def get_optimal_worker_count() -> int:
    try:
        cpu_count = getattr(os, "process_cpu_count", os.cpu_count)() or 1
    except Exception:
        cpu_count = os.cpu_count() or 1
    return max(1, min(16, cpu_count))


def find_batch_folders(base_dir: Path) -> list[Path]:
    return sorted(
        d for d in base_dir.iterdir() if d.is_dir() and "batch" in d.name.lower()
    )


def load_json(filepath: Path) -> dict:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def load_local_flow(batch_dir: Path) -> dict | None:
    flow_path = batch_dir / "flow_local.json"
    if not flow_path.exists():
        return None
    return load_json(flow_path)


def load_global_flow(base_dir: Path) -> dict | None:
    flow_path = base_dir / "global_flow.json"
    if not flow_path.exists():
        return None
    return load_json(flow_path)


def compute_final_matrix(local_matrix: list, global_matrix: list) -> np.ndarray:
    """
    Resolução por Multiplicação (Nível 2).

    Combina o Flow Local (Frame -> Âncora da Batch), calculado
    pelo AstroFlow e salvo em flow_local.json, com o Flow Global
    (Âncora da Batch -> Global_Master), salvo em global_flow.json,
    numa única matriz homogênea absoluta:

        Final = Batch_Offset_Matrix @ Local_Matrix

    Nenhuma estrela precisa ser reprocessada: é pura álgebra
    linear sobre matrizes já calculadas.
    """
    local = np.asarray(local_matrix, dtype=np.float64)
    offset = np.asarray(global_matrix, dtype=np.float64)
    return offset @ local


def load_fits_data(filepath: Path) -> tuple[np.ndarray, fits.Header]:
    # Usamos ignore_missing_end=True e, se necessário, context manager seguro
    with fits.open(filepath, memmap=False, do_not_scale_image_data=True) as hdul:
        for hdu in hdul:
            if hdu.is_image and hdu.data is not None and hdu.data.ndim == 2:
                # Copia o header ignorando erros est严ritos de conformidad do FITS original
                header = hdu.header.copy(strip=False)
                data = np.asarray(hdu.data, dtype=np.float32)
                return data, header
    raise ValueError(f"Imagem 2D não encontrada em {filepath.name}")


def warp_frame(data: np.ndarray, final_matrix: np.ndarray, interpolation_flag: int) -> np.ndarray:
    """
    Transformação Final (Warping).

    Aplica a matriz absoluta via cv2.warpAffine, preenchendo as
    bordas vazias geradas pela translação/rotação com zeros
    (fundo preto absoluto) para não interferir na estatística do
    empilhamento futuro. O tamanho do canvas é preservado
    (mesma dimensão do frame original).
    """
    h, w = data.shape
    matrix_2x3 = final_matrix[:2, :].astype(np.float32)
    return cv2.warpAffine(
        data,
        matrix_2x3,
        (w, h),
        flags=interpolation_flag,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )


def save_fits(data: np.ndarray, header: fits.Header | None, output_path: Path):
    hdu = fits.PrimaryHDU(data=data.astype(np.float32), header=header)
    
    # Cria o HDU List explicitamente permitindo ignorar avisos de verificação padrão do FITS
    hdul = fits.HDUList([hdu])
    hdul.writeto(str(output_path), overwrite=True, output_verify='ignore')


def _process_single_alignment(
    frame_name: str,
    frame_info: dict,
    batch_dir: Path,
    output_dir: Path,
    global_matrix: list,
    interpolation_flag: int,
    config: "AlignConfig",
) -> tuple[str, str | None]:
    """
    Executada em paralelo.

    Responsabilidade única: ler a imagem original, resolver a
    matriz absoluta (Local x Global) e aplicar o warp. Não há
    dependência entre frames, portanto todos podem ser
    processados simultaneamente — diferente do AstroFlow, que
    precisa da cadeia sequencial para detectar estrelas.
    """
    try:
        filepath = batch_dir / frame_name
        if not filepath.exists():
            return frame_name, f"Aviso: arquivo original não encontrado: {filepath}"

        data, header = load_fits_data(filepath)

        final_matrix = compute_final_matrix(frame_info["matrix"], global_matrix)
        warped = warp_frame(data, final_matrix, interpolation_flag)

        output_path = output_dir / frame_name
        if output_path.exists() and not config.overwrite:
            return frame_name, f"ERRO: destino já existe, arquivo ignorado: {output_path}"

        if not config.dry_run:
            output_dir.mkdir(parents=True, exist_ok=True)
            save_fits(warped, header if config.keep_header else None, output_path)

        return frame_name, None
    except Exception as exc:
        return frame_name, f"Erro ao alinhar {frame_name}: {exc}"


def process_batch_alignment(
    batch_dir: Path,
    global_flow: dict,
    config: AlignConfig,
    app_print,
    app_progress,
    cancel_event: threading.Event,
    progress_state: dict,
) -> tuple[int, int]:
    local_flow = load_local_flow(batch_dir)
    if local_flow is None:
        app_print(f"[{batch_dir.name}] AVISO: flow_local.json não encontrado, batch ignorada.\n")
        return 0, 0

    batch_entry = global_flow["batches"].get(batch_dir.name)
    if batch_entry is None:
        app_print(f"[{batch_dir.name}] AVISO: batch ausente no global_flow.json, ignorada.\n")
        return 0, 0

    global_matrix = batch_entry["matrix"]
    frames = local_flow.get("frames", {})
    total_frames = len(frames)

    if total_frames == 0:
        return 0, 0

    output_dir = config.output_dir / batch_dir.name
    interpolation_flag = INTERPOLATION_MODES.get(config.interpolation, cv2.INTER_LANCZOS4)

    worker_count = get_optimal_worker_count()
    processed = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="astroalign") as executor:
        futures = {
            executor.submit(
                _process_single_alignment,
                frame_name,
                frame_info,
                batch_dir,
                output_dir,
                global_matrix,
                interpolation_flag,
                config,
            ): frame_name
            for frame_name, frame_info in frames.items()
        }

        for future in as_completed(futures):
            if cancel_event.is_set():
                executor.shutdown(wait=False, cancel_futures=True)
                break

            frame_name, error = future.result()
            progress_state["done"] += 1

            if error:
                app_print(f"  [{frame_name}] {error}\n")
                failed += 1
            else:
                processed += 1

            if progress_state["done"] % 10 == 0 or progress_state["done"] == progress_state["total"]:
                app_progress(
                    progress_state["done"],
                    progress_state["total"],
                    f"Alinhando frames ({progress_state['done']}/{progress_state['total']})...",
                )

    app_print(
        f"[{batch_dir.name}] Concluído: {processed} alinhados, {failed} falhas de {total_frames} frames.\n"
    )
    return processed, failed


def process_all_alignments(
    base_dir: Path,
    output_dir: Path,
    config_dict: dict,
    app_print,
    app_progress,
    cancel_event: threading.Event,
) -> tuple[int, int]:
    """
    Orquestrador principal do AstroAlign.

    O AstroAlign atua como módulo puramente executor: não detecta
    estrelas nem recalcula alinhamentos. Ele apenas lê as matrizes
    já calculadas pelo AstroFlow (Flow Local por Batch + Flow
    Global entre Âncoras), resolve a matriz absoluta de cada frame
    por multiplicação de matrizes e aplica a transformação
    geométrica final (warpAffine), gerando as bordas pretas do
    mosaico. Nenhum cálculo de alinhamento no céu é refeito aqui —
    o resultado é instantâneo.
    """
    if not isinstance(config_dict, dict):
        config_dict = {}

    align_config = AlignConfig(
        base_dir=base_dir,
        output_dir=output_dir,
        interpolation=config_dict.get("interpolation", "Lanczos"),
        overwrite=bool(config_dict.get("overwrite", False)),
        dry_run=bool(config_dict.get("dry_run", False)),
        keep_header=bool(config_dict.get("keep_header", True)),
    )

    app_print(f"Lendo Flow Global de: {base_dir / 'global_flow.json'}\n")
    global_flow = load_global_flow(base_dir)

    if global_flow is None:
        app_print("ERRO: global_flow.json não encontrado. Execute o AstroFlow primeiro.\n")
        return 0, 0

    batch_folders = find_batch_folders(base_dir)
    if not batch_folders:
        app_print(f"Nenhuma subpasta de Batch encontrada em {base_dir}\n")
        return 0, 0

    app_print(f"Master Global: {global_flow.get('global_master_batch', '???')}\n")

    # Pré-carrega o Flow Local de cada Batch e soma o total de
    # frames para uma barra de progresso global unificada.
    total_frames = 0
    batches_with_flow = []
    for b_folder in batch_folders:
        local_flow = load_local_flow(b_folder)
        if local_flow is None:
            continue
        batches_with_flow.append(b_folder)
        total_frames += len(local_flow.get("frames", {}))

    if total_frames == 0:
        app_print("Nenhum frame com Flow Local calculado foi encontrado.\n")
        return 0, 0

    progress_state = {"done": 0, "total": total_frames}
    app_progress(0, total_frames, "Iniciando AstroAlign...")

    if not align_config.dry_run:
        align_config.output_dir.mkdir(parents=True, exist_ok=True)

    total_processed = 0
    total_failed = 0

    for b_folder in batches_with_flow:
        if cancel_event.is_set():
            app_print("\nCancelamento solicitado...\n")
            break

        app_print(f"\nAlinhando Batch: {b_folder.name}\n")
        processed, failed = process_batch_alignment(
            b_folder,
            global_flow,
            align_config,
            app_print,
            app_progress,
            cancel_event,
            progress_state,
        )
        total_processed += processed
        total_failed += failed

    app_progress(total_frames, total_frames, "Concluído.")
    app_print(
        f"\n>>> AstroAlign Finalizado! {total_processed} frames alinhados, "
        f"{total_failed} falhas, em {len(batches_with_flow)} batches. <<<\n"
    )
    return total_processed, total_failed
