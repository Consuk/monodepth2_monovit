# eval_endovis_corruptions.py
from __future__ import absolute_import, division, print_function

import os
import csv
import argparse
from collections import defaultdict

import cv2
import numpy as np
import torch

import networks
import datasets

from utils import readlines
from layers import disp_to_depth


STEREO_SCALE_FACTOR = 5.4


def compute_errors(gt, pred):
    """Métricas estándar de profundidad."""
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = np.sqrt(((gt - pred) ** 2).mean())
    rmse_log = np.sqrt(((np.log(gt) - pred.__array_priority__ * 0 + np.log(pred)) ** 2).mean())  # se corrige abajo
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    # corrección explícita para evitar cualquier rareza
    rmse_log = np.sqrt(((np.log(gt) - np.log(pred)) ** 2).mean())

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def debug_print(debug, msg):
    if debug:
        print(msg)


def describe_tensor(x, name="tensor"):
    if isinstance(x, torch.Tensor):
        return f"{name}: shape={tuple(x.shape)} dtype={x.dtype} device={x.device}"
    return f"{name}: type={type(x)}"


def load_model(load_weights_folder, num_layers, device, debug=False):
    """
    Carga encoder y decoder de MonoIIT una sola vez.
    num_layers se conserva por compatibilidad, aunque mpvit_small no lo use.
    """
    encoder_path = os.path.join(load_weights_folder, "encoder.pth")
    decoder_path = os.path.join(load_weights_folder, "depth.pth")

    if not os.path.isdir(load_weights_folder):
        raise FileNotFoundError(f"No existe la carpeta de pesos: {load_weights_folder}")
    if not os.path.isfile(encoder_path):
        raise FileNotFoundError(f"No existe {encoder_path}")
    if not os.path.isfile(decoder_path):
        raise FileNotFoundError(f"No existe {decoder_path}")

    debug_print(debug, f"[DEBUG] device = {device}")
    debug_print(debug, f"[DEBUG] encoder_path = {encoder_path}")
    debug_print(debug, f"[DEBUG] decoder_path = {decoder_path}")

    # ===== MonoIIT backbone =====
    encoder = networks.mpvit_small()
    encoder.num_ch_enc = [64, 128, 216, 288, 288]
    depth_decoder = networks.DepthDecoderT()

    encoder_dict = torch.load(encoder_path, map_location=device)
    decoder_dict = torch.load(decoder_path, map_location=device)

    encoder_state = encoder.state_dict()
    filtered_dict = {k: v for k, v in encoder_dict.items() if k in encoder_state}

    missing_in_ckpt = [k for k in encoder_state.keys() if k not in encoder_dict]
    extra_in_ckpt = [k for k in encoder_dict.keys() if k not in encoder_state]

    debug_print(debug, f"[DEBUG] encoder ckpt keys = {len(encoder_dict)}")
    debug_print(debug, f"[DEBUG] encoder model keys = {len(encoder_state)}")
    debug_print(debug, f"[DEBUG] encoder matched keys = {len(filtered_dict)}")
    debug_print(debug, f"[DEBUG] encoder missing keys (first 10) = {missing_in_ckpt[:10]}")
    debug_print(debug, f"[DEBUG] encoder extra keys (first 10) = {extra_in_ckpt[:10]}")
    debug_print(debug, f"[DEBUG] depth_decoder ckpt keys = {len(decoder_dict)}")

    encoder.load_state_dict(filtered_dict, strict=False)
    depth_decoder.load_state_dict(decoder_dict)

    encoder.to(device).eval()
    depth_decoder.to(device).eval()

    debug_print(debug, f"[DEBUG] Encoder class = {encoder.__class__.__name__}")
    debug_print(debug, f"[DEBUG] Decoder class = {depth_decoder.__class__.__name__}")

    return encoder, depth_decoder


def build_dataset(dataset_name, data_path_root, filenames, height, width, png=False):
    """
    Construye el dataset correcto según el nombre indicado.
    """
    img_ext = ".png" if png else ".jpg"

    if dataset_name.lower() == "hamlyn":
        return datasets.HamlynDataset(
            data_path_root,
            filenames,
            height,
            width,
            [0],
            4,
            is_train=False,
            img_ext=img_ext,
        )

    if dataset_name.lower() in ("endovis", "scared"):
        return datasets.SCAREDRAWDataset(
            data_path_root,
            filenames,
            height,
            width,
            [0],
            4,
            is_train=False,
            img_ext=img_ext,
        )

    raise ValueError(f"Dataset no soportado: {dataset_name}")


def evaluate_one_root(
    data_path_root,
    filenames,
    gt_depths,
    encoder,
    depth_decoder,
    dataset_name="hamlyn",
    height=256,
    width=320,
    batch_size=16,
    png=False,
    disable_median_scaling=False,
    pred_depth_scale_factor=1.0,
    strict=False,
    min_depth=1e-3,
    max_depth=150.0,
    device="cuda",
    debug=False,
):
    """
    Evalúa una raíz concreta.
    """
    dataset = build_dataset(dataset_name, data_path_root, filenames, height, width, png=png)

    debug_print(debug, f"[DEBUG] dataset_name = {dataset_name}")
    debug_print(debug, f"[DEBUG] data_path_root = {data_path_root}")
    debug_print(debug, f"[DEBUG] total split lines = {len(filenames)}")
    debug_print(debug, f"[DEBUG] dataset len = {len(dataset)}")

    preds_list = []
    kept_indices = []

    buffer_imgs = []
    buffer_ids = []

    first_forward_done = False

    def flush_buffer():
        nonlocal first_forward_done

        if not buffer_imgs:
            return

        with torch.no_grad():
            batch = torch.stack(buffer_imgs, dim=0).to(device)

            if debug and not first_forward_done:
                print("[DEBUG] " + describe_tensor(batch, "input_batch"))

            features = encoder(batch)

            if debug and not first_forward_done:
                if isinstance(features, (list, tuple)):
                    print(f"[DEBUG] encoder returned {type(features).__name__} with {len(features)} feature maps")
                    for j, feat in enumerate(features):
                        print("[DEBUG] " + describe_tensor(feat, f"features[{j}]"))
                else:
                    print(f"[DEBUG] encoder returned type={type(features)}")
                    try:
                        print("[DEBUG] " + describe_tensor(features, "features"))
                    except Exception:
                        pass

            outputs = depth_decoder(features)

            if debug and not first_forward_done:
                print(f"[DEBUG] decoder output type = {type(outputs)}")
                if isinstance(outputs, dict):
                    print(f"[DEBUG] decoder output keys = {list(outputs.keys())}")
                    if ("disp", 0) in outputs:
                        print("[DEBUG] " + describe_tensor(outputs[("disp", 0)], 'outputs[("disp", 0)]'))
                    else:
                        print('[DEBUG] key ("disp", 0) NO encontrada en outputs')
                else:
                    print(f"[DEBUG] decoder output repr = {repr(outputs)}")

            if not isinstance(outputs, dict) or ("disp", 0) not in outputs:
                raise KeyError(
                    f'El decoder no devolvió la llave ("disp", 0). '
                    f"Tipo de salida: {type(outputs)}"
                )

            pred_disp, _ = disp_to_depth(outputs[("disp", 0)], min_depth, max_depth)

            if debug and not first_forward_done:
                print("[DEBUG] " + describe_tensor(pred_disp, "pred_disp_after_disp_to_depth"))
                first_forward_done = True

            preds_list.append(pred_disp[:, 0].cpu().numpy())

    missing = 0
    total = len(filenames)
    debug_shown = 0

    for i in range(total):
        try:
            sample = dataset[i]

            if debug and i == 0:
                print(f"[DEBUG] sample[0] keys = {list(sample.keys())}")

            img_t = sample[("color", 0, 0)]

            if not isinstance(img_t, torch.Tensor):
                img_t = torch.as_tensor(img_t)

            if debug and i == 0:
                print("[DEBUG] " + describe_tensor(img_t, 'sample[("color", 0, 0)]'))

            buffer_imgs.append(img_t)
            buffer_ids.append(i)

            if len(buffer_imgs) == batch_size:
                flush_buffer()
                kept_indices.extend(buffer_ids)
                buffer_imgs.clear()
                buffer_ids.clear()

        except FileNotFoundError as e:
            missing += 1
            if debug_shown < 10:
                print(f"   [DEBUG] idx={i} file='{filenames[i]}' FileNotFoundError: {e}")
                debug_shown += 1
            if strict:
                raise FileNotFoundError(
                    f"[STRICT] Falta la muestra idx={i} en {data_path_root}: {e}"
                )
        except Exception as e:
            missing += 1
            if debug_shown < 10:
                print(f"   [DEBUG] idx={i} file='{filenames[i]}' error={repr(e)}")
                debug_shown += 1
            if strict:
                raise RuntimeError(
                    f"[STRICT] Error cargando idx={i} en {data_path_root}: {e}"
                )

    if buffer_imgs:
        flush_buffer()
        kept_indices.extend(buffer_ids)

    if len(kept_indices) == 0:
        mode = "STRICT" if strict else "LENIENT"
        raise FileNotFoundError(
            f"[{mode}] Ninguna muestra utilizable en {data_path_root}. "
            f"Faltantes/errores: {missing}/{total}"
        )

    if (not strict) and missing > 0:
        print(
            f"   [INFO] {data_path_root}: usando {len(kept_indices)}/{total} frames "
            f"(faltaron {missing})."
        )

    pred_disps = np.concatenate(preds_list, axis=0)

    if isinstance(gt_depths, list):
        sel_gt = [gt_depths[idx] for idx in kept_indices]
    else:
        sel_gt = gt_depths[kept_indices]

    debug_print(debug, f"[DEBUG] pred_disps.shape = {pred_disps.shape}")
    debug_print(debug, f"[DEBUG] kept_indices count = {len(kept_indices)}")

    errors = []
    ratios = []
    invalid_metric_samples = 0

    for i in range(pred_disps.shape[0]):
        gt_depth = np.asarray(sel_gt[i]).astype(np.float32)
        gt_h, gt_w = gt_depth.shape[:2]

        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_w, gt_h))
        pred_depth = 1.0 / (pred_disp + 1e-8)

        mask = np.logical_and(gt_depth > min_depth, gt_depth < max_depth)

        gd = gt_depth[mask]
        pd = pred_depth[mask]

        if pd.size == 0 or gd.size == 0:
            invalid_metric_samples += 1
            if debug and invalid_metric_samples <= 5:
                print(f"[DEBUG] sample {i}: máscara vacía para métricas")
            continue

        if pred_depth_scale_factor != 1.0:
            pd *= pred_depth_scale_factor

        if not disable_median_scaling:
            ratio = np.median(gd) / (np.median(pd) + 1e-8)
            ratios.append(ratio)
            pd *= ratio

        pd[pd < min_depth] = min_depth
        pd[pd > max_depth] = max_depth

        errors.append(compute_errors(gd, pd))

    if debug:
        print(f"[DEBUG] valid metric samples = {len(errors)}")
        print(f"[DEBUG] invalid metric samples = {invalid_metric_samples}")

    if len(errors) == 0:
        raise RuntimeError(f"No se pudieron calcular métricas válidas en {data_path_root}")

    if not disable_median_scaling and len(ratios) > 0:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(f"    Scaling ratios | med: {med:0.3f} | std: {np.std(ratios / med):0.3f}")

    return np.array(errors).mean(0), len(kept_indices), total, missing


def list_corruption_dirs(root):
    """
    Si root ya apunta a una sola corrupción (contiene severity_*), devuelve [root].
    Si root contiene múltiples corrupciones, devuelve sus subdirectorios.
    """
    if not os.path.isdir(root):
        return []

    severities = [
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d)) and d.startswith("severity_")
    ]
    if len(severities) > 0:
        return [root]

    return [
        os.path.join(root, d)
        for d in sorted(os.listdir(root))
        if os.path.isdir(os.path.join(root, d))
    ]


def safe_makedirs(path):
    os.makedirs(path, exist_ok=True)


def save_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser("Evaluate Hamlyn/EndoVIS corruptions with MonoIIT + debug")

    parser.add_argument("--corruptions_root", type=str, required=True,
                        help="Raíz de corrupciones o carpeta de una corrupción")
    parser.add_argument("--load_weights_folder", type=str, required=True,
                        help="Carpeta con encoder.pth y depth.pth")
    parser.add_argument("--splits_dir", type=str, default=os.path.join(os.path.dirname(__file__), "splits"))
    parser.add_argument("--split", type=str, default="hamlyn",
                        help="Nombre del split dentro de splits/")
    parser.add_argument("--dataset", type=str, default="hamlyn",
                        choices=["hamlyn", "endovis", "scared"],
                        help="Dataset a usar para construir el loader")
    parser.add_argument("--data_subdir", type=str, default="",
                        help="Subcarpeta dentro de severity_X para datasets no-Hamlyn")

    parser.add_argument("--num_layers", type=int, default=18)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--batch_size", type=int, default=16)

    parser.add_argument("--png", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--eval_stereo", action="store_true")
    parser.add_argument("--debug", action="store_true",
                        help="Imprime información de debug del modelo, features, decoder y dataset")
    parser.add_argument("--debug_first_only", action="store_true",
                        help="Hace debug detallado solo en la primera severidad utilizable")

    parser.add_argument("--min_depth", type=float, default=1.0)
    parser.add_argument("--max_depth", type=float, default=50.0)

    parser.add_argument("--run_name", type=str, default="hamlyn_corruptions_eval",
                        help="Nombre base de la corrida/salida")
    parser.add_argument("--output_dir", type=str, default="eval_outputs",
                        help="Directorio donde se guardarán los CSV")
    parser.add_argument("--summary_filename", type=str, default="summary_by_severity.csv",
                        help="Nombre del CSV principal")
    parser.add_argument("--per_corruption_filename", type=str, default="summary_by_corruption.csv",
                        help="Nombre del CSV con promedio por corrupción")
    parser.add_argument("--global_avg_filename", type=str, default="global_average.csv",
                        help="Nombre del CSV con promedio global")

    args = parser.parse_args()

    cv2.setNumThreads(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_files_path = os.path.join(args.splits_dir, args.split, "test_files.txt")
    gt_path = os.path.join(args.splits_dir, args.split, "gt_depths.npz")

    if not os.path.isfile(test_files_path):
        raise FileNotFoundError(f"No se encontró: {test_files_path}")
    if not os.path.isfile(gt_path):
        raise FileNotFoundError(f"No se encontró: {gt_path}")

    test_files = readlines(test_files_path)

    gt_npz = np.load(
        gt_path,
        allow_pickle=True,
        fix_imports=True,
        encoding="latin1"
    )
    gt_depths = gt_npz["data"]

    if isinstance(gt_depths, np.ndarray) and gt_depths.dtype == object:
        gt_depths = list(gt_depths)

    if len(test_files) != len(gt_depths):
        print(
            "[WARN] test_files y gt_depths no tienen la misma longitud. "
            "El script seguirá y filtrará según las muestras realmente utilizables."
        )

    disable_median_scaling = args.eval_stereo
    pred_depth_scale_factor = STEREO_SCALE_FACTOR if args.eval_stereo else 1.0

    print("-> Cargando modelo MonoIIT desde:", args.load_weights_folder)
    encoder, depth_decoder = load_model(
        args.load_weights_folder,
        args.num_layers,
        device,
        debug=args.debug
    )

    corr_dirs = list_corruption_dirs(args.corruptions_root)
    if len(corr_dirs) == 0:
        raise FileNotFoundError(f"No se encontraron corrupciones en {args.corruptions_root}")

    run_output_dir = os.path.join(args.output_dir, args.run_name)
    safe_makedirs(run_output_dir)

    rows = []

    print("-> Iniciando evaluación")
    detailed_debug_consumed = False

    for corr_dir in corr_dirs:
        corr_name = os.path.basename(corr_dir.rstrip("/"))

        severities = sorted(
            [
                d for d in os.listdir(corr_dir)
                if os.path.isdir(os.path.join(corr_dir, d)) and d.startswith("severity_")
            ],
            key=lambda s: int(s.split("_")[-1]) if s.split("_")[-1].isdigit() else 9999
        )

        for sev in severities:
            if args.dataset.lower() == "hamlyn":
                data_root = os.path.join(corr_dir, sev)
            else:
                data_root = (
                    os.path.join(corr_dir, sev, args.data_subdir)
                    if args.data_subdir
                    else os.path.join(corr_dir, sev)
                )

            print(f"\n>> {corr_name} / {sev} :: {data_root}")

            if not os.path.isdir(data_root):
                print(f"   [WARN] No existe {data_root}, se omite.")
                continue

            this_debug = args.debug
            if args.debug_first_only and detailed_debug_consumed:
                this_debug = False

            try:
                mean_errors, used_count, total_count, missing_count = evaluate_one_root(
                    data_path_root=data_root,
                    filenames=test_files,
                    gt_depths=gt_depths,
                    encoder=encoder,
                    depth_decoder=depth_decoder,
                    dataset_name=args.dataset,
                    height=args.height,
                    width=args.width,
                    batch_size=args.batch_size,
                    png=args.png,
                    disable_median_scaling=disable_median_scaling,
                    pred_depth_scale_factor=pred_depth_scale_factor,
                    strict=args.strict,
                    min_depth=args.min_depth,
                    max_depth=args.max_depth,
                    device=device,
                    debug=this_debug,
                )

                abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = mean_errors.tolist()
                rows.append([
                    corr_name, sev,
                    abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3,
                    used_count, total_count, missing_count
                ])

                print(
                    f"   used={used_count}/{total_count} | missing={missing_count} | "
                    f"abs_rel={abs_rel:.3f} | sq_rel={sq_rel:.3f} | rmse={rmse:.3f} | "
                    f"rmse_log={rmse_log:.3f} | a1={a1:.3f} | a2={a2:.3f} | a3={a3:.3f}"
                )

                if args.debug_first_only and not detailed_debug_consumed:
                    detailed_debug_consumed = True

            except Exception as e:
                print(f"   [SKIP] {e}")

    if not rows:
        print("\n-> No se generaron resultados.")
        return

    header = [
        "corruption", "severity",
        "abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3",
        "used_samples", "total_split_samples", "missing_or_failed_samples"
    ]

    summary_csv = os.path.join(run_output_dir, args.summary_filename)
    save_csv(summary_csv, header, rows)

    print(f"\n-> CSV principal guardado en: {summary_csv}")

    # Promedio por corrupción
    bucket = defaultdict(list)
    for r in rows:
        bucket[r[0]].append(r)

    per_corr_rows = []
    for corr in sorted(bucket.keys()):
        vals = np.array([r[2:9] for r in bucket[corr]], dtype=np.float64)
        means = vals.mean(axis=0).tolist()

        used_sum = int(sum(r[9] for r in bucket[corr]))
        total_sum = int(sum(r[10] for r in bucket[corr]))
        missing_sum = int(sum(r[11] for r in bucket[corr]))

        per_corr_rows.append([corr] + means + [used_sum, total_sum, missing_sum])

    per_corr_header = [
        "corruption",
        "abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3",
        "used_samples_sum", "total_split_samples_sum", "missing_or_failed_samples_sum"
    ]
    per_corr_csv = os.path.join(run_output_dir, args.per_corruption_filename)
    save_csv(per_corr_csv, per_corr_header, per_corr_rows)

    print(f"-> Promedio por corrupción guardado en: {per_corr_csv}")

    # Promedio global
    all_vals = np.array([r[2:9] for r in rows], dtype=np.float64)
    global_means = all_vals.mean(axis=0).tolist()
    global_used = int(sum(r[9] for r in rows))
    global_total = int(sum(r[10] for r in rows))
    global_missing = int(sum(r[11] for r in rows))

    global_csv = os.path.join(run_output_dir, args.global_avg_filename)
    save_csv(
        global_csv,
        per_corr_header,
        [["global"] + global_means + [global_used, global_total, global_missing]]
    )

    print(f"-> Promedio global guardado en: {global_csv}")

    print("\n======= RESUMEN =======")
    print("Archivo principal:", summary_csv)
    print("Por corrupción   :", per_corr_csv)
    print("Global           :", global_csv)


if __name__ == "__main__":
    main()