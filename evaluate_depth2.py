from __future__ import absolute_import, division, print_function
import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import cv2

from options import MonodepthOptions
from networks import ResnetEncoder, DepthDecoder

def readlines(p):
    with open(p, "r") as f:
        return f.read().splitlines()

def disp_to_depth(disp, min_depth, max_depth):
    min_disp = 1.0 / max_depth
    max_disp = 1.0 / min_depth
    scaled = min_disp + (max_disp - min_disp) * disp
    depth = 1.0 / scaled
    return scaled, depth

def compute_depth_errors(gt, pred):
    mask = gt > 0
    if not np.any(mask):
        return None
    gt = gt[mask]
    pred = pred[mask]

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel  = np.mean(((gt - pred) ** 2) / gt)
    rmse    = np.sqrt(np.mean((gt - pred) ** 2))
    rmse_log = np.sqrt(np.mean((np.log(gt + 1e-8) - np.log(pred + 1e-8)) ** 2))

    thresh = np.maximum(gt / (pred + 1e-8), (pred + 1e-8) / gt)
    a1 = (thresh < 1.25    ).mean()
    a2 = (thresh < 1.25**2 ).mean()
    a3 = (thresh < 1.25**3 ).mean()
    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

def load_model(opt, device):
    enc = ResnetEncoder(opt.num_layers, opt.weights_init == "pretrained")
    dec = DepthDecoder(enc.num_ch_enc, scales=opt.scales)

    enc_path = os.path.join(opt.load_weights_folder, "encoder.pth")
    dec_path = os.path.join(opt.load_weights_folder, "depth.pth")
    enc_dict = torch.load(enc_path, map_location=device)
    dec_dict = torch.load(dec_path, map_location=device)

    feed_h = enc_dict.get("height", opt.height)
    feed_w = enc_dict.get("width",  opt.width)

    enc.load_state_dict({k: v for k, v in enc_dict.items() if k in enc.state_dict()})
    dec.load_state_dict(dec_dict)

    enc.to(device).eval()
    dec.to(device).eval()
    return enc, dec, feed_h, feed_w

def load_gt_npz(npz_path):
    npz = np.load(npz_path, allow_pickle=True)
    # Soporta claves comunes
    for k in ["data", "depths", "arr_0"]:
        if k in npz:
            depths = npz[k]
            break
    else:
        # Si no hay clave conocida, toma el primer arreglo
        first_key = list(npz.keys())[0]
        depths = npz[first_key]
    depths = depths.astype(np.float32)
    return depths  # [N, H, W]

def main():
    # Extra flags (no rompen options.py)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gt_npz", type=str, default=None,
                        help="Ruta a gt_depths.npz; por defecto usa splits/<eval_split>/gt_depths.npz")
    extra, _ = parser.parse_known_args()

    options = MonodepthOptions()
    opt = options.parse()

    device = torch.device("cuda" if torch.cuda.is_available() and not opt.no_cuda else "cpu")

    # Archivos del split
    split_dir = os.path.join(os.path.dirname(__file__), "splits", opt.eval_split)
    list_name = "test_files.txt" if os.path.isfile(os.path.join(split_dir, "test_files.txt")) else "val_files.txt"
    filenames = readlines(os.path.join(split_dir, list_name))
    N = len(filenames)

    # GT depths
    gt_npz_path = extra.gt_npz or os.path.join(split_dir, "gt_depths.npz")
    assert os.path.isfile(gt_npz_path), f"No existe GT NPZ en: {gt_npz_path}"
    gt_depths = load_gt_npz(gt_npz_path)
    assert gt_depths.shape[0] >= N, f"gt_depths ({gt_depths.shape[0]}) < num samples ({N})"

    # Predicciones de disparidad
    if opt.ext_disp_to_eval:
        pred_disps = np.load(opt.ext_disp_to_eval, allow_pickle=True)
        if pred_disps.ndim == 4:
            pred_disps = pred_disps.squeeze(-1)
        print(f"-> Loaded disparities: {opt.ext_disp_to_eval} | shape={pred_disps.shape}")
        feed_h, feed_w = pred_disps.shape[1], pred_disps.shape[2]
    else:
        assert opt.load_weights_folder, "Especifica --ext_disp_to_eval o --load_weights_folder"
        print("-> Loading model from:", opt.load_weights_folder)
        enc, dec, feed_h, feed_w = load_model(opt, device)

        # Dataset para imágenes
        from datasets import SCAREDDataset
        ds = SCAREDDataset(opt.data_path, filenames, feed_h, feed_w, [0], 4, is_train=False)
        preds = []
        with torch.no_grad():
            for i in range(N):
                sample = ds[i]
                inp = sample[("color", 0, 0)].unsqueeze(0).to(device)
                out = dec(enc(inp))
                disp = out[("disp", 0)]
                disp = F.interpolate(disp, (feed_h, feed_w), mode="bilinear", align_corners=False)
                preds.append(disp.squeeze().cpu().numpy())
        pred_disps = np.stack(preds, axis=0)
        print(f"-> Computed disparities for {N} images")

    # Sanity & alineación
    M = min(N, pred_disps.shape[0], gt_depths.shape[0])
    if M < N:
        print(f"[warn] Ajustando a {M} muestras por tamaño de disps/GT")

    accum = np.zeros(7, dtype=np.float64)
    evaluated = 0

    for i in range(M):
        disp = pred_disps[i]
        # Pasamos disp -> depth
        disp_t = torch.from_numpy(disp).unsqueeze(0).unsqueeze(0)
        _, depth_pred_t = disp_to_depth(disp_t, opt.min_depth, opt.max_depth)
        depth_pred = depth_pred_t.squeeze().numpy()

        depth_gt = gt_depths[i]

        # Si hay mismatch de tamaño, redimensiona la predicción a la forma del GT
        if depth_pred.shape != depth_gt.shape:
            depth_pred = cv2.resize(depth_pred, (depth_gt.shape[1], depth_gt.shape[0]), interpolation=cv2.INTER_LINEAR)

        # Median scaling (si aplica)
        valid = depth_gt > 0
        if np.any(valid) and (not opt.disable_median_scaling) and (not opt.eval_stereo):
            scale = np.median(depth_gt[valid]) / (np.median(depth_pred[valid]) + 1e-6)
            depth_pred = depth_pred * scale

        # Clamps por seguridad
        depth_pred = np.clip(depth_pred, opt.min_depth, opt.max_depth)

        metrics = compute_depth_errors(depth_gt, depth_pred)
        if metrics is None:
            continue

        accum += np.array(metrics, dtype=np.float64)
        evaluated += 1

    assert evaluated > 0, "No se evaluó ningún ejemplo válido."
    mean = accum / evaluated

    print("\n-> Depth evaluation on '{}' ({} samples)".format(opt.eval_split, evaluated))
    print("   abs_rel:  {:.4f}".format(mean[0]))
    print("   sq_rel:   {:.4f}".format(mean[1]))
    print("   rmse:     {:.3f}".format(mean[2]))
    print("   rmse_log: {:.3f}".format(mean[3]))
    print("   a1:       {:.3f}".format(mean[4]))
    print("   a2:       {:.3f}".format(mean[5]))
    print("   a3:       {:.3f}".format(mean[6]))

if __name__ == "__main__":
    main()
