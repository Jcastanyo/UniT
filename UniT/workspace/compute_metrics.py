"""
Compute SSIM and FID metrics between original and reconstructed images.

- Original images  : /UniT_shared/data/julio_wth_markers/train/splitted_images/test/
- Reconstructed    : /UniT_shared/data/julio_wth_markers/train/img_results/vqgan_test_big_model/

SSIM  – computed per-pair (original resized to 160x128 vs reconstructed).
FID   – computed over both full image sets using torchmetrics / torch-fidelity.

Usage (inside Docker):
    python /UniT/workspace/compute_metrics.py
"""

import os
import sys
import glob
import argparse
import random
import torch
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


# ── Reproducibility ──
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ORIG_DIR  = "/UniT_shared/data/julio_wth_markers/train/splitted_images/test/"
REC_DIR   = "/UniT_shared/data/julio_wth_markers/train/img_results/test_color_jitter/vqgan_test_small_model/"
REC_SIZE  = (160, 128)   # (width, height) – same as what the workspace saves

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sorted_images(folder, extensions=("*.png", "*.jpg", "*.jpeg")):
    """Return images sorted by the numeric suffix in the filename."""
    paths = []
    for ext in extensions:
        paths.extend(glob.glob(os.path.join(folder, ext)))

    def _key(p):
        stem = os.path.splitext(os.path.basename(p))[0]
        # try to extract trailing integer, fall back to stem string
        parts = stem.split("_")
        try:
            return int(parts[-1])
        except ValueError:
            return stem

    paths.sort(key=_key)
    return paths


# ---------------------------------------------------------------------------
# SSIM
# ---------------------------------------------------------------------------

def compute_ssim(orig_paths, rec_paths):
    """Compute mean SSIM over matched pairs of images."""
    assert len(orig_paths) == len(rec_paths), (
        f"Number of original images ({len(orig_paths)}) != "
        f"reconstructed images ({len(rec_paths)})"
    )

    scores = []
    for idx, (op, rp) in enumerate(zip(orig_paths, rec_paths)):
        orig = cv2.imread(op)
        rec  = cv2.imread(rp)

        if orig is None:
            print(f"[WARN] Cannot read original image: {op}")
            continue
        if rec is None:
            print(f"[WARN] Cannot read reconstructed image: {rp}")
            continue

        # Resize original to the same size as the reconstruction
        orig_resized = cv2.resize(orig, REC_SIZE)   # (W, H)

        # Convert BGR -> RGB for consistency (doesn't affect SSIM value)
        orig_rgb = cv2.cvtColor(orig_resized, cv2.COLOR_BGR2RGB)
        rec_rgb  = cv2.cvtColor(rec,          cv2.COLOR_BGR2RGB)

        score = ssim(orig_rgb, rec_rgb, channel_axis=2, data_range=255)
        scores.append(score)

        if (idx + 1) % 100 == 0:
            print(f"  [{idx+1}/{len(orig_paths)}] running SSIM mean = {np.mean(scores):.4f}")

    mean_ssim = float(np.mean(scores))
    std_ssim  = float(np.std(scores))
    return mean_ssim, std_ssim, scores


# ---------------------------------------------------------------------------
# FID  (using torch-fidelity if available, otherwise torchmetrics)
# ---------------------------------------------------------------------------

def compute_fid_torch_fidelity(orig_dir, rec_dir):
    """Compute FID with the torch-fidelity library."""
    import torch_fidelity
    metrics = torch_fidelity.calculate_metrics(
        input1=orig_dir,
        input2=rec_dir,
        cuda=True,
        fid=True,
        isc=False,
        kid=False,
        verbose=True,
    )
    return metrics["frechet_inception_distance"]


def compute_fid_torchmetrics(orig_dir, rec_dir):
    """Compute FID with torchmetrics (FrechetInceptionDistance)."""
    import torch
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchvision import transforms
    from torch.utils.data import DataLoader, Dataset

    class ImageFolder(Dataset):
        def __init__(self, folder, transform=None):
            exts = ("*.png", "*.jpg", "*.jpeg")
            self.paths = []
            for ext in exts:
                self.paths.extend(glob.glob(os.path.join(folder, ext)))
            self.paths.sort()
            self.transform = transform

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):
            img = cv2.cvtColor(cv2.imread(self.paths[idx]), cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, REC_SIZE)
            t = torch.from_numpy(img).permute(2, 0, 1)   # C H W, uint8
            if self.transform:
                t = self.transform(t)
            return t

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    fid.reset()

    for folder, real_flag in [(orig_dir, True), (rec_dir, False)]:
        ds     = ImageFolder(folder)
        loader = DataLoader(ds, batch_size=64, num_workers=4)
        for batch in loader:
            batch = batch.to(device)
            fid.update(batch, real=real_flag)

    return float(fid.compute().item())


def compute_fid(orig_dir, rec_dir):
    """Try torch-fidelity first, fall back to torchmetrics."""
    # Resize all originals to a temp folder so both sets are the same resolution
    import tempfile
    import shutil

    tmp_dir = tempfile.mkdtemp(prefix="orig_resized_")
    try:
        orig_paths = sorted_images(orig_dir)
        print(f"  Resizing {len(orig_paths)} original images to {REC_SIZE} for FID …")
        for p in orig_paths:
            img = cv2.imread(p)
            img = cv2.resize(img, REC_SIZE)
            cv2.imwrite(os.path.join(tmp_dir, os.path.basename(p)), img)

        try:
            print("  Trying torch-fidelity …")
            fid_score = compute_fid_torch_fidelity(tmp_dir, rec_dir)
        except ImportError:
            print("  torch-fidelity not found, falling back to torchmetrics …")
            fid_score = compute_fid_torchmetrics(tmp_dir, rec_dir)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return fid_score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compute SSIM and FID metrics")
    parser.add_argument("--orig",  default=ORIG_DIR,  help="Original image directory")
    parser.add_argument("--rec",   default=REC_DIR,   help="Reconstructed image directory")
    parser.add_argument("--no-fid",  action="store_true", help="Skip FID computation")
    parser.add_argument("--no-ssim", action="store_true", help="Skip SSIM computation")
    args = parser.parse_args()

    print("=" * 60)
    print("  Image Quality Metrics")
    print("=" * 60)
    print(f"  Original dir    : {args.orig}")
    print(f"  Reconstructed   : {args.rec}")

    orig_paths = sorted_images(args.orig)
    rec_paths  = sorted_images(args.rec)
    print(f"  Original images : {len(orig_paths)}")
    print(f"  Reconst. images : {len(rec_paths)}")
    print("=" * 60)

    # ---- SSIM ----------------------------------------------------------------
    if not args.no_ssim:
        print("\n[SSIM] Computing per-pair SSIM …")
        mean_ssim, std_ssim, scores = compute_ssim(orig_paths, rec_paths)
        print(f"\n  ✓ SSIM  mean = {mean_ssim:.4f}  ±  {std_ssim:.4f}")
        print(f"         min  = {min(scores):.4f}   max = {max(scores):.4f}")
    else:
        print("\n[SSIM] Skipped.")

    # ---- FID -----------------------------------------------------------------
    if not args.no_fid:
        print("\n[FID]  Computing Fréchet Inception Distance …")
        try:
            fid_score = compute_fid(args.orig, args.rec)
            print(f"\n  ✓ FID  = {fid_score:.4f}")
        except Exception as e:
            print(f"\n  ✗ FID computation failed: {e}")
            print("    Install either:  pip install torch-fidelity")
            print("                 or  pip install torchmetrics[image]")
    else:
        print("\n[FID]  Skipped.")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
