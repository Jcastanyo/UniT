"""Estima la imagen sin contacto del GelSight a partir del zarr de entrenamiento.

Mediana de una muestra de imágenes -> se puntúa cuánto se aleja cada imagen de esa mediana (tras quitar el
brillo global por canal) -> se quedan las que casi no se alejan (sin contacto) -> mediana de ellas.
La salida es un PNG BGR uint8 con el tamaño nativo del zarr.

Uso: python scripts/compute_no_contact_background.py --zarr <dataset_rep_learning.zarr> --out gelsight_no_contact.png
"""
import argparse
import random

import cv2
import numpy as np
import zarr


def norm(x):
    return x - x.mean(axis=(0, 1), keepdims=True)


def scores(imgs, ref):
    return np.array([cv2.GaussianBlur(np.abs(norm(i) - norm(ref)).mean(2), (0, 0), 3).max() for i in imgs])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--zarr', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=6000, help='imágenes a muestrear')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    z = zarr.open(a.zarr + '/data/tactile_image', mode='r')
    total = z.shape[0]
    step = max(total // a.n, 1)
    imgs = []
    chunk = z.chunks[0]
    for s in range(0, total, chunk):  # por bloques de frames para no descomprimir todo a la vez
        imgs += list(np.asarray(z[s:min(s + chunk, total):step]).astype(np.float32))
    random.Random(a.seed).shuffle(imgs)
    print('imágenes muestreadas:', len(imgs), 'forma:', imgs[0].shape)

    ref = np.median(np.stack(imgs[:1500]), 0)
    s = scores(imgs, ref)
    p5 = np.percentile(s, 5)
    nocon = [i for i, v in zip(imgs, s) if v < 1.5 * p5]
    print('candidatas sin contacto:', len(nocon))
    assert len(nocon) >= 30, 'muy pocas imágenes sin contacto para estimar el fondo'
    bg = np.median(np.stack(nocon[:1500]), 0)
    s2 = scores(imgs, bg)
    print('fracción con contacto (> 4*p5):', float((s2 > 4 * np.percentile(s2, 5)).mean()))
    cv2.imwrite(a.out, np.clip(bg, 0, 255).astype(np.uint8))
    print('guardado', a.out)


if __name__ == '__main__':
    main()
