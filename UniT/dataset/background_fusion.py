import glob
import os
import random

import cv2
import numpy as np


def _list_images(path):
    if path is None:
        return []
    if os.path.isfile(path):
        return [path]
    files = []
    for ext in ('png', 'jpg', 'jpeg'):
        files += glob.glob(os.path.join(path, f'*.{ext}'))
    return sorted(files)


class BackgroundFusion:
    """Aumentado de datos por imagen de diferencia y fusión con fondos de otros sensores.

    Para una imagen de contacto C del GelSight y su imagen sin contacto G:

        D = clip(C - G + 0.5, 0, 1)                    (imagen de diferencia, 0.5 = 128/255)
        F = clip(D + strength * (S - 0.5), 0, 1)       (fusión aditiva con el fondo S de otro sensor)

    Con probabilidad `prob_diff` se sustituye la imagen por D; dentro de ese caso, con probabilidad
    `prob_background` se le suma un fondo S (si no, se devuelve D sola, que imita sensores de
    imagen neutra). El fondo S se elige del conjunto sintético con probabilidad `prob_synthetic`
    y, si no, del real. A cada fondo se le aplica un volteo/rotación aleatorios; si su orientación
    no coincide con la de la imagen (apaisado/vertical) se rota 90 grados antes de redimensionarlo.
    Por último, con probabilidad `prob_gray` el resultado (D o F) se pasa a escala de grises
    (replicada en 3 canales) para imitar sensores monocromos.

    Recibe y devuelve float32 HxWx3 en [0, 1]. Los fondos y G se leen con cv2.imread (BGR) y se
    pasan a RGB si bgr=False, para coincidir con el orden de canales de las imágenes de entrada.
    El azar usa el módulo `random` de Python (distinto en cada worker del DataLoader).
    """

    def __init__(self,
                 no_contact_path,
                 real_dir=None,
                 synthetic_dir=None,
                 prob_diff=0.5,
                 prob_background=0.7,
                 prob_synthetic=0.5,
                 prob_gray=0.15,
                 strength=0.7,
                 bgr=True):
        assert os.path.isfile(no_contact_path), f'no existe la imagen sin contacto: {no_contact_path}'
        self.prob_diff = prob_diff
        self.prob_background = prob_background
        self.prob_synthetic = prob_synthetic
        self.prob_gray = prob_gray
        self.strength = strength
        self.bgr = bgr
        self._to_gray = cv2.COLOR_BGR2GRAY if bgr else cv2.COLOR_RGB2GRAY
        self._no_contact_raw = self._read(no_contact_path)
        self.real = [self._read(p) for p in _list_images(real_dir)]
        self.synthetic = [self._read(p) for p in _list_images(synthetic_dir)]
        assert prob_background == 0 or self.real or self.synthetic, \
            'prob_background > 0 pero no se ha encontrado ningún fondo'
        self._cache = {}

    def _read(self, path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        assert img is not None, f'no se pudo leer {path}'
        if not self.bgr:
            img = img[..., ::-1]
        return np.ascontiguousarray(img).astype(np.float32) / 255.0

    @staticmethod
    def _resize(img, h, w):
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

    def _no_contact(self, h, w):
        key = (h, w)
        if key not in self._cache:
            self._cache[key] = self._resize(self._no_contact_raw, h, w)
        return self._cache[key]

    def _random_background(self, h, w):
        if self.synthetic and random.random() < self.prob_synthetic:
            pool = self.synthetic
        else:
            pool = self.real or self.synthetic
        bg = random.choice(pool)
        if (bg.shape[0] > bg.shape[1]) != (h > w):  # orientación distinta: girar 90 grados
            bg = np.rot90(bg, random.choice((1, 3)))
        if random.random() < 0.5:
            bg = bg[:, ::-1]
        if random.random() < 0.5:
            bg = bg[::-1]
        return self._resize(np.ascontiguousarray(bg), h, w)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if random.random() >= self.prob_diff:
            return image
        h, w = image.shape[:2]
        out = np.clip(image - self._no_contact(h, w) + 0.5, 0.0, 1.0)
        if random.random() < self.prob_background:
            bg = self._random_background(h, w)
            out = np.clip(out + self.strength * (bg - 0.5), 0.0, 1.0)
        if random.random() < self.prob_gray:
            gray = cv2.cvtColor(out.astype(np.float32), self._to_gray)
            out = np.repeat(gray[..., None], 3, axis=2)
        return out.astype(np.float32)
