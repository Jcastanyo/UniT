import random

import cv2
import numpy as np


class ColorAugmentation:
    """Aumentado de datos de color para imágenes táctiles.

    Recibe imágenes float32 HxWx3 en RGB (o BGR con bgr=True) con valores en [0, 1] y devuelve una
    imagen del mismo tipo y rango. Las magnitudes siguen la semántica de
    torchvision.ColorJitter: el factor de brillo, contraste y saturación se
    muestrea uniformemente en [1 - m, 1 + m] y el desplazamiento de tono en
    [-h, h] (fracción del círculo de tono, h <= 0.5).

    El azar usa el módulo `random` de Python (que PyTorch reinicia con una
    semilla distinta en cada worker del DataLoader), no `np.random` global.
    """

    def __init__(self,
                 brightness=0.3,
                 contrast=0.3,
                 saturation=0.3,
                 hue=0.1,
                 noise_std=0.0,
                 prob=1.0,
                 bgr=False):
        assert 0.0 <= hue <= 0.5, 'hue val outside range [0.0-0.5]'
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.noise_std = noise_std
        self.prob = prob
        # True si los canales están en orden BGR (imágenes leídas con cv2.imread)
        self._to_gray = cv2.COLOR_BGR2GRAY if bgr else cv2.COLOR_RGB2GRAY
        self._to_hsv = cv2.COLOR_BGR2HSV if bgr else cv2.COLOR_RGB2HSV
        self._from_hsv = cv2.COLOR_HSV2BGR if bgr else cv2.COLOR_HSV2RGB

    @staticmethod
    def _factor(magnitude):
        return random.uniform(1.0 - magnitude, 1.0 + magnitude)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if random.random() >= self.prob:
            return image
        img = image.astype(np.float32, copy=True)

        if self.brightness > 0:
            img = img * self._factor(self.brightness)
            img = np.clip(img, 0.0, 1.0)

        if self.contrast > 0:
            mean = cv2.cvtColor(img, self._to_gray).mean()
            img = (img - mean) * self._factor(self.contrast) + mean
            img = np.clip(img, 0.0, 1.0)

        if self.saturation > 0:
            gray = cv2.cvtColor(img, self._to_gray)[..., None]
            img = gray + (img - gray) * self._factor(self.saturation)
            img = np.clip(img, 0.0, 1.0)

        if self.hue > 0:
            hsv = cv2.cvtColor(img, self._to_hsv)  # H en [0, 360]
            shift = random.uniform(-self.hue, self.hue) * 360.0
            hsv[..., 0] = (hsv[..., 0] + shift) % 360.0
            img = np.clip(cv2.cvtColor(hsv, self._from_hsv), 0.0, 1.0)

        if self.noise_std > 0:
            rng = np.random.default_rng(random.getrandbits(32))
            img = img + rng.normal(0.0, self.noise_std, img.shape).astype(np.float32)
            img = np.clip(img, 0.0, 1.0)

        return img.astype(np.float32)
