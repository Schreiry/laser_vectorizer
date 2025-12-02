"""
image_processor.py
Модуль обработки изображений промышленного уровня.
Pipeline: Upscale -> Gamma -> DoG -> Threshold -> Despeckle -> Skeleton.
"""
import cv2
import numpy as np
from config import VectorizerConfig

class ImageProcessor:
    def __init__(self, config: VectorizerConfig):
        self.cfg = config 

    def load_image(self, path: str) -> np.ndarray:
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Файл не найден: {path}")
        return img

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """
        Превращает "грязный" скан/фото в идеальную топологическую карту линий.
        """
        # 1. UPSCALING (Суперсэмплинг)
        if self.cfg.SCALE_FACTOR > 1:
            h, w = img.shape[:2]
            img = cv2.resize(img, (w * self.cfg.SCALE_FACTOR, h * self.cfg.SCALE_FACTOR), 
                             interpolation=cv2.INTER_CUBIC)

        # 2. GRAYSCALE
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        # 3. GAMMA CORRECTION (Нормализация света)
        # Позволяет вытянуть детали из теней
        invGamma = 1.0 / self.cfg.GAMMA
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        gray = cv2.LUT(gray, table)

        # 4. DIFFERENCE OF GAUSSIANS (DoG)
        # Магический метод для extract line drawings.
        # Вычитаем сильно размытое из слабо размытого -> остаются только края нужной частоты.
        g1 = cv2.GaussianBlur(gray, (0, 0), sigmaX=self.cfg.DOG_K1)
        g2 = cv2.GaussianBlur(gray, (0, 0), sigmaX=self.cfg.DOG_K2)
        dog = cv2.subtract(g1, g2)

        # 5. BINARIZATION
        # DoG возвращает темные линии на сером фоне. Инвертируем.
        dog = cv2.bitwise_not(dog)
        # Применяем жесткий порог для выделения линий
        _, binary = cv2.threshold(dog, self.cfg.BINARY_THRESHOLD, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        # Инвертируем обратно: нам нужны БЕЛЫЕ линии на ЧЕРНОМ фоне для скелетизации
        binary = cv2.bitwise_not(binary)

        # 6. MORPHOLOGICAL CLEANING (Despeckle)
        # Удаляем шум
        nb_components, output, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        sizes = stats[1:, -1]
        nb_components = nb_components - 1
        
        clean_img = np.zeros((output.shape), dtype=np.uint8)
        min_size = self.cfg.MIN_SPECKLE_AREA * (self.cfg.SCALE_FACTOR ** 2) / 2 # Корректируем под масштаб
        
        for i in range(0, nb_components):
            if sizes[i] >= min_size:
                clean_img[output == i + 1] = 255

        # 7. CLOSING (Залечивание разрывов)
        # Небольшое закрытие дыр внутри линий
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        closed = cv2.morphologyEx(clean_img, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        return closed

    def skeletonize(self, binary_img: np.ndarray) -> np.ndarray:
        """Получение однопиксельного скелета (Centerline extraction)."""
        try:
            # Zhang-Suen - золотой стандарт топологии
            skel = cv2.ximgproc.thinning(binary_img, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
            return skel
        except AttributeError:
            raise ImportError("Critical: cv2.ximgproc not found. Install opencv-contrib-python-headless.")