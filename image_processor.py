"""
image_processor.py
Модуль компьютерного зрения High-End класса.
Логика: CLAHE Contrast -> Adaptive Threshold -> Morphological Cleaning -> Skeleton.
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
        Превращает цветную картинку в идеальную бинарную маску для скелетизации.
        """
        # 1. Ч/Б + Умное размытие (убирает шум бумаги/текстуры)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Bilateral Filter очень важен: он замыливает "шерсть", но оставляет контур
        blurred = cv2.bilateralFilter(gray, 9, self.cfg.BILATERAL_SIGMA, self.cfg.BILATERAL_SIGMA)

        # 2. Усиление локального контраста (CLAHE)
        # Это делает слабые линии карандаша четкими и черными.
        clahe = cv2.createCLAHE(clipLimit=self.cfg.CLAHE_CLIP, tileGridSize=(8, 8))
        contrast = clahe.apply(blurred)

        # 3. Адаптивная бинаризация (Gaussian)
        # Лучший метод для рисунков. Ищет линии относительно фона, а не абсолютного цвета.
        binary = cv2.adaptiveThreshold(
            contrast, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            self.cfg.ADAPTIVE_BLOCK_SIZE, 
            self.cfg.ADAPTIVE_C
        )

        # 4. Инверсия (если нужно получить белые линии на черном)
        # Если INVERT_INPUT=True (черный рисунок на белом), то мы инвертируем,
        # чтобы линии стали белыми (255) для алгоритма скелетизации.
        if self.cfg.INVERT_INPUT:
            binary = cv2.bitwise_not(binary)
        else:
            # Если фон уже черный, а линии белые - инвертировать обратно
            binary = cv2.bitwise_not(binary)
            binary = cv2.bitwise_not(binary) # костыль для логики, оставляем как есть

        # 5. Чистка "соли и перца" (мелких точек)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        # Morph Open: убирает белый шум (точки)
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        # Morph Close: закрывает дырки внутри линий
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)

        # 6. Удаление мусора по площади (очень важно для чистоты вектора)
        cleaned = self._remove_small_components(closed, self.cfg.MIN_COMPONENT_AREA)
        
        # 7. Финальное утолщение (Dilation)
        # Чуть утолщаем линии перед скелетизацией, чтобы они гарантированно сцепились
        dilated = cv2.dilate(cleaned, kernel, iterations=1)

        return dilated

    def _remove_small_components(self, binary_img: np.ndarray, min_area: int) -> np.ndarray:
        """Удаляет все изолированные объекты меньше min_area."""
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_img, connectivity=8)
        output = np.zeros_like(binary_img)
        
        for i in range(1, num_labels): # 0 - это фон
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                output[labels == i] = 255
        return output

    def skeletonize(self, binary_img: np.ndarray) -> np.ndarray:
        try:
            # Алгоритм Чжан-Суеня (Zhang-Suen) - идеален для получения центров линий
            return cv2.ximgproc.thinning(binary_img, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
        except AttributeError:
            raise ImportError("Нет модуля cv2.ximgproc. Выполните: pip install opencv-contrib-python-headless")