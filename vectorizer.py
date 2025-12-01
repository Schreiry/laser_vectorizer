"""
vectorizer.py
Преобразование растрового скелета в векторные пути (SVG).
Реализует логику графов для single-stroke трассировки.
"""
import numpy as np
import cv2
import svgwrite
from typing import List, Tuple
from config import VectorizerConfig

class VectorConverter:
    """
    Отвечает за анализ топологии скелета и генерацию SVG.
    """
    def __init__(self, config: VectorizerConfig):
        self.cfg = config

    def _find_paths(self, skeleton: np.ndarray) -> List[np.ndarray]:
        """
        Преобразует скелет в список замкнутых контуров (полилиний).
        
        Улучшенный алгоритм:
        1. Находим все контуры скелета (замкнутые области).
        2. Для каждого контура вычисляем граничные точки.
        3. Упрощаем контуры для векторизации.
        4. Отбрасываем мелкие артефакты.
        """
        h, w = skeleton.shape
        
        # Нормализуем скелет (0 и 255)
        skel_norm = skeleton.copy()
        skel_norm[skel_norm > 0] = 255
        
        # 1. Находим ВСЕ контуры скелета (включая замкнутые)
        # RETR_TREE сохраняет иерархию (внешние и внутренние контуры)
        contours, hierarchy = cv2.findContours(skel_norm, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        
        if not contours:
            return []
        
        paths = []
        
        for idx, cnt in enumerate(contours):
            # Пропускаем очень маленькие контуры (шум)
            area = cv2.contourArea(cnt)
            if area < 50:  # Мин. площадь
                continue
            
            # Периметр для отсечения микроконтуров
            perimeter = cv2.arcLength(cnt, True)
            if perimeter < 20:  # Мин. периметр
                continue
            
            # 2. Упрощение контура (Рамер-Дугласс-Пойкер)
            # epsilon зависит от размера контура
            epsilon = self.cfg.SIMPLIFICATION_EPSILON * max(1, perimeter / 100)
            approx = cv2.approxPolyDP(cnt, epsilon, closed=True)
            
            # Преобразуем в плоский массив точек (N, 2)
            pts = approx.reshape(-1, 2).astype(np.float32)
            
            # 3. Фильтрация: контур должен иметь >= 3 точек
            if len(pts) >= 3:
                # Сглаживание точек для лучшей векторизации
                smoothed_pts = self._smooth_path(pts)
                paths.append(smoothed_pts)
        
        # Сортируем по площади контура (большие контуры важнее)
        paths.sort(key=lambda p: cv2.contourArea(p.reshape(-1, 1, 2)), reverse=True)
        
        return paths
    
    def _smooth_path(self, pts: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """
        Применяет легкое сглаживание к точкам контура.
        Это убирает зубчатость, которую создает скелетизация.
        """
        if len(pts) < kernel_size:
            return pts
        
        smoothed = np.zeros_like(pts)
        kernel_size = min(kernel_size, len(pts) // 2) if len(pts) > 2 else 1
        
        if kernel_size == 1:
            return pts
        
        for i in range(len(pts)):
            start = max(0, i - kernel_size // 2)
            end = min(len(pts), i + kernel_size // 2 + 1)
            smoothed[i] = np.mean(pts[start:end], axis=0)
        
        return smoothed

    def save_to_svg(self, paths: List[np.ndarray], output_path: str, img_size: Tuple[int, int]):
        """Генерирует SVG файл из списка полносвязных контуров."""
        height, width = img_size
        dwg = svgwrite.Drawing(output_path, size=(width, height), profile='tiny')
        
        # Группа для контуров
        group = dwg.g(stroke=self.cfg.STROKE_COLOR, stroke_width=self.cfg.STROKE_WIDTH, fill='none')
        
        for pts in paths:
            if len(pts) < 2:
                continue
            
            # Преобразуем в список кортежей (целые координаты)
            points_list = [(float(p[0]), float(p[1])) for p in pts]
            
            # Строим path команду
            # M = moveto, L = lineto, Z = close path
            path_data = []
            if len(points_list) > 0:
                path_data.append(f"M {points_list[0][0]:.1f},{points_list[0][1]:.1f}")
                for p in points_list[1:]:
                    path_data.append(f"L {p[0]:.1f},{p[1]:.1f}")
                # Z закрывает контур (связывает последнюю точку с первой)
                path_data.append("Z")
                
                group.add(dwg.path(d=" ".join(path_data)))
            
        dwg.add(group)
        dwg.save()

    def process_and_save(self, skeleton: np.ndarray, output_path: str):
        paths = self._find_paths(skeleton)
        self.save_to_svg(paths, output_path, skeleton.shape)
        return len(paths)