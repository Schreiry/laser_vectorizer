"""
vectorizer.py
Модуль векторизации.
"""
import numpy as np
import cv2
import svgwrite
from typing import List, Tuple
from config import VectorizerConfig

class VectorConverter:
    def __init__(self, config: VectorizerConfig):
        self.cfg = config

    def _prune_skeleton(self, skeleton: np.ndarray) -> np.ndarray:
        """
        Удаляет короткие паразитные ветки (spurs/hairs) скелета.
        Использует морфологическую hit-or-miss трансформацию для поиска концов.
        """
        pruned = skeleton.copy()
        prune_len = self.cfg.PRUNE_LENGTH * self.cfg.SCALE_FACTOR # Масштабируем длину стрижки
        
        # Ядро для поиска пикселей, у которых всего 1 сосед (концы линий)
        # В свертке: центр(10) + 1 сосед(1) = 11.
        kernel = np.array([[1, 1, 1],
                           [1, 10, 1],
                           [1, 1, 1]], dtype=np.uint8)
        
        # Итеративное удаление. Да, это O(N*K), но на i9 это миллисекунды.
        for _ in range(int(prune_len)):
            binary = (pruned > 0).astype(np.uint8)
            neighbors = cv2.filter2D(binary, -1, kernel)
            
            # Находим концы (значение 11)
            ends_mask = (neighbors == 11)
            
            # НО! Не удаляем, если линия совсем короткая и исчезнет полностью?
            # В данном методе мы просто стрижем все концы. 
            # Для лазера это хорошо - убирает микро-артефакты.
            pruned[ends_mask] = 0
            
        return pruned

    def _smooth_path(self, pts: np.ndarray) -> np.ndarray:
        """Скользящее среднее для удаления дрожи (Jitter removal)."""
        window = self.cfg.SMOOTHING_WINDOW
        if len(pts) < window:
            return pts
            
        kernel = np.ones(window) / window
        
        # Сглаживаем X и Y раздельно
        # mode='same' сохраняет размер массива
        x_smooth = np.convolve(pts[:, 0], kernel, mode='same')
        y_smooth = np.convolve(pts[:, 1], kernel, mode='same')
        
        # Края массива будут искажены сверткой, оставляем оригинальные концы
        pad = window // 2
        x_smooth[:pad] = pts[:pad, 0]
        x_smooth[-pad:] = pts[-pad:, 0]
        y_smooth[:pad] = pts[:pad, 1]
        y_smooth[-pad:] = pts[-pad:, 1]
        
        return np.column_stack((x_smooth, y_smooth))

    def _find_paths(self, skeleton: np.ndarray) -> List[np.ndarray]:
        # 1. Pruning (Стрижка)
        clean_skel = self._prune_skeleton(skeleton)
        
        # 2. Поиск контуров на чистом скелете
        # CHAIN_APPROX_NONE - берем все сырые точки, упростим сами
        contours, _ = cv2.findContours(clean_skel, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        
        paths = []
        for cnt in contours:
            # Пропускаем микро-мусор
            if cv2.arcLength(cnt, False) < 10 * self.cfg.SCALE_FACTOR:
                continue
                
            # 3. Ramer-Douglas-Peucker (Геометрическое упрощение)
            epsilon = self.cfg.SIMPLIFICATION_EPSILON
            approx = cv2.approxPolyDP(cnt, epsilon, closed=False)
            
            pts = approx.reshape(-1, 2).astype(np.float32)
            
            # 4. Сглаживание (Эстетика)
            if len(pts) > self.cfg.SMOOTHING_WINDOW:
                pts = self._smooth_path(pts)
            
            # 5. Downscaling (Возвращаем оригинальный размер)
            if self.cfg.SCALE_FACTOR > 1:
                pts /= self.cfg.SCALE_FACTOR
                
            paths.append(pts)
            
        return paths

    def save_to_svg(self, paths: List[np.ndarray], output_path: str, img_size: Tuple[int, int]):
        # Размер холста корректируем обратно
        h, w = img_size
        h //= self.cfg.SCALE_FACTOR
        w //= self.cfg.SCALE_FACTOR
        
        dwg = svgwrite.Drawing(output_path, size=(w, h), profile='tiny')
        group = dwg.g(stroke=self.cfg.STROKE_COLOR, stroke_width=self.cfg.STROKE_WIDTH, fill='none')
        
        for pts in paths:
            if len(pts) < 2: continue
            
            # SVG Path Syntax: M x y L x y ...
            d_commands = [f"M {pts[0][0]:.3f},{pts[0][1]:.3f}"]
            for p in pts[1:]:
                d_commands.append(f"L {p[0]:.3f},{p[1]:.3f}")
            
            group.add(dwg.path(d=" ".join(d_commands)))
            
        dwg.add(group)
        dwg.save()

    def process_and_save(self, skeleton: np.ndarray, output_path: str):
        paths = self._find_paths(skeleton)
        self.save_to_svg(paths, output_path, skeleton.shape)
        return len(paths)