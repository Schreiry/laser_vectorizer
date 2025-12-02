"""
config.py
Централизованное хранилище настроек приложения.
High-End конфигурация для обработки карандашных рисунков и скетчей.
"""
from dataclasses import dataclass

@dataclass
class VectorizerConfig:
    # --- 1. SUPER-SAMPLING (ПОВЫШЕНИЕ РАЗРЕШЕНИЯ) ---
    # Увеличиваем изображение перед обработкой для субпиксельной точности.
    # 2 = 200% (рекомендуется для RTX/i9), 1 = оригинал.
    SCALE_FACTOR: int = 5
    
    # --- 2. PRE-PROCESSING (DIFFERENCE OF GAUSSIANS) ---
    # Гамма-коррекция: < 1.0 делает тени светлее, > 1.0 темнее.
    # Помогает выровнять освещение бумаги.
    GAMMA: float = 0.70
    
    # DoG параметры: имитация band-pass фильтра.
    # K1 - размытие шума, K2 - ширина штриха.
    DOG_K1: int = 3   
    DOG_K2: int = 9   
    
    # --- 3. CLEANING & SKELETONIZATION ---
    # Порог бинаризации после DoG (0-255).
    # Выше значение = остаются только самые жирные линии.
    BINARY_THRESHOLD: int = 240 # Инвертированный порог (т.к. работаем с DoG)
    
    # Удалять пятна площадью меньше N пикселей (в масштабе Upscale)
    MIN_SPECKLE_AREA: int = 150
    
    # --- 4. TOPOLOGICAL PRUNING (СТРИЖКА ГРАФА) ---
    # Критически важно для чистого вектора!
    # Удаляет тупиковые ветки скелета короче N пикселей.
    PRUNE_LENGTH: int = 18

    # --- 5. VECTORIZATION (SVG) ---
    # Точность Ramer-Douglas-Peucker (меньше = больше точек/деталей).
    SIMPLIFICATION_EPSILON: float = 0.8
    # Окно сглаживания координат (Moving Average) для удаления "дрожи" руки.
    SMOOTHING_WINDOW: int = 5
    
    # Параметры экспорта
    STROKE_WIDTH: str = "0.1mm"
    STROKE_COLOR: str = "red" # Цвет реза
    OUTPUT_SUFFIX: str = "_laser_ready"