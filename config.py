"""
config.py
Централизованное хранилище настроек приложения.
"""
from dataclasses import dataclass

@dataclass
class VectorizerConfig:
    # --- ПАРАМЕТРЫ ОБРАБОТКИ (IMAGE PROCESSING) ---
    
    # Инверсия: True, если исходник - темные линии на светлом фоне (рисунок)
    # Лазеру нужно наоборот: белые линии на черном фоне.
    INVERT_INPUT: bool = True 

    # 1. Подготовка (Контраст и размытие)
    # CLAHE вытягивает бледные линии карандаша
    CLAHE_CLIP: float = 3.0          
    # Bilateral Filter: убирает текстуру бумаги/шерсти, оставляя жесткие края
    BILATERAL_SIGMA: int = 75        
    
    # 2. Адаптивная бинаризация (Block Size должен быть нечетным!)
    # Определяет, насколько большую область смотреть вокруг пикселя.
    ADAPTIVE_BLOCK_SIZE: int = 21    
    ADAPTIVE_C: int = 10             # Чем выше, тем меньше мусора, но линии тоньше

    # 3. Очистка от мусора
    # Удалять любые замкнутые пятна меньше N пикселей
    MIN_COMPONENT_AREA: int = 100    
    
    # --- ПАРАМЕТРЫ ВЕКТОРИЗАЦИИ (SVG) ---
    SIMPLIFICATION_EPSILON: float = 1.2  # Сглаживание линий (1.0 = детально, 2.0 = плавно)
    STROKE_WIDTH: str = "0.1mm"          # "Волосяная" линия для реза
    STROKE_COLOR: str = "red"            # Красный - стандарт для слоя Cut

    OUTPUT_SUFFIX: str = "_cut"