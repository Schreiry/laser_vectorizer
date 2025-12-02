"""
tests.py
Модуль автоматического тестирования (Unit Tests).
Запускается перед основной работой для проверки целостности системы.
"""
import unittest
import numpy as np
import cv2
from config import VectorizerConfig
from image_processor import ImageProcessor
from vectorizer import VectorConverter

class TestLaserVectorizer(unittest.TestCase):
    
    def setUp(self):
        """Подготовка окружения перед каждым тестом."""
        # 1. Загружаем стандартный конфиг
        self.config = VectorizerConfig()
        
        # 2. КРИТИЧЕСКАЯ АДАПТАЦИЯ ДЛЯ ТЕСТОВ:
        # Реальный конфиг настроен на огромные изображения (Upscale x5).
        # Тестовые картинки маленькие (100x100), поэтому мощные фильтры
        # стирают тестовые линии целиком.
        # Мы временно "ослабляем" настройки только для тестов.
        self.config.SCALE_FACTOR = 1        # Отключаем апскейл
        self.config.MIN_SPECKLE_AREA = 5    # Порог мусора делаем крошечным
        self.config.PRUNE_LENGTH = 5        # Стрижку делаем короткой
        
        self.processor = ImageProcessor(self.config)
        self.vectorizer = VectorConverter(self.config)
        
        # Создаем фейковое изображение (черный квадрат с белой линией)
        self.test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.line(self.test_img, (10, 10), (90, 90), (255, 255, 255), 2)
        
    def test_config_validity(self):
        """Проверка валидности конфигурации."""
        # Проверяем, что в базовом конфиге (не self.config, а классе) значения адекватны
        base_cfg = VectorizerConfig()
        self.assertGreater(base_cfg.SCALE_FACTOR, 0, "Scale Factor должен быть > 0")
        self.assertGreater(base_cfg.DOG_K2, base_cfg.DOG_K1, "K2 должен быть больше K1 для DoG")

    def test_image_preprocessing(self):
        """Проверка конвейера обработки изображения."""
        # Проверяем, что возвращается одноканальное изображение
        result = self.processor.preprocess(self.test_img)
        self.assertEqual(len(result.shape), 2, "Результат должен быть Grayscale")
        # Линия должна сохраниться (благодаря ослабленным настройкам в setUp)
        self.assertTrue(result.max() > 0, "Результат не должен быть полностью черным (линия стерлась!)")

    def test_skeletonization(self):
        """Проверка скелетизации."""
        binary = np.zeros((100, 100), dtype=np.uint8)
        cv2.line(binary, (10, 10), (90, 90), 255, 5) # Толстая линия
        
        skel = self.processor.skeletonize(binary)
        
        # Скелет должен быть тоньше оригинала (проверка суммы пикселей)
        self.assertLess(np.sum(skel), np.sum(binary), "Скелет должен быть тоньше исходника")
        # Скелет должен содержать только 0 и 255
        unique = np.unique(skel)
        self.assertTrue(np.all(np.isin(unique, [0, 255])), "Скелет должен быть бинарным")

    def test_pruning(self):
        """Проверка алгоритма стрижки (Pruning)."""
        # Создаем крест с коротким хвостом
        skel = np.zeros((50, 50), dtype=np.uint8)
        # Горизонтальная линия (длинная, должна остаться)
        cv2.line(skel, (10, 25), (40, 25), 255, 1)
        # Короткий "ус" вверх (короткий, должен подрезаться, но не исчезнуть совсем, так как база осталась)
        cv2.line(skel, (25, 25), (25, 20), 255, 1) 
        
        # Запускаем стрижку
        pruned = self.vectorizer._prune_skeleton(skel)
        
        # Проверка: что-то должно остаться
        self.assertGreater(np.sum(pruned), 0, "Скелет не должен исчезнуть полностью")

    def test_vector_output(self):
        """Проверка генерации путей."""
        skel = np.zeros((100, 100), dtype=np.uint8)
        cv2.line(skel, (10, 10), (90, 90), 255, 1)
        
        paths = self.vectorizer._find_paths(skel)
        self.assertTrue(len(paths) > 0, "Должен быть найден хотя бы один путь")
        self.assertEqual(paths[0].shape[1], 2, "Путь должен состоять из координат (x, y)")

if __name__ == '__main__':
    unittest.main()