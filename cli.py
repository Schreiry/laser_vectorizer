"""
cli.py
Обработка аргументов командной строки и UI.
"""
import time
import argparse
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.table import Table
from rich.panel import Panel
from rich import box

from image_processor import ImageProcessor
from vectorizer import VectorConverter
from config import VectorizerConfig

console = Console()

class ConsoleApp:
    def __init__(self):
        # 1. Загружаем конфиг
        self.config = VectorizerConfig()
        
        # 2. ИСПРАВЛЕНИЕ: Передаем весь конфиг. Ошибка USE_OTSU была тут.
        self.img_proc = ImageProcessor(config=self.config)
        
        # 3. Инициализируем векторизатор
        self.vectorizer = VectorConverter(self.config)

    def parse_args(self):
        parser = argparse.ArgumentParser(description="Laser Vectorizer")
        parser.add_argument("input_dir", type=str, help="Папка с картинками")
        parser.add_argument("--out", type=str, default="output", help="Папка для сохранения")
        return parser.parse_args()

    def run(self):
        args = self.parse_args()
        input_path = Path(args.input_dir)
        output_path = Path(args.out)
        output_path.mkdir(exist_ok=True)

        # Поддержка разных форматов
        extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
        files = []
        for ext in extensions:
            files.extend(input_path.glob(ext.lower()))
            files.extend(input_path.glob(ext.upper()))
        
        # Удаляем дубликаты
        files = sorted(list(set(files)))

        if not files:
            console.print("[bold red]Ошибка:[/bold red] Файлы не найдены.")
            return

        console.print(Panel.fit(
            f"Файлов: [bold cyan]{len(files)}[/bold cyan]\n"
            f"Метод: [bold green]Adaptive Contrast -> Skeleton[/bold green]", 
            title="🚀 Laser Vectorizer Pro", border_style="blue"
        ))

        results = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.percentage:>3.0f}%"),
        ) as progress:
            task = progress.add_task("[cyan]Обработка...", total=len(files))

            for file in files:
                start_time = time.time()
                status = "OK"
                nodes_count = 0
                
                try:
                    # 1. Загрузка
                    raw_img = self.img_proc.load_image(str(file))
                    
                    # 2. Препроцессинг (Получаем чистую ч/б маску)
                    clean_img = self.img_proc.preprocess(raw_img)
                    
                    # 3. Скелетизация (Получаем линии в 1 пиксель)
                    skeleton = self.img_proc.skeletonize(clean_img)
                    
                    # 4. Векторизация и сохранение
                    out_file = output_path / (file.stem + self.config.OUTPUT_SUFFIX + ".svg")
                    nodes_count = self.vectorizer.process_and_save(skeleton, str(out_file))
                    
                except Exception as e:
                    status = f"ERROR: {str(e)}"
                    console.print(f"\n[red]Сбой на {file.name}: {e}[/red]")
                
                elapsed = time.time() - start_time
                results.append((file.name, f"{elapsed:.2f}s", str(nodes_count), status))
                progress.advance(task)

        self.print_summary(results)

    def print_summary(self, data):
        table = Table(title="Результаты", box=box.ROUNDED)
        table.add_column("Файл", style="cyan")
        table.add_column("Время", justify="right")
        table.add_column("Вектора", justify="right")
        table.add_column("Статус", justify="center")

        for row in data:
            status_style = "green" if "OK" in row[3] else "red"
            short_status = row[3] if len(row[3]) < 20 else "ERROR"
            table.add_row(row[0], row[1], row[2], f"[{status_style}]{short_status}[/{status_style}]")

        console.print(table)