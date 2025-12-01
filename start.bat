@echo off
chcp 65001 > nul
setlocal

:: --- НАСТРОЙКИ ---
set VENV_NAME=venv_py312
set TARGET_PY=3.12
set INPUT_DIR=input_images
set OUTPUT_DIR=output_vectors

echo ========================================================
echo  🚀 LASER VECTORIZER SYSTEM SETUP (Python %TARGET_PY%)
echo ========================================================

:: 1. Проверка наличия Python 3.12 через Python Launcher (py)
py -%TARGET_PY% --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python %TARGET_PY% не найден! 
    echo Пожалуйста, установите Python 3.12 с python.org и повторите попытку.
    pause
    exit /b
)
echo [OK] Python %TARGET_PY% обнаружен.

:: 2. Создание виртуального окружения, если его нет
if not exist %VENV_NAME% (
    echo [INFO] Создание виртуального окружения (%VENV_NAME%)...
    py -%TARGET_PY% -m venv %VENV_NAME%
) else (
    echo [INFO] Виртуальное окружение уже существует.
)

:: 3. Активация окружения и установка зависимостей
echo [INFO] Установка/Проверка библиотек...
call %VENV_NAME%\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

:: 4. Проверка входных данных
if not exist %INPUT_DIR% (
    echo [INFO] Папка %INPUT_DIR% не найдена. Создаю ее...
    mkdir %INPUT_DIR%
    echo [WARNING] Папка %INPUT_DIR% пуста. Положите туда картинки!
    pause
    exit /b
)

:: 5. Запуск программы
echo.
echo [START] Запуск движка векторизации...
echo ========================================================
python main.py %INPUT_DIR% --out %OUTPUT_DIR%

echo.
if %ERRORLEVEL% NEQ 0 (
    echo [FAILURE] Программа завершилась с ошибкой.
) else (
    echo [SUCCESS] Обработка завершена успешно.
)
pause