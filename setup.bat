@echo off
REM SignBridge AI setup (Windows, Python 3.12).
cd /d "%~dp0"
py -3.12 --version >nul 2>&1
if errorlevel 1 (
  echo Install Python 3.12 from https://www.python.org/downloads/ and re-run setup.bat
  exit /b 1
)
if not exist .venv py -3.12 -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip -q
pip install -r requirements.txt
if errorlevel 1 exit /b 1
python scripts\check_env.py
if errorlevel 1 exit /b 1
if exist data\raw\sign_language_mnist\sign_mnist_train.csv (
  if not exist models\trained\image_cnn.keras python scripts\train_image.py
)
if exist data\raw\sign_mnist_train.csv (
  if not exist models\trained\image_cnn.keras python scripts\train_image.py
)
echo Done. Start the app with run_app.bat
exit /b 0
