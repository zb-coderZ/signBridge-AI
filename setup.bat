@echo off
REM SignBridge AI setup (Windows). Tries Python 3.14, then 3.12 / 3.11 / 3.13 via the "py" launcher.
cd /d "%~dp0"
for %%V in (3.14 3.12 3.11 3.13) do (
  py -%%V --version >nul 2>&1
  if not errorlevel 1 (
    echo === Trying Python %%V ===
    if exist venv rmdir /s /q venv
    py -%%V -m venv venv
    if not errorlevel 1 (
      call venv\Scripts\activate.bat
      python -m pip install --upgrade pip -q
      pip install -r requirements.txt
      if not errorlevel 1 (
        python scripts\check_env.py
        if not errorlevel 1 (
          echo Python %%V > venv\PYTHON_USED.txt
          echo === Environment OK with Python %%V ===
          if not exist models\demo\static_ASL_meta.json python scripts\build_demo.py
          echo Done. Start the app with run_app.bat
          exit /b 0
        )
      )
      echo !!! Python %%V failed - trying the next one...
      call venv\Scripts\deactivate.bat
    )
  )
)
echo No suitable Python found. Install Python 3.12 from https://www.python.org/downloads/ and re-run setup.bat
exit /b 1
