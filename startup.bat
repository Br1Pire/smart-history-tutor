@echo off
echo 🚀 Iniciando Smart History Tutor...

REM Verificar si el entorno virtual existe
IF NOT EXIST venv (
    echo 📦 Creando entorno virtual...
    python -m venv venv
    IF %ERRORLEVEL% NEQ 0 (
        echo ❌ Error al crear el entorno virtual.
        pause
        exit /b
    )
    echo ✅ Entorno virtual creado.
)

REM Activar entorno virtual
echo ⚙ Activando entorno virtual...
call venv\Scripts\activate

REM Instalar dependencias
echo 📦 Instalando dependencias desde requirements.txt...
pip install -r requirements.txt

REM Ejecutar Streamlit tutor
echo 🚀 Ejecutando Smart History Tutor...
streamlit run src\visual\app.py

pause
