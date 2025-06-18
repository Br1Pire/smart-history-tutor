#!/bin/bash

echo "🚀 Iniciando Smart History Tutor..."

# Crear entorno si no existe
if [ ! -d "venv" ]; then
    echo "📦 Creando entorno virtual..."
    python3 -m venv venv
    echo "✅ Entorno virtual creado."
fi

# Activar entorno
echo "⚙ Activando entorno virtual..."
source venv/bin/activate

# Instalar dependencias
echo "📦 Instalando dependencias desde requirements.txt..."
pip install -r requirements.txt

# Ejecutar el tutor
echo "🚀 Ejecutando Smart History Tutor..."
python -m src.agents.tutor_agent
