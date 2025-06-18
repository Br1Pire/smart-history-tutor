# Smart History Tutor

## ✍ Autor
- Bruno Pire Ricardo C311


## 🎯 Problema
El objetivo de este proyecto es desarrollar un sistema inteligente que actúe como tutor en Historia Universal.  
El tutor responde preguntas utilizando:
- **Vectorización semántica** (SentenceTransformer + FAISS)
- **Generación de texto** (Google Gemini API)
- **Document processing** (preprocesamiento, chunking, vectorización)

El sistema permite consultar información histórica, refinar preguntas y evaluar su rendimiento.

## ⚙ Requerimientos generales
- Python 3.10 o superior
- Al menos 8 GB de RAM (recomendado 16 GB para mayor rendimiento)
- Conexión a internet (para la API de Gemini y descargas de modelos)
- Sistema operativo: Windows 

## 🌐 APIs utilizadas
- **Google Gemini API**: generación de respuestas, validación y refinado de preguntas

## 🚀 Forma de uso

### 1️⃣ Descarga / clona el proyecto
```bash
git clone https://github.com/Br1Pire/smart-history-tutor
```
### 2️⃣ Crea el entorno virtual (solo la primera vez)
```bash
python -m venv venv
```
### 3️⃣ Activa el entorno virtual
En PowerShell:
```bash
venv\Scripts\Activate.ps1
```
En CMD:
```bash
venv\Scripts\activate.bat
```
### 4️⃣ Instala las dependencias
```bash
pip install -r requirements.txt
```
### 5️⃣ Ejecuta el proyecto
```bash
python -m src.agents.tutor_agent
```
### 💡 Descarga previa del modelo (IMPORTANTE)
```bash
python src\tools\download_mpnet.py
```