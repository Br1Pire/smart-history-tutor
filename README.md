# Smart History Tutor

## ✍ Autor
- Bruno Jesús Pire Ricardo (C311)

## 🎯 Descripción del proyecto
*Smart History Tutor* es un sistema inteligente que actúa como tutor de Historia Universal, integrando:

### 🔹 Tutor RAG multiagente
- Recuperación semántica (SentenceTransformer + FAISS)
- Generación de texto (Google Gemini API)
- Procesamiento de documentos: crawler, preprocesador, vectorizador, retriever y generador

### 🔹 Simulación educativa con metaheurística
- Optimización de planes de clases mediante algoritmos genéticos
- Modelado de estudiantes con:
  - Tasa base de aprendizaje
  - Probabilidades de omisión y olvido
  - Motivación, estado y ambiente (modulados con sistemas difusos)
- Evaluación de planes según el rendimiento simulado de los estudiantes

El sistema permite **consultar información histórica, generar respuestas fundamentadas, optimizar planes de clases y simular su efectividad educativa**.

---

## ⚙ Requerimientos generales
- Python 3.10 o superior
- Al menos 8 GB de RAM (16 GB recomendado para simulación)
- Conexión a internet (para la API de Gemini y descargas de modelos)
- Sistema operativo: Windows

---

## 🌐 APIs y bibliotecas principales
- Google Gemini API
- FAISS
- Scikit-Fuzzy
- Streamlit (opcional para interfaz visual)

---

## 🚀 Forma de uso

### 1️⃣ Clona el proyecto
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
### 5️⃣. Descarga previa del modelo de embeddings (IMPORTANTE):
    ```bash
    python src\tools\download_mpnet.py
    ```
### 6️⃣. Ejecuta la aplicación con Streamlit (opcional):
    ```bash
    streamlit run src/visual/app.py
    ```

### 7️⃣. Ejecuta la simulación metaheurística:
    ```bash
    python -m src.core.pipeline_metaheuristic_simulation.py
    ```