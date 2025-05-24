import wikipediaapi
import json
import os

# Configuraciones iniciales
LANG = 'es'  # 'es' para español, 'en' para inglés
MAX_DOCS = 2000

# Calcular ruta del directorio de salida usando path relativo
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "raw")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"wikipedia_historia_{LANG}.json")

# Crear carpeta si no existe
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Inicializar Wikipedia API
wiki = wikipediaapi.Wikipedia(
    language=LANG,
    user_agent="smart-history-tutor/1.0 (contacto@ejemplo.com)"
)

# Categorías principales de historia (puedes agregar más si quieres)
CATEGORIES = [
    "Historia antigua",
    "Edad Media",
    "Edad Moderna",
    "Edad Contemporánea",
    "Imperio romano",
    "Segunda Guerra Mundial",
    "Primera Guerra Mundial",
    "Revolución francesa",
    "Historia de América",
    "Imperio español",
    "Historia mundial",
    "Historia de Europa",
    "Historia de Asia",
    "Batallas",
    "Imperialismo",
    "Historia militar",
    "Colonialismo",
    "Historia de China",
    "Historia de la India",
    "Historia del Caribe",
    "Historia de Mesopotamia",
    "Historia de Egipto",
    "Renacimiento",
    "Revoluciones",
    "Independencias de América",
    "Ilustración",
    "Líderes militares",
    "Reyes y reinas"
]

# Función para recolectar artículos únicos desde categorías
visited_titles = set()
documents = []


def fetch_articles_from_category(cat_name, depth=0, max_depth=3):
    if depth > max_depth:
        return
    category = wiki.page(f"Categoría:{cat_name}")
    if not category.exists():
        print(f"Categoría no encontrada: {cat_name}")
        return

    for title, page in category.categorymembers.items():
        if len(documents) >= MAX_DOCS:
            break

        if page.ns == 14:  # Namespace 14 = categoría

            # Quitar prefijo "Categoría:" del título si existe
            subcat_name = title
            if subcat_name.startswith("Categoría:"):
                subcat_name = subcat_name[len("Categoría:"):]

            # Recursivamente buscar en subcategorías
            fetch_articles_from_category(subcat_name, depth=depth+1, max_depth=max_depth)
        elif page.ns == 0 and title not in visited_titles:
            content = page.text
            if len(content) > 500:  # Opcional: filtro mínimo para evitar artículos muy cortos
                documents.append({
                    "title": title,
                    "content": content,
                    "category": cat_name
                })
                visited_titles.add(title)
                print(f"[+] {title} ({len(documents)}/{MAX_DOCS})")


# Ejecutar extracción
for cat in CATEGORIES:
    if len(documents) >= MAX_DOCS:
        break
    fetch_articles_from_category(cat)

# Guardar documentos
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(documents, f, ensure_ascii=False, indent=2)

print(f"\n✅ Guardados {len(documents)} documentos en {OUTPUT_FILE}")
