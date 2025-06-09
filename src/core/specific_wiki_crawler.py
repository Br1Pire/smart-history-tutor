import json
import requests
import time
import os

# Archivos
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUERIES_FILE = "specific_wiki_titles"
OUTPUT_FILE = os.path.join(BASE_DIR, "data", "raw", "specific_wiki_articles")

# Función para consultar la API de Wikipedia en español
def search_wikipedia_article(query):
    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": query,
        "utf8": 1
    }
    response = requests.get("https://es.wikipedia.org/w/api.php", params=params)
    data = response.json()
    results = data.get("query", {}).get("search", [])
    return results[0]["title"] if results else None

def get_article_content(title):
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "explaintext": 1,
        "titles": title,
        "utf8": 1
    }
    response = requests.get("https://es.wikipedia.org/w/api.php", params=params)
    data = response.json()
    pages = data.get("query", {}).get("pages", {})
    for page in pages.values():
        return page.get("extract", "")
    return ""

# Cargar queries
with open(QUERIES_FILE, "r", encoding="utf-8") as f:
    queries = json.load(f)

# Cargar artículos ya guardados si existen
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        saved_articles = json.load(f)
else:
    saved_articles = []

# Crear set de títulos ya guardados
saved_titles = set(article["titulo"] for article in saved_articles)

# Artículos nuevos a guardar
new_articles = []

for item in queries:
    query = item["query"]
    print(f"🔍 Buscando: {query}")

    try:
        title = search_wikipedia_article(query)
        if not title:
            print(f"⚠️ No se encontró artículo para: {query}")
            continue

        if title in saved_titles:
            print(f"✅ Ya guardado: {title}")
            continue

        content = get_article_content(title)
        if not content.strip():
            print(f"⚠️ Artículo vacío: {title}")
            continue

        article = {
            "query": query,
            "titulo": title,
            "contenido": content
        }

        new_articles.append(article)
        saved_titles.add(title)
        print(f"✅ Guardado: {title}")
        time.sleep(0.5)

    except Exception as e:
        print(f"❌ Error con '{query}': {e}")

# Guardar combinación
all_articles = saved_articles + new_articles
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(all_articles, f, ensure_ascii=False, indent=2)

print(f"\n✅ Proceso completo. Se añadieron {len(new_articles)} nuevos artículos.")
