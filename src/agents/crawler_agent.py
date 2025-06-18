import requests
import json
import logging
import os
import time

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

WIKI_API_URL = "https://es.wikipedia.org/w/api.php"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(BASE_DIR,"data", "titles", "specific_wiki_titles.json")
OUTPUT_FILE = os.path.join(BASE_DIR,"data", "raw", "wiki_articles_raw.json")


def safe_get(url, params, max_retries=3, timeout=10):
    """Realiza un GET con reintentos y timeout."""
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.warning(f"Intento {attempt} falló: {e}")
            if attempt < max_retries:
                time.sleep(2 * attempt)  # backoff exponencial suave
    logging.error(f"Fallo tras {max_retries} intentos para URL: {url} con params: {params}")
    return None

def search_article(query):
    """Busca el título real en Wikipedia para un término dado."""
    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": query
    }
    resp = safe_get(WIKI_API_URL, params)
    if not resp:
        return None
    search_results = resp.get("query", {}).get("search", [])
    if search_results:
        return search_results[0]["title"]
    else:
        return None

def fetch_article_data(title):
    """Obtiene extracto plano y categorías de un artículo dado."""
    params_extract = {
        "action": "query",
        "format": "json",
        "prop": "extracts|categories",
        "titles": title,
        "explaintext": 1,
        "cllimit": "max"
    }
    resp = safe_get(WIKI_API_URL, params_extract)
    extract = ""
    categories = []

    if resp:
        pages = resp.get("query", {}).get("pages", {})
        for page in pages.values():
            extract = page.get("extract", "")
            category_list = page.get("categories", [])
            categories = [
                cat.get("title", "").replace("Categoría:", "").strip()
                for cat in category_list
                if not cat.get("title", "").startswith("Categoría:Wikipedia:")
            ]

    return extract, categories


def process_article(query, title):
    """Procesa un artículo individual dado un query y un título."""
    logging.info(f"Procesando artículo: {title}")
    extract, categories = fetch_article_data(title)
    return {
        "query": query,
        "title": title,
        "content": extract,
        "categories": categories
    }

def load_existing_articles(output_file):
    if not os.path.exists(output_file):
        return []
    with open(output_file, "r", encoding="utf-8") as f:
        return json.load(f)

def save_articles(output_file, articles):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)

def crawl_titles(input_file = INPUT_FILE, output_file = OUTPUT_FILE ):
    with open(input_file, "r", encoding="utf-8") as f:
        topics = json.load(f)

    existing_articles = load_existing_articles(output_file)
    existing_titles = {a["title"] for a in existing_articles}

    results = existing_articles.copy()

    for item in topics:
        query = item["query"]
        logging.info(f"Buscando artículo real para: {query}")
        title = search_article(query)
        if not title:
            logging.warning(f"No se encontró artículo para: {query}")
            continue
        if title in existing_titles:
            logging.info(f"Artículo ya extraído previamente: {title}")
            continue

        result = process_article(query, title)
        results.append(result)
        existing_titles.add(title)

    save_articles(output_file, results)
    logging.info(f"Proceso completado. Total temas procesados: {len(results)}")


def crawl_single_title(query, output_file = OUTPUT_FILE):
    """Procesa un solo query, verifica si el artículo ya existe y, si no, lo añade al corpus."""

    logging.info(f"🌐 Crawler intentando buscar artículo para: '{query}'")

    # Cargar artículos existentes
    existing_articles = load_existing_articles(output_file)
    existing_titles = {a["title"] for a in existing_articles}

    # Buscar el título real en Wikipedia
    title = search_article(query)
    if not title:
        logging.warning(f"No se encontró artículo para: {query}")
        return None

    logging.info(f"📄 Título elegido por el crawler: '{title}'")

    # Verificar duplicado
    if title in existing_titles:
        logging.info(f"Artículo '{title}' ya existe en el corpus. No se añade.")
        return None

    # Procesar y añadir
    result = process_article(query, title)
    existing_articles.append(result)
    save_articles(output_file, existing_articles)
    logging.info(f"Artículo '{title}' añadido al corpus.")

    return result


if __name__ == "__main__":
    crawl_titles(INPUT_FILE, OUTPUT_FILE)
