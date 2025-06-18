import requests
import json
import logging
import os
import time
from src.config import LOG_FILES, TITLES_FILE, RAW_FILE

# Configuración de logging
LOG_FILE = LOG_FILES["crawler"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

WIKI_API_URL = "https://es.wikipedia.org/w/api.php"
INPUT_FILE = TITLES_FILE
OUTPUT_FILE = RAW_FILE


def safe_get(url, params, max_retries=3, timeout=10):
    """
    Realiza una solicitud GET con reintentos y manejo de errores.

    Args:
        url (str): URL base de la API.
        params (dict): Parámetros de la solicitud.
        max_retries (int): Número máximo de reintentos.
        timeout (int): Tiempo máximo de espera por solicitud (segundos).

    Returns:
        dict or None: Respuesta JSON si es exitosa; None en caso de fallo.
    """
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            logging.info(f"✅ GET exitoso en intento {attempt} para {params}")
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.warning(f"⚠️ Intento {attempt} falló: {e}")
            if attempt < max_retries:
                time.sleep(2 * attempt)
    logging.error(f"❌ Fallo tras {max_retries} intentos para URL: {url} con params: {params}")
    return None


def search_article(query):
    """
    Busca el título real de un artículo en Wikipedia para un término dado.

    Args:
        query (str): Término de búsqueda.

    Returns:
        str or None: Título del primer resultado o None si no se encuentra.
    """
    logging.info(f"🔎 Buscando artículo para query: '{query}'")
    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": query
    }
    resp = safe_get(WIKI_API_URL, params)
    if not resp:
        logging.error(f"❌ No se obtuvo respuesta de la API para: '{query}'")
        return None
    search_results = resp.get("query", {}).get("search", [])
    if search_results:
        title = search_results[0]["title"]
        logging.info(f"✅ Artículo encontrado: '{title}' para query '{query}'")
        return title
    logging.warning(f"⚠️ No se encontraron resultados para: '{query}'")
    return None


def fetch_article_data(title):
    """
    Obtiene el extracto y las categorías de un artículo de Wikipedia.

    Args:
        title (str): Título del artículo.

    Returns:
        tuple: Extracto (str) y lista de categorías (list).
    """
    logging.info(f"📄 Descargando datos del artículo: '{title}'")
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
    logging.info(f"✅ Datos obtenidos para '{title}' (Extracto: {len(extract)} caracteres, Categorías: {len(categories)})")
    return extract, categories


def process_article(query, title):
    """
    Procesa un artículo: obtiene su extracto y categorías.

    Args:
        query (str): Término original de búsqueda.
        title (str): Título del artículo.

    Returns:
        dict: Diccionario con los datos del artículo.
    """
    logging.info(f"🚀 Procesando artículo: '{title}' para query: '{query}'")
    extract, categories = fetch_article_data(title)
    return {
        "query": query,
        "title": title,
        "content": extract,
        "categories": categories
    }


def load_existing_articles(output_file):
    """
    Carga los artículos previamente almacenados.

    Args:
        output_file (str): Ruta al archivo JSON de salida.

    Returns:
        list: Lista de artículos existentes.
    """
    if not os.path.exists(output_file):
        logging.info("ℹ️ No se encontró archivo previo. Se iniciará nuevo corpus.")
        return []
    with open(output_file, "r", encoding="utf-8") as f:
        logging.info(f"📂 Archivo existente cargado: {output_file}")
        return json.load(f)


def save_articles(output_file, articles):
    """
    Guarda los artículos en un archivo JSON.

    Args:
        output_file (str): Ruta al archivo de salida.
        articles (list): Lista de artículos a guardar.
    """
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)
    logging.info(f"💾 Corpus guardado en: {output_file} (Total artículos: {len(articles)})")


def crawl_titles(input_file=INPUT_FILE, output_file=OUTPUT_FILE):
    """
    Realiza crawling para una lista de títulos.

    Args:
        input_file (str): Ruta al archivo con queries.
        output_file (str): Ruta al archivo de salida.
    """
    logging.info("🚀 Inicio del crawling por títulos")
    with open(input_file, "r", encoding="utf-8") as f:
        topics = json.load(f)

    existing_articles = load_existing_articles(output_file)
    existing_titles = {a["title"] for a in existing_articles}
    results = existing_articles.copy()

    for item in topics:
        query = item["query"]
        title = search_article(query)
        if not title:
            continue
        if title in existing_titles:
            logging.info(f"⏩ Artículo ya existente: '{title}'")
            continue
        result = process_article(query, title)
        results.append(result)
        existing_titles.add(title)

    save_articles(output_file, results)
    logging.info(f"🏁 Proceso completado. Total temas procesados: {len(results)}")


def crawl_single_title(query, output_file=OUTPUT_FILE):
    """
    Procesa un solo término, descarga y guarda el artículo si es nuevo.

    Args:
        query (str): Término de búsqueda.
        output_file (str): Ruta al archivo de salida.

    Returns:
        dict or None: Artículo procesado o None si no fue añadido.
    """
    logging.info(f"🌐 Crawler intentando buscar artículo para: '{query}'")
    existing_articles = load_existing_articles(output_file)
    existing_titles = {a["title"] for a in existing_articles}
    title = search_article(query)
    if not title:
        return None
    if title in existing_titles:
        logging.info(f"⏩ Artículo '{title}' ya existe en el corpus.")
        return None
    result = process_article(query, title)
    existing_articles.append(result)
    save_articles(output_file, existing_articles)
    logging.info(f"✅ Artículo '{title}' añadido al corpus.")
    return result


if __name__ == "__main__":
    crawl_titles(INPUT_FILE, OUTPUT_FILE)
