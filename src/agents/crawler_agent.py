import requests
import json
import logging
import os
import time
from src.core.document_manager import DocumentManager
from src.config import LOG_FILES

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

class Crawler:

    def __init__(self, document_manager: DocumentManager):
        self.document_manager = document_manager
        logging.info("Crawler inicializado.")


    @staticmethod
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


    def search_article(self, query):
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
        resp = self.safe_get(WIKI_API_URL, params)
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


    def fetch_article_data(self, title):
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
        resp = self.safe_get(WIKI_API_URL, params_extract)
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


    def process_article(self, query, title):
        """
        Procesa un artículo: obtiene su extracto y categorías.

        Args:
            query (str): Término original de búsqueda.
            title (str): Título del artículo.

        Returns:
            dict: Diccionario con los datos del artículo.
        """
        logging.info(f"🚀 Procesando artículo: '{title}' para query: '{query}'")
        extract, categories = self.fetch_article_data(title)
        return {
            "query": query,
            "title": title,
            "content": extract,
            "categories": categories
        }

    def crawl_titles(self):
        """
        Realiza crawling para una lista de títulos.

        Args:
            input_file (str): Ruta al archivo con queries.
            output_file (str): Ruta al archivo de salida.
        """
        logging.info("🚀 Inicio del crawling por títulos")

        existing_articles = self.document_manager.raw_documents
        existing_titles = {a["title"] for a in existing_articles}
        results = existing_articles.copy()

        for item in self.document_manager.titles:
            query = item["query"]
            title = self.search_article(query)
            if not title:
                continue
            if title in existing_titles:
                logging.info(f"⏩ Artículo ya existente: '{title}'")
                continue
            result = self.process_article(query, title)
            results.append(result)
            existing_titles.add(title)

        self.document_manager.add_articles(results)
        logging.info(f"🏁 Proceso completado. Total temas procesados: {len(results)}")


    def crawl_single_title(self, query):
        """
        Procesa un solo término, descarga y guarda el artículo si es nuevo.

        Args:
            query (str): Término de búsqueda.
            output_file (str): Ruta al archivo de salida.

        Returns:
            dict or None: Artículo procesado o None si no fue añadido.
        """
        logging.info(f"🌐 Crawler intentando buscar artículo para: '{query}'")
        existing_articles = self.document_manager.raw_documents
        existing_titles = {a["title"] for a in existing_articles}
        title = self.search_article(query)
        if not title:
            return None
        if title in existing_titles:
            logging.info(f"⏩ Artículo '{title}' ya existe en el corpus.")
            return None
        result = self.process_article(query, title)
        existing_articles.append(result)
        self.document_manager.add_articles(existing_articles)
        logging.info(f"✅ Artículo '{title}' añadido al corpus.")
        return result



