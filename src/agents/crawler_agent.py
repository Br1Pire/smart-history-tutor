# crawler_agent.py
import json
import requests
import os
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_FILE = os.path.join(BASE_DIR, "data", "raw", "specific_wiki_articles.json")

def search_wikipedia_title(query):
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

def load_existing_articles():
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_articles(articles):
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)

def crawl_batch(queries):
    saved_articles = load_existing_articles()
    saved_titles = set(article.get("title") for article in saved_articles)
    new_articles = []

    for query in queries:
        title = search_wikipedia_title(query)
        if not title or title in saved_titles:
            continue
        content = get_article_content(title)
        if not content.strip():
            continue
        article = {
            "query": query,
            "title": title,
            "content": content
        }
        new_articles.append(article)
        saved_titles.add(title)
        time.sleep(0.5)

    if new_articles:
        save_articles(saved_articles + new_articles)
    return new_articles

def crawl_single(query):
    saved_articles = load_existing_articles()
    saved_titles = set(article.get("title") for article in saved_articles)

    title = search_wikipedia_title(query)
    if not title or title in saved_titles:
        return None
    content = get_article_content(title)
    if not content.strip():
        return None

    article = {
        "query": query,
        "title": title,
        "content": content
    }
    #save_articles(saved_articles + [article])
    return article

# Para pruebas individuales
if __name__ == "__main__":
    from sys import argv
    if len(argv) > 1:
        result = crawl_single(argv[1])
        if result:
            print("✅ Article saved:")
            print(f"\n📝 Title: {result['title']}\n")
            print(f"📄 Content:\n{result['content']}\n")
        else:
            print("⚠️ Article not found or already exists.")
    else:
        print("ℹ️ Uso: python crawler_agent.py 'Nombre del tema'")
