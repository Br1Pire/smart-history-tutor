import wikipediaapi

def fetch_wikipedia_article(query, lang="es"):
    wiki_wiki = wikipediaapi.Wikipedia(lang)
    page = wiki_wiki.page(query)

    if not page.exists():
        return None
    return page.text