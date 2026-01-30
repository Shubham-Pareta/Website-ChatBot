import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

HEADERS = {"User-Agent": "Mozilla/5.0"}

def get_all_links(base_url, max_pages=6):
    visited = set()
    to_visit = [base_url]
    domain = urlparse(base_url).netloc

    while to_visit and len(visited) < max_pages:
        url = to_visit.pop()
        if url in visited:
            continue

        try:
            res = requests.get(url, timeout=6, headers=HEADERS)
            if res.status_code != 200:
                continue
        except:
            continue

        visited.add(url)
        soup = BeautifulSoup(res.text, "html.parser")

        for link in soup.find_all("a", href=True):
            full_url = urljoin(url, link["href"])
            if urlparse(full_url).netloc == domain and full_url not in visited:
                to_visit.append(full_url)

    return list(visited)


def extract_text_and_title(url):
    try:
        res = requests.get(url, timeout=6, headers=HEADERS)
        if res.status_code != 200:
            return "", ""
    except:
        return "", ""

    soup = BeautifulSoup(res.text, "html.parser")

    # Remove junk sections
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()


    title = soup.title.string.strip() if soup.title else ""
    elements = soup.find_all(["p", "h1", "h2", "h3", "li"])
    text = " ".join(el.get_text(" ", strip=True) for el in elements)

    return text, title
