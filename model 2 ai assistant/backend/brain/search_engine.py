"""OMEGA MODEL 2: Real Web Search and Extraction Engine.

Performs live search queries, extracts clean text from webpages, generates
grounded summaries, and provides verifiable URL citations.
"""

import html
import re
from typing import Any, Dict, List, Optional
from urllib.parse import unquote, urlparse
import requests

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    BeautifulSoup = None
    HAS_BS4 = False


class SearchEngine:
    """Production web search and webpage text extraction engine."""

    def __init__(self):
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        }

    def search(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """Perform real web search via DuckDuckGo HTML without requiring API keys."""
        results = []
        if not query.strip():
            return results

        try:
            url = "https://html.duckduckgo.com/html/"
            data = {"q": query, "b": ""}
            resp = requests.post(url, data=data, headers=self.headers, timeout=10)

            if resp.status_code == 200:
                if HAS_BS4 and BeautifulSoup is not None:
                    soup = BeautifulSoup(resp.text, "html.parser")
                    web_results = soup.find_all("div", class_="result")

                    for res in web_results:
                        title_elem = res.find("a", class_="result__a")
                        snippet_elem = res.find("a", class_="result__snippet")

                        if title_elem:
                            raw_href = title_elem.get("href", "")
                            # Parse actual target URL from DDG redirect wrapper
                            match = re.search(r"uddg=([^&]+)", raw_href)
                            actual_url = unquote(match.group(1)) if match else raw_href

                            title = title_elem.get_text(strip=True)
                            snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""

                            if actual_url and actual_url.startswith("http"):
                                results.append({
                                    "title": title,
                                    "url": actual_url,
                                    "snippet": snippet,
                                    "source": urlparse(actual_url).netloc,
                                })

                        if len(results) >= max_results:
                            break
                else:
                    wiki_results = self._search_wikipedia(query, limit=max_results)
                    results.extend(wiki_results)

        except Exception as e:
            # Fallback to Wikipedia API for conceptual knowledge
            try:
                wiki_results = self._search_wikipedia(query, limit=max_results)
                results.extend(wiki_results)
            except Exception:
                pass

        return results

    def _search_wikipedia(self, query: str, limit: int = 3) -> List[Dict[str, str]]:
        """Fallback to Wikipedia OpenSearch API."""
        url = f"https://en.wikipedia.org/w/api.php?action=opensearch&search={query}&limit={limit}&namespace=0&format=json"
        resp = requests.get(url, headers=self.headers, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            titles = data[1]
            snippets = data[2]
            urls = data[3]
            out = []
            for t, s, u in zip(titles, snippets, urls):
                out.append({"title": t, "url": u, "snippet": s, "source": "wikipedia.org"})
            return out
        return []

    def fetch_page_content(self, url: str, max_chars: int = 2500) -> str:
        """Fetch and extract clean plain text from a web page."""
        try:
            resp = requests.get(url, headers=self.headers, timeout=10)
            if resp.status_code == 200:
                if HAS_BS4 and BeautifulSoup is not None:
                    soup = BeautifulSoup(resp.text, "html.parser")
                    # Strip script, style, navigation, and advertisement tags
                    for element in soup(["script", "style", "nav", "footer", "header", "noscript", "aside"]):
                        element.decompose()
                    text = soup.get_text(separator=" ", strip=True)
                else:
                    clean_no_script = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", resp.text, flags=re.DOTALL | re.IGNORECASE)
                    text = re.sub(r"<[^>]+>", " ", clean_no_script)

                # Clean redundant whitespace
                clean_text = re.sub(r"\s+", " ", text)
                return clean_text[:max_chars]
        except Exception as e:
            return f"[Failed to fetch content from {url}: {e}]"
        return ""

    def search_and_synthesize(self, query: str, max_results: int = 3) -> Dict[str, Any]:
        """Execute search, retrieve top snippets, and prepare cited context."""
        results = self.search(query, max_results=max_results)
        if not results:
            return {
                "query": query,
                "found": False,
                "summary": f"No live search results found for '{query}'.",
                "citations": [],
            }

        citations = []
        synthesized_blocks = []
        for i, item in enumerate(results, 1):
            citations.append({"index": i, "title": item["title"], "url": item["url"], "source": item["source"]})
            synthesized_blocks.append(
                f"[{i}] {item['title']} ({item['source']})\n"
                f"URL: {item['url']}\n"
                f"Excerpt: {item['snippet']}\n"
            )

        context_text = "\n".join(synthesized_blocks)
        return {
            "query": query,
            "found": True,
            "context": context_text,
            "citations": citations,
            "raw_results": results,
        }


# Singleton instance
search_engine = SearchEngine()

