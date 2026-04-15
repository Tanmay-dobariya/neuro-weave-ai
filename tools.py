import requests
import os
from langchain.tools import tool
from tavily import TavilyClient
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


@tool(
    name="research_tool",
    description="Use this tool whenever you need to do some research on the web from certain topics",
)
def web_search(query: str) -> str:
    """Use this tool to do research on web for provided topic. Returns Titles, URLs and Snippets."""
    try:
        tavily = TavilyClient(api_key=TavilyClient)
        response = tavily.search(query=query, max_results=5)

        outputs = []

        for result in response.get("results"):
            outputs.append(
                f"Title: {result.get('title')}\n"
                f"URL: {result.get('url')}\n"
                f"Snippet: {result.get('content')}\n"
            )

        return "\n___\n".join(outputs)
    except Exception as e:
        return f"Error while searching on the web: {str(e)}"


@tool(
    name="reader_tool",
    description="Use this tool to scrape contents from web pages for more deeper readings.",
)
def scrape_webpage(url: str) -> str:
    """Use this tool to scrape contents from web pages for more deeper readings"""
    try:
        response = requests.get(
            url=url,
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        soup = BeautifulSoup(response.text, "lxml")

        for tag in soup(["style", "script", "nav", "footer", "header"]):
            soup.decompose()

        return soup.get_text(separator=" ", strip=True)[:3000]
    except Exception as e:
        return f"Error while scraping webpage: {str(e)}"
