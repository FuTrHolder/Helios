from __future__ import annotations

from dataclasses import dataclass
from html import unescape
import re
from typing import Optional


@dataclass
class BlogArticle:
    title: str
    content: str
    url: Optional[str] = None


def clean_html(text: str) -> str:
    """
    Convert basic HTML into plain text.
    This intentionally avoids requiring BeautifulSoup for the first PoC.
    """

    if not text:
        return ""

    text = unescape(text)

    # Remove script/style blocks.
    text = re.sub(
        r"<(script|style).*?>.*?</\1>",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    # Convert common block elements into line breaks.
    text = re.sub(
        r"</(p|div|h1|h2|h3|h4|li|section|article)>",
        "\n",
        text,
        flags=re.IGNORECASE,
    )

    # Remove remaining HTML tags.
    text = re.sub(r"<[^>]+>", " ", text)

    # Normalize whitespace.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    return text.strip()


def parse_article(
    title: str,
    content: str,
    url: Optional[str] = None,
) -> BlogArticle:
    """
    Create a normalized BlogArticle object.
    """

    if not title or not title.strip():
        raise ValueError("Article title is required.")

    if not content or not content.strip():
        raise ValueError("Article content is required.")

    cleaned_content = clean_html(content)

    return BlogArticle(
        title=title.strip(),
        content=cleaned_content,
        url=url,
    )


def split_into_paragraphs(article: BlogArticle) -> list[str]:
    """
    Split article content into meaningful paragraphs.
    """

    paragraphs = [
        p.strip()
        for p in re.split(r"\n\s*\n", article.content)
        if p.strip()
    ]

    if not paragraphs:
        paragraphs = [
            line.strip()
            for line in article.content.splitlines()
            if line.strip()
        ]

    return paragraphs


def truncate_text(text: str, max_chars: int = 500) -> str:
    """
    Truncate text without cutting excessively long content.
    """

    text = text.strip()

    if len(text) <= max_chars:
        return text

    return text[:max_chars].rsplit(" ", 1)[0] + "..."
