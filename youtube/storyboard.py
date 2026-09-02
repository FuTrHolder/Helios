from __future__ import annotations

from dataclasses import dataclass, asdict
import re

from .article_parser import BlogArticle, split_into_paragraphs


@dataclass
class Scene:
    scene_id: int
    narration: str
    visual_description: str
    duration: float = 10.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Storyboard:
    title: str
    scenes: list[Scene]

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "scenes": [scene.to_dict() for scene in self.scenes],
        }


def extract_topic(text: str) -> str:
    """
    Extract a simple topic from a paragraph.
    """

    text = re.sub(r"\s+", " ", text).strip()

    if len(text) <= 120:
        return text

    return text[:120].rsplit(" ", 1)[0]


def create_visual_description(
    title: str,
    paragraph: str,
) -> str:
    """
    Convert article content into a visual description suitable for Helios.

    This is intentionally conservative for the first PoC.
    """

    topic = extract_topic(paragraph)

    return (
        f"Cinematic visual illustrating the topic '{title}'. "
        f"The scene visually represents: {topic}. "
        "Professional editorial style, realistic environment, "
        "natural lighting, subtle camera movement, "
        "high visual quality, detailed composition, "
        "no text, no subtitles, no logos, no watermark."
    )


def build_storyboard(
    article: BlogArticle,
    max_scenes: int = 3,
    scene_duration: float = 10.0,
) -> Storyboard:
    """
    Convert a blog article into a simple storyboard.

    The first paragraph becomes the first scene, followed by
    subsequent meaningful paragraphs.
    """

    paragraphs = split_into_paragraphs(article)

    if not paragraphs:
        raise ValueError("No usable paragraphs found in article.")

    selected = paragraphs[:max_scenes]

    scenes: list[Scene] = []

    for index, paragraph in enumerate(selected, start=1):
        scenes.append(
            Scene(
                scene_id=index,
                narration=paragraph,
                visual_description=create_visual_description(
                    article.title,
                    paragraph,
                ),
                duration=scene_duration,
            )
        )

    return Storyboard(
        title=article.title,
        scenes=scenes,
    )
