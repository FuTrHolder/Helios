from __future__ import annotations

from .storyboard import Scene


BASE_STYLE = (
    "cinematic documentary footage, "
    "photorealistic, "
    "professional editorial video, "
    "natural realistic lighting, "
    "high detail, "
    "smooth realistic motion, "
    "subtle camera movement, "
    "coherent composition, "
    "consistent visual identity, "
    "no text, "
    "no subtitles, "
    "no captions, "
    "no logos, "
    "no watermark"
)


def build_scene_prompt(scene: Scene) -> str:
    """
    Build the final Helios prompt for a scene.
    """

    prompt = (
        f"{scene.visual_description}. "
        f"{BASE_STYLE}. "
        "The camera movement should be smooth and controlled. "
        "The main subject should remain visually coherent throughout "
        "the entire shot. "
        "Avoid sudden scene changes, object duplication, "
        "unnatural deformation, flickering, or excessive motion."
    )

    return prompt


def build_hook_prompt(title: str) -> str:
    """
    Prompt used for a future YouTube intro/hook scene.
    """

    return (
        f"A cinematic opening shot introducing the topic '{title}'. "
        "Strong visual hook, visually compelling composition, "
        "professional documentary style, "
        "realistic lighting, "
        "slow cinematic camera movement, "
        "photorealistic, "
        "high detail, "
        "no text, no subtitles, no logos, no watermark."
    )
