from __future__ import annotations

from pathlib import Path
import json
import time

from .article_parser import parse_article
from .helios_generator import HeliosGenerator
from .prompt_builder import build_scene_prompt
from .storyboard import build_storyboard
from .video_composer import concatenate_videos


class BlogToYouTubePipeline:
    """
    Convert a blog article into a YouTube-ready video.

    First PoC:
    - Rule-based article parsing
    - Rule-based storyboard generation
    - Helios visual generation
    - Scene concatenation

    TTS, subtitles, music and YouTube upload are intentionally
    separated for later stages.
    """

    def __init__(
        self,
        output_dir: str = "outputs",
        helios_generator: HeliosGenerator | None = None,
    ):
        self.output_dir = Path(output_dir)

        self.scene_dir = (
            self.output_dir / "scenes"
        )

        self.final_dir = (
            self.output_dir / "final"
        )

        self.scene_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.final_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.helios = (
            helios_generator
            if helios_generator is not None
            else HeliosGenerator()
        )

    def generate(
        self,
        title: str,
        content: str,
        url: str | None = None,
        max_scenes: int = 1,
        scene_duration: float = 10.0,
        num_frames: int = 33,
        height: int = 384,
        width: int = 640,
        steps: int = 2,
        seed: int = 42,
    ) -> dict:

        started = time.time()

        article = parse_article(
            title=title,
            content=content,
            url=url,
        )

        storyboard = build_storyboard(
            article=article,
            max_scenes=max_scenes,
            scene_duration=scene_duration,
        )

        safe_title = self._safe_filename(
            title
        )

        storyboard_path = (
            self.output_dir
            / f"{safe_title}_storyboard.json"
        )

        with open(
            storyboard_path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                storyboard.to_dict(),
                file,
                ensure_ascii=False,
                indent=2,
            )

        generated_scenes: list[str] = []

        for scene in storyboard.scenes:

            prompt = build_scene_prompt(
                scene
            )

            scene_path = (
                self.scene_dir
                / f"{safe_title}_scene_{scene.scene_id}.mp4"
            )

            self.helios.generate(
                prompt=prompt,
                output_path=str(scene_path),
                height=height,
                width=width,
                num_frames=num_frames,
                num_inference_steps=steps,
                seed=seed + scene.scene_id - 1,
            )

            generated_scenes.append(
                str(scene_path)
            )

        final_path = (
            self.final_dir
            / f"{safe_title}.mp4"
        )

        concatenate_videos(
            generated_scenes,
            str(final_path),
        )

        elapsed = time.time() - started

        return {
            "title": title,
            "storyboard": storyboard.to_dict(),
            "storyboard_path": str(
                storyboard_path
            ),
            "scenes": generated_scenes,
            "video": str(final_path),
            "elapsed_seconds": round(
                elapsed,
                2,
            ),
        }

    @staticmethod
    def _safe_filename(
        value: str,
        max_length: int = 80,
    ) -> str:

        safe = "".join(
            char
            if char.isalnum()
            or char in (
                " ",
                "_",
                "-",
            )
            else "_"
            for char in value
        )

        safe = "_".join(
            safe.split()
        )

        return safe[:max_length]
