from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile

import imageio_ffmpeg


def concatenate_videos(
    video_paths: list[str],
    output_path: str,
) -> str:

    if not video_paths:
        raise ValueError(
            "At least one video is required."
        )

    output = Path(output_path)
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".txt",
        delete=False,
        encoding="utf-8",
    ) as concat_file:

        concat_path = Path(concat_file.name)

        for video_path in video_paths:
            path = Path(video_path).resolve()

            if not path.exists():
                raise FileNotFoundError(
                    f"Video not found: {path}"
                )

            # FFmpeg concat format.
            escaped = str(path).replace(
                "'",
                "'\\''",
            )

            concat_file.write(
                f"file '{escaped}'\n"
            )

    command = [
        ffmpeg,
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_path),
        "-c",
        "copy",
        str(output),
    ]

    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "FFmpeg video concatenation failed.\n"
            f"{exc.stderr}"
        ) from exc
    finally:
        concat_path.unlink(
            missing_ok=True
        )

    return str(output)
