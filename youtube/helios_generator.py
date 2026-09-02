from __future__ import annotations

import os
import time
from pathlib import Path

import torch
from diffusers import (
    AutoencoderKLWan,
    HeliosDMDScheduler,
    HeliosPyramidPipeline,
)
from diffusers.utils import export_to_video


DEFAULT_MODEL_ID = "BestWishYsh/Helios-Distilled"

DEFAULT_HEIGHT = 384
DEFAULT_WIDTH = 640

# Helios works with frame counts based on 33-frame chunks.
DEFAULT_NUM_FRAMES = 33

DEFAULT_FPS = 24

DEFAULT_STEPS = 2


class HeliosGenerator:
    """
    Reusable Helios video generator.

    The model is loaded once and reused for multiple generations.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.model_id = model_id
        self.device = device
        self.dtype = dtype

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. Helios generation requires a CUDA GPU."
            )

        self.pipe = self._load_pipeline()

    def _load_pipeline(self):
        print(f"[Helios] Loading model: {self.model_id}")

        vae = AutoencoderKLWan.from_pretrained(
            self.model_id,
            subfolder="vae",
            torch_dtype=torch.float32,
        )

        scheduler = HeliosDMDScheduler.from_pretrained(
            self.model_id,
            subfolder="scheduler",
        )

        pipe = HeliosPyramidPipeline.from_pretrained(
            self.model_id,
            vae=vae,
            scheduler=scheduler,
            torch_dtype=self.dtype,
            is_distilled=True,
        )

        pipe.to(self.device)

        self._configure_attention_backend(pipe)

        print("[Helios] Model loaded.")

        return pipe

    def _configure_attention_backend(self, pipe):
        """
        Configure the same attention backend strategy used by the
        original Helios application when possible.
        """

        if not torch.cuda.is_available():
            return

        cuda_major = torch.cuda.get_device_capability()[0]

        try:
            if cuda_major >= 9:
                try:
                    pipe.transformer.set_attention_backend(
                        "_flash_3_hub"
                    )
                except Exception:
                    pipe.transformer.set_attention_backend(
                        "flash_hub"
                    )
            else:
                pipe.transformer.set_attention_backend(
                    "flash_hub"
                )

            print("[Helios] Flash attention backend configured.")

        except Exception as exc:
            print(
                "[Helios] Warning: could not configure "
                f"flash attention backend: {exc}"
            )

    def generate(
        self,
        prompt: str,
        output_path: str | os.PathLike,
        height: int = DEFAULT_HEIGHT,
        width: int = DEFAULT_WIDTH,
        num_frames: int = DEFAULT_NUM_FRAMES,
        num_inference_steps: int = DEFAULT_STEPS,
        seed: int = 42,
        is_amplify_first_chunk: bool = True,
        fps: int = DEFAULT_FPS,
    ) -> str:

        if not prompt or not prompt.strip():
            raise ValueError("Prompt is required.")

        if num_frames < 33:
            raise ValueError(
                "num_frames must be at least 33."
            )

        if num_frames % 33 != 0:
            raise ValueError(
                "num_frames must be a multiple of 33."
            )

        output_path = Path(output_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        print()
        print("[Helios] Starting generation")
        print(f"[Helios] Prompt: {prompt}")
        print(f"[Helios] Resolution: {width}x{height}")
        print(f"[Helios] Frames: {num_frames}")
        print(f"[Helios] Steps: {num_inference_steps}")
        print(f"[Helios] Seed: {seed}")

        generator = torch.Generator(
            device=self.device
        ).manual_seed(int(seed))

        kwargs = {
            "prompt": prompt,
            "height": int(height),
            "width": int(width),
            "num_frames": int(num_frames),
            "guidance_scale": 1.0,
            "generator": generator,
            "output_type": "np",
            "pyramid_num_inference_steps_list": [
                int(num_inference_steps),
                int(num_inference_steps),
                int(num_inference_steps),
            ],
            "is_amplify_first_chunk": bool(
                is_amplify_first_chunk
            ),
        }

        start_time = time.time()

        with torch.inference_mode():
            output = self.pipe(**kwargs).frames[0]

        elapsed = time.time() - start_time

        export_to_video(
            output,
            str(output_path),
            fps=fps,
        )

        print(
            f"[Helios] Generation completed in "
            f"{elapsed:.1f}s"
        )

        print(
            f"[Helios] Output: {output_path}"
        )

        return str(output_path)
