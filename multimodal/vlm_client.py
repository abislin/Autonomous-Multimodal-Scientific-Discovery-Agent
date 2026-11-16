import base64
import json
from pathlib import Path
from typing import Dict, Any
import os

VLM_SYSTEM = """You are a scientific figure analyst. Analyse the figure and
return STRICT JSON with these keys:
{
  "chart_type": "bar_chart"|"line_graph"|"scatter_plot"|"table"|"diagram"|"other",
  "title_guess": str,
  "axes": {"x": str, "y": str},
  "legend": [str, ...],
  "trends": [str, ...],
  "key_values": [str, ...],
  "explanation": str,     // 2-3 sentence summary of what the figure communicates
  "caveats": [str, ...]   // anything unreadable / ambiguous
}
Return ONLY the JSON object, no markdown fences."""


class VLMClient:
    def __init__(self):
        self.mode = os.getenv("VLM_MODE", "openai").lower()
        self.model = os.getenv("VLM_MODEL", "gpt-4o")
        self._local_pipe = None
        if self.mode == "local-llava":
            self._load_local()

    # ---------------- local LLaVA ----------------
    def _load_local(self):
        from transformers import pipeline
        self._local_pipe = pipeline(
            "image-to-text", model=self.model, device_map="auto"
        )

    # ---------------- openai-compatible ----------------
    def _openai_describe(self, image_path: str, caption: str) -> str:
        from openai import OpenAI
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        )
        b64 = base64.b64encode(Path(image_path).read_bytes()).decode()
        user_prompt = (
            f"Figure caption (may be empty): {caption or '(none)'}\n"
            "Analyse the image and return the required JSON."
        )
        resp = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": VLM_SYSTEM},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    ],
                },
            ],
            temperature=0.1,
        )
        return resp.choices[0].message.content

    # ---------------- public API ----------------
    def describe(self, image_path: str, caption: str = "") -> Dict[str, Any]:
        try:
            if self.mode == "local-llava":
                prompt = (f"USER: <image>\n{VLM_SYSTEM}\n"
                          f"Caption: {caption or '(none)'}\nASSISTANT:")
                raw = self._local_pipe(
                    images=image_path, prompt=prompt, max_new_tokens=512
                )[0]["generated_text"]
            else:
                raw = self._openai_describe(image_path, caption)
        except Exception as e:
            return {"_error": f"{type(e).__name__}: {e}", "chart_type": "unknown"}

        # strip fences if the model ignored instructions
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw
            if raw.rstrip().endswith("```"):
                raw = raw.rstrip()[:-3]
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"_parse_error": True, "raw": raw[:2000],
                    "chart_type": "unknown"}