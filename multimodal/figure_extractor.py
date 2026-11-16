import hashlib
import re
from pathlib import Path
from typing import List, Dict, Any
import fitz  # PyMuPDF

FIG_DIR = Path("artifacts/figures")

CAPTION_RE = re.compile(
    r"^(Figure|Fig\.?|Table)\s*(\d+)[:\.\s]+(.+)$",
    re.IGNORECASE | re.MULTILINE,
)


class FigureExtractor:
    """Extract figures/images + captions from a research PDF."""

    def __init__(self, out_dir: Path = FIG_DIR):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def extract(self, pdf_path: Path, paper_id: str) -> List[Dict[str, Any]]:
        doc = fitz.open(pdf_path)
        figures = []

        for page_num, page in enumerate(doc, start=1):
            # 1. Extract embedded images
            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                try:
                    pix = fitz.Pixmap(doc, xref)
                    if pix.n - pix.alpha >= 4:       # CMYK → RGB
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    if pix.width < 80 or pix.height < 80:   # skip tiny icons
                        continue
                    safe_pid = re.sub(r"[^A-Za-z0-9_.-]", "_", paper_id)
                    fname = f"{safe_pid}_p{page_num}_img{img_index}.png"
                    out_path = self.out_dir / fname
                    pix.save(out_path)
                    figures.append({
                        "figure_id": hashlib.sha1(
                            f"{paper_id}:{page_num}:{img_index}".encode()
                        ).hexdigest()[:12],
                        "paper_id": paper_id,
                        "page": page_num,
                        "image_path": str(out_path),
                        "width": pix.width,
                        "height": pix.height,
                        "caption": None,           # filled below
                        "nearby_text": "",
                    })
                except Exception as e:
                    print(f"[fig] skip xref {xref}: {e}")

            # 2. Extract caption candidates from page text
            page_text = page.get_text("text")
            captions = [m.group(0).strip()
                        for m in CAPTION_RE.finditer(page_text)]

            # 3. Attach the nearest caption to figures on this page
            page_figs = [f for f in figures if f["page"] == page_num]
            for i, fig in enumerate(page_figs):
                if i < len(captions):
                    fig["caption"] = captions[i]
                # capture a text window for extra context
                idx = page_text.find(captions[i]) if i < len(captions) else -1
                if idx >= 0:
                    fig["nearby_text"] = page_text[max(0, idx - 100): idx + 800]
        doc.close()
        return figures