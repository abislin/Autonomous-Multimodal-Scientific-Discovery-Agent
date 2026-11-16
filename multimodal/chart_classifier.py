from typing import Dict, Any
from PIL import Image
import numpy as np


class ChartClassifier:
    """Heuristic chart-type guess from image statistics.

    The VLM output is authoritative; this is only used as a hint/metadata.
    """

    def classify(self, image_path: str) -> Dict[str, Any]:
        try:
            img = Image.open(image_path).convert("L")
        except Exception:
            return {"chart_type": "unknown", "confidence": 0.0}

        arr = np.asarray(img)
        # column/row ink density profile
        ink_cols = (arr < 200).sum(axis=0)
        ink_rows = (arr < 200).sum(axis=1)
        col_peaks = self._count_peaks(ink_cols)
        row_peaks = self._count_peaks(ink_rows)
        aspect = arr.shape[1] / max(1, arr.shape[0])

        if col_peaks >= 3 and row_peaks <= 2:
            ctype = "bar_chart"
        elif row_peaks >= 3 and col_peaks <= 2:
            ctype = "line_graph"
        elif col_peaks >= 3 and row_peaks >= 3 and aspect > 0.8:
            ctype = "scatter_plot"
        else:
            ctype = "unknown"
        return {"chart_type": ctype, "confidence": 0.4}

    @staticmethod
    def _count_peaks(profile, thresh_ratio: float = 0.5) -> int:
        if profile.size == 0:
            return 0
        t = profile.max() * thresh_ratio
        above = profile > t
        # count contiguous runs
        runs = 0
        prev = False
        for v in above:
            if v and not prev:
                runs += 1
            prev = v
        return runs