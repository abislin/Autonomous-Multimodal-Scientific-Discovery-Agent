from typing import Dict, Any, List, Optional
import numpy as np
from scipy import stats


class StatisticalValidator:
    """Confidence intervals, paired tests, and multiple-comparison correction."""

    # ---------- confidence intervals ----------
    def bootstrap_ci(self, scores: List[float], confidence: float = 0.95,
                     n_boot: int = 2000, seed: int = 42) -> Dict[str, float]:
        if not scores:
            return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n": 0}
        rng = np.random.default_rng(seed)
        arr = np.asarray(scores, dtype=float)
        boots = np.array([
            rng.choice(arr, size=arr.size, replace=True).mean()
            for _ in range(n_boot)
        ])
        alpha = (1 - confidence) / 2
        return {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
            "ci_low": float(np.quantile(boots, alpha)),
            "ci_high": float(np.quantile(boots, 1 - alpha)),
            "n": int(arr.size),
        }

    # ---------- paired comparison ----------
    def paired_test(self, baseline: List[float], proposed: List[float],
                    design_hint: Optional[str] = None) -> Dict[str, Any]:
        """Pick a test based on design hint; default = paired t-test.

        design_hint:
            "paired_normal"     -> paired t-test
            "paired_nonnormal"  -> Wilcoxon signed-rank
            "unpaired"          -> Welch's t-test
            "cv_folds"          -> paired t-test on fold scores (correct for CV)
            None                -> auto: Shapiro-Wilk on diffs
        """
        b = np.asarray(baseline, dtype=float)
        p = np.asarray(proposed, dtype=float)
        if b.size == 0 or p.size == 0:
            return {"test": "none", "reason": "empty input"}
        if b.size != p.size:
            return self._welch(b, p)

        diffs = p - b
        if design_hint == "paired_nonnormal":
            return self._wilcoxon(b, p, diffs)
        if design_hint == "unpaired":
            return self._welch(b, p)
        if design_hint in ("paired_normal", "cv_folds", None):
            if diffs.size >= 3:
                try:
                    _, p_sw = stats.shapiro(diffs)
                    if p_sw < 0.05:
                        return self._wilcoxon(b, p, diffs)
                except Exception:
                    pass
            return self._paired_t(b, p, diffs)
        return self._paired_t(b, p, diffs)

    def _paired_t(self, b, p, diffs) -> Dict[str, Any]:
        t, pval = stats.ttest_rel(p, b)
        return {
            "test": "paired_t",
            "statistic": float(t),
            "p_value": float(pval),
            "mean_diff": float(diffs.mean()),
            "ci_diff": self.bootstrap_ci(list(diffs)),
            "significant": bool(pval < 0.05),
            "n_pairs": int(b.size),
        }

    def _wilcoxon(self, b, p, diffs) -> Dict[str, Any]:
        try:
            stat, pval = stats.wilcoxon(p, b)
        except Exception as e:
            return {"test": "wilcoxon", "error": str(e)}
        return {
            "test": "wilcoxon_signed_rank",
            "statistic": float(stat),
            "p_value": float(pval),
            "median_diff": float(np.median(diffs)),
            "significant": bool(pval < 0.05),
            "n_pairs": int(b.size),
        }

    def _welch(self, b, p) -> Dict[str, Any]:
        t, pval = stats.ttest_ind(p, b, equal_var=False)
        return {
            "test": "welch_t",
            "statistic": float(t),
            "p_value": float(pval),
            "mean_diff": float(p.mean() - b.mean()),
            "significant": bool(pval < 0.05),
            "n_baseline": int(b.size),
            "n_proposed": int(p.size),
        }

    # ---------- convenience: full report ----------
    def report(self, baseline: List[float], proposed: List[float],
               design_hint: Optional[str] = None) -> Dict[str, Any]:
        return {
            "baseline_ci": self.bootstrap_ci(baseline),
            "proposed_ci": self.bootstrap_ci(proposed),
            "comparison": self.paired_test(baseline, proposed, design_hint),
            "interpretation": self._interpret(baseline, proposed, design_hint),
        }

    def _interpret(self, b, p, hint) -> str:
        if not b or not p:
            return "Insufficient data."
        bm, pm = float(np.mean(b)), float(np.mean(p))
        diff = pm - bm
        direction = "improvement" if diff > 0 else "degradation"
        return (
            f"Proposed model shows a {abs(diff):.4f} {direction} "
            f"(baseline {bm:.4f} → proposed {pm:.4f}). "
            f"Design hint: {hint or 'auto'}. "
            "See comparison.p_value for significance; a single split "
            "with small N is not sufficient evidence of generalisation."
        )