import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, List


class LaTeXCompiler:
    """Compile .tex → .pdf with pdflatex. Validates references and citations."""

    def __init__(self, timeout_s: int = 120):
        self.timeout_s = timeout_s
        self.pdflatex = shutil.which("pdflatex")
        self.bibtex = shutil.which("bibtex")

    def compile(self, tex_path: str) -> Dict[str, Any]:
        tex = Path(tex_path)
        if not tex.exists():
            return {"ok": False, "error": "tex file missing"}
        if not self.pdflatex:
            return {
                "ok": False,
                "error": "pdflatex not installed (install TeX Live or MiKTeX)",
                "pdf_path": None,
            }
        workdir = tex.parent
        stem = tex.stem

        # pass 1
        r1 = self._run([self.pdflatex, "-interaction=nonstopmode",
                        "-halt-on-error", tex.name], workdir)

        # bibtex (best effort)
        if self.bibtex and (workdir / f"{stem}.bib").exists():
            self._run([self.bibtex, stem], workdir)

        # pass 2 + 3 (resolve refs)
        r2 = self._run([self.pdflatex, "-interaction=nonstopmode",
                        "-halt-on-error", tex.name], workdir)
        self._run([self.pdflatex, "-interaction=nonstopmode",
                   "-halt-on-error", tex.name], workdir)

        pdf_path = workdir / f"{stem}.pdf"
        log_path = workdir / f"{stem}.log"

        warnings: List[str] = []
        errors: List[str] = []

        if log_path.exists():
            log = log_path.read_text(errors="replace")
            for m in re.finditer(r"^! (.+)$", log, flags=re.MULTILINE):
                errors.append(m.group(1).strip())
            if "There were undefined references" in log:
                warnings.append("Undefined references (check .bib keys)")
            if "Citation" in log and "undefined" in log:
                warnings.append("Undefined citations")

        return {
            "ok": pdf_path.exists() and not errors,
            "pdf_path": str(pdf_path) if pdf_path.exists() else None,
            "log_tail": (r2.get("stderr") or "")[-3000:],
            "warnings": warnings,
            "errors": errors,
        }

    def _run(self, cmd: List[str], cwd: Path) -> Dict[str, Any]:
        try:
            r = subprocess.run(
                cmd, cwd=str(cwd), capture_output=True,
                timeout=self.timeout_s, text=True,
            )
            return {"rc": r.returncode, "stdout": r.stdout, "stderr": r.stderr}
        except Exception as e:
            return {"rc": -1, "stdout": "", "stderr": str(e)}