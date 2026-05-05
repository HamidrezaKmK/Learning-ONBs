"""Batch-execute notebooks and export every figure as an editable PDF for Overleaf.

Usage
-----
    python extract_notebook_figures.py

Edit the ``NOTEBOOKS`` list near the top of this file to control which notebooks
run and what parameters they receive.  Everything else is automatic.


What it does
------------
For each entry in ``NOTEBOOKS`` the script:

1.  Reads the notebook with nbformat and injects two code cells:
      - A *preamble* cell at position 0 (matplotlib rcParams, LaTeX setup,
        safety-net patches — see "Compiler modes" below).
      - An *overrides* cell immediately after the notebook's ``parameters``-
        tagged cell (papermill convention) that assigns the key/value pairs
        from the ``params`` dict, overriding any defaults in the notebook.

2.  Executes the notebook cell-by-cell via nbclient using the
    ``infidictionary`` Jupyter kernel (configurable via ``KERNEL_NAME``).
    On the first ``CellExecutionError`` the rest of the notebook is skipped
    and the failure is recorded.

3.  After each cell, every figure output is decoded from base64 and saved
    under ``ASSET_FOLDER`` as both a vector PDF and a high-fidelity PNG::

        <OUTPUT_DIR>/<notebook-stem>/<ASSET_FOLDER>/cell_<name>_<fig>.pdf
        <OUTPUT_DIR>/<notebook-stem>/<ASSET_FOLDER>/cell_<name>_<fig>.png

    ``<name>`` comes from the cell's ``report:<name>`` tag (set in JupyterLab's
    Property Inspector); cells without a ``report:`` tag fall back to their
    zero-based index.

4.  Errors are reported on stdout and also appended to ``extraction_err_log.log``
    (configurable via ``LOG_FILE``) with full pdflatex output and tracebacks.


Outputs
-------
For each figure produced by a cell, two files are written::

    <OUTPUT_DIR>/<notebook-stem>/<ASSET_FOLDER>/cell_<name>_<fig>.pdf   ← vector
    <OUTPUT_DIR>/<notebook-stem>/<ASSET_FOLDER>/cell_<name>_<fig>.png   ← 300 DPI raster

``<name>`` is the value after ``report:`` in the cell's tags, or the cell's
zero-based index when no ``report:`` tag is present.  ``<fig>`` counts figures
within that cell (zero-based).  Only user cells are counted — the injected
preamble and override cells are excluded.

``OUTPUT_DIR`` defaults to ``../overleaf/Learning-Basis-Functions/notebook_generations``
(sibling Overleaf repo).

``extraction_err_log.log`` is created / overwritten at the start of each run
and accumulates pdflatex error logs, unknown-Unicode warnings, and full
CellExecutionError tracebacks.


Compiler modes  (``COMPILER_MODE`` key in params)
--------------------------------------------------
Each notebook entry can set ``"COMPILER_MODE"`` to one of:

``"latex"`` (default)
    Enables ``text.usetex=True``, Times-roman font family, and the full
    LaTeX safety-net described below.  Produces PDF text that matches the
    NeurIPS paper body font and is editable as live text in Illustrator /
    Affinity Designer.

``"normal"``
    Uses matplotlib's built-in mathtext renderer (``text.usetex=False``).
    No pdflatex dependency.  Useful when a notebook has strings that are
    hard to escape, or when you just want a quick render without LaTeX.


System requirements
-------------------
The script is designed to run inside the ``infidictionary`` conda environment.
That environment contains ``texlive-core`` (conda-forge), but as of the current
package version its Perl helper scripts (``mktexlsr.pl`` etc.) and pre-built
format files (``latex.fmt``) are absent, so its ``pdflatex`` cannot initialise.

The preamble therefore **prepends ``/usr/bin`` to PATH** inside each kernel so
that the system TeX Live installation is used instead.  You need:

- **System TeX Live** installed and reachable at ``/usr/bin/pdflatex``.
  On Ubuntu/Debian::

      sudo apt install texlive-latex-base texlive-fonts-recommended

  Verify with::

      pdflatex --version   # should print TeX Live 2023 or later

- The following LaTeX packages must be available in the system TeX Live
  (all present in ``texlive-latex-base`` / ``texlive-fonts-recommended``):

      times     amsmath     amssymb

- **Python packages** (all in the conda env):  ``nbformat``, ``nbclient``,
  ``matplotlib >= 3.6``, ``matplotlib-inline``.

If you move to a machine where the system TeX Live is at a different path,
update the ``_os.environ['PATH']`` line near the top of ``_LATEX_BASE``.


LaTeX safety net (latex mode only)
-----------------------------------
Notebook strings are written for human readers, not pdflatex.  To avoid
hand-escaping every label the preamble monkey-patches
``matplotlib.texmanager.TexManager._get_tex_source`` so each string is
normalised just before pdflatex sees it.  Three layers:

1.  **String normaliser** — maps ~260 Unicode codepoints (Greek, blackboard
    bold, calligraphic, arrows, relations, sub/superscripts, …) to their LaTeX
    commands; escapes ``# % & ~`` in text mode; tracks ``$...$`` so user-
    written math is passed through unchanged.

2.  **Unknown-Unicode drop** — any non-ASCII codepoint not in the map is
    silently dropped with a one-time warning written to stderr *and* to
    ``extraction_err_log.log``.  Prevents "missing glyph" crashes from rare
    symbols.  To render them, add the codepoint to ``_UNICODE_MATH`` in the
    script and re-run.

3.  **Figure-level fallback** — if pdflatex still crashes (mismatched ``$``,
    broken custom command, etc.) the figure is re-rendered with
    ``text.usetex=False`` (mathtext) so the build never loses a figure.  The
    full pdflatex error log is written to ``extraction_err_log.log``.


Adding a new notebook
---------------------
Append a tuple to ``NOTEBOOKS``::

    ("notebooks/my_notebook.ipynb", {
        "ASSET_FOLDER":   "my-output-folder",   # required; names the output dir
        "COMPILER_MODE":  "latex",               # or "normal"
        "MY_PARAM":       42,                    # any variable the notebook reads
    }),

The notebook must have a code cell tagged ``parameters`` (the first cell that
has ``"parameters"`` in its ``metadata.tags`` list).  The override cell is
inserted immediately after it.  If the tag is missing the run fails with a
clear error message.
"""
from __future__ import annotations

import base64
import shutil
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError


# ── Edit me ─────────────────────────────────────────────────────────────────
NOTEBOOKS: list[tuple[str, dict]] = [
    # 1D PCA with all of the ablations
    # ("notebooks/fpca_1d.ipynb", {
    #     "ASSET_FOLDER": "1d-analysis",
    #     "COMPILER_MODE": "latex",
    #     "CHECKPOINT_DIR":  "../outputs/checkpoints/wandb-5yylqjxg",
    #     "CHECKPOINT_FILE": "step_3520.pt",
    # }),
    # FPCA experiment on INRs
    # ("notebooks/inr_fpca.ipynb", {
    #     "ASSET_FOLDER": "cifar10-inr-zoo",
    #     "COMPILER_MODE": "latex",
    #     "CHECKPOINT_DIR":  "../outputs/checkpoints/wandb-39iw3oae",
    #     "CHECKPOINT_FILE": "step_2400.pt",
    #     "EXP_NAME": "CIFAR10"
    # }),
    # ("notebooks/inr_fpca.ipynb", {
    #     "ASSET_FOLDER": "mnist-inr-zoo",
    #     "COMPILER_MODE": "latex",
    #     "CHECKPOINT_DIR":  "../outputs/checkpoints/wandb-a2rwqque",
    #     "CHECKPOINT_FILE": "step_10000.pt",
    #     "EXP_NAME": "MNIST"
    # }),
    # NTK experiment
    # ("notebooks/ntk.ipynb", {
    #     "ASSET_FOLDER": "ntk-two-moons",
    #     "COMPILER_MODE": "latex",
    #     "INSET_COL": 0,
    #     "INSET_LEFT": 1/5.,
    #     "INSET_RIGHT": 1/2. + 0.1,
    #     "INSET_DOWN": 1/5.,
    #     "INSET_UP": 1/2. + 0.1,
    #     "INSET_TARGET": 64,
    # }),
    # ## FPCA on CelebA
    # ("notebooks/fpca_celeba.ipynb", {
    #     "ASSET_FOLDER": "celeba-fpca-first-1024",
    #     "COMPILER_MODE": "latex",
    #     "CHECKPOINT_DIR": "../outputs/checkpoints/wandb-i28fso8p",
    #     "CHECKPOINT_FILE": "step_3480.pt",
    # }),
    # ("notebooks/fpca_celeba.ipynb", {
    #     "ASSET_FOLDER": "celeba-fpca-first-64",
    #     "COMPILER_MODE": "latex",
    #     "CHECKPOINT_DIR": "../outputs/checkpoints/wandb-n4tfx41g",
    #     "CHECKPOINT_FILE": "step_5100.pt",
    # }),
    # ("notebooks/fpca_celeba.ipynb", {
    #     "ASSET_FOLDER": "celeba-fpca-first-6",
    #     "COMPILER_MODE": "latex",
    #     "CHECKPOINT_DIR": "../outputs/checkpoints/wandb-4g6w8iu7",
    #     "CHECKPOINT_FILE": "step_2160.pt",
    # }),
    # ("notebooks/fpca_celeba.ipynb", {
    #     "ASSET_FOLDER": "celeba-fpca-shuffled-1024",
    #     "COMPILER_MODE": "latex",
    #     "CHECKPOINT_DIR": "../outputs/checkpoints/wandb-at6x2m3n",
    #     "CHECKPOINT_FILE": "step_1540.pt",
    # }),
    # # Volume Preserving
    # ("notebooks/volume_preserving.ipynb", {
    #     "ASSET_FOLDER": "taylor-green",
    #     "COMPILER_MODE": "latex",
    #     "CKPT_DIR": "outputs/checkpoints/wandb-kay59tlz",
    #     "CKPT_FILE": "step_12486.pt",
    # }),
    # Volume Preserving (higher frequency)
    ("notebooks/volume_preserving.ipynb", {
        "ASSET_FOLDER": "taylor-green-high-freq",
        "COMPILER_MODE": "latex",
        "CKPT_DIR": "outputs/checkpoints/wandb-kay59tlz",
        "CKPT_FILE": "step_114132.pt",
    }),
    # # Volume Preserving (Mirroring)
    # ("notebooks/volume_preserving.ipynb", {
    #     "ASSET_FOLDER": "mirrors",
    #     "COMPILER_MODE": "latex",
    #     "CKPT_DIR": "outputs/checkpoints/wandb-2ymvx2m8",
    #     "CKPT_FILE": "step_3650.pt",
    # }),
    # # Concept bases
    # ("notebooks/concept_basis.ipynb", {
    #     "ASSET_FOLDER": "art",
    #     "COMPILER_MODE": "latex",
    #     "NAME": "SDS",
    #     "CHECKPOINT_DIR": "../outputs/checkpoints/wandb-ubqul893",
    #     "CHECKPOINT_FILE": "step_3008.pt",
    # }),
]

# Overleaf path for sync
OUTPUT_DIR = Path("../overleaf/Learning-Basis-Functions/notebook_generations")
LOG_FILE   = Path("extraction_err_log.log")  # pdflatex errors + cell failures
DPI     = 200   # DPI for rasterised content embedded inside PDFs
PNG_DPI = 300   # DPI for exported PNG files
EXECUTION_TIMEOUT = -1  # per-cell, in seconds; -1 = no timeout
KERNEL_NAME = "infidictionary"   # override the notebook's kernelspec; None = use what the .ipynb says
# ────────────────────────────────────────────────────────────────────────────


# Shared LaTeX / NeurIPS font setup injected as the first cell of every run.
# pdf.fonttype=42 stores text as editable TrueType in Illustrator / Affinity.
# DPI only affects raster content embedded inside the vector PDF.
# Notebook display strings are written for human readers, not for LaTeX. With
# text.usetex=True they're piped straight into pdflatex, which crashes on:
#   - raw Unicode (λ, ×, —, ‖, ≥, ²…) — Times has no Greek; em/en-dashes need ---/--;
#   - LaTeX-special chars (#, %, &, ~) appearing literally in text mode.
# Rather than escape every notebook string by hand, we monkey-patch
# matplotlib.texmanager.TexManager so each LaTeX-bound string is normalised
# right before pdflatex sees it. The parser tracks $...$ math mode so user-
# written math is left alone (\lambda, \times, etc. keep working).
_LATEX_PREAMBLE = (
    r'\usepackage{times}'
    r'\usepackage{amsmath}'
    r'\usepackage{amssymb}'
)

_LATEX_BASE = f"""\
import os as _os
# conda's texlive-core is incomplete (missing Perl scripts / format files);
# put /usr/bin first so pdflatex and mktexfmt resolve to the system TeX Live.
_os.environ['PATH'] = '/usr/bin:' + _os.environ.get('PATH', '')
import matplotlib as _mpl
_mpl.rcParams.update({{
    'text.usetex':         True,
    'font.family':         'serif',
    'text.latex.preamble': {_LATEX_PREAMBLE!r},
    'pdf.fonttype':        42,
    'figure.dpi':          {PNG_DPI},
    'font.size':           10,
    'axes.labelsize':      10,
    'axes.titlesize':      10,
    'xtick.labelsize':     8,
    'ytick.labelsize':     8,
    'legend.fontsize':     8,
    'savefig.bbox':        'tight',
}})
try:
    from matplotlib_inline.backend_inline import set_matplotlib_formats
    set_matplotlib_formats('pdf', 'png')
except Exception:
    pass
_LOG_FILE = {str(LOG_FILE.resolve())!r}

# ── LaTeX safety net: normalise strings on the way to pdflatex ──────────────
_UNICODE_MATH = {{
    # Lowercase Greek
    'α': r'\\alpha', 'β': r'\\beta', 'γ': r'\\gamma', 'δ': r'\\delta',
    'ε': r'\\varepsilon', 'ζ': r'\\zeta', 'η': r'\\eta', 'θ': r'\\theta',
    'ι': r'\\iota', 'κ': r'\\kappa', 'λ': r'\\lambda', 'μ': r'\\mu',
    'ν': r'\\nu', 'ξ': r'\\xi', 'ο': 'o', 'π': r'\\pi', 'ρ': r'\\rho',
    'ς': r'\\varsigma', 'σ': r'\\sigma', 'τ': r'\\tau', 'υ': r'\\upsilon',
    'φ': r'\\varphi', 'χ': r'\\chi', 'ψ': r'\\psi', 'ω': r'\\omega',
    # Variant Greek
    'ϑ': r'\\vartheta', 'ϕ': r'\\phi', 'ϖ': r'\\varpi', 'ϱ': r'\\varrho',
    'ϵ': r'\\epsilon', 'ϰ': r'\\varkappa', 'ϐ': r'\\beta',
    # Uppercase Greek (those that don't coincide with Latin letters)
    'Γ': r'\\Gamma', 'Δ': r'\\Delta', 'Θ': r'\\Theta', 'Λ': r'\\Lambda',
    'Ξ': r'\\Xi', 'Π': r'\\Pi', 'Σ': r'\\Sigma', 'Υ': r'\\Upsilon',
    'Φ': r'\\Phi', 'Ψ': r'\\Psi', 'Ω': r'\\Omega',
    # Hebrew
    'ℵ': r'\\aleph', 'ℶ': r'\\beth', 'ℷ': r'\\gimel', 'ℸ': r'\\daleth',
    # Blackboard-bold (number sets)
    'ℕ': r'\\mathbb{{N}}', 'ℤ': r'\\mathbb{{Z}}', 'ℚ': r'\\mathbb{{Q}}',
    'ℝ': r'\\mathbb{{R}}', 'ℂ': r'\\mathbb{{C}}', 'ℙ': r'\\mathbb{{P}}',
    'ℍ': r'\\mathbb{{H}}', '𝔼': r'\\mathbb{{E}}',
    # Script / calligraphic
    'ℒ': r'\\mathcal{{L}}', 'ℳ': r'\\mathcal{{M}}', 'ℋ': r'\\mathcal{{H}}',
    'ℱ': r'\\mathcal{{F}}', 'ℬ': r'\\mathcal{{B}}', 'ℛ': r'\\mathcal{{R}}',
    'ℰ': r'\\mathcal{{E}}', 'ℐ': r'\\mathcal{{I}}', 'ℓ': r'\\ell',
    # Binary operators
    '×': r'\\times', '·': r'\\cdot', '∗': r'\\ast', '⋆': r'\\star',
    '∘': r'\\circ', '∙': r'\\bullet', '±': r'\\pm', '∓': r'\\mp',
    '⊕': r'\\oplus', '⊖': r'\\ominus', '⊗': r'\\otimes', '⊘': r'\\oslash',
    '⊙': r'\\odot', '⊞': r'\\boxplus', '⊟': r'\\boxminus', '⊠': r'\\boxtimes',
    '⊡': r'\\boxdot', '∧': r'\\wedge', '∨': r'\\vee',
    # Big operators
    '⨁': r'\\bigoplus', '⨂': r'\\bigotimes', '⨀': r'\\bigodot',
    '⋁': r'\\bigvee', '⋀': r'\\bigwedge', '⨆': r'\\bigsqcup', '⨄': r'\\biguplus',
    # Relations
    '≤': r'\\leq', '≥': r'\\geq', '≠': r'\\neq', '≈': r'\\approx',
    '≡': r'\\equiv', '≅': r'\\cong', '≃': r'\\simeq', '∼': r'\\sim',
    '≪': r'\\ll', '≫': r'\\gg', '≺': r'\\prec', '≻': r'\\succ',
    '⪯': r'\\preceq', '⪰': r'\\succeq', '∝': r'\\propto', '⊥': r'\\perp',
    '∥': r'\\parallel', '⊢': r'\\vdash', '⊣': r'\\dashv', '⊨': r'\\models',
    # Set / logic
    '∈': r'\\in', '∉': r'\\notin', '∋': r'\\ni',
    '∪': r'\\cup', '∩': r'\\cap', '⊂': r'\\subset', '⊃': r'\\supset',
    '⊆': r'\\subseteq', '⊇': r'\\supseteq', '⊊': r'\\subsetneq', '⊋': r'\\supsetneq',
    '∅': r'\\emptyset', '∀': r'\\forall', '∃': r'\\exists', '∄': r'\\nexists',
    '¬': r'\\neg', '⊤': r'\\top', '⟂': r'\\perp',
    # Calculus / analysis
    '∞': r'\\infty', '∂': r'\\partial', '∇': r'\\nabla',
    '∫': r'\\int', '∬': r'\\iint', '∭': r'\\iiint',
    '∮': r'\\oint', '∯': r'\\oiint', '∰': r'\\oiiint',
    '∑': r'\\sum', '∏': r'\\prod', '∐': r'\\coprod',
    # Arrows
    '→': r'\\rightarrow', '←': r'\\leftarrow', '↔': r'\\leftrightarrow',
    '↑': r'\\uparrow', '↓': r'\\downarrow', '↕': r'\\updownarrow',
    '⇒': r'\\Rightarrow', '⇐': r'\\Leftarrow', '⇔': r'\\Leftrightarrow',
    '⇑': r'\\Uparrow', '⇓': r'\\Downarrow', '⇕': r'\\Updownarrow',
    '⟶': r'\\longrightarrow', '⟵': r'\\longleftarrow',
    '⟷': r'\\longleftrightarrow', '⟹': r'\\Longrightarrow',
    '⟸': r'\\Longleftarrow', '⟺': r'\\Longleftrightarrow',
    '↦': r'\\mapsto', '⟼': r'\\longmapsto', '↪': r'\\hookrightarrow',
    '↩': r'\\hookleftarrow', '⇀': r'\\rightharpoonup', '⇁': r'\\rightharpoondown',
    # Brackets / norms / dots
    '‖': r'\\|', '⟨': r'\\langle', '⟩': r'\\rangle',
    '⌈': r'\\lceil', '⌉': r'\\rceil', '⌊': r'\\lfloor', '⌋': r'\\rfloor',
    '…': r'\\ldots', '⋯': r'\\cdots', '⋮': r'\\vdots', '⋱': r'\\ddots',
    # Geometric shapes (often used as markers)
    '△': r'\\triangle', '▽': r'\\bigtriangledown', '◊': r'\\diamond',
    '□': r'\\square', '■': r'\\blacksquare', '○': r'\\circ',
    # Sub/superscripts — digits
    '⁰': '^{{0}}', '¹': '^{{1}}', '²': '^{{2}}', '³': '^{{3}}',
    '⁴': '^{{4}}', '⁵': '^{{5}}', '⁶': '^{{6}}', '⁷': '^{{7}}',
    '⁸': '^{{8}}', '⁹': '^{{9}}', '⁺': '^{{+}}', '⁻': '^{{-}}',
    '⁼': '^{{=}}', '⁽': '^{{(}}', '⁾': '^{{)}}',
    '₀': '_{{0}}', '₁': '_{{1}}', '₂': '_{{2}}', '₃': '_{{3}}',
    '₄': '_{{4}}', '₅': '_{{5}}', '₆': '_{{6}}', '₇': '_{{7}}',
    '₈': '_{{8}}', '₉': '_{{9}}', '₊': '_{{+}}', '₋': '_{{-}}',
    '₌': '_{{=}}', '₍': '_{{(}}', '₎': '_{{)}}',
    # Sub/superscripts — letters (the ones Unicode actually defines)
    'ₐ': '_{{a}}', 'ₑ': '_{{e}}', 'ₕ': '_{{h}}', 'ᵢ': '_{{i}}',
    'ⱼ': '_{{j}}', 'ₖ': '_{{k}}', 'ₗ': '_{{l}}', 'ₘ': '_{{m}}',
    'ₙ': '_{{n}}', 'ₒ': '_{{o}}', 'ₚ': '_{{p}}', 'ᵣ': '_{{r}}',
    'ₛ': '_{{s}}', 'ₜ': '_{{t}}', 'ᵤ': '_{{u}}', 'ᵥ': '_{{v}}', 'ₓ': '_{{x}}',
    'ᵃ': '^{{a}}', 'ᵇ': '^{{b}}', 'ᶜ': '^{{c}}', 'ᵈ': '^{{d}}',
    'ᵉ': '^{{e}}', 'ᶠ': '^{{f}}', 'ᵍ': '^{{g}}', 'ʰ': '^{{h}}',
    'ⁱ': '^{{i}}', 'ʲ': '^{{j}}', 'ᵏ': '^{{k}}', 'ˡ': '^{{l}}',
    'ᵐ': '^{{m}}', 'ⁿ': '^{{n}}', 'ᵒ': '^{{o}}', 'ᵖ': '^{{p}}',
    'ʳ': '^{{r}}', 'ˢ': '^{{s}}', 'ᵗ': '^{{t}}', 'ᵘ': '^{{u}}',
    'ᵛ': '^{{v}}', 'ʷ': '^{{w}}', 'ˣ': '^{{x}}', 'ʸ': '^{{y}}',
    'ᶻ': '^{{z}}',
    # Misc
    '°': r'^{{\\circ}}', '′': r'^{{\\prime}}', '″': r'^{{\\prime\\prime}}',
    'ℏ': r'\\hbar', 'ı': r'\\imath', 'ȷ': r'\\jmath',
    '∠': r'\\angle', '√': r'\\surd',
}}
# Plain-text replacements (no math wrap when in text mode).
# Keep lean — only chars that don't have a sensible math mapping or that we
# explicitly want to render as text (em-dash, smart quotes, etc.). Anything
# requiring extra LaTeX packages (€, ™, ®, …) stays out unless we add the
# package to _LATEX_PREAMBLE.
_UNICODE_TEXT = {{
    '—': '---', '–': '--', '−': '-',
    '“': '``', '”': "''", '‘': '`', '’': "'",
    '\xa0': '~',           # U+00A0 non-breaking space → LaTeX tie
    '\u200b': '',          # U+200B zero-width space → drop
    '\u2009': r'\\,',     # U+2009 thin space
    '\u202f': r'\\,',     # U+202F narrow no-break space
    '\u2002': r'\\enspace{{}}',  # U+2002 en space
    '\u2003': r'\\quad{{}}',     # U+2003 em space
    '§': r'\\S{{}}', '¶': r'\\P{{}}',
}}
_TEXT_SPECIALS = set('#%&~')

_UNKNOWN_UNICODE_SEEN = set()
def _drop_unknown_unicode(c):
    # Any non-ASCII codepoint outside our maps would crash pdflatex.
    # Replace silently with empty; emit a one-time warning per codepoint so
    # the user can extend _UNICODE_MATH / _UNICODE_TEXT if it matters.
    if ord(c) <= 127:
        return c
    if c not in _UNKNOWN_UNICODE_SEEN:
        _UNKNOWN_UNICODE_SEEN.add(c)
        _msg = (f'[extract_notebook_figures] unmapped Unicode {{c!r}} (U+{{ord(c):04X}}) '
                f'dropped from LaTeX-bound string — add to _UNICODE_MATH or _UNICODE_TEXT '
                f'if you want it rendered.\\n')
        import sys as _sys
        _sys.stderr.write(_msg)
        try:
            with open(_LOG_FILE, 'a') as _lf:
                _lf.write(_msg)
        except Exception:
            pass
    return ''

def _normalize_latex_string(s):
    if not s or not isinstance(s, str):
        return s
    out, in_math, i, n = [], False, 0, len(s)
    while i < n:
        c = s[i]
        if c == '\\\\' and i + 1 < n:
            out.append(c); out.append(s[i+1]); i += 2; continue
        if c == '$':
            in_math = not in_math; out.append(c); i += 1; continue
        if in_math:
            if c in _UNICODE_MATH:
                out.append(_UNICODE_MATH[c])
            elif c in _UNICODE_TEXT:
                out.append(_UNICODE_TEXT[c])
            else:
                out.append(_drop_unknown_unicode(c))
        else:
            if c in _TEXT_SPECIALS:
                out.append('\\\\' + c)
            elif c in _UNICODE_TEXT:
                out.append(_UNICODE_TEXT[c])
            elif c in _UNICODE_MATH:
                out.append('$' + _UNICODE_MATH[c] + '$')
            else:
                out.append(_drop_unknown_unicode(c))
        i += 1
    return ''.join(out)

try:
    from matplotlib import texmanager as _tm
    _orig_get_tex_source = _tm.TexManager._get_tex_source  # bound classmethod
    def _patched_get_tex_source(cls, tex, fontsize):
        normalised = _normalize_latex_string(tex)
        try:
            return _orig_get_tex_source(normalised, fontsize)
        except Exception:
            # Last-ditch fallback: strip ALL non-ASCII chars and retry.
            ascii_only = ''.join(c if ord(c) <= 127 else '' for c in normalised)
            return _orig_get_tex_source(ascii_only, fontsize)
    _tm.TexManager._get_tex_source = classmethod(_patched_get_tex_source)

    # Per-figure fallback: if pdflatex still fails (e.g. mismatched $...$, broken
    # custom command), retry the whole figure with text.usetex=False so the run
    # doesn't lose every figure for one bad string. mathtext can render most
    # math fine even without LaTeX.
    #
    # We MUST preserve the original __module__ via functools.wraps — matplotlib's
    # _switch_canvas_and_return_print_method does
    # `print_method.__module__.startswith("matplotlib.")` to recognise native
    # backends; a wrapper with __module__=None breaks the whole PDF pipeline.
    import functools as _functools
    from matplotlib.backends.backend_pdf import FigureCanvasPdf as _CanvasPdf
    _orig_print_pdf = _CanvasPdf.print_pdf
    @_functools.wraps(_orig_print_pdf)
    def _safe_print_pdf(self, *args, **kwargs):
        try:
            return _orig_print_pdf(self, *args, **kwargs)
        except RuntimeError as e:
            if 'latex' not in str(e).lower():
                raise
            import sys as _sys
            _sys.stderr.write(
                '[extract_notebook_figures] pdflatex failed on this figure; '
                'retrying with text.usetex=False (mathtext fallback).\\n')
            try:
                with open(_LOG_FILE, 'a') as _lf:
                    _lf.write('[extract_notebook_figures] pdflatex error:\\n')
                    _lf.write(str(e))
                    _lf.write('\\n' + '-' * 60 + '\\n')
            except Exception:
                pass
            _prev = _mpl.rcParams['text.usetex']
            _mpl.rcParams['text.usetex'] = False
            try:
                return _orig_print_pdf(self, *args, **kwargs)
            finally:
                _mpl.rcParams['text.usetex'] = _prev
    _CanvasPdf.print_pdf = _safe_print_pdf
except Exception as _patch_err:
    import warnings
    warnings.warn('LaTeX text-normaliser patch failed: ' + str(_patch_err))
"""

# Preamble for COMPILER_MODE="normal": no usetex, no LaTeX font wiring.
_NORMAL_BASE = f"""\
import matplotlib as _mpl
_mpl.rcParams.update({{
    'text.usetex':         False,
    'pdf.fonttype':        42,
    'figure.dpi':          {PNG_DPI},
    'font.size':           10,
    'axes.labelsize':      10,
    'axes.titlesize':      10,
    'xtick.labelsize':     8,
    'ytick.labelsize':     8,
    'legend.fontsize':     8,
    'savefig.bbox':        'tight',
}})
try:
    from matplotlib_inline.backend_inline import set_matplotlib_formats
    set_matplotlib_formats('pdf', 'png')
except Exception:
    pass
"""

PREAMBLE          = _LATEX_BASE   # compiler_mode="latex"
PREAMBLE_NO_LATEX = _NORMAL_BASE  # compiler_mode="normal"


def _append_log(text: str) -> None:
    try:
        with LOG_FILE.open("a") as fh:
            fh.write(text)
    except Exception:
        pass


def find_parameters_cell(nb) -> int | None:
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "code" and "parameters" in cell.metadata.get("tags", []):
            return i
    return None


def render_overrides(params: dict) -> str:
    body = "\n".join(f"{k} = {v!r}" for k, v in params.items())
    return f"# Injected overrides from extract_notebook_figures.py\n{body}\n"


def save_figures_from_cell(cell, target_dir: Path, cell_index: int) -> int:
    """Write every PDF figure in a cell.

    Filename: ``cell_<label>_<fig>.pdf`` when ``cell.metadata["label"]`` is
    set, otherwise ``cell_<cell_index>_<fig>.pdf``.
    """
    if cell.cell_type != "code":
        return 0
    report_tag = next((t for t in cell.metadata.get("tags", []) if t.startswith("report:")), None)
    prefix = report_tag[len("report:"):] if report_tag else str(cell_index)
    written = 0
    for output in cell.get("outputs", []):
        data = output.get("data", {})
        pdf_b64 = data.get("application/pdf")
        png_b64 = data.get("image/png")
        if pdf_b64 is None and png_b64 is None:
            continue
        stem = target_dir / f"cell_{prefix}_{written}"
        if pdf_b64 is not None:
            stem.with_suffix(".pdf").write_bytes(base64.b64decode(pdf_b64))
        if png_b64 is not None:
            stem.with_suffix(".png").write_bytes(base64.b64decode(png_b64))
        written += 1
    return written


def run_notebook(
    nb_path: Path, params: dict, target_dir: Path,
    compiler_mode: str = "latex",
) -> tuple[bool, str | None, int]:
    """Execute ``nb_path`` cell-by-cell and save PDFs.

    compiler_mode="latex" enables Times/usetex rendering; "normal" skips LaTeX.
    Returns (ok, error_msg, figure_count).
    """
    nb = nbformat.read(str(nb_path), as_version=4)

    # Inject parameter overrides right after the user's `parameters` cell.
    if params:
        param_idx = find_parameters_cell(nb)
        if param_idx is None:
            return (
                False,
                f"no `parameters`-tagged cell found; cannot inject {list(params)}",
                0,
            )
        override = nbformat.v4.new_code_cell(source=render_overrides(params))
        override.metadata["__figure_extractor_injected__"] = True
        nb.cells.insert(param_idx + 1, override)

    preamble_src = PREAMBLE if compiler_mode == "latex" else PREAMBLE_NO_LATEX
    preamble = nbformat.v4.new_code_cell(source=preamble_src)
    preamble.metadata["__figure_extractor_injected__"] = True
    nb.cells.insert(0, preamble)

    kernel_name = KERNEL_NAME or nb.metadata.get("kernelspec", {}).get("name", "python3")
    client = NotebookClient(
        nb,
        timeout=EXECUTION_TIMEOUT,
        kernel_name=kernel_name,
        resources={"metadata": {"path": str(nb_path.parent.resolve())}},
        allow_errors=False,
    )

    user_index = 0   # cell index excluding any injected cells
    total_figs = 0

    with client.setup_kernel():
        for raw_index, cell in enumerate(nb.cells):
            try:
                client.execute_cell(cell, raw_index)
            except CellExecutionError as exc:
                first_line = exc.evalue.splitlines()[0] if exc.evalue else ""
                msg = f"cell {user_index}: {exc.ename}: {first_line}"
                _append_log(
                    f"\n=== CellExecutionError in {nb_path} ===\n"
                    f"{msg}\n"
                    f"{exc.evalue or ''}\n"
                    + "-" * 60 + "\n"
                )
                return False, msg, total_figs
            if cell.metadata.get("__figure_extractor_injected__"):
                continue
            total_figs += save_figures_from_cell(cell, target_dir, user_index)
            user_index += 1

    return True, None, total_figs


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    import datetime
    LOG_FILE.write_text(
        f"=== extract_notebook_figures run {datetime.datetime.now().isoformat()} ===\n\n"
    )
    failures: list[tuple[str, int, str]] = []

    seen: dict[str, int] = {}
    for nb_str, params in NOTEBOOKS:
        nb_path = Path(nb_str)
        params = dict(params)  # copy so we don't mutate the global
        asset_folder = params.pop("ASSET_FOLDER", None)
        compiler_mode = params.pop("COMPILER_MODE", "latex")
        if asset_folder is not None:
            folder_name = asset_folder
        else:
            version = seen.get(nb_path.stem, 0)
            folder_name = f"v{version}"
        seen[nb_path.stem] = seen.get(nb_path.stem, 0) + 1
        target_dir = OUTPUT_DIR / nb_path.stem / folder_name

        if not nb_path.exists():
            print(f"[SKIP] {nb_str} [{folder_name}]: file not found")
            failures.append((nb_str, folder_name, "file not found"))
            continue

        # Wipe and recreate target dir so stale figures from earlier runs are removed.
        if target_dir.exists():
            shutil.rmtree(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        print(f"[RUN ] {nb_path} [{folder_name}]  ->  {target_dir}")
        ok, err, n_figs = run_notebook(nb_path, params, target_dir, compiler_mode)
        plural = "s" if n_figs != 1 else ""
        if ok:
            print(f"[ OK ] {nb_path} [{folder_name}]  ({n_figs} figure{plural})")
        else:
            print(f"[FAIL] {nb_path} [{folder_name}]: {err}  ({n_figs} figure{plural} before failure)")
            failures.append((nb_str, folder_name, err or "unknown error"))

    print()
    if failures:
        print("Notebooks that failed:")
        for nb_str, folder, msg in failures:
            print(f"  - {nb_str} [{folder}]: {msg}")
        return 1

    print("All notebooks executed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
