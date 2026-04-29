"""Run notebooks and extract every PNG output to outputs/figures/<name>/v<i>/cell_<j>.png.

Each entry in ``NOTEBOOKS`` is a ``(notebook_path, params)`` tuple. ``params``
is a dict of variable overrides; they are injected via a code cell placed
right after the notebook's ``parameters``-tagged cell (papermill convention).
When the same notebook appears multiple times, outputs go into ``v0``,
``v1``, ... in order of appearance.

Cell-by-cell execution: on the first error, the notebook is flagged with the
failing cell index and the rest of its cells are skipped.

Usage:
    python extract_notebook_figures.py
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
    ## FPCA on CelebA
    ("notebooks/fpca_celeba.ipynb", {
        "ASSET_FOLDER": "celeba-fpca-first-1024",
        "COMPILER_MODE": "latex",
        "CHECKPOINT_DIR": "../outputs/checkpoints/wandb-i28fso8p",
        "CHECKPOINT_FILE": "step_3480.pt",
    }),
    ("notebooks/fpca_celeba.ipynb", {
        "ASSET_FOLDER": "celeba-fpca-first-64",
        "COMPILER_MODE": "normal",
        "CHECKPOINT_DIR": "../outputs/checkpoints/wandb-n4tfx41g",
        "CHECKPOINT_FILE": "step_5100.pt",
    }),
    ("notebooks/fpca_celeba.ipynb", {
        "ASSET_FOLDER": "celeba-fpca-first-6",
        "COMPILER_MODE": "normal",
        "CHECKPOINT_DIR": "../outputs/checkpoints/wandb-4g6w8iu7",
        "CHECKPOINT_FILE": "step_2160.pt",
    }),
    ("notebooks/fpca_celeba.ipynb", {
        "ASSET_FOLDER": "celeba-fpca-shuffled-1024",
        "COMPILER_MODE": "latex",
        "CHECKPOINT_DIR": "../outputs/checkpoints/wandb-at6x2m3n",
        "CHECKPOINT_FILE": "step_1540.pt",
    }),
    # Volume Preserving
    # ("notebooks/volume_preserving.ipynb", {
    #     "ASSET_FOLDER": "taylor-green",
    #     "COMPILER_MODE": "latex",
    #     "CKPT_DIR": "outputs/checkpoints/wandb-kay59tlz",
    #     "CKPT_FILE": "step_12486.pt",
    # }),
    # Volume Preserving (Mirroring)
    # ("notebooks/volume_preserving.ipynb", {
    #     "ASSET_FOLDER": "mirrors",
    #     "COMPILER_MODE": "latex",
    #     "CKPT_DIR": "outputs/checkpoints/wandb-2ymvx2m8",
    #     "CKPT_FILE": "step_3650.pt",
    # }),
]

# Overleaf path for sync
OUTPUT_DIR = Path("../overleaf/Learning-Basis-Functions/notebook_generations")
DPI = 200           # resolution for any rasterised content inside the PDFs
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
import matplotlib as _mpl
_mpl.rcParams.update({{
    'text.usetex':         True,
    'font.family':         'serif',
    'text.latex.preamble': {_LATEX_PREAMBLE!r},
    'pdf.fonttype':        42,
    'figure.dpi':          {DPI},
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
    set_matplotlib_formats('pdf')
except Exception:
    pass

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
        import sys as _sys
        _sys.stderr.write(
            f'[extract_notebook_figures] unmapped Unicode {{c!r}} (U+{{ord(c):04X}}) '
            f'dropped from LaTeX-bound string — add to _UNICODE_MATH or _UNICODE_TEXT '
            f'if you want it rendered.\\n')
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
    'figure.dpi':          {DPI},
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
    set_matplotlib_formats('pdf')
except Exception:
    pass
"""

# Mode A: full labels / captions, editable PDF.
NORMAL_PREAMBLE = _LATEX_BASE            # latex mode
NORMAL_PREAMBLE_NO_LATEX = _NORMAL_BASE  # normal mode

# Mode B: all text and axes stripped, transparent background, editable PDF.
# Patches IPython's display formatter for Figure so every rendering path
# (plt.show, display(fig), auto-display, flush_figures) goes through
# _clean_figure before the PDF bytes are produced.
_CLEAN_EXTRA = """\

_mpl.rcParams['savefig.transparent'] = True

def _clean_figure(_fig):
    _fig.patch.set_alpha(0.0)
    if getattr(_fig, '_suptitle', None) is not None:
        _fig._suptitle.set_visible(False)
    for _txt in list(_fig.texts):
        _txt.set_visible(False)
    for _ax in _fig.get_axes():
        _ax.set_facecolor((0, 0, 0, 0))
        try:
            _ax.set_title('', loc='left')
            _ax.set_title('', loc='center')
            _ax.set_title('', loc='right')
        except Exception:
            _ax.set_title('')
        _ax.set_xlabel('')
        _ax.set_ylabel('')
        if hasattr(_ax, 'set_zlabel'):
            try:
                _ax.set_zlabel('')
            except Exception:
                pass
        _ax.axis('off')
        _leg = _ax.get_legend()
        if _leg is not None:
            _leg.remove()
        for _txt in list(_ax.texts):
            _txt.set_visible(False)
        try:
            _ax.set_xticklabels([])
            _ax.set_yticklabels([])
        except Exception:
            pass

import matplotlib.pyplot as _plt_init
del _plt_init

try:
    from IPython import get_ipython as _get_ipython
    from matplotlib.figure import Figure as _Figure_cls
    _ip = _get_ipython()
    if _ip is not None:
        for _mime in ('application/pdf', 'image/png', 'image/jpeg', 'image/svg+xml'):
            _formatter = _ip.display_formatter.formatters.get(_mime)
            if _formatter is None:
                continue
            try:
                _orig = _formatter.lookup_by_type(_Figure_cls)
            except KeyError:
                continue
            def _make_wrapped(_orig_func):
                def _wrapped(_fig):
                    _clean_figure(_fig)
                    return _orig_func(_fig)
                return _wrapped
            _formatter.for_type(_Figure_cls, _make_wrapped(_orig))
except Exception as _patch_err:
    import warnings
    warnings.warn('clean-mode formatter patch failed: ' + str(_patch_err))
"""

CLEAN_PREAMBLE = _LATEX_BASE + _CLEAN_EXTRA            # latex mode
CLEAN_PREAMBLE_NO_LATEX = _NORMAL_BASE + _CLEAN_EXTRA  # normal mode


def find_parameters_cell(nb) -> int | None:
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "code" and "parameters" in cell.metadata.get("tags", []):
            return i
    return None


def render_overrides(params: dict) -> str:
    body = "\n".join(f"{k} = {v!r}" for k, v in params.items())
    return f"# Injected overrides from extract_notebook_figures.py\n{body}\n"


def save_figures_from_cell(cell, target_dir: Path, prefix: str, cell_index: int) -> int:
    """Write every PDF figure in a cell as cell_{prefix}_{cell_index}_{fig_index}.pdf."""
    if cell.cell_type != "code":
        return 0
    written = 0
    for output in cell.get("outputs", []):
        pdf_b64 = output.get("data", {}).get("application/pdf")
        if pdf_b64 is None:
            continue
        path = target_dir / f"cell_{prefix}_{cell_index}_{written}.pdf"
        path.write_bytes(base64.b64decode(pdf_b64))
        written += 1
    return written


def run_notebook(
    nb_path: Path, params: dict, target_dir: Path, prefix: str,
    compiler_mode: str = "latex",
) -> tuple[bool, str | None, int]:
    """Execute ``nb_path`` cell-by-cell and save PDFs with the given prefix.

    prefix="A" uses the normal preamble; prefix="B" uses the clean preamble
    (transparent background, all text/axes/legends stripped).
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

    if compiler_mode == "latex":
        preamble_src = NORMAL_PREAMBLE if prefix == "A" else CLEAN_PREAMBLE
    else:
        preamble_src = NORMAL_PREAMBLE_NO_LATEX if prefix == "A" else CLEAN_PREAMBLE_NO_LATEX
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
                return False, msg, total_figs
            if cell.metadata.get("__figure_extractor_injected__"):
                continue
            total_figs += save_figures_from_cell(cell, target_dir, prefix, user_index)
            user_index += 1

    return True, None, total_figs


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
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

        # Fresh directory shared by both A and B passes.
        if target_dir.exists():
            shutil.rmtree(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        all_ok = True
        total = 0
        for prefix in ("A", "B"):
            print(f"[RUN ] {nb_path} [{folder_name}] [{prefix}]  ->  {target_dir}")
            ok, err, n_figs = run_notebook(nb_path, params, target_dir, prefix, compiler_mode)
            plural = "s" if n_figs != 1 else ""
            total += n_figs
            if ok:
                print(f"[ OK ] {nb_path} [{folder_name}] [{prefix}]  ({n_figs} figure{plural})")
            else:
                print(f"[FAIL] {nb_path} [{folder_name}] [{prefix}]: {err}  ({n_figs} figure{plural} before failure)")
                failures.append((nb_str, folder_name, f"[{prefix}] {err or 'unknown error'}"))
                all_ok = False
                break  # don't run B if A already failed

        if all_ok:
            print(f"[ OK ] {nb_path} [{folder_name}]  ({total} figures total, A+B)")

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
