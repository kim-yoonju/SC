"""utils_math.py — 수학 정답 추출 및 판정 유틸리티."""

import re

from generate_utils import _extract_gsm8k_answer


def _normalize_latex(s: str) -> str:
    s = s.strip().replace(" ", "")
    s = re.sub(r"\\[,;!]|~", "", s)          # thin/thick spaces: \, \; \! ~
    s = re.sub(r"\\dfrac|\\tfrac", r"\\frac", s)
    s = re.sub(r"\\text\{([^}]*)\}|\\mathrm\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\\(|\\\)", "", s)
    s = re.sub(r"\\left|\\right|[()]", "", s)
    s = re.sub(r"\\mathbb\{Z\}_\{(\d+)\}", r"\\mathbb{Z}/\1\\mathbb{Z}", s)
    s = re.sub(r"\\mathbb\{Z\}_(\d+)", r"\\mathbb{Z}/\1\\mathbb{Z}", s)
    s = re.sub(r"\bC_\{?(\d+)\}?", r"\\mathbb{Z}/\1\\mathbb{Z}", s)
    s = re.sub(r"\bZ_\{?(\d+)\}?", r"\\mathbb{Z}/\1\\mathbb{Z}", s)
    s = re.sub(r"\\frac([^{\\])([^{\\])", r"\\frac{\1}{\2}", s)
    s = re.sub(r"\\frac([^{\\])\{", r"\\frac{\1}{", s)
    s = re.sub(r"\\frac\{([^}]*)\}([^{\\])", r"\\frac{\1}{\2}", s)
    s = re.sub(r"\\sqrt([^{\s\\])", r"\\sqrt{\1}", s)  # \sqrt3 → \sqrt{3}
    # \chi^2(n) and \chi^2_{(n)} → \chi^2_{n}  (must run before paren removal)
    s = re.sub(r"\\chi\^2_?\{?\(?(\d+)\)?\}?", r"\\chi^2_{\1}", s)
    # \mathbb{F}_{p^k} → \mathbb{F}_{p^k evaluated}  e.g. F_{2^3} → F_8
    def _eval_ff(m: re.Match) -> str:
        try: return f"\\mathbb{{F}}_{int(m.group(1)) ** int(m.group(2))}"
        except: return m.group(0)
    s = re.sub(r"\\mathbb\{F\}_\{(\d+)\^(\d+)\}", _eval_ff, s)
    return s


_BOOL_NORM: dict[str, str] = {
    "true": "yes", "false": "no",
    "yes": "yes",  "no": "no",
}

def _normalize_bool(s: str) -> str | None:
    return _BOOL_NORM.get(s.strip().lower())


_DEG_RE = re.compile(
    r"^([+-]?\d+(?:\.\d+)?)\s*\^?\s*(?:\\circ|°|\\degree)$"
)

def _parse_angle_rad(s: str) -> float | None:
    import math
    s = s.strip().replace(" ", "")
    m = _DEG_RE.match(s)
    if m:
        return float(m.group(1)) * math.pi / 180
    m = re.match(r"^\\frac\{([+-]?\d*(?:\.\d+)?)\\pi\}\{([+-]?\d+(?:\.\d+)?)\}$", s)
    if m:
        num = float(m.group(1)) if m.group(1) not in ("", "+", "-") else (1.0 if m.group(1) != "-" else -1.0)
        return num * math.pi / float(m.group(2))
    m = re.match(r"^\\frac([+-]?\d*(?:\.\d+)?)\\pi\{([+-]?\d+(?:\.\d+)?)\}$", s)
    if m:
        num = float(m.group(1)) if m.group(1) not in ("", "+", "-") else (1.0 if m.group(1) != "-" else -1.0)
        return num * math.pi / float(m.group(2))
    m = re.match(r"^([+-]?\d*(?:\.\d+)?)\\pi(?:/([+-]?\d+(?:\.\d+)?))?$", s)
    if m:
        num = float(m.group(1)) if m.group(1) not in ("", "+", "-") else (1.0 if m.group(1) != "-" else -1.0)
        denom = float(m.group(2)) if m.group(2) else 1.0
        return num * math.pi / denom
    return None


def _angle_equal(a: str, b: str) -> bool:
    import math
    ra, rb = _parse_angle_rad(a), _parse_angle_rad(b)
    if ra is None or rb is None:
        return False
    return abs(ra - rb) < 1e-9


def extract_boxed(text: str, is_gsm8k: bool = False) -> str | None:
    if not text:
        return None
    if not is_gsm8k:
        marker = r"\boxed{"
        pos = text.rfind(marker)
        if pos != -1:
            start, depth = pos + len(marker), 1
            for i in range(start, len(text)):
                if text[i] == "{": depth += 1
                elif text[i] == "}": depth -= 1
                if depth == 0: return text[start:i].strip()
    m = None
    for match in re.finditer(r"####\s*(.+)", text):
        m = match
    if m:
        return m.group(1).strip().replace(",", "")
    if is_gsm8k:
        return None
    m = None
    for match in re.finditer(r"###\s*(.+)", text):
        m = match
    if m:
        return m.group(1).strip()
    return None


_KNOWN_EQUIV: dict[str, str] = {
    r"\mathfrak{c}": r"2^{\aleph_0}",
}
_KNOWN_EQUIV.update({v: k for k, v in list(_KNOWN_EQUIV.items())})

def _canonicalize(s: str) -> str:
    for variant, canonical in _KNOWN_EQUIV.items():
        s = s.replace(variant.replace(" ", ""), canonical.replace(" ", ""))
    return s


def _latex2sympy_equal(a: str, b: str) -> bool:
    try:
        from latex2sympy2_extended import latex2sympy
        from sympy import simplify, Matrix
        la, lb = latex2sympy(a), latex2sympy(b)
        if isinstance(la, Matrix) or isinstance(lb, Matrix):
            return la == lb
        diff = simplify(la - lb)
        if diff == 0:
            return True
        try:
            return abs(float(diff)) < 1e-9
        except Exception:
            return False
    except Exception:
        return False


def _polynomial_form_equal(a: str, b: str) -> bool:
    try:
        from latex2sympy2_extended import latex2sympy
        from sympy import Poly, symbols
        x = symbols('x')
        pa = Poly(latex2sympy(a), x)
        pb = Poly(latex2sympy(b), x)
        if pa.degree() != pb.degree() or pa.degree() < 1:
            return False
        has_numeric_match = False
        for ci, cj in zip(pa.all_coeffs(), pb.all_coeffs()):
            if ci.is_number and cj.is_number:
                if ci != cj:
                    return False
                has_numeric_match = True
            elif ci.is_number != cj.is_number:
                return False
        return has_numeric_match
    except Exception:
        return False


def _numeric_approx_equal(a: str, b: str, rel_tol: float = 2e-3) -> bool:
    try:
        from latex2sympy2_extended import latex2sympy
        import sympy
        # Replace standalone imaginary unit i → I so latex2sympy treats it as √(-1)
        def _sub_imag(s: str) -> str:
            return re.sub(r"(?<=[0-9})])\s*i\b|(?<=\s)i\b|^i\b", "I", s)
        va = sympy.N(latex2sympy(_sub_imag(a)), 15)
        vb = sympy.N(latex2sympy(_sub_imag(b)), 15)
        va = va.subs(sympy.Symbol("i"), sympy.I)
        vb = vb.subs(sympy.Symbol("i"), sympy.I)
        va, vb = sympy.N(va, 15), sympy.N(vb, 15)
        ca = complex(float(sympy.re(va)), float(sympy.im(va)))
        cb = complex(float(sympy.re(vb)), float(sympy.im(vb)))
        if ca == cb == 0:
            return True
        return abs(ca - cb) / max(abs(ca), abs(cb), 1e-12) < rel_tol
    except Exception:
        return False


def _matrix_elements(s: str) -> list[str] | None:
    m = re.search(r"\\begin\{[pbvBsS]?matrix\}(.*?)\\end\{[pbvBsS]?matrix\}", s, re.DOTALL)
    if not m:
        return None
    inner = m.group(1)
    elements = []
    for row in re.split(r"\\\\", inner):
        for elem in re.split(r"&", row):
            elem = elem.strip()
            if elem:
                elements.append(elem)
    return elements or None


def _matrix_equal(a: str, b: str) -> bool:
    ea, eb = _matrix_elements(a), _matrix_elements(b)
    if ea is None or eb is None or len(ea) != len(eb):
        return False
    for x, y in zip(ea, eb):
        nx, ny = x.replace(" ", ""), y.replace(" ", "")
        if nx == ny:
            continue
        if _normalize_latex(nx) == _normalize_latex(ny):
            continue
        if not _numeric_approx_equal(nx, ny):
            return False
    return True


def _extract_approx_value(s: str) -> str | None:
    m = re.search(r"\\approx\s*([+-]?\d+(?:\.\d+)?)", s)
    return m.group(1) if m else None


def _normalize_pred(pred: str) -> list[str]:
    candidates = [pred]

    def _add(s: str) -> None:
        s = s.strip().rstrip(".,")
        if s and s not in candidates:
            candidates.append(s)

    approx = _extract_approx_value(pred)
    if approx:
        _add(approx)
    # Extract value before trailing \text{...} qualifier: "-1 \text{ is an eigenvalue of } T" → "-1"
    m = re.match(r"^(.+?)\s*\\text\{", pred)
    if m:
        _add(m.group(1).strip())
    m = re.search(r"=\s*(.+?)\s*$", pred.replace(" ", ""))
    if m:
        _add(m.group(1))
    # Extract value after = but before \;\text{for...} qualifier: "f(u)=2u\;\text{for }0≤u≤1" → "2u"
    m = re.search(r"=\s*(.+?)(?:\s*\\[,;]\s*\\text\{(?:for|where|when|with|and)\b|\s*,\s*\\text\{(?:for|where|when|with|and)\b)", pred)
    if m:
        _add(m.group(1).strip())
    m = re.search(r"=\s*([^\\,]+?)\s*(?:\\text|,\s*(?:\\text|\\quad))", pred)
    if m:
        _add(m.group(1).strip())
    m = re.match(r"\\text\{(yes|no|true|false)[^}]*\}", pred, re.IGNORECASE)
    if m:
        _add(m.group(1).capitalize())
    m = re.match(r"\((.+?),\s*(?:\\cdot|\\times|\\circ|\+|\-)\s*\)", pred.replace(" ", ""))
    if m:
        _add(m.group(1))
    m = re.search(r",\s*(?:\\quad|\\text\{\s*(?:where|for|with|and)\b)", pred)
    if m:
        _add(pred[:m.start()])
    stripped = re.sub(r"\\text\{[^}]*\}", "", pred).strip().rstrip(".,")
    if stripped:
        _add(stripped)
    # Extract expression after the last \text{...} block: "\text{...homeomorphic to }M\times[0,1]." → "M\times[0,1]"
    parts = re.split(r"\\text\{[^}]*\}", pred)
    if len(parts) > 1:
        last = parts[-1].strip().rstrip(".,").strip()
        if last:
            _add(last)
    return candidates


def _normalize_gold(gold: str) -> list[str]:
    candidates = [gold]
    stripped = re.sub(r"\\text\{[^}]*\}", "", gold).strip().rstrip(".,")
    if stripped and stripped not in candidates:
        candidates.append(stripped)
    # Extract RHS from gold equation "A = B" → also try "B"
    m = re.search(r"=\s*(.+?)\s*$", gold.replace(" ", ""))
    if m:
        rhs = m.group(1).strip().rstrip(".,")
        if rhs and rhs not in candidates:
            candidates.append(rhs)
    return candidates


def _times_sorted(s: str) -> str:
    parts = re.split(r"\\times", s)
    if len(parts) < 2:
        return s
    return r"\times".join(sorted(p.strip() for p in parts))


_INFINITY_RE = re.compile(
    r"^(?:\\infty|[+]?\\infty"
    r"|\\text\{(?:diverges?|divergent|infinite?|infinity|\\infty|thelimitdoesnotexist[^}]*)\}"
    r"|diverges?|divergent|infinite?|infinity"
    r"|thelimitdoesnotexist(?:diverges?)?)$",
    re.IGNORECASE,
)
_NEG_INFINITY_RE = re.compile(
    r"^(?:-\\infty|\\text\{-\\infty\}|-infinity|-infinite?)$",
    re.IGNORECASE,
)

def _normalize_infinity(s: str) -> str | None:
    s = s.replace(" ", "")
    if _INFINITY_RE.match(s):
        return "\\infty"
    if _NEG_INFINITY_RE.match(s):
        return "-\\infty"
    return None


def _parse_interval(s: str) -> tuple | None:
    """Parse interval notation like (-\\infty,-1) or [0,1] into (lo, hi, lo_open, hi_open)."""
    s = re.sub(r"\\[,;!]|~| ", "", s)
    s = re.sub(r"^[a-zA-Z\\]+\\in", "", s)  # strip "x\in" prefix
    m = re.match(r"^([\(\[])(.+),(.+)([\)\]])$", s)
    if not m:
        return None
    lo_br, lo_val, hi_val, hi_br = m.groups()
    return (lo_val, hi_val, lo_br == "(", hi_br == ")")


def _parse_inequality_as_interval(s: str) -> tuple | None:
    """Parse inequalities like x<5, 0<x<=2, alpha<-1 into (lo, hi, lo_open, hi_open)."""
    s = re.sub(r"\\[,;!]|~| ", "", s)
    s = s.replace("\\leq", "≤").replace("\\geq", "≥").replace("\\le", "≤").replace("\\ge", "≥")
    s = s.replace("<=", "≤").replace(">=", "≥")
    # Variable pattern: plain letter or LaTeX command like \alpha, \beta
    _var_pat = r"(?:[a-zA-Z][a-zA-Z_]*|\\[a-zA-Z]+)"
    # Double inequality: lo OP var OP hi
    m = re.match(rf"^([^<>≤≥]+)([<≤])({_var_pat})([<≤])([^<>≤≥]+)$", s)
    if m:
        lo, op1, _var, op2, hi = m.groups()
        return (lo, hi, op1 == "<", op2 == "<")
    # Single: var < hi  or  var <= hi
    m = re.match(rf"^({_var_pat})([<>≤≥])(.+)$", s)
    if m:
        _var, op, val = m.groups()
        if op in ("<", "≤"):
            return ("-\\infty", val, True, op == "<")
        else:
            return (val, "\\infty", op == ">", True)
    return None


def _interval_bounds_equal(a: str, b: str) -> bool:
    a, b = a.strip(), b.strip()
    if a == b:
        return True
    ia, ib = _normalize_infinity(a), _normalize_infinity(b)
    if ia is not None and ib is not None:
        return ia == ib
    return _numeric_approx_equal(a, b)


def _interval_equal(a: str, b: str) -> bool:
    """Check if two interval/inequality expressions represent the same set."""
    pa = _parse_interval(a) or _parse_inequality_as_interval(a)
    pb = _parse_interval(b) or _parse_inequality_as_interval(b)
    if pa is None or pb is None:
        return False
    lo_a, hi_a, lo_open_a, hi_open_a = pa
    lo_b, hi_b, lo_open_b, hi_open_b = pb
    if lo_open_a != lo_open_b or hi_open_a != hi_open_b:
        return False
    return _interval_bounds_equal(lo_a, lo_b) and _interval_bounds_equal(hi_a, hi_b)


def _compare_single(pred: str, gold: str) -> bool:
    pred, gold = pred.replace(" ", ""), gold.replace(" ", "")
    if pred == gold: return True
    if pred.lower() == gold.lower(): return True
    pi, gi = _normalize_infinity(pred), _normalize_infinity(gold)
    if pi is not None and gi is not None and pi == gi: return True
    pb, gb = _normalize_bool(pred), _normalize_bool(gold)
    if pb is not None and gb is not None and pb == gb: return True
    try:
        if abs(float(pred) - float(gold)) < 1e-6: return True
    except ValueError:
        pass
    np_, ng = _normalize_latex(pred), _normalize_latex(gold)
    if np_ == ng: return True
    if _times_sorted(np_) == _times_sorted(ng): return True
    if _canonicalize(pred) == _canonicalize(gold): return True
    if _angle_equal(pred, gold): return True
    if _matrix_equal(pred, gold): return True
    if _interval_equal(pred, gold): return True
    if _latex2sympy_equal(pred, gold): return True
    if _polynomial_form_equal(pred, gold): return True
    return _numeric_approx_equal(pred, gold)


def _extract_mc_options(problem: str) -> dict[str, str]:
    options = {}
    pattern = re.compile(
        r'[\(\[]?([A-F])[\)\]\.]\)?[\s\)]*'
        r'((?:(?![\(\[]?[A-F][\)\]\.]|\Z).)+)',
        re.DOTALL
    )
    for m in pattern.finditer(problem):
        key = m.group(1).upper()
        val = m.group(2).strip().rstrip(' \n,;')
        if val:
            options[key] = val
    return options


def check_solved(step_text: str, gold_answer, is_gsm8k: bool = False,
                 problem: str = "") -> bool:
    pred_raw = extract_boxed(step_text, is_gsm8k=is_gsm8k)
    if not pred_raw:
        return False
    gold_str = str(gold_answer).strip()
    if "####" in gold_str:
        gold_str = _extract_gsm8k_answer(gold_str)
    if problem and re.fullmatch(r"[A-F]", gold_str):
        options = _extract_mc_options(problem)
        if gold_str in options:
            option_val = options[gold_str].strip()
            for pred_cand in _normalize_pred(pred_raw):
                for gold_cand in _normalize_gold(option_val):
                    if _compare_single(pred_cand, gold_cand):
                        return True
    for pred_cand in _normalize_pred(pred_raw):
        for gold_cand in _normalize_gold(gold_str):
            if _compare_single(pred_cand, gold_cand):
                return True
    return False


def has_boxed(text: str) -> bool:
    return bool(re.search(r"\\boxed\{", text))
