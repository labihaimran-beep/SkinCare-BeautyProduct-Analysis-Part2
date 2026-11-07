"""CSV and Excel export helpers. Text is sanitised: spreadsheet formulas are neutralised and
characters Excel rejects are removed."""
from __future__ import annotations
import io
import pandas as pd
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE

EXCEL_MAX_ROWS = 1_000_000
EXCEL_MAX_CHARS = 32_000          # Excel cell limit is 32,767
_FORMULA_START = ("=", "+", "-", "@", "\t", "\r")


def _safe_cell(v):
    if isinstance(v, str):
        v = ILLEGAL_CHARACTERS_RE.sub(" ", v)
        if v.startswith(_FORMULA_START):
            v = "'" + v        # CSV/Excel formula injection guard (review text is untrusted)
    return v


def unsanitize_text(v):
    """Undo the formula guard so exported text can be matched back to the original (e.g. labelled samples)."""
    if isinstance(v, str) and v.startswith("'") and v[1:].startswith(_FORMULA_START):
        return v[1:]
    return v


def sanitize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object or pd.api.types.is_string_dtype(out[c]):
            out[c] = out[c].map(_safe_cell)
    return out


def csv_bytes(df: pd.DataFrame) -> bytes:
    return sanitize(df).to_csv(index=False).encode("utf-8-sig")


def excel_bytes(sheets: dict) -> bytes:
    """Write {sheet_name: DataFrame} to an .xlsx; sheet names trimmed to 31 chars, rows and cell length capped."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        for name, df in sheets.items():
            if df is None:
                continue
            d = sanitize(df.head(EXCEL_MAX_ROWS))
            for c in d.columns:
                if d[c].dtype == object or pd.api.types.is_string_dtype(d[c]):
                    d[c] = d[c].map(lambda v: v[:EXCEL_MAX_CHARS] if isinstance(v, str) else v)
            d.to_excel(xw, sheet_name=name[:31], index=False)
    return buf.getvalue()
