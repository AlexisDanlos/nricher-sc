import re
from functools import lru_cache

@lru_cache()
def extract_dimensions(text: str):
    """
    Extract dimensions from text as tuple (raw, cm), or (None, None) if no match.
    raw: e.g. "80*40*15cm"
    cm: e.g. "80*40*15"
    """
    # special case for 3D dimensions with letters before numbers e.g. '12x12xh22cm' or '60l x 24l x 148h cm'
    special_3d = re.compile(
        # match 3D with optional dimension letters l/p/h before numbers
        r"(?<!\w)(?:[lph]\s*)?(?P<w>\d+(?:[., ]\d+)?)(?:[lph])?\s*[x×]\s*"
        r"(?:[lph]\s*)?(?P<h>\d+(?:[., ]\d+)?)(?:[lph])?\s*[x×]\s*"
        r"(?:[lph]\s*)?(?P<d>\d+(?:[., ]\d+)?)(?:[lph])?(?:\s*(?P<unit>mm|cm|m))?\b",
        re.IGNORECASE
    )
    m0 = special_3d.search(text)
    if m0:
        # treat as multi-dimension
        w = m0.group('w').replace(',', '.').replace(' ', '.')
        h = m0.group('h').replace(',', '.').replace(' ', '.')
        d = m0.group('d').replace(',', '.').replace(' ', '.')
        unit = (m0.group('unit') or 'cm').lower()
        parts = [w, h, d]
        raw = '*'.join(parts) + unit
        # convert to cm
        factor = {'m': 100.0, 'cm': 1.0, 'mm': 0.1}[unit]
        conv = []
        for x in parts:
            val = round(float(x) * factor, 6)
            conv.append(str(int(val)) if val.is_integer() else str(val).rstrip('0').rstrip('.'))
        return raw, '*'.join(conv)
    pattern = re.compile(
        # ensure not preceded by word char and allow optional dimension letters after w and h
        r'(?<!\w)(?P<w>\d+(?:[., ]\d+)?)(?:[lph])?(?:\s*(?P<unit1>mm|cm|m))?\s*[x×]\s*'
        r'(?P<h>\d+(?:[., ]\d+)?)(?:[lph])?(?:\s*(?P<unit2>mm|cm|m))?'
        r'(?:\s*[x×]\s*(?:[lph]\s*)?(?P<d>\d+(?:[., ]\d+)?)(?:\s*(?P<unit3>mm|cm|m))?)?'
        r'\b',
        re.IGNORECASE
    )
    # find all multidimensional matches and select best by number of parts and sum of values
    matches = list(pattern.finditer(text))
    single = False
    if matches:
        def match_score(m):
            parts_vals = [m.group('w'), m.group('h')]
            if m.group('d'):
                parts_vals.append(m.group('d'))
            nums = [float(p.replace(',', '.').replace(' ', '.')) for p in parts_vals]
            return (len(nums), sum(nums))
        m = max(matches, key=match_score)
    else:
        # pattern pour dimension simple
        single_pattern = re.compile(
            r'(?P<w>\d+(?:[., ]\d+)?)(?:\s*(?P<unit>mm|cm|m))\b',
            re.IGNORECASE
        )
        m = single_pattern.search(text)
        if not m:
            return None, None
        single = True

    # reject 2-part matches without explicit units when indicating power specs like '2x2.5 W'
    if not single and m.group('d') is None and not (m.group('unit1') or m.group('unit2')):
        suffix = text[m.end():]
        if re.match(r"\s*[Ww]\b", suffix):
            return None, None
    # normalize numbers en remplaçant virgule et espace par point
    if single:
        w = m.group('w').replace(',', '.').replace(' ', '.')
        unit = m.group('unit').lower()
        parts = [w]
        raw = f"{w}{unit}"
    else:
        w = m.group('w').replace(',', '.').replace(' ', '.')
        h = m.group('h').replace(',', '.').replace(' ', '.')
        d = m.group('d')
        if d:
            d = d.replace(',', '.').replace(' ', '.')
        # déterminer l'unité : priorité unit3 > unit2 > unit1 > défaut cm
        unit = (m.group('unit3') or m.group('unit2') or m.group('unit1') or 'cm').lower()
        parts = [w, h] + ([d] if d else [])
        raw = '*'.join(parts) + unit

    # convert to cm
    factor = {'m': 100.0, 'cm': 1.0, 'mm': 0.1}[unit]
    conv = []
    for x in parts:
        val = float(x) * factor
        # éviter les artefacts de flottant
        val = round(val, 6)
        if val.is_integer():
            conv.append(str(int(val)))
        else:
            conv.append(str(val).rstrip('0').rstrip('.'))
    cm = '*'.join(conv)

    return raw, cm
