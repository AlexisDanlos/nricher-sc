import re

def extract_dimensions(text):
    """
    Extrait les dimensions du texte avec gestion avancée des cas spéciaux.
    Retourne les dimensions sous format standardisé ou None si aucune trouvée.
    """
    text = str(text)
    original_text = text
    # Count separators
    sep_count = len(re.findall(r"[xX×*]", original_text))
    # Handle 3D decimal triplet with meter suffix: '3 33x2 06x1 17m'
    m_m3d = re.search(r"\b(\d+)\s+(\d+)[xX×*]\s*(\d+)\s+(\d+)[xX×*]\s*(\d+)\s+(\d+)m\b", original_text)
    if m_m3d:
        a, b, c, d, e, f = m_m3d.groups()
        return f"{a}.{b}*{c}.{d}*{e}.{f}"
    # Handle 2D decimal with meter suffix: '0 45 x 2m'
    m_m2d = re.search(r"\b(\d+)\s+(\d+)\s*[xX×*]\s*(\d+)m\b", original_text)
    if m_m2d:
        a, b, c = m_m2d.groups()
        return f"{a}.{b}*{c}"
    # Remove standalone meter suffixes to avoid treating as dimension: e.g., '3m'
    text = re.sub(r"\b(\d+(?:[.,]\d+)?)m\b", "", text)
    # Special: headboard single dimension, matching accented or unaccented 'Tête de lit'
    if re.search(r"\b(?:[tT] ?te|[tT]ête)\s+de\s+lit\b", original_text, re.IGNORECASE):
        # look for number with unit
        m_head = re.search(r"\b(\d+(?:[.,]\d+)?)\s*(?:cm|mm|m)\b", original_text, re.IGNORECASE)
        if m_head:
            return m_head.group(1).replace(',', '.')
        # fallback to any number
        m_head2 = re.search(r"\b(\d+)\b", original_text)
        if m_head2:
            return m_head2.group(1)
    # Quick exit: skip texts without 'x' separators unless special keywords
    if sep_count == 0 and not re.search(r"\b(Surmatelas|prot[eéè]ge matelas|dimensions|Matelas)\b", original_text, re.IGNORECASE):
        return None
    # Filter magnification patterns like '1x 2x 3x' to avoid false positives
    if re.search(r"\b\d+x\s+\d+x\s+\d+x\b", original_text):
        return None
    # Special case: drap housse (sheet cover) dimensions
    if re.search(r"\bdrap\s+housse\b", original_text, re.IGNORECASE):
        m_dh = re.search(r"(\d+(?:[.,]\d+)?)\s*[xX×*]\s*(\d+(?:[.,]\d+)?)(?=\s*(?:\d+(?:[.,]\d+)?\s*)?cm\b)", original_text)
        if m_dh:
            d1 = m_dh.group(1).replace(',', '.')
            d2 = m_dh.group(2).replace(',', '.')
            return f"{d1}*{d2}"
    # Special case: housse de couette (duvet cover) dimensions, ignore thickness
    if re.search(r"\bhousse\s+de\s+couette\b", original_text, re.IGNORECASE):
        # find all two-part dimensions and pick the first reasonable pair
        dims = re.findall(r"(\d+(?:[.,]\d+)?)\s*[xX×*]\s*(\d+(?:[.,]\d+)?)", original_text)
        for d1, d2 in dims:
            d1f = d1.replace(',', '.')
            d2f = d2.replace(',', '.')
            try:
                # choose dimensions likely of a duvet cover
                if float(d1f) >= 50 and float(d2f) >= 50:
                    return f"{d1f}*{d2f}"
            except:
                # fallback for non-numeric matches
                return f"{d1f}*{d2f}"
    # Handle noisy 'dim ' prefix with unit letters and extra decimal: 'dim 94l x 38l x 95 105h'
    m_dim_noise = re.search(
        r"\bdim\s+(\d+)[lL]\s*[xX×*]\s*(\d+)[lL]\s*[xX×*]\s*(\d+)\s+\d+[hH]",
        original_text,
        re.IGNORECASE
    )
    if m_dim_noise:
        return f"{m_dim_noise.group(1)}*{m_dim_noise.group(2)}*{m_dim_noise.group(3)}"
    # Special: prefix 'l X x l Y x h Z cm' allowing decimal parts separated by space or comma
    m_prefix_double_any = re.search(
        r"\b[lL]\s*(\d+(?:[.,\s]\d+)?)\s*[xX×*]\s*[lL]\s*(\d+(?:[.,\s]\d+)?)\s*[xX×*]\s*[hH]\s*(\d+)\s*cm\b",
        original_text,
        re.IGNORECASE
    )
    if m_prefix_double_any:
        a, b, c = m_prefix_double_any.groups()
        # normalize decimals
        a_norm = a.replace(',', '.').replace(' ', '.')
        b_norm = b.replace(',', '.').replace(' ', '.')
        return f"{a_norm}*{b_norm}*{c}"
    # Special: prefix pattern with two space-decimals for first and second dims: 'l 59 5 x p 23 3 x h 118 cm'
    m_prefix_two_dec = re.search(
        r"\b[lL]\s*(\d+)[\s,]+(\d+)\s*[xX×*]\s*[pP]\s*(\d+)[\s,]+(\d+)\s*[xX×*]\s*[hH]\s*(\d+)\s*cm\b",
        original_text,
        re.IGNORECASE
    )
    if m_prefix_two_dec:
        i1_int, i1_frac, d2_int, d2_frac, h = m_prefix_two_dec.groups()
        return f"{i1_int}.{i1_frac}*{d2_int}.{d2_frac}*{h}"
    # Special: prefix pattern with fractional first dim and integer second dim both prefixed by 'l': 'l 76 5 x l 38 x h 38 cm'
    m_prefix_double_l = re.search(
        r"\b[lL]\s*(\d+)[\s,]+(\d+)\s*[xX×*]\s*[lL]\s*(\d+)\s*[xX×*]\s*[hH]\s*(\d+)\s*cm\b",
        original_text,
        re.IGNORECASE
    )
    if m_prefix_double_l:
        i1_int, i1_frac, d2, h = m_prefix_double_l.groups()
        return f"{i1_int}.{i1_frac}*{d2}*{h}"
    # Special: prefix pattern with space-decimal for 'l 115 x p 50 1 x h 203 cm'
    m_prefix_space_dec = re.search(
        r"\b[lL]\s*(\d+)\s*[xX×*]\s*[pP]\s*(\d+)[\s,]+(\d+)\s*[xX×*]\s*[hH]\s*(\d+)\s*cm\b",
        original_text,
        re.IGNORECASE
    )
    if m_prefix_space_dec:
        i1, d2_int, d2_frac, h = m_prefix_space_dec.groups()
        return f"{i1}*{d2_int}.{d2_frac}*{h}"
    # Special: prefix 'l X x p Y x h Z' without 'cm'
    m_prefix_nocm = re.search(
        r"\b[lL]\s*(\d+(?:[.,]\d+)?)\s*[xX×*]\s*[pP]\s*(\d+(?:[.,]\d+)?)\s*[xX×*]\s*[hH]\s*(\d+(?:[.,]\d+)?)\b",
        original_text,
        re.IGNORECASE
    )
    if m_prefix_nocm:
        a, b, c = m_prefix_nocm.groups()
        # normalize decimals and spaces
        a_norm = a.replace(',', '.').replace(' ', '')
        b_norm = b.replace(',', '.').replace(' ', '')
        c_norm = c.replace(',', '.').replace(' ', '')
        return f"{a_norm}*{b_norm}*{c_norm}"
    # Special: dual 'l' prefix without 'p', e.g., 'l 76 5 x l 38 x h 38 cm'
    m_prefix_l_l = re.search(
        r"\b[lL]\s*(\d+(?:[.,]?\d+)?)\s*[xX×*]\s*[lL]\s*(\d+(?:[.,]?\d+)?)\s*[xX×*]\s*[hH]\s*(\d+)\s*cm\b",
        original_text,
        re.IGNORECASE
    )
    if m_prefix_l_l:
        g1, g2, g3 = m_prefix_l_l.groups()
        g1n = g1.replace(',', '.').replace(' ', '')
        g2n = g2.replace(',', '.').replace(' ', '')
        return f"{g1n}*{g2n}*{g3}"
    # Special: 3D space-decimal triplet with cm: integer and fractional parts separated by space for each dimension
    m_space_dec3_cm = re.search(
        r"\b(\d+)[\s,]+(\d+)[xX×*](\d+)[\s,]+(\d+)[xX×*](\d+)[\s,]+(\d+)\s*cm\b",
        original_text,
        re.IGNORECASE
    )
    if m_space_dec3_cm:
        a, af, b, bf, c, cf = m_space_dec3_cm.groups()
        return f"{a}.{af}*{b}.{bf}*{c}.{cf}"

    # Special: 2D space-decimal pair with cm: integer and fractional parts separated by space
    m_space_dec2_cm = re.search(
        r"\b(\d+)[\s,]+(\d+)\s*[xX×*](\d+)[\s,]+(\d+)\s*cm\b",
        original_text,
        re.IGNORECASE
    )
    if m_space_dec2_cm:
        a, af, b, bf = m_space_dec2_cm.groups()
        return f"{a}.{af}*{b}.{bf}"
    # Handle 'dimensions' prefix with decimal triplet: e.g., 'dimensions 50 0 x 50 0 x 177 8 cm'
    # Disabled to allow fallback for dimensions prefix
    # m_dec3_prefix = re.search(
    #     r"\bdimensions\s+(\d+)\s+(\d+)\s*[xX×*]\s*(\d+)\s+(\d+)\s*[xX×*]\s*(\d+)\s+(\d+)\s*cm\b",
    #     original_text,
    #     re.IGNORECASE
    # )
    # if m_dec3_prefix:
    #     a, b, c, d, e, f = m_dec3_prefix.groups()
    #     return f"{a}.{b}*{c}.{d}*{e}.{f}"
    # Special: 3D decimal cm: catch patterns like '25 5x25 5x55cm' or '92 1x29 5x123cm'
    m_dec3d_cm = re.search(r"\b(\d+)[\s,]+(\d+)[xX×*](\d+)[\s,]+(\d+)[xX×*](\d+)\s*cm\b", original_text)
    if m_dec3d_cm:
        g = m_dec3d_cm.groups()
        # require single-digit fractional parts to avoid stray matches
        if len(g[1]) == 1 and len(g[3]) == 1:
            dim1 = f"{g[0]}.{g[1]}"
            dim2 = f"{g[2]}.{g[3]}"
            dim3 = g[4]
            return f"{dim1}*{dim2}*{dim3}"
    # Special: stray leading code then decimal 3-part, only if leading code >100: '102 164 x 48 5 x 198 cm'
    m_stray3 = re.search(r"\b(\d+)[\s,]+(\d+)\s*[xX×*]\s*(\d+)[\s,]+(\d+)[xX×*]\s*(\d+)\s*cm\b", original_text)
    if m_stray3:
        s, a, b, c, d = m_stray3.groups()
        try:
            if float(s) > 100:
                return f"{a}*{b}.{c}*{d}"
        except:
            pass
    # Handle 3D integer + fractional third dimension with 'c' or 'cm': '92 x 30 x 7 5 cm' or '84 5 c'
    m_third_dec = re.search(
        r"\b(\d+)\s*[xX×*]\s*(\d+)\s*[xX×*]\s*(\d+)[\s,]+(\d+)\s*(?:c|cm)\b",
        original_text,
        re.IGNORECASE
    )
    if m_third_dec:
        a, b, c_int, c_frac = m_third_dec.groups()
        return f"{a}*{b}*{c_int}.{c_frac}"
    # Special: 3D with decimal in first dimension without 'cm' or unit: '61 3 x 53 x 15'
    m_dec3_no_unit = re.search(
        r"\b(\d+)[\s,]+(\d+)\s*[xX×*]\s*(\d+)\s*[xX×*]\s*(\d+)\b",
        original_text
    )
    if m_dec3_no_unit:
        g = m_dec3_no_unit.groups()
        # only treat as decimal if fractional part is single-digit
        if len(g[1]) == 1:
            return f"{g[0]}.{g[1]}*{g[2]}*{g[3]}"
    # Plain 3D integer without unit: '140x40x76' -> '140*40*76'
    # Only match when exactly three parts, not followed by another sep or 'cm'
    m_plain3 = re.search(
        r"\b(\d+)\s*[xX×*]\s*(\d+)\s*[xX×*]\s*(\d+)\b(?!\s*x|\s*cm) ",
        original_text
    )
    if m_plain3:
        return "*".join(m_plain3.groups())
    # Special: 3D decimal on first dimension only: '77 5 x 160 x 48 cm'
    if sep_count >= 2:
        m_first_dec = re.search(
            r"\b(\d+)\s+(\d+)\s*[xX×*]\s*(\d+)\s*[xX×*]\s*(\d+)\s*cm\b",
            original_text,
            re.IGNORECASE
        )
        if m_first_dec:
            g = m_first_dec.groups()
            # Only treat as decimal if fractional part is single-digit
            if len(g[1]) == 1:
                return f"{g[0]}.{g[1]}*{g[2]}*{g[3]}"
    # Generic 3D cm dimensions: '139 x 144 x 33 cm' or drop first if too small
    m_3d_cm = re.search(
        r"\b(\d+(?:[.,]\d+)?)\s*[xX×*]\s*(\d+(?:[.,]\d+)?)\s*[xX×*]\s*(\d+(?:[.,]\d+)?)(?=\s*cm\b)",
        original_text,
        re.IGNORECASE
    )
    if m_3d_cm:
        d1, d2, d3 = m_3d_cm.groups()
        # decide triple or pair based on first dim value and separator count
        sep_count = len(re.findall(r"[xX×*]", original_text))
        v1 = float(d1.replace(',', '.'))
        if sep_count >= 2 and v1 >= 10:
            return f"{d1.replace(',', '.')}*{d2.replace(',', '.')}*{d3.replace(',', '.')}"
        elif sep_count >= 2 and v1 < 10:
            # likely count indicator, return only width and depth
            return f"{d2.replace(',', '.')}*{d3.replace(',', '.')}"
    # Generic 2D cm dimensions: '140x70 cm' (only when exactly one separator)
    if sep_count == 1:
        m_2d_cm = re.search(
            r"\b(\d+(?:[.,]\d+)?)\s*[xX×*]\s*(\d+(?:[.,]\d+)?)(?=\s*cm\b)",
            original_text,
            re.IGNORECASE
        )
        if m_2d_cm:
            d1, d2 = m_2d_cm.groups()
            return f"{d1.replace(',', '.')}*{d2.replace(',', '.')}"
    # Special: catch '12x12xh22cm' style (no spaces between x, h and number)
    m_xxh = re.search(
        r"\b(\d+)\s*[xX×*]\s*(\d+)\s*[xX×*]\s*[hH]\s*(\d+)\s*cm\b",
        original_text,
        re.IGNORECASE
    )
    if m_xxh:
        return f"{m_xxh.group(1)}*{m_xxh.group(2)}*{m_xxh.group(3)}"
    # Simple numeric unit prefix for l x l x h (e.g., '94l x 38l x 95h'), support decimals with comma, dot or space
    m_simple_numunit = re.search(
        r"(\d+(?:[.,]\d+|\s+\d+)?)[lL]\s*[xX×*]\s*(\d+(?:[.,]\d+|\s+\d+)?)[lL]\s*[xX×*]\s*(\d+(?:[.,]\d+|\s+\d+)?)[hH]\b",
        text
    )
    if m_simple_numunit:
        # Normalize dimensions: replace comma or space with dot
        groups = m_simple_numunit.groups()
        parts = []
        for g in groups:
            parts.append(g.replace(',', '.').replace(' ', '.'))
        return "*".join(parts)
    # Remove standalone meter suffixes to avoid treating as dimension: e.g., '3m'
    text = re.sub(r"\b(\d+(?:[.,]\d+)?)m\b", "", text)

    # Pattern: numeric unit prefixes before x and h (e.g., '90l x 42l x 58h cm')
    # Pattern: numeric unit prefixes before x and h (e.g., '90l x 42l x 58h cm'), supports '54 5h'
    m_numunit3 = re.search(
        r"\b(\d+(?:[.,]\d+|\s+\d+)?)\s*[lL]\s*[xX×*]\s*(\d+(?:[.,]\d+|\s+\d+)?)\s*[lL]\s*[xX×*]\s*(\d+(?:[.,]\d+|\s+\d+)?)\s*[hH]\b",
        text,
        re.IGNORECASE
    )
    if m_numunit3:
        d1, d2, d3 = m_numunit3.groups()
        # For d3, drop extra decimal segments if space exists
        d3_clean = d3.split()[0] if ' ' in d3 else d3
        # normalize comma decimals to dot
        def norm(val): return val.replace(',', '.')
        return f"{norm(d1)}*{norm(d2)}*{norm(d3_clean)}"
    # Special case: 'Lit' entries
    if re.search(r'\bLit\b', text, re.IGNORECASE):
        # First, try triplet decimals: '150 1 x 76 3 x 217 3 cm'
        m_trip = re.search(r"(\d+)\s+(\d+)\s*[xX×*]\s*(\d+)\s+(\d+)\s*[xX×*]\s*(\d+)\s+(\d+)\s*cm\b", text)
        if m_trip:
            a, b, c, d, e, f = m_trip.groups()
            if int(a) >= 10:
                return f"{a}.{b}*{c}.{d}*{e}.{f}"
        # Otherwise, catch 2D dims before stray cm: '140x190 200cm'
        m2 = re.search(r"(\d+(?:[.,]\d+)?)\s*[xX×*]\s*(\d+(?:[.,]\d+)?)(?=\s+\d+(?:[.,]\d+)?\s*cm\b)", text)
        if m2:
            d1 = m2.group(1).replace(',', '.')
            d2 = m2.group(2).replace(',', '.')
            return f"{d1}*{d2}"
    # Special case: 'Lit' entries with three decimal dimensions e.g. '150 1 x 76 3 x 217 3 cm'
    if re.search(r'\bLit\b', text, re.IGNORECASE):
        m_lit = re.search(r'(\d+)\s+(\d+)\s*[xX×*]\s*(\d+)\s+(\d+)\s*[xX×*]\s*(\d+)\s+(\d+)\s*cm\b', text)
        if m_lit:
            a, b, c, d, e, f = m_lit.groups()
            return f"{a}.{b}*{c}.{d}*{e}.{f}"
    # Special case: 'Lit' entries with three decimal dimensions like '150 1 x 76 3 x 217 3 cm'
    if re.search(r'\bLit\b', text, re.IGNORECASE):
        m_lit = re.search(
            r"\b(\d+)\s+(\d+)\s*[xX×*]\s*(\d+)\s+(\d+)\s*[xX×*]\s*(\d+)\s+(\d+)\s*cm\b",
            text
        )
        if m_lit:
            a, b, c, d, e, f = m_lit.groups()
            return f"{a}.{b}*{c}.{d}*{e}.{f}"
    # Special case: Lit entries with three decimal dimensions e.g. 'Lit 2 places ... 150 1 x 76 3 x 217 3 cm'
    if re.search(r'\bLit\b', text, re.IGNORECASE):
        m_lit = re.search(
            r"(\d+)\s+(\d+)\s*[xX×*]\s*(\d+)\s+(\d+)\s*[xX×*]\s*(\d+)\s+(\d+)\s*cm\b",
            text
        )
        if m_lit:
            a, b, c, d, e, f = m_lit.groups()
            return f"{a}.{b}*{c}.{d}*{e}.{f}"
    # Special case: Surmatelas dimensions before 'mousse'
    if re.search(r'\bSurmatelas\b', text, re.IGNORECASE):
        m_sur = re.search(r'(\d+(?:[.,]\d+)?)\s*[xX×*]\s*(\d+(?:[.,]\d+)?)(?=\s+mousse\b)', text, re.IGNORECASE)
        if m_sur:
            dim1 = m_sur.group(1).replace(',', '.')
            dim2 = m_sur.group(2).replace(',', '.')
            return f"{dim1}*{dim2}"
    # Special case: mattress protector ('protège matelas') dims
    # Special case: mattress protector ('protège matelas') dims
    # Match both acute and grave accent on 'e'
    if re.search(r'\bprot[eéè]ge matelas\b', text, re.IGNORECASE):
        m_pm = re.search(r'(\d+(?:[.,]\d+)?)\s*[xX×*]\s*(\d+(?:[.,]\d+)?)', text)
        if m_pm:
            d1 = m_pm.group(1).replace(',', '.')
            d2 = m_pm.group(2).replace(',', '.')
            return f"{d1}*{d2}"
    # Special case: 'Matelas' entries, extract 2D dims only when text contains exactly one 'x' separator and is followed by thickness in cm
    if re.search(r'\bMatelas\b', text, re.IGNORECASE):
        # Only consider when exactly one dimension separator
        if len(re.findall(r'[xX×*]', text)) == 1:
            # Match two dimensions with optional spaces around 'x', followed by thickness in cm
            m_mat = re.search(
                r"(\d+(?:[.,]\d+)?)(?:\s*[xX×*]\s*)(\d+(?:[.,]\d+)?)(?=[^\n]*\b\d+(?:[.,]\d+)?\s*cm\b)",
                text,
                re.IGNORECASE
            )
            if m_mat:
                d1 = m_mat.group(1).replace(',', '.')
                d2 = m_mat.group(2).replace(',', '.')
                # Filter out stray small values (e.g., 0x50) by requiring dims >= 10
                try:
                    if float(d1) >= 10 and float(d2) >= 10:
                        return f"{d1}*{d2}"
                except:
                    return f"{d1}*{d2}"
    # Special: 'dimensions' prefix with decimal triplet: 'dimensions 50 0 x 50 0 x 177 8 cm'
    m_dec3_prefix = re.search(
        r"\bdimensions\s+(\d+)\s+(\d+)\s*[xX×*]\s*(\d+)\s+(\d+)\s*[xX×*]\s*(\d+)\s+(\d+)\s*cm\b",
        original_text,
        re.IGNORECASE
    )
    if m_dec3_prefix:
        a, b, c, d, e, f = m_dec3_prefix.groups()
        return f"{a}.{b}*{c}.{d}*{e}.{f}"
    # Special case: 'dimensions' prefix, extract three numbers after first one and before 'cm'
    if re.search(r"\bdimensions\b", text, re.IGNORECASE):
        # Extract substring between 'dimensions' and 'cm'
        sub = re.search(r"\bdimensions\b(.*?)(?=cm\b)", text, re.IGNORECASE)
        if sub:
            nums = re.findall(r"\d+(?:[.,]\d+)?", sub.group(1))
            # Expect at least four numbers: skip the first, take the next three
            if len(nums) >= 4:
                d1, d2, d3 = [n.replace(',', '.') for n in nums[1:4]]
                return f"{d1}*{d2}*{d3}"
    # Remove thickness mentions (e.g., '22cm épaisseur') to avoid misinterpreting as decimal dimensions
    if 'épaisseur' in text.lower():
        text = re.sub(r"\b\d+(?:[.,]\d+)?\s*cm\b", "", text, flags=re.IGNORECASE)
    # Special case: three decimal dimensions like '150 1 x 76 3 x 217 3 cm'
    m_triplet_dec = re.search(
        r"\b(\d+)\s+(\d+)\s*[xX×*]\s*(\d+)\s+(\d+)\s*[xX×*]\s*(\d+)\s+(\d+)\s*cm\b",
        text,
        re.IGNORECASE
    )
    if m_triplet_dec:
        a, b, c, d, e, f = m_triplet_dec.groups()
        # Ensure first dimension is not a small stray number (e.g., '2 places')
        try:
            if int(a) < 10:
                pass
            else:
                return f"{a}.{b}*{c}.{d}*{e}.{f}"
        except:
            return f"{a}.{b}*{c}.{d}*{e}.{f}"
    # Generic Matelas two-dimensional dims before thickness
    if re.search(r"\bMatelas\b", text, re.IGNORECASE):
        m_mat2 = re.search(
            r"(\d+(?:[.,]\d+)?)\s*[xX×*]\s*(\d+(?:[.,]\d+)?)(?=[^\n]*\b\d+(?:[.,]\d+)?\s*cm\b)",
            text,
            re.IGNORECASE
        )
        if m_mat2:
            d1s = m_mat2.group(1).replace(',', '.')
            d2s = m_mat2.group(2).replace(',', '.')
            try:
                if float(d1s) >= 10 and float(d2s) >= 10:
                    return f"{d1s}*{d2s}"
            except:
                return f"{d1s}*{d2s}"
    # Special case: double 'l' prefix format 'l 10 x l 10 x h 250 cm' -> '10*10*250'
    m_llh = re.search(r"\b[lL]\s*(\d+(?:[.,]\d+)?)\s*[xX×*]\s*[lL]\s*(\d+(?:[.,]\d+)?)\s*[xX×*]\s*[hH]\s*(\d+(?:[.,]\d+)?)\s*cm\b", text)
    if m_llh:
        parts = [g.replace(',', '.') for g in m_llh.groups()]
        return "*".join(parts)
    # Special: prefix pattern with any decimal separator (comma or dot) for cm after each: both 'l 76,5 cm x p 29,5 cm x h 70 cm' and 'l 76.5 cm x p 29.5 cm x h 70 cm'
    m_prefix_cm_any = re.search(
        r"\b[lL]\s*([0-9]+(?:[.,]\d+)?)\s*cm\s*[xX×*]\s*[pP]\s*([0-9]+(?:[.,]\d+)?)\s*cm\s*[xX×*]\s*[hH]\s*([0-9]+(?:[.,]\d+)?)\s*cm\b",
        text,
        re.IGNORECASE
    )
    if m_prefix_cm_any:
        parts = [g.replace(',', '.') for g in m_prefix_cm_any.groups()]
        return "*".join(parts)
    # Special: prefix pattern with cm after each: 'l 76 5 cm x p 29 5 cm x h 70 cm' -> '76.5*29.5*70'
    m_generic_prefix = re.search(
        r"\b[lL]\s*(\d+)\s+(\d+)\s*cm.*?p\s*(\d+)\s+(\d+)\s*cm.*?h\s*(\d+)\s*cm\b",
        text,
        re.IGNORECASE
    )
    if m_generic_prefix:
        i1, f1, i2, f2, i3 = m_generic_prefix.groups()
        return f"{i1}.{f1}*{i2}.{f2}*{i3}"
    # Top priority: prefix pattern decimals 'l 76.5 cm x p 29.5 cm x h 70 cm' -> '76.5*29.5*70'
    m_prefix3_cm_dec = re.search(
        r"\b[lL]\s*([0-9]+(?:[.,]\d+)?)\s*cm\s*[xX×*]\s*[pP]\s*([0-9]+(?:[.,]\d+)?)\s*cm\s*[xX×*]\s*[hH]\s*([0-9]+(?:[.,]\d+)?)\s*cm\b",
        text,
        re.IGNORECASE
    )
    if m_prefix3_cm_dec:
        g1, g2, g3 = m_prefix3_cm_dec.groups()
        return f"{g1.replace(',','.')}*{g2.replace(',','.')}*{g3.replace(',','.') }"
    # Special: prefix pattern 'l dim1 x dim2 x h dim3 cm' with comma, dot, or split decimals
    m_prefix2_xh = re.search(
        r"\b[lL]\s*([0-9]+(?:[.,]\d+)?(?:\s+\d+)?)\s*[xX×*]\s*([0-9]+(?:[.,]\d+)?(?:\s+\d+)?)\s*[xX×*]\s*[hH]\s*([0-9]+(?:[.,]\d+)?(?:\s+\d+)?)\s*cm\b",
        text,
        re.IGNORECASE
    )
    if m_prefix2_xh:
        parts = [re.sub(r"[\s,]+", ".", g) for g in m_prefix2_xh.groups()]
        return "*".join(parts)
    # Top priority: 3D integer with height prefix and no unit: '7 x 70 x h150' -> '7*70*150'
    m_h3_int = re.search(r"\b(\d+)\s*[xX×*]\s*(\d+)\s*[xX×*]\s*[hH](\d+)\b", text)
    if m_h3_int:
        a, b, c = m_h3_int.groups()
        return f"{a}*{b}*{c}"
    # Special: prefix pattern with 'cm' after each dimension: 'l 76 5 cm x p 29 5 cm x h 70 cm' -> '76.5*29.5*70'
    m_prefix3_cm = re.search(
        r"\b[lL]\s+(\d+)\s+(\d+)\s*cm\s*[xX×*]\s*[pP]\s+(\d+)\s+(\d+)\s*cm\s*[xX×*]\s*[hH]\s*(\d+)\s*cm\b",
        text,
        re.IGNORECASE
    )
    if m_prefix3_cm:
        i1, f1, i2, f2, i3 = m_prefix3_cm.groups()
        return f"{i1}.{f1}*{i2}.{f2}*{i3}"
    # Top priority: 3D decimal last dimension: '92 x 30 x 88 5 cm', '94 x 50 x 63 5 cm'
    m3_last = re.search(r"\b(\d+(?:[.,]\d+)?)\s*[xX×*]\s*(\d+(?:[.,]\d+)?)\s*[xX×*]\s*(\d+)\s+(\d+)\s*cm\b", text)
    if m3_last:
        g0, g1, g2, g3 = m3_last.groups()
        return f"{g0.replace(',','.')}*{g1.replace(',','.')}*{g2}.{g3}"
    # Special: 3D decimals with zero prefix: '50 0 x 50 0 x 177 8 cm'
    m_zero3d = re.search(r"\b(\d{2,})\s+0\s*[xX×*]\s*(\d+)\s+0\s*[xX×*]\s*(\d+)\s+(\d+)\s*cm\b", text)
    if m_zero3d:
        a, b, c, d = m_zero3d.groups()
        return f"{a}.0*{b}.0*{c}.{d}"
    # Special: 3D decimal with 'c' suffix: '100 x 35 x 84 5 c' -> '100*35*84.5'
    m_3d_c = re.search(r"\b(\d+)\s*[xX×*]\s*(\d+)\s*[xX×*]\s*(\d+)\s+(\d+)\s*c\b", text)
    if m_3d_c:
        g = m_3d_c.groups()
        return f"{g[0]}*{g[1]}*{g[2]}.{g[3]}"
    # Special: 3D with double decimals: '120 x 40 3 x 34 7 cm' -> '120*40.3*34.7'
    m_3d_double = re.search(r"\b(\d+)\s*[xX×*]\s*(\d+)\s+(\d+)\s*[xX×*]\s*(\d+)\s+(\d+)\s*cm\b", text)
    if m_3d_double:
        a, b, c, d, e = m_3d_double.groups()
        return f"{a}*{b}.{c}*{d}.{e}"
    # Special: 3D decimal on first dimension only: '77 5 x 160 x 48 cm' -> '77.5*160*48'
    m_first_dec = re.search(r"\b(\d+)\s+(\d+)\s*[xX×*]\s*(\d+)\s*[xX×*]\s*(\d+)\s*cm\b", text)
    if m_first_dec:
        g = m_first_dec.groups()
        return f"{g[0]}.{g[1]}*{g[2]}*{g[3]}"
    # Special: 3D decimal last dimension with comma: '94 x 50 x 63,5 cm' -> '94*50*63.5'
    m3_last_comma = re.search(
        r"\b(\d+(?:[.,]\d+)?)\s*[xX×*]\s*(\d+(?:[.,]\d+)?)\s*[xX×*]\s*(\d+(?:[.,]\d+))\s*cm\b",
        text
    )
    if m3_last_comma:
        a, b, c = m3_last_comma.groups()
        return f"{a.replace(',','.')}*{b.replace(',','.')}*{c.replace(',','.') }"
    # Special: 2D integer dims followed by height unit: '160 x 200 ... h22 cm' -> '160*200'
    m_2d_h_cm = re.search(
        r"\b(\d+(?:[.,]\d+)?)\s*[xX×*]\s*(\d+(?:[.,]\d+)?).*?h\s*\d+(?:[.,]\d+)?\s*cm\b",
        text,
        re.IGNORECASE
    )
    if m_2d_h_cm:
        d1, d2 = m_2d_h_cm.groups()
        return f"{d1.replace(',','.')}*{d2.replace(',','.') }"
    # Special: 2D comma meter: '1,5x5m' -> '1.5*5'
    m_2d_comma_m = re.search(r"\b(\d+(?:[.,]\d+))x(\d+(?:[.,]\d+)?)m\b", text, re.IGNORECASE)
    if m_2d_comma_m:
        a, b = m_2d_comma_m.groups()
        return f"{a.replace(',','.')}*{b.replace(',','.')}"
    # Special: prefix pattern 'l 80 x 39 5 x h 75cm' -> '80*39.5*75'
    m_l_pref = re.search(r"\bl\s*(\d+)\s*[xX×*]\s*(\d+)\s+(\d+)\s*[xX×*]\s*[hH]\s*(\d+)\s*cm\b", text, re.IGNORECASE)
    if m_l_pref:
        i1, i2, f2, h = m_l_pref.groups()
        return f"{i1}*{i2}.{f2}*{h}"
    # Special: 3D with double decimals: '120 x 40 3 x 34 7 cm' -> '120*40.3*34.7'
    m_3d_double = re.search(
        r"\b(\d+)\s*[xX×*]\s*(\d+)\s+(\d+)\s*[xX×*]\s*(\d+)\s+(\d+)\s*cm\b",
        text
    )
    if m_3d_double:
        a, b, c, d, e = m_3d_double.groups()
        return f"{a}*{b}.{c}*{d}.{e}"
    # Special: 3D decimal on first dimension only: '77 5 x 160 x 48 cm' -> '77.5*160*48'
    m_first_dec = re.search(
        r"\b(\d+)\s+(\d+)\s*[xX×*]\s*(\d+)\s*[xX×*]\s*(\d+)\s*cm\b",
        text
    )
    if m_first_dec:
        g = m_first_dec.groups()
        return f"{g[0]}.{g[1]}*{g[2]}*{g[3]}"
    # Special: 3D decimal in last dimension: '92 x 30 x 88 5 cm' or '94 x 50 x 63 5 cm'
    m_3d_last = re.search(
        r"\b(\d+(?:[.,]\d+)?)\s*[xX×*]\s*(\d+(?:[.,]\d+)?)\s*[xX×*]\s*(\d+)\s+(\d+)\s*cm\b",
        text
    )
    if m_3d_last:
        d1, d2, i3, f3 = m_3d_last.groups()
        return f"{d1.replace(',','.')}*{d2.replace(',','.')}*{i3}.{f3}"
    # Special: prefix pattern decimals supporting missing 'x': 'l 122 6 x p 34 2 h 88 1 cm'
    m_prefix3_dec = re.search(
        r"\b[lL]\s+(\d+)\s+(\d+)(?:\s*[xX×*]\s*|\s+)[pP]\s+(\d+)\s+(\d+)(?:\s*[xX×*]\s*|\s+)[hH]\s+(\d+)\s+(\d+)\s*cm\b",
        text,
        re.IGNORECASE
    )
    if m_prefix3_dec:
        i1, f1, i2, f2, i3, f3 = m_prefix3_dec.groups()
        return f"{i1}.{f1}*{i2}.{f2}*{i3}.{f3}"
    # Special: l/p/h style 3D cm: 'l 132 x p 1 x h 81 cm' -> '132*1*81'
    m_lph = re.search(r"\b[lL]\s*(\d+(?:[.,]\d+)?)\s*[xX×*]\s*[pP]\s*(\d+(?:[.,]\d+)?)\s*[xX×*]\s*[hH]\s*(\d+(?:[.,]\d+)?)\s*cm\b", text, re.IGNORECASE)
    if m_lph:
        parts = [p.replace(',', '.') for p in m_lph.groups()]
        return "*".join(parts)
    # Special: h/l/p style 3D cm: 'h 170 x l 120 x p 2 cm' -> '170*120*2'
    m_hlp = re.search(r"\b[hH]\s*(\d+(?:[.,]\d+)?)\s*[xX×*]\s*[lL]\s*(\d+(?:[.,]\d+)?)\s*[xX×*]\s*[pP]\s*(\d+(?:[.,]\d+)?)\s*cm\b", text, re.IGNORECASE)
    if m_hlp:
        parts = [p.replace(',', '.') for p in m_hlp.groups()]
        return "*".join(parts)
    # Special: 4-part decimal cm '50 7x46x1x142 5cm' -> '50.7*46*142.5'
    m_4dec_cm = re.search(
        r"\b(\d+)\s+(\d+)[xX×*](\d+)[xX×*](\d+)[xX×*](\d+)\s+(\d+)cm\b",
        text
    )
    if m_4dec_cm:
        a, b, c, d, e, f = m_4dec_cm.groups()
        # drop fourth part if small stray (<10)
        try:
            if float(d) < 10:
                return f"{a}.{b}*{c}*{e}.{f}"
        except:
            pass
    # Special: stray leading code then decimal 3-part, only if stray code >100: '102 164 x 48 5 x 198 cm' -> '164*48.5*198'
    m_stray3 = re.search(
        r"\b(\d+)\s+(\d+)\s*[xX×*]\s*(\d+)\s+(\d+)\s*[xX×*]\s*(\d+)\s*cm\b",
        text
    )
    if m_stray3:
        s, a, b, c, d = m_stray3.groups()
        try:
            if float(s) > 100:
                return f"{a}*{b}.{c}*{d}"
        except:
            pass
    # Special: 2D decimal meters '0 45 x 2m' -> '0.45*2'
    m_2d_dec_m = re.search(r"\b(\d+)\s+(\d+)\s*[xX×*]\s*(\d+(?:[.,]\d+)?)m\b", text)
    if m_2d_dec_m:
        g = m_2d_dec_m.groups()
        return f"{g[0]}.{g[1]}*{g[2].replace(',', '.')}"
    # Special: prefix pattern 'l 80 x 39 5 x h 75cm' -> '80*39.5*75'
    m_prefix3 = re.search(r"\bl\s*(\d+(?:[.,]\d+)?)\s*[xX×*]\s*(\d+)\s+(\d+)\s*[xX×*]\s*[hH]\s*(\d+(?:[.,]\d+)?)cm\b", text)
    if m_prefix3:
        g = m_prefix3.groups()
        dim1 = g[0].replace(',', '.')
        dim2 = f"{g[1]}.{g[2]}"
        dim3 = g[3].replace(',', '.')
        return f"{dim1}*{dim2}*{dim3}"
    # Special: 3D decimal with single 'c' suffix: '100 x 35 x 84 5 c' -> '100*35*84.5'
    m_3d_dec_c = re.search(r"\b(\d+)\s*[xX×*]\s*(\d+)\s*[xX×*]\s*(\d+)\s+(\d+)\s*c\b", text)
    if m_3d_dec_c:
        g = m_3d_dec_c.groups()
        return f"{g[0]}*{g[1]}*{g[2]}.{g[3]}"
    # Quick 3D integer cm: catch '110 x 40 x 45 cm' before other regex
    m_3int_cm = re.search(r"\b(\d{2,3})\s*[xX×*]\s*(\d{1,4})\s*[xX×*]\s*(\d{1,4})\s*cm\b", text, re.IGNORECASE)
    if m_3int_cm:
        g = m_3int_cm.groups()
        return f"{g[0]}*{g[1]}*{g[2]}"
    # Simplify cases like '95 105h' to '95h' (ignore secondary number before 'h')
    text = re.sub(r"(\d+)\s+\d+h", r"\1h", text)
    # Drop first small multiplier in 3D: '2 x 60 x 160 cm' -> '60*160'
    m_drop3 = re.search(r"\b(\d{1,2})\s*[xX×*]\s*(\d+(?:[.,]\d+)?)\s*[xX×*]\s*(\d+(?:[.,]\d+)?)\s*cm\b", text)
    if m_drop3:
        g = m_drop3.groups()
        try:
            if float(g[0]) < 10 and float(g[1]) >= 10 and float(g[2]) >= 10:
                return f"{g[1].replace(',','.')}*{g[2].replace(',','.') }"
        except:
            pass
    # Strict integer 3D cm: '110 x 40 x 45 cm' -> '110*40*45'
    m_int3d_cm = re.search(r"\b(\d{2,3})\s*[xX×*]\s*(\d{1,4})\s*[xX×*]\s*(\d{1,4})\s*cm\b", text)
    if m_int3d_cm:
        g = m_int3d_cm.groups()
        return f"{g[0]}*{g[1]}*{g[2]}"
    # Standard 3D cm: '110 x 40 x 45 cm' -> '110*40*45'
    m_std3d_cm = re.search(r"\b(\d+(?:[.,]\d+)?)\s*[xX×*]\s*(\d+(?:[.,]\d+)?)\s*[xX×*]\s*(\d+(?:[.,]\d+)?)\s*cm\b", text)
    if m_std3d_cm:
        g = m_std3d_cm.groups()
        return f"{g[0].replace(',','.')}*{g[1].replace(',','.')}*{g[2].replace(',','.') }"
    # Simple 2D cm: '200 x 200cm' or '200x200 cm' -> '200*200', skip for Matelas entries
    if not re.search(r"\bMatelas\b", text, re.IGNORECASE):
        m_2d_cm = re.search(r"\b(\d+(?:[.,]\d+)?)\s*[xX×*]\s*(\d+(?:[.,]\d+)?)(?:\s*cm)\b", text)
        if m_2d_cm:
            g = m_2d_cm.groups()
            try:
                if float(g[0].replace(',','.')) >= 10 and float(g[1].replace(',','.')) >= 10:
                    return f"{g[0].replace(',','.')}*{g[1].replace(',','.') }"
            except:
                pass
    # Special: 3D decimal cm: '25 5x25 5x55cm' or '92 1x29 5x123cm' -> '25.5*25.5*55'/'92.1*29.5*123'
    m_dec3d_cm = re.search(r"\b(\d+)\s+(\d+)[xX×*](\d+)\s+(\d+)[xX×*](\d+)\s*cm\b", text)
    if m_dec3d_cm:
        g = m_dec3d_cm.groups()
        dim1 = f"{g[0]}.{g[1]}"
        dim2 = f"{g[2]}.{g[3]}"
        dim3 = g[4]
        return f"{dim1}*{dim2}*{dim3}"
    # Simple 3D cm: '110 x 40 x 45 cm' -> '110*40*45'
    m_simple3d_cm = re.search(r"\b(\d+(?:[.,]\d+)?)\s*[xX×*]\s*(\d+(?:[.,]\d+)?)\s*[xX×*]\s*(\d+(?:[.,]\d+)?)\s*cm\b", text)
    if m_simple3d_cm:
        g = m_simple3d_cm.groups()
        return f"{g[0].replace(',','.')}*{g[1].replace(',','.')}*{g[2].replace(',','.') }"
    
    # Filtrer les codes produits longs et références techniques (comme 484000008582, 481281718533, etc.)
    # Ces codes sont souvent de longs nombres ou alphanumériques qu'on veut exclure
    # Détecter si le texte contient beaucoup de codes produits (indicateur: plusieurs longs codes numériques)
    long_codes = re.findall(r'\b\d{8,}\b', text)  # Codes de 8+ chiffres
    alphanumeric_codes = re.findall(r'\b[a-zA-Z]\d+[a-zA-Z]*\d*\b', text, re.IGNORECASE)  # Codes comme d085, chf85, amc859, ac40
    
    if len(long_codes) >= 2 or (len(long_codes) >= 1 and len(alphanumeric_codes) >= 3):
        # Ce texte semble être principalement des codes produits, très peu de chance d'avoir de vraies dimensions
        # Exemples détectés: "484000008582 nyttig fil 100 481281718533 chf85 1 type 10 amc859 ac40"
        return None
    
    # Filtrer les téléphones et appareils mobiles (Nokia, iPhone, Samsung, etc.)
    # Ces produits n'ont généralement pas de dimensions physiques utiles à extraire
    # Être plus précis pour éviter de filtrer les meubles qui mentionnent ces marques
    phone_indicators = 0
    
    # Vérifier les patterns de téléphones
    if re.search(r'\b(nokia|iphone)\s+\d+', text, re.IGNORECASE):
        phone_indicators += 2  # Nokia/iPhone + numéro = très probablement un téléphone
    
    if re.search(r'\b(samsung|huawei|xiaomi|sony|lg)\s+(galaxy|redmi|xperia|p\d+|v\d+)', text, re.IGNORECASE):
        phone_indicators += 2  # Marque + modèle de téléphone connu
    
    if re.search(r'\b(smartphone|mobile\s+phone)\b', text, re.IGNORECASE):
        phone_indicators += 1  # Mots-clés téléphone explicites
    
    if re.search(r'\b(double|dual)\s+(sim|carte)', text, re.IGNORECASE):
        phone_indicators += 1  # Double sim
    
    if re.search(r'\b\d+gb\s+(ram|stockage|mémoire)', text, re.IGNORECASE):
        phone_indicators += 1  # Spécifications mobiles
    
    # Ne filtrer que si on a plusieurs indicateurs de téléphone ET pas de dimensions claires
    if phone_indicators >= 2:
        # Vérifier s'il n'y a pas de dimensions explicites avec unités (cm, mm)
        if not re.search(r'\d+\s*[xX×*]\s*\d+\s*(cm|mm)', text, re.IGNORECASE):
            # Ce texte semble décrire un téléphone sans dimensions physiques claires
            # Exemple détecté: "Nokia 105 2017 double sim blanc"
            return None
    
    # Exclure les unités de poids et autres unités non-dimensionnelles
    if re.search(r'\d+\s*kg\b', text, re.IGNORECASE):
        # Si le texte contient des kg, filtrer pour ne garder que les vraies dimensions
        text_filtered = re.sub(r'\d+\s*[xX×*]\s*\d+\s*kg\b', '', text, flags=re.IGNORECASE)
        text_filtered = re.sub(r'\d+\s*kg\b', '', text_filtered, flags=re.IGNORECASE)
        text = text_filtered
    
    # Exclure les tailles TV (pouces)
    if re.search(r'tv\s+\d+\s+\d+', text, re.IGNORECASE):
        return None
    
    # Filtrer les appareils électriques avec puissance (wattage, voltage, etc.)
    # Exemples: "grille pain pc ta 1073 1500 w", "mixeur 500w", "aspirateur 1200w"
    electrical_indicators = 0
    
    if re.search(r'\b\d+\s*w\b', text, re.IGNORECASE):  # Wattage
        electrical_indicators += 2
    
    if re.search(r'\b\d+\s*(watts?|volts?|ampères?|amps?)\b', text, re.IGNORECASE):  # Autres unités électriques
        electrical_indicators += 2
    
    if re.search(r'\b(grille.pain|mixeur|aspirateur|sèche.cheveux|micro.onde|four|frigo|lave)', text, re.IGNORECASE):  # Appareils électriques
        electrical_indicators += 1
    
    if re.search(r'\b(proficook|moulinex|philips|bosch|siemens)\b', text, re.IGNORECASE):  # Marques d'électroménager
        electrical_indicators += 1
    
    # Filtrer si plusieurs indicateurs d'appareil électrique et pas de dimensions avec unités claires
    if electrical_indicators >= 2:
        if not re.search(r'\d+\s*[xX×*]\s*\d+\s*(cm|mm)', text, re.IGNORECASE):
            # Ce texte semble décrire un appareil électrique sans dimensions physiques claires
            # Exemple détecté: "Proficook grille pain pc ta 1073 1500 w"
            return None
    
    # Filtrer les accessoires informatiques et électroniques (adaptateurs, chargeurs, etc.)
    # Exemples: "adaptateur alimentation chargeur pour ordinateur portable dell inspiron 15 3521 7537"
    computer_indicators = 0
    
    if re.search(r'\b(adaptateur|chargeur|alimentation|ordinateur|portable|laptop)\b', text, re.IGNORECASE):
        computer_indicators += 1
    
    if re.search(r'\b(dell|hp|lenovo|asus|acer|toshiba|apple|macbook|rowenta|hobby\s*tech)\b', text, re.IGNORECASE):
        computer_indicators += 1
    
    if re.search(r'\b(inspiron|thinkpad|pavilion|aspire|air\s*force)\b', text, re.IGNORECASE):  # Gammes de PC et aspirateurs
        computer_indicators += 1
    
    # Ajouter indicateur pour voltage spécifique aux chargeurs
    if re.search(r'\b\d+v\s+\d+\s+\d+a\b', text, re.IGNORECASE):  # Pattern voltage + ampérage avec espaces
        computer_indicators += 2
    
    # Filtrer si plusieurs indicateurs d'informatique et beaucoup de références numériques
    if computer_indicators >= 2:
        model_numbers = re.findall(r'\d{4,5}', text)  # Codes modèles 4-5 chiffres (match anywhere)
        if len(model_numbers) >= 1:  # Au moins une référence de modèle (réduit de 3 à 1)
            # Ce texte semble décrire des accessoires informatiques avec références
            # Exemple détecté: "Chargeur compatible rh5664 29v 0 75a pour aspirateur rowenta air force 360"
            return None
    
    # Spécial: 3D decimal cm: "25 5x25 5x55cm" and "92 1x29 5x123cm" -> "25.5*25.5*55", "92.1*29.5*123"
    m_3d_dec_cm = re.search(r"\b(\d+)\s+(\d+)[xX×*](\d+)\s+(\d+)[xX×*](\d+)\s*cm\b", text)
    if m_3d_dec_cm:
        g = m_3d_dec_cm.groups()
        dim1 = f"{g[0]}.{g[1]}"
        dim2 = f"{g[2]}.{g[3]}"
        dim3 = g[4]
        return f"{dim1}*{dim2}*{dim3}"
    m_h2 = re.search(r"\b(\d+(?:[.,]\d+)?)\s*[xX×*]\s*[hH]\s*(\d+(?:[.,]\d+)?)\s*cm\b", text)
    if m_h2:
        g = m_h2.groups()
        return f"{g[0].replace(',','.')}*{g[1].replace(',','.') }"
    # Spécial: prefixes with units collées for 3D: "l32cm x p30cm x h170cm" -> "32*30*170"
    m4 = re.search(r"\b[lL](\d+(?:[.,]\d+)?)cm\s*[xX×*]\s*[pP](\d+(?:[.,]\d+)?)cm\s*[xX×*]\s*[hH](\d+(?:[.,]\d+)?)cm\b", text)
    if m4:
        g = m4.groups()
        return f"{g[0].replace(',','.')}*{g[1].replace(',','.')}*{g[2].replace(',','.') }"
    # Spécial: dimensions l/h 3D: "94l x 38l x 95h cm" -> "94*38*95"
    m_l3 = re.search(r"\b(\d+(?:[,\.]\d+)?)l\s*[xX×*]\s*(\d+(?:[,\.]\d+)?)l\s*[xX×*]\s*(\d+(?:[,\.]\d+)?)h\s*cm\b", text, re.IGNORECASE)
    if m_l3:
        g = m_l3.groups()
        return f"{g[0].replace(',','.')}*{g[1].replace(',','.')}*{g[2].replace(',','.')}"
    # Spécial: dimensions 3D décimales en mètres: "3 33x2 06x1 17m" -> "3.33*2.06*1.17"
    m3 = re.search(r"\b(\d+)\s+(\d+)[xX×*](\d+)\s+(\d+)[xX×*](\d+)\s+(\d+)m\b", text)
    if m3:
        g = m3.groups()
        dim1 = f"{g[0]}.{g[1]}"
        dim2 = f"{g[2]}.{g[3]}"
        dim3 = f"{g[4]}.{g[5]}"
        return f"{dim1}*{dim2}*{dim3}"
    # Patterns pour différents formats de dimensions (ordre de priorité important)
    patterns = [
        # Format extensible avec range: "extensible 70 105 x 22 x 4 cm" -> "105*22*4" (prendre les vraies dimensions, pas le range)
        r'extensible\s+\d+\s+(\d+(?:[,\.]\d+)?)\s*[xX×*]\s*(\d+(?:[,\.]\d+)?)\s*[xX×*]\s*(\d+(?:[,\.]\d+)?)\s*cm',
        # Format avec espaces décimaux 2D à la fin: "11x28 3 cm" -> "11*28.3" (HIGH PRIORITY - must not have additional x)  
        r'(\d+(?:[,\.]\d+)?)[xX×*](\d+)\s+(\d+)\s*cm(?!\s*[xX×*])',
        # Format avec espaces décimaux 3D à la fin: "55 x 198 x 40 5 cm" -> "55*198*40.5" (HIGH PRIORITY)
        r'(\d+(?:[,\.]\d+)?)\s*[xX×*]\s*(\d+(?:[,\.]\d+)?)\s*[xX×*]\s*(\d+)\s+(\d+)\s*cm',
        # Format avec espaces décimaux avec 0: "50 0 x 50 0 x 177 8 cm" -> "50.0*50.0*177.8"
        r'(\d+)\s+0\s*[xX×*]\s*(\d+)\s+0\s*[xX×*]\s*(\d+)\s+(\d+)\s*cm',
        # Format avec espaces décimaux 3D: "25 5x25 5x55cm" -> "25.5*25.5*55" (priorité très haute)
        r'(\d+)\s+(\d+)\s*[xX×*]\s*(\d+)\s+(\d+)\s*[xX×*]\s*(\d+(?:[,\.]\d+)?)\s*cm',
        # Format dimensions principales 3D: "100x38x38 cm" -> "100*38*38" (priorité haute)
        r'(\d+(?:[,\.]\d+)?)[xX×*](\d+(?:[,\.]\d+)?)[xX×*](\d+(?:[,\.]\d+)?)\s*cm\b',
        # Format dimensions principales 2D avec cm: "160x230cm" -> "160*230" (priorité très haute)
        r'(\d+(?:[,\.]\d+)?)[xX×*](\d+(?:[,\.]\d+)?)cm\b',
        # Format dimensions principales 3D avec mm: "230x210x30mm" -> "230*210*30" (priorité très haute)
        r'(\d+(?:[,\.]\d+)?)[xX×*](\d+(?:[,\.]\d+)?)[xX×*](\d+(?:[,\.]\d+)?)mm\b',
        # Format dimensions principales 2D avec mm: "160x230mm" -> "160*230" (priorité très haute)
        r'(\d+(?:[,\.]\d+)?)[xX×*](\d+(?:[,\.]\d+)?)mm\b',
        # Format dimensions principales 2D: "160x200" -> "160*200" (priorité haute, avant les petits nombres)
        r'\b(\d{2,3})[xX×*](\d{2,3})\b(?!\s*kg)(?!\s*[xX×*]\d+)',
        # Format avec l/h notation avec espaces décimaux: "l 80 x 39 5 x h 75cm" -> "80*39.5*75"
        r'[lh]\s*(\d+(?:[,\.]\d+)?)\s*[xX×*]\s*(\d+)\s+(\d+)\s*[xX×*]\s*[hH]\s*(\d+(?:[,\.]\d+)?)cm',
        # Format avec l/h notation: "l 43 x l 6 x h 6 cm" -> "43*6*6"
        r'[lh]\s*(\d+(?:[,\.]\d+)?)\s*[xX×*]\s*[lh]\s*(\d+(?:[,\.]\d+)?)\s*[xX×*]\s*[lh]\s*(\d+(?:[,\.]\d+)?)\s*cm',
        # Format avec diam notation: "diam 35 x h 40 cm" -> "35*40"
        r'diam\s*(\d+(?:[,\.]\d+)?)\s*[xX×*]\s*[lh]\s*(\d+(?:[,\.]\d+)?)\s*cm',
        # Format dimensions avec l/h et espaces décimaux: "80l x 43l x 54 5h cm" -> "80*43*54.5"
        r'(\d+(?:[,\.]\d+)?)[lh]\s*[xX×*]\s*(\d+(?:[,\.]\d+)?)[lh]\s*[xX×*]\s*(\d+)\s+(\d+)[lh]\s*cm',
        # Format dimensions avec l/h sans x: "90l x 42l x 58h cm" -> "90*42*58"
        r'(\d+(?:[,\.]\d+)?)[lh]\s*[xX×*]\s*(\d+(?:[,\.]\d+)?)[lh]\s*[xX×*]\s*(\d+(?:[,\.]\d+)?)[lh]\s*cm',
        # Format avec l/p/h mixtes 3D: "l 120 x p 50 x h 41 cm" -> "120*50*41"
        r'[lph]\s*(\d+(?:[,\.]\d+)?)\s*[xX×*]\s*[lph]\s*(\d+(?:[,\.]\d+)?)\s*[xX×*]\s*[lph]\s*(\d+(?:[,\.]\d+)?)\s*cm',
        # Format avec l/p mixtes 2D: "l 205 x p 80 cm" -> "205*80"
        r'[lph]\s*(\d+(?:[,\.]\d+)?)\s*[xX×*]\s*[lph]\s*(\d+(?:[,\.]\d+)?)\s*cm',
        # Format avec préfixes collés 3D: "lo45xla45xh170 cm" -> "45*45*170"
        r'[a-z]{1,3}(\d+(?:[,\.]\d+)?)[xX×*][a-z]{1,3}(\d+(?:[,\.]\d+)?)[xX×*][a-z]{1,3}(\d+(?:[,\.]\d+)?)\s*cm',
        # Format 3D avec cm: "108 x 32 5 x 48 cm" ou "143 x 36 x 178 cm" -> "108 x 32.5 x 48" (HIGH PRIORITY for 4 groups)
        r'(\d+(?:[,\.]\d+)?)\s*[xX×*]\s*(\d+)\s+(\d+)\s*[xX×*]\s*(\d+)\s*cm',
        # Format avec espaces décimaux au milieu: "38 5 x 54 cm" -> "38.5*54" (HIGH PRIORITY for 3 groups)
        r'(\d+)\s+(\d+)\s*[xX×*]\s*(\d+(?:[,\.]\d+)?)\s*cm',
        # Format cm après chaque dimension: "112 cm x 207 cm x 57 cm" -> "112*207*57"
        r'(\d+(?:[,\.]\d+)?)\s*cm\s*[xX×*]\s*(\d+(?:[,\.]\d+)?)\s*cm\s*[xX×*]\s*(\d+(?:[,\.]\d+)?)\s*cm',
        # Format sommier éviter multiplicateur: "2x90x200" -> "90*200"
        r'\d+[xX×*](\d+(?:[,\.]\d+)?)[xX×*](\d+(?:[,\.]\d+)?)\s*cm',
        # Format avec espaces décimaux sans cm: "3x7 5 m" -> "3*7.5"
        r'(\d+(?:[,\.]\d+)?)[xX×*](\d+)\s+(\d+)\s*m',
        # Format avec espaces décimaux 2D avec m: "1 5 x 10m" -> "1.5*10"
        r'(\d+)\s+(\d+)\s*[xX×*]\s*(\d+(?:[,\.]\d+)?)m',
        # Format avec espaces décimaux 3D avec m: "3 33x2 06x1 17m" -> "3.33*2.06*1.17"
        r'(\d+)\s+(\d+)\s*[xX×*]\s*(\d+)\s+(\d+)\s*[xX×*]\s*(\d+)\s+(\d+)m',
        # Format avec espaces dans les décimaux: "39 5" -> "39.5"
        r'(\d+)\s+(\d)\s*[xX×*]\s*(\d+(?:[,\.]\d+)?)\s*[xX×*]\s*(\d+(?:[,\.]\d+)?)',
        # Format unités mixtes: "27 5 cm 3m" -> "27.5*3"
        r'(\d+)\s+(\d)\s*cm\s*(\d+(?:[,\.]\d+)?)m',
        # Format avec cm collé: "200cm x 180 cm" -> "200*180"
        r'(\d+(?:[,\.]\d+)?)cm\s*[xX×*]\s*(\d+(?:[,\.]\d+)?)\s*cm',
        # Format 3D standard: "143 x 36 x 178 cm" ou "143 x 36 x 178"
        r'(\d+(?:[,\.]\d+)?)\s*[xX×*]\s*(\d+(?:[,\.]\d+)?)\s*[xX×*]\s*(\d+(?:[,\.]\d+)?)',
        # Format 2D: "45 x 75" (plus restrictif pour éviter H20 22)
        r'\b(\d+(?:[,\.]\d+)?)\s*[xX×*]\s*(\d+(?:[,\.]\d+)?)\b(?!\s*mousse)(?!\s*H\d+)'
    ]
    
    found_dimensions = []
    main_dimension_found = False
    
    for pattern_idx, pattern in enumerate(patterns):
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            dimensions = []
            for match in matches:
                if len(match) == 6:
                    if pattern_idx == 24:  # Format avec espaces décimaux 3D avec m: "3 33x2 06x1 17m" (updated index +1)
                        dim1 = match[0] + '.' + match[1]  # "3" + "." + "33" = "3.33"
                        dim2 = match[2] + '.' + match[3]  # "2" + "." + "06" = "2.06"
                        dim3 = match[4] + '.' + match[5]  # "1" + "." + "17" = "1.17"
                        dimensions.append(f"{dim1}*{dim2}*{dim3}")
                        main_dimension_found = True
                elif len(match) == 5:
                    if pattern_idx == 4:  # Format avec espaces décimaux 3D: "25 5x25 5x55cm" (index shifted +1)
                        dim1 = match[0] + '.' + match[1]  # "25" + "." + "5" = "25.5"
                        dim2 = match[2] + '.' + match[3]  # "25" + "." + "5" = "25.5"
                        dim3 = match[4].replace(',', '.')
                        dimensions.append(f"{dim1}*{dim2}*{dim3}")
                        main_dimension_found = True
                elif len(match) == 4:
                    if pattern_idx == 2:  # Format avec espaces décimaux 3D à la fin: "55 x 198 x 40 5 cm" -> "55*198*40.5"
                        dim1 = match[0].replace(',', '.')
                        dim2 = match[1].replace(',', '.')
                        dim3 = match[2] + '.' + match[3]  # "40" + "." + "5" = "40.5"
                        dimensions.append(f"{dim1}*{dim2}*{dim3}")
                        main_dimension_found = True
                    elif pattern_idx == 3:  # Format avec espaces décimaux avec 0 (index shifted +1)
                        dim1 = match[0] + '.0'
                        dim2 = match[1] + '.0'
                        dim3 = match[2] + '.' + match[3]
                        dimensions.append(f"{dim1}*{dim2}*{dim3}")
                        main_dimension_found = True
                    elif pattern_idx == 10:  # Format l/h avec espaces décimaux (index shifted +1)
                        dim1 = match[0].replace(',', '.')
                        dim2 = match[1] + '.' + match[2]  # "39" + "." + "5" = "39.5"
                        dim3 = match[3].replace(',', '.')
                        dimensions.append(f"{dim1}*{dim2}*{dim3}")
                        main_dimension_found = True
                    elif pattern_idx == 13:  # Format l/h avec espaces décimales à la fin (index shifted +1)
                        dim1 = match[0].replace(',', '.')
                        dim2 = match[1].replace(',', '.')
                        dim3 = match[2] + '.' + match[3]  # "54" + "." + "5" = "54.5"
                        dimensions.append(f"{dim1}*{dim2}*{dim3}")
                        main_dimension_found = True
                    elif pattern_idx == 25:  # Format avec espaces dans les décimaux (index shifted +1)
                        # "39 5" -> "39.5"
                        dim1 = match[0] + '.' + match[1]
                        dim2 = match[2].replace(',', '.')
                        dim3 = match[3].replace(',', '.')
                        dimensions.append(f"{dim1}*{dim2}*{dim3}")
                    elif pattern_idx == 18:  # Format spécial "108 x 32 5 x 48" (index shifted +1)
                        # Combine le 2e et 3e groupe: "32" + "5" = "32.5"
                        dim1 = match[0].replace(',', '.')
                        dim2 = match[1] + '.' + match[2]  # "32" + "." + "5" = "32.5"
                        dim3 = match[3].replace(',', '.')
                        dimensions.append(f"{dim1}*{dim2}*{dim3}")
                        main_dimension_found = True
                elif len(match) == 3:
                    if pattern_idx == 0:  # Format extensible: "extensible 70 105 x 22 x 4 cm" -> "105*22*4"
                        # Prendre les 3 groupes capturés comme dimensions (ignore le range extensible)
                        dim_parts = [part.replace(',', '.') for part in match if part]
                        if len(dim_parts) == 3:
                            dimensions.append("*".join(dim_parts))
                            main_dimension_found = True
                    elif pattern_idx == 1:  # Format avec espaces décimaux 2D à la fin: "11x28 3 cm" -> "11*28.3"
                        dim1 = match[0].replace(',', '.')
                        dim2 = match[1] + '.' + match[2]
                        dimensions.append(f"{dim1}*{dim2}")
                        main_dimension_found = True
                    elif pattern_idx in [5, 7, 11, 14, 15, 17, 20, 29]:  # Formats avec l/h/p/cm/mm (updated indices +1)
                        dim_parts = [part.replace(',', '.') for part in match if part]
                        if len(dim_parts) == 3:
                            # Vérifier si c'est une vraie dimension (pas trop petite)
                            dims = [float(d) for d in dim_parts]
                            if min(dims) >= 10 or pattern_idx in [5, 7]:  # Dimensions principales ou format avec cm/mm (updated indices +1)
                                dimensions.append("*".join(dim_parts))
                                if pattern_idx in [5, 7]:  # Format avec cm/mm = priorité haute (updated indices +1)
                                    main_dimension_found = True
                    elif pattern_idx == 20:  # Format avec espaces décimaux au milieu: "38 5 x 54 cm" (updated index +1)
                        dim1 = match[0] + '.' + match[1]  # "38" + "." + "5" = "38.5"
                        dim2 = match[2].replace(',', '.')
                        dimensions.append(f"{dim1}*{dim2}")
                        main_dimension_found = True
                    elif pattern_idx == 23:  # Format avec espaces décimaux sans cm: "3x7 5 m" (updated index +1)
                        dim1 = match[0].replace(',', '.')
                        dim2 = match[1] + '.' + match[2]  # "7" + "." + "5" = "7.5"
                        dimensions.append(f"{dim1}*{dim2}")
                        main_dimension_found = True
                    elif pattern_idx == 25:  # Format avec espaces décimaux 2D avec m: "1 5 x 10m" (updated index +1)
                        dim1 = match[0] + '.' + match[1]  # "1" + "." + "5" = "1.5"
                        dim2 = match[2].replace(',', '.')
                        dimensions.append(f"{dim1}*{dim2}")
                        main_dimension_found = True
                    elif pattern_idx == 27:  # Format unités mixtes "27 5 cm 3m" (updated index +1)
                        # "27 5" -> "27.5"
                        dim1 = match[0] + '.' + match[1]
                        dim2 = match[2].replace(',', '.')
                        dimensions.append(f"{dim1}*{dim2}")
                        main_dimension_found = True
                    else:  # Format 3D standard
                        dim_parts = [part.replace(',', '.') for part in match if part]
                        if len(dim_parts) == 3:
                            dimensions.append("*".join(dim_parts))
                elif len(match) == 2:
                    if pattern_idx == 7:  # Format dimensions principales 2D avec cm: "160x230cm" (updated index +1)
                        dim1 = match[0].replace(',', '.')
                        dim2 = match[1].replace(',', '.')
                        dimensions.append(f"{dim1}*{dim2}")
                        main_dimension_found = True
                    elif pattern_idx == 9:  # Format dimensions principales 2D avec mm: "160x230mm" (updated index +1)
                        dim1 = match[0].replace(',', '.')
                        dim2 = match[1].replace(',', '.')
                        dimensions.append(f"{dim1}*{dim2}")
                        main_dimension_found = True
                    elif pattern_idx == 10:  # Format dimensions principales 2D - priorité haute (updated index +1)
                        dim1 = match[0].replace(',', '.')
                        dim2 = match[1].replace(',', '.')
                        # Vérifier que ce sont des dimensions raisonnables (pas des codes produit)
                        try:
                            d1, d2 = float(dim1), float(dim2)
                            if 10 <= d1 <= 500 and 10 <= d2 <= 500:  # Dimensions réalistes en cm
                                dimensions.append(f"{dim1}*{dim2}")
                                main_dimension_found = True
                        except:
                            pass
                    elif pattern_idx in [12, 16]:  # Format diam ou l/p 2D (updated indices +1)
                        dim_parts = [part.replace(',', '.') for part in match if part]
                        if len(dim_parts) == 2:
                            dimensions.append("*".join(dim_parts))
                            main_dimension_found = True
                    elif pattern_idx in [21, 22]:  # Format éviter multiplicateurs (updated indices +1)
                        dim1 = match[0].replace(',', '.')
                        dim2 = match[1].replace(',', '.')
                        # Éviter les petites valeurs qui sont probablement des multiplicateurs
                        # Aussi éviter les nombres commençant par 0 (codes produits)
                        try:
                            d1, d2 = float(dim1), float(dim2)
                            if (d1 >= 20 and d2 >= 20 and 
                                not match[0].startswith('0') and not match[1].startswith('0')):  
                                dimensions.append(f"{dim1}*{dim2}")
                        except:
                            pass
                    elif pattern_idx == 28:  # Format avec cm collé (updated index +1)
                        dim_parts = [part.replace(',', '.') for part in match if part]
                        if len(dim_parts) == 2:
                            dimensions.append("*".join(dim_parts))
                            main_dimension_found = True
                    else:  # Format 2D standard
                        dim_parts = [part.replace(',', '.') for part in match if part]
                        if len(dim_parts) == 2:
                            # Éviter les dimensions trop petites sauf si c'est le seul match
                            try:
                                d1, d2 = float(dim_parts[0]), float(dim_parts[1])
                                if d1 >= 15 and d2 >= 15:  # Seulement dimensions >= 15
                                    dimensions.append("*".join(dim_parts))
                            except:
                                pass
            
            if dimensions:
                found_dimensions.extend(dimensions)
                # Si on a trouvé une dimension principale, arrêter la recherche
                if main_dimension_found and pattern_idx <= 6:
                    break
    
    if found_dimensions:
        # Supprimer les doublons tout en gardant l'ordre
        unique_dimensions = []
        seen = set()
        for dim in found_dimensions:
            if dim not in seen:
                unique_dimensions.append(dim)
                seen.add(dim)
        
        # Filtrer les dimensions contenant des segments avec zéro en tête (codes produits)
        filtered_dims = [dim for dim in unique_dimensions if not any(part.startswith('0') and not part.startswith('0.') for part in dim.split('*'))]
        if filtered_dims:
            unique_dimensions = filtered_dims
        
        # Si plusieurs dimensions, privilégier la plus grande (dimension principale)
        if len(unique_dimensions) > 1:
            # Calculer la "taille" de chaque dimension pour prioriser
            def dimension_size(dim_str):
                try:
                    parts = dim_str.split('*')
                    nums = [float(p) for p in parts]
                    return sum(nums) * len(nums)  # Somme pondérée par le nombre de dimensions
                except:
                    return 0
            
            unique_dimensions.sort(key=dimension_size, reverse=True)
            return unique_dimensions[0]  # Retourner seulement la plus grande
        else:
            return unique_dimensions[0]
    
    return None