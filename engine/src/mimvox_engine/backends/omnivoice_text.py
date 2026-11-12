"""Text normalization for the OmniVoice backend: digits, symbols and common abbreviations spelled out.

OmniVoice reads text as written: "2026" may come out digit by digit and "т.д." letter by letter. This module turns
them into words before synthesis. Ukrainian gets the full treatment (cases after prepositions, ordinal years and
dates, decimals, percent and currency with plural agreement, abbreviations); English gets years, ordinals, decimals,
percent and currency; every other language gets bare integers and decimals through ``num2words`` when it knows the
language, else the text is left as it is.

Standalone on purpose: the OmniVoice worker loads this file by its path (it runs in its own interpreter and imports
nothing else from ``mimvox_engine``). Needs only ``re`` and ``num2words`` (installed in the OmniVoice venv).

    normalize(text, lang) -> text
"""
import re

try:
    from num2words import num2words as _n2w
except ImportError:  # pragma: no cover - the worker venv has it; without it digits stay digits
    _n2w = None

# ------------------------------------------------------------------------------------------------- shared helpers

_STRESS = re.compile("[́̀]")
_SPACES = re.compile(r"[ \t  ]+")
# 1 000 000 / 1,000,000 / 1 000,5 -> plain digits (a thin or normal space between groups of three)
_GROUPED = re.compile(r"(?<![\d,.])(\d{1,3})((?:[   ]\d{3})+)(?![\d])")


def _ungroup(text):
    return _GROUPED.sub(lambda m: m.group(1) + re.sub(r"\D", "", m.group(2)), text)


def _words(n, lang, **kw):
    if _n2w is None:
        return str(n)
    try:
        return _n2w(n, lang=lang, **kw)
    except (NotImplementedError, OverflowError, ValueError, TypeError, KeyError, AttributeError):
        return str(n)


def _n2w_lang(lang):
    """num2words' name for an ISO 639-1 code, or None when it does not know the language."""
    if _n2w is None:
        return None
    alias = {"zh": "zh", "no": "no", "pt": "pt", "sr": "sr", "fa": "fa", "he": "he", "be": "be", "kk": "kz"}
    code = alias.get(lang, lang)
    try:
        _n2w(1, lang=code)
        return code
    except (NotImplementedError, OverflowError, ValueError, KeyError, AttributeError):
        return None


def normalize(text, lang):
    """Spell out what OmniVoice would misread. ``lang``: ISO 639-1 code of the text."""
    lang = (lang or "").lower()
    text = _STRESS.sub("", text)
    text = _SPACES.sub(" ", text)
    if lang == "uk":
        return _uk(text)
    if lang == "en":
        return _en(text)
    code = _n2w_lang(lang)
    if code is None:
        return text
    text = _ungroup(text)
    text = re.sub(r"(\d+)[.,](\d+)", lambda m: _words(float(f"{m.group(1)}.{m.group(2)}"), code), text)
    return re.sub(r"\d+(?!\.\s+[A-ZÀ-ÞĀ-ſА-Я])", lambda m: _words(int(m.group()), code), text)


# ------------------------------------------------------------------------------------------------------- English

_EN_UNITS = {"%": ("percent", "percent"), "$": ("dollar", "dollars"), "€": ("euro", "euros"), "£": ("pound", "pounds")}


def _en(text):
    text = _ungroup(re.sub(r"(?<=\d),(?=\d{3}\b)", "", text))
    text = re.sub(r"([$€£])\s?(\d+)\.(\d\d)\b", lambda m: f"{m.group(2)} {_EN_UNITS[m.group(1)][1]} and "
                  f"{int(m.group(3))} {'cent' if int(m.group(3)) == 1 else 'cents'}", text)
    text = re.sub(r"([$€£])\s?(\d+(?:\.\d+)?)", lambda m: f"{m.group(2)} {_EN_UNITS[m.group(1)][1]}", text)
    text = re.sub(r"\b(\d{1,2}):(\d{2})\b", lambda m: _words(int(m.group(1)), "en") + " " + (
        "o'clock" if m.group(2) == "00" else ("oh " if m.group(2)[0] == "0" else "") + _words(int(m.group(2)), "en")),
        text)
    text = re.sub(r"\b(\d+)(st|nd|rd|th)\b", lambda m: _words(int(m.group(1)), "en", to="ordinal"), text)
    text = re.sub(r"(\d+)\.(\d+)", lambda m: _words(int(m.group(1)), "en") + " point " +
                  " ".join(_words(int(d), "en") for d in m.group(2)), text)
    text = re.sub(r"(\d+(?:\s\w+)*?)\s?%", lambda m: m.group(1) + " percent", text)

    def num(m):
        s = m.group()
        n = int(s)
        if len(s) == 4 and 1100 <= n <= 2099:
            return _words(n, "en", to="year")
        return _words(n, "en")

    text = re.sub(r"\d+", num, text)
    return text.replace("&", " and ")


# ----------------------------------------------------------------------------------------------------- Ukrainian

# Prepositions after which a number is in the genitive ("до 5 років", "з 10 учасників", "близько 300 людей").
_UK_GEN = {"до", "від", "з", "із", "зі", "після", "близько", "без", "для", "біля", "коло", "крім", "замість",
           "протягом", "понад?", "більше", "менше", "щонайменше", "щонайбільше", "кількох", "серед", "навколо"}
_UK_GEN.discard("понад?")  # "понад" takes the accusative ("понад триста років")
_UK_LOC = {"у", "в", "на", "при", "об"}  # + a number + a locative noun ("у 5 містах") -- only with a known noun ending

_UK_MONTHS_GEN = ("січня", "лютого", "березня", "квітня", "травня", "червня", "липня", "серпня", "вересня",
                  "жовтня", "листопада", "грудня")

# Abbreviations replaced by words wherever they stand (case-insensitive).
_UK_ABBR = [
    (r"і т\.\s?д\.", "і так далі"), (r"т\.\s?д\.", "так далі"), (r"і т\.\s?п\.", "і тому подібне"),
    (r"т\.\s?п\.", "тому подібне"), (r"і т\.\s?ін\.", "і таке інше"), (r"та ін\.", "та інші"),
    (r"т\.\s?зв\.", "так званий"), (r"т\.\s?ч\.", "тому числі"), (r"напр\.", "наприклад"), (r"див\.", "дивись"),
    (r"вул\.", "вулиця"), (r"просп\.", "проспект"), (r"обл\.", "область"), (r"р-н", "район"), (r"ім\.", "імені"),
    (r"проф\.", "професор"), (r"акад\.", "академік"), (r"д-р", "доктор"), (r"№", "номер"),
]
# Abbreviated units, recognized only right after a number ("5 км", "100 грн", "3 тис.", "2026 р.").
_UK_UNIT_ABBR = r"(грн|тис|млн|млрд|км|кг|см|мм|год|хв|сек|рр|р|ст)\.?"
# Units that agree with the number before them: (one, few, many, genitive singular for fractions).
_UK_UNITS = {
    "грн": ("гривня", "гривні", "гривень", "гривні"), "тис": ("тисяча", "тисячі", "тисяч", "тисячі"),
    "млн": ("мільйон", "мільйони", "мільйонів", "мільйона"), "млрд": ("мільярд", "мільярди", "мільярдів", "мільярда"),
    "км": ("кілометр", "кілометри", "кілометрів", "кілометра"), "кг": ("кілограм", "кілограми", "кілограмів", "кілограма"),
    "см": ("сантиметр", "сантиметри", "сантиметрів", "сантиметра"),
    "мм": ("міліметр", "міліметри", "міліметрів", "міліметра"),
    "год": ("година", "години", "годин", "години"), "хв": ("хвилина", "хвилини", "хвилин", "хвилини"),
    "сек": ("секунда", "секунди", "секунд", "секунди"),
    "%": ("відсоток", "відсотки", "відсотків", "відсотка"), "$": ("долар", "долари", "доларів", "долара"),
    "€": ("євро", "євро", "євро", "євро"), "₴": ("гривня", "гривні", "гривень", "гривні"),
    "рр": ("роки", "роки", "років", "року"), "р": ("рік", "роки", "років", "року"),
}
_UK_FEM_UNITS = {"грн", "тис", "год", "хв", "сек", "₴"}

_ORD_ENDINGS = {  # nominative masculine ending -> (gen m/n, dat/loc m/n, nom f, gen f, loc f, nom n, nom pl)
    "ий": ("ого", "ому", "а", "ої", "ій", "е", "і"),
    "ій": ("ього", "ьому", "я", "ьої", "ій", "є", "і"),
}


def _plural(n, forms):
    """forms = (one, few, many): 1 рік, 2 роки, 5 років (11-14 are 'many')."""
    n = abs(int(n))
    if n % 10 == 1 and n % 100 != 11:
        return forms[0]
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return forms[1]
    return forms[2]


def _uk_card(n, case="nominative", feminine=False):
    kw = {}
    if case != "nominative":
        kw["case"] = case
    if feminine:
        kw["gender"] = "feminine"
    return _words(int(n), "uk", **kw)


def _uk_ord(n, form):
    """Ordinal of ``n``: form is one of nom_m, gen_m, loc_m, nom_f, gen_f, loc_f, nom_n."""
    words = _words(int(n), "uk", to="ordinal")
    if form == "nom_m" or not isinstance(words, str):
        return words
    head, _, last = words.rpartition(" ")
    for end, forms in _ORD_ENDINGS.items():
        if last.endswith(end):
            idx = {"gen_m": 0, "loc_m": 1, "nom_f": 2, "gen_f": 3, "loc_f": 4, "nom_n": 5}[form]
            last = last[: -len(end)] + forms[idx]
            break
    return f"{head} {last}".strip()


def _uk_decimal(whole, frac, case="nominative"):
    """3,5 -> три цілих п'ять десятих (1 ціла, 2-4/5+ цілих)."""
    w = int(whole)
    whole_words = _uk_card(w, case, feminine=True)
    cil = "ціла" if w % 10 == 1 and w % 100 != 11 else "цілих"
    f = int(frac)
    denom = {1: ("десята", "десятих"), 2: ("сота", "сотих"), 3: ("тисячна", "тисячних")}.get(len(frac))
    if denom is None:
        return f"{whole_words} кома {' '.join(_uk_card(int(d)) for d in frac)}"
    frac_words = _uk_card(f, case, feminine=True)
    dn = denom[0] if f % 10 == 1 and f % 100 != 11 else denom[1]
    if frac == "5" and case == "nominative":
        return f"{whole_words} з половиною" if w else "половина"
    return f"{whole_words} {cil} {frac_words} {dn}"


_UK_NUM = r"(\d+)(?:[,.](\d+))?"


def _uk(text):
    text = _ungroup(text)
    # abbreviations: plain words, or a \0 marker for units that agree with a number
    for pat, rep in _UK_ABBR:
        # an abbreviation that ends a sentence keeps the full stop it shared with it
        text = re.sub(r"(?<![\wʼ'’])" + pat + r"(?=\s*$|\s+[A-ZА-ЯІЇЄҐ])", rep + ("." if pat.endswith(".") else ""),
                      text, flags=re.IGNORECASE)
        text = re.sub(r"(?<![\wʼ'’])" + pat + r"(?![\w])", rep, text, flags=re.IGNORECASE)
    text = re.sub(r"(\d)\s*" + _UK_UNIT_ABBR + r"(?![\w])", "\\1 \0\\2", text)
    # time 10:30 -> десята тридцять (after "о" -> о десятій тридцять)
    text = re.sub(r"\b(о|об)\s+(\d{1,2}):(\d{2})\b",
                  lambda m: f"{m.group(1)} {_uk_ord(m.group(2), 'loc_f')} {_uk_minutes(m.group(3))}", text)
    text = re.sub(r"\b(\d{1,2}):(\d{2})\b", lambda m: f"{_uk_ord(m.group(1), 'nom_f')} {_uk_minutes(m.group(2))}",
                  text)
    # years: "у 2026 році", "2026 року", "2026 р.", "2026 рік", "у 1990-х"
    text = re.sub(r"(\d{1,4})(?:-?(?:му|ому|ім|ому))?\s+(році)\b", lambda m: f"{_uk_ord(m.group(1), 'loc_m')} році",
                  text)
    text = re.sub(r"(\d{1,4})(?:-?(?:го|ого))?\s+(року|\0р)(?![\w])", lambda m: f"{_uk_ord(m.group(1), 'gen_m')} року",
                  text)
    text = re.sub(r"(\d{4})\s+(рік)\b", lambda m: f"{_uk_ord(m.group(1), 'nom_m')} рік", text)
    text = re.sub(r"(\d{1,4})-(?:х|их)\b", lambda m: _uk_ord(m.group(1), "nom_m")[:-2] + "их", text)
    text = re.sub(r"\b([IVXL]{1,6})\s*(ст\.|столітт[яі])", lambda m: f"{_roman(m.group(1))} \0ст", text)
    text = re.sub(r"(\d+)\s*\0ст(?![\w])", lambda m: f"{_uk_ord(m.group(1), 'nom_n')} століття", text)
    # dates: "5 травня" -> п'ятого травня, "1-го" -> першого
    text = re.sub(r"(\d{1,2})(?:-?(?:го|ого))?\s+(" + "|".join(_UK_MONTHS_GEN) + r")\b",
                  lambda m: f"{_uk_ord(m.group(1), 'gen_m')} {m.group(2)}", text, flags=re.IGNORECASE)
    # ordinal suffixes: 5-й, 3-я, 2-га, 10-те, 1-го, 7-му
    suffix = {"й": "nom_m", "ий": "nom_m", "я": "nom_f", "а": "nom_f", "га": "nom_f", "та": "nom_f", "е": "nom_n",
              "те": "nom_n", "го": "gen_m", "ого": "gen_m", "му": "loc_m", "ому": "loc_m", "ої": "gen_f", "ій": "loc_f"}
    text = re.sub(r"(\d+)-(" + "|".join(sorted(suffix, key=len, reverse=True)) + r")\b",
                  lambda m: _uk_ord(m.group(1), suffix[m.group(2)]), text)
    # currency symbols before the number: $100 -> 100 $
    text = re.sub(r"([$€₴])\s?" + _UK_NUM, lambda m: f"{m.group(2)}{',' + m.group(3) if m.group(3) else ''} {m.group(1)}",
                  text)

    # a number (optionally after a preposition) followed by a unit that agrees with it
    unit_re = "|".join(re.escape(u) for u in sorted(("%", "$", "€", "₴"), key=len, reverse=True))
    marker_re = "|".join(sorted((k for k in _UK_UNITS if k not in ("%", "$", "€", "₴")), key=len, reverse=True))

    def with_unit(m):
        prep, whole, frac, unit = m.group(1), m.group(2), m.group(3), m.group(4) or m.group(5)
        case = "genitive" if prep and prep.strip().lower() in _UK_GEN else "nominative"
        forms = _UK_UNITS[unit]
        fem = unit in _UK_FEM_UNITS
        if frac:
            words = _uk_decimal(whole, frac, case)
            return f"{prep or ''}{words} {forms[3]}"
        n = int(whole)
        words = _uk_card(n, case, feminine=fem)
        if case == "genitive":
            noun = forms[3] if n % 10 == 1 and n % 100 != 11 else forms[2]
        else:
            noun = _plural(n, forms[:3])
        return f"{prep or ''}{words} {noun}"

    text = re.sub(r"(\b\w+\s+)?" + _UK_NUM + r"\s*(?:(" + unit_re + r")|\0(" + marker_re + r"))(?![\w])", with_unit,
                  text)
    text = re.sub(r"\0(" + marker_re + r")(?![\w])", lambda m: _UK_UNITS[m.group(1)][2], text)
    text = text.replace("\0", "")
    text = re.sub(r"(?<!\d)%", " відсотків", text)
    text = re.sub(r"\$", " доларів", text)

    # remaining numbers: decimals, then integers (genitive after a genitive preposition)
    def plain(m):
        prep, whole, frac = m.group(1), m.group(2), m.group(3)
        case = "genitive" if prep and prep.strip().lower() in _UK_GEN else "nominative"
        words = _uk_decimal(whole, frac, case) if frac else _uk_card(int(whole), case)
        return f"{prep or ''}{words}"

    text = re.sub(r"(\b\w+\s+)?" + r"(\d+)(?:,(\d+))?", plain, text)
    text = re.sub(r"(\d+)\.(\d+)", lambda m: _uk_decimal(m.group(1), m.group(2)), text)
    text = re.sub(r"\bодна тисяча\b", "тисяча", text)
    text = re.sub(r"\+(?=\w)|(?<=\w)\+", "", text)  # "+" stress marks used by other Ukrainian TTS front-ends
    return _SPACES.sub(" ", text).strip()


def _roman(r):
    vals = {"I": 1, "V": 5, "X": 10, "L": 50}
    total = 0
    for a, b in zip(r, r[1:] + " "):
        v = vals[a]
        total += -v if b != " " and vals[b] > v else v
    return str(total)


def _uk_minutes(mm):
    n = int(mm)
    if n == 0:
        return "нуль нуль"
    return ("нуль " if mm.startswith("0") else "") + _uk_card(n, feminine=True)
