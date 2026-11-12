"""Names of the speech languages (``hello`` ``languages``): English name, name in the language itself, and the
"common" set the app lists first. Codes are those of ``backends.languages()`` (ISO 639-1, ``yue`` for Cantonese)."""
from __future__ import annotations

from typing import Any

# code -> (English name, native name)
NAMES: dict[str, tuple[str, str]] = {
    "en": ("English", "English"), "uk": ("Ukrainian", "Українська"), "de": ("German", "Deutsch"),
    "fr": ("French", "Français"), "es": ("Spanish", "Español"), "pl": ("Polish", "Polski"),
    "it": ("Italian", "Italiano"), "pt": ("Portuguese", "Português"), "cs": ("Czech", "Čeština"),
    "ru": ("Russian", "Русский"), "zh": ("Chinese", "中文"), "ja": ("Japanese", "日本語"),
    "yue": ("Cantonese", "粵語"), "th": ("Thai", "ไทย"), "ko": ("Korean", "한국어"),
    "vi": ("Vietnamese", "Tiếng Việt"), "id": ("Indonesian", "Bahasa Indonesia"), "no": ("Norwegian", "Norsk"),
    "ca": ("Catalan", "Català"), "hr": ("Croatian", "Hrvatski"), "lt": ("Lithuanian", "Lietuvių"),
    "sk": ("Slovak", "Slovenčina"), "sv": ("Swedish", "Svenska"), "el": ("Greek", "Ελληνικά"),
    "nl": ("Dutch", "Nederlands"), "bg": ("Bulgarian", "Български"), "sr": ("Serbian", "Српски"),
    "be": ("Belarusian", "Беларуская"), "da": ("Danish", "Dansk"), "kk": ("Kazakh", "Қазақ тілі"),
    "lv": ("Latvian", "Latviešu"), "sl": ("Slovenian", "Slovenščina"), "ar": ("Arabic", "العربية"),
    "et": ("Estonian", "Eesti"), "bs": ("Bosnian", "Bosanski"), "is": ("Icelandic", "Íslenska"),
    "mt": ("Maltese", "Malti"), "eu": ("Basque", "Euskara"), "fi": ("Finnish", "Suomi"),
    "ta": ("Tamil", "தமிழ்"), "sw": ("Swahili", "Kiswahili"), "fa": ("Persian", "فارسی"),
    "bn": ("Bengali", "বাংলা"), "as": ("Assamese", "অসমীয়া"), "mn": ("Mongolian", "Монгол"),
    "hu": ("Hungarian", "Magyar"), "ba": ("Bashkir", "Башҡортса"), "te": ("Telugu", "తెలుగు"),
    "ur": ("Urdu", "اردو"), "gl": ("Galician", "Galego"), "ml": ("Malayalam", "മലയാളം"),
    "ka": ("Georgian", "ქართული"), "mr": ("Marathi", "मराठी"), "pa": ("Punjabi", "ਪੰਜਾਬੀ"),
    "cy": ("Welsh", "Cymraeg"), "kn": ("Kannada", "ಕನ್ನಡ"), "tr": ("Turkish", "Türkçe"),
    "hi": ("Hindi", "हिन्दी"), "uz": ("Uzbek", "Oʻzbek"), "ne": ("Nepali", "नेपाली"),
}

# Listed first by the app ("Common"), in this order.
COMMON = ("en", "uk", "de", "fr", "es", "pl", "it", "pt", "cs", "ru")


def name(code: str) -> str:
    """English name of ``code`` (the code upper-cased when unknown)."""
    return NAMES.get(code, (code.upper(), code))[0]


def describe(codes: set[str] | list[str]) -> list[dict[str, Any]]:
    """``[{"code","name","native","common"}]``: the common languages first (in ``COMMON`` order), then the rest by
    English name."""
    have = set(codes)
    first = [c for c in COMMON if c in have]
    rest = sorted((c for c in have if c not in COMMON), key=name)
    return [{"code": c, "name": name(c), "native": NAMES.get(c, (c, c))[1], "common": c in COMMON}
            for c in first + rest]
