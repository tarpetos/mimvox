"""Language tables: Whisper codes -> NLLB-200 codes. (The languages the voice can speak: ``backends.languages()``.)"""
from __future__ import annotations

import os

NLLB_DIR_NAME = "nllb-200-distilled-600M"
NLLB_REPO = "facebook/nllb-200-distilled-600M"

# Translators (NLLB-200), best first: id -> folder under <data>/runtime/models (setup downloads them; the 1.3B is
# optional and stored as fp16 safetensors).
TRANSLATORS = {"nllb-1.3b": "nllb-200-distilled-1.3B", "nllb-600m": NLLB_DIR_NAME}


def installed_translators(models_root: str) -> list[str]:
    """The translators whose files are in ``models_root``, best first (``hello`` -> ``translators``)."""
    out = []
    for name, folder in TRANSLATORS.items():
        d = os.path.join(models_root, folder)
        if (all(os.path.isfile(os.path.join(d, f)) for f in ("config.json", "sentencepiece.bpe.model"))
                and any(os.path.isfile(os.path.join(d, f)) for f in ("model.safetensors", "pytorch_model.bin"))):
            out.append(name)
    return out

# Whisper language code -> NLLB-200 language code (languages NLLB does not have are absent).
NLLB = {
    "en": "eng_Latn", "zh": "zho_Hans", "de": "deu_Latn", "es": "spa_Latn", "ru": "rus_Cyrl", "ko": "kor_Hang",
    "fr": "fra_Latn", "ja": "jpn_Jpan", "pt": "por_Latn", "tr": "tur_Latn", "pl": "pol_Latn", "ca": "cat_Latn",
    "nl": "nld_Latn", "ar": "arb_Arab", "sv": "swe_Latn", "it": "ita_Latn", "id": "ind_Latn", "hi": "hin_Deva",
    "fi": "fin_Latn", "vi": "vie_Latn", "he": "heb_Hebr", "uk": "ukr_Cyrl", "el": "ell_Grek", "ms": "zsm_Latn",
    "cs": "ces_Latn", "ro": "ron_Latn", "da": "dan_Latn", "hu": "hun_Latn", "ta": "tam_Taml", "no": "nob_Latn",
    "th": "tha_Thai", "ur": "urd_Arab", "hr": "hrv_Latn", "bg": "bul_Cyrl", "lt": "lit_Latn", "mi": "mri_Latn",
    "ml": "mal_Mlym", "cy": "cym_Latn", "sk": "slk_Latn", "te": "tel_Telu", "fa": "pes_Arab", "lv": "lvs_Latn",
    "bn": "ben_Beng", "sr": "srp_Cyrl", "az": "azj_Latn", "sl": "slv_Latn", "kn": "kan_Knda", "et": "est_Latn",
    "mk": "mkd_Cyrl", "eu": "eus_Latn", "is": "isl_Latn", "hy": "hye_Armn", "ne": "npi_Deva", "mn": "khk_Cyrl",
    "bs": "bos_Latn", "kk": "kaz_Cyrl", "sq": "als_Latn", "sw": "swh_Latn", "gl": "glg_Latn", "mr": "mar_Deva",
    "pa": "pan_Guru", "si": "sin_Sinh", "km": "khm_Khmr", "sn": "sna_Latn", "yo": "yor_Latn", "so": "som_Latn",
    "af": "afr_Latn", "ka": "kat_Geor", "be": "bel_Cyrl", "tg": "tgk_Cyrl", "sd": "snd_Arab", "gu": "guj_Gujr",
    "am": "amh_Ethi", "lo": "lao_Laoo", "uz": "uzn_Latn", "fo": "fao_Latn", "ht": "hat_Latn", "ps": "pbt_Arab",
    "tk": "tuk_Latn", "nn": "nno_Latn", "mt": "mlt_Latn", "lb": "ltz_Latn", "my": "mya_Mymr", "bo": "bod_Tibt",
    "tl": "tgl_Latn", "mg": "plt_Latn", "as": "asm_Beng", "tt": "tat_Cyrl", "ln": "lin_Latn", "ha": "hau_Latn",
    "ba": "bak_Cyrl", "su": "sun_Latn", "yue": "yue_Hant", "jw": "jav_Latn", "sa": "san_Deva",
}

# Common names Whisper may return that differ from the table above.
_ALIAS = {"iw": "he", "in": "id", "jv": "jw", "nb": "no", "cmn": "zh"}


def norm(code: str | None) -> str:
    c = (code or "").strip().lower().replace("_", "-").split("-")[0]
    return _ALIAS.get(c, c)


def nllb_code(code: str | None) -> str | None:
    return NLLB.get(norm(code))


def script_language_guess(text: str) -> str | None:
    """A crude language guess from the script, used only when the caller does not say what language a text is in.
    Returns ``None`` for Latin text (assumed speakable), else a code that is not speakable."""
    cyr = sum(1 for ch in text if "Ѐ" <= ch <= "ӿ")
    lat = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    if cyr > lat:
        return "uk" if any(ch in "іїєґІЇЄҐ" for ch in text) else "ru"
    return None
