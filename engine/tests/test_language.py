import os
import subprocess
import sys
from pathlib import Path

import mimvox_engine
import pytest
from factories import text_file
from mimvox_engine.dataset import lang

SAMPLES = {
    "uk": "Він сказав, що вона вже пішла додому, але ми ще чекали її біля школи. Це була їхня улюблена пісня.",
    "ru": "Он сказал, что она уже ушла домой, но мы ещё ждали её у школы. Это была их любимая песня.",
    "be": "Ён сказаў, што яна ўжо пайшла дадому, але мы яшчэ чакалі яе ля школы. Гэта была іх любімая песня.",
    "bg": "Той каза, че тя вече си е тръгнала, но ние още я чакахме пред училището. Това беше любимата им песен.",
    "en": "He said that she had already gone home, but we were still waiting for her at the school.",
    "de": "Er sagte, dass sie schon nach Hause gegangen ist, aber wir warteten noch vor der Schule auf sie.",
    "fr": "Il a dit qu'elle était déjà rentrée à la maison, mais nous l'attendions encore devant l'école.",
    "es": "Él dijo que ella ya se había ido a casa, pero nosotros todavía la esperábamos en la escuela.",
    "it": "Lui ha detto che lei era già tornata a casa, ma noi la aspettavamo ancora davanti alla scuola.",
    "pl": "Powiedział, że ona już poszła do domu, ale my wciąż czekaliśmy na nią przed szkołą.",
    "ro": "El a spus că ea a plecat deja acasă, dar noi încă o așteptam în fața școlii.",
    "pt": "Ele disse que ela já tinha ido para casa, mas nós ainda esperávamos por ela na escola. Não sei.",
    "cs": "Řekl, že už odešla domů, ale my jsme na ni stále čekali před školou. To byla její oblíbená píseň.",
    "nl": "Hij zei dat zij al naar huis was gegaan, maar wij wachtten nog op haar voor de school.",
}


@pytest.mark.parametrize("code", sorted(SAMPLES))
def test_detects_language(code):
    assert lang.detect(SAMPLES[code], among=lang.DETECTABLE) == code


@pytest.mark.parametrize(
    ("code", "text"),
    [
        ("be", "Гэта была цёплая летняя ноч. Дзеці гулялі на беразе ракі, а маці глядзела на іх з акна."),
        ("be", "Яна не ведала, што сказаць, бо ён зноў маўчаў."),
        ("be", "Сёння вечарам мы паедзем да бабулі, калі не будзе дажджу."),
        ("ru", "Это был тёплый летний вечер. Дети играли на берегу реки, а мать смотрела на них из окна."),
        ("ru", "Мы долго шли по лесу, и только к вечеру увидели огни деревни."),
        ("ru", "Он сказал, что она ушла домой."),
        ("uk", "Це була тепла літня ніч. Діти гралися на березі річки, а мати дивилася на них з вікна."),
        ("uk", "Сьогодні ввечері ми поїдемо до бабусі, якщо не буде дощу."),
        ("bg", "Това беше топла лятна нощ. Децата играеха на брега на реката, а майка им ги гледаше от прозореца."),
        ("bg", "Тази вечер ще отидем при баба, ако не вали."),
    ],
)
def test_tells_cyrillic_languages_apart(code, text):
    assert lang.detect(text, among=lang.DETECTABLE) == code


def test_belarusian_with_its_own_letters():
    text = "Ён усё ўбачыў і ўздыхнуў: усе ўжо пайшлі, а яна яшчэ стаяла ў двары."
    assert lang.detect(text, among=lang.DETECTABLE) == "be"


TIE = "Я ведаю, што ты прыйдзеш заўтра. Ён не хацеў гэтага, але яна ўсё ж такі пайшла да яго."


@pytest.mark.parametrize("text", [TIE, "la"])
def test_detection_does_not_depend_on_the_hash_seed(text):
    src = str(Path(mimvox_engine.__file__).parent.parent)
    code = f"from mimvox_engine.dataset import lang; print(lang.detect({text!r}, among=lang.DETECTABLE))"
    seen = set()
    for seed in range(4):
        env = {**os.environ, "PYTHONHASHSEED": str(seed), "PYTHONPATH": src}
        out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
        seen.add(out.stdout.strip())
    assert len(seen) == 1, seen


@pytest.mark.parametrize("text", ["", "   ", "12345 678.90", "!!! ??? ...", "— 42 —"])
def test_no_letters_gives_the_default(text):
    assert lang.detect(text, among=lang.DETECTABLE) == lang.DEFAULT


def test_default_pool_is_the_recording_languages():
    assert lang.detect(SAMPLES["de"]) == "de"
    assert lang.detect(SAMPLES["nl"]) in lang.LANGUAGES


@pytest.mark.parametrize(
    ("value", "code"), [("uk-UA", "uk"), ("EN", "en"), ("pl_PL", "pl"), ("auto", None), ("", None)]
)
def test_normalize_code(value, code):
    assert lang.normalize_code(value) == code


def test_normalize_code_rejects_unknown():
    with pytest.raises(ValueError, match="Unsupported language 'xx'"):
        lang.normalize_code("xx")


@pytest.mark.parametrize("code", ["uk", "ru", "bg", "de", "pl"])
def test_import_detects_the_book_language(tmp_path, import_book, code):
    r = import_book(text_file(tmp_path / "b.txt", (SAMPLES[code] + " ") * 5))
    assert r["language"] == code and r["language_detected"] is True


def test_import_keeps_a_given_language(tmp_path, import_book):
    path = text_file(tmp_path / "b.txt", SAMPLES["ru"])
    r = import_book(path, language="uk-UA")
    assert r["language"] == "uk" and r["language_detected"] is False
    assert import_book(path, language="it")["language"] == "it"
