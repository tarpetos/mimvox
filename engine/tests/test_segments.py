import random

import pytest
from mimvox_engine import autoparts, narration
from mimvox_engine.dataset.text import count_words


def tokens(s):
    return max(1, len(s) // 4)


SENTENCES = [
    "The wind came down from the hills and rattled every shutter in the village.",
    "Nobody remembered a storm like it, not even the oldest fisherman by the harbour.",
    "She closed the book, set it on the table, and listened to the rain for a while.",
    "In the morning the roads were rivers, and the bakery opened two hours late.",
    "He counted the coins twice before he dared to knock on the door of the inn.",
    "Somewhere far away a dog barked, and the lamps along the quay went out one by one.",
]


def prose(words, seed=0):
    rnd = random.Random(seed)
    out, n = [], 0
    while n < words:
        para = " ".join(rnd.choice(SENTENCES) for _ in range(rnd.randint(3, 6)))
        out.append(para)
        n += count_words(para)
    return out


def test_title_is_its_own_segment():
    segs = narration.segments("Chapter One", "First sentence here. Second one.", tokens)
    text, pause, kind = segs[0]
    assert (text, kind, pause) == ("Chapter One.", "title", narration.PAUSES["title"])
    assert all(k != "title" for _, _, k in segs[1:])


def test_no_title_no_title_segment():
    segs = narration.segments(None, "Just a sentence.", tokens)
    assert segs == [("Just a sentence.", narration.PAUSES["end"], "end")]
    assert narration.segments("   ", "Just a sentence.", tokens) == segs


def test_long_title_is_cut_and_its_last_piece_is_the_title():
    title = ", ".join(["A very long chapter title that goes on and on"] * 12)
    segs = narration.segments(title, "Body.", tokens)
    kinds = [k for _, _, k in segs]
    n = kinds.index("title") + 1
    assert n > 1 and kinds[: n - 1] == ["clause"] * (n - 1)
    assert kinds[n:] == ["end"]


def test_pauses_follow_paragraphs_and_dialogue():
    text = "First paragraph ends here.\n\n— Said someone, opening a dialogue.\n\nNarration again.\n\nThe end."
    segs = narration.segments(None, text, tokens)
    assert [k for _, _, k in segs] == ["dialogue", "paragraph", "paragraph", "end"]
    assert [p for _, p, _ in segs] == [narration.PAUSES[k] for k in ("dialogue", "paragraph", "paragraph", "end")]


def test_long_paragraph_is_cut_within_the_budget():
    para = " ".join(SENTENCES * 4)
    segs = narration.segments(None, para, tokens)
    assert len(segs) > 1
    assert all(tokens(s) <= narration.BUDGET_TOKENS for s, _, _ in segs)
    assert " ".join(s for s, _, _ in segs).split() == narration.clean_text(para).split()
    assert segs[-1][2] == "end"
    assert all(k in ("sentence", "clause", "question", "exclaim", "ellipsis") for _, _, k in segs[:-1])


def test_clean_text():
    assert narration.clean_text("Wait... what--no!!") == "Wait… what — no!"
    assert narration.clean_text("a ,b;c\n\n\n- Yes.") == "a, b; c\n\n— Yes."


def test_suggest_leaves_short_chapters_alone():
    chapters = [{"title": "One", "text": "\n\n".join(prose(1000))}]
    assert autoparts.suggest(chapters, "en") is None


def test_suggest_splits_on_hidden_headings():
    paras = []
    for n in (1, 2, 3):
        paras += [f"Chapter {n}", *prose(1500, seed=n)]
    text = "\n\n".join(paras)
    out = autoparts.suggest([{"title": "Everything", "text": text}], "en")
    assert out["how"] == "headings"
    assert [c["title"] for c in out["chapters"]] == ["Chapter 1", "Chapter 2", "Chapter 3"]
    assert all(not c["text"].startswith("Chapter") for c in out["chapters"])


def test_suggest_splits_on_scene_breaks():
    scenes = [prose(1500, seed=n) for n in range(4)]
    text = "\n\n* * *\n\n".join("\n\n".join(s) for s in scenes)
    out = autoparts.suggest([{"title": "Long", "text": text}], "en")
    assert out["how"] == "breaks"
    parts = out["chapters"]
    assert len(parts) >= 2
    assert all("* * *" not in c["text"] for c in parts)
    assert sum(count_words(c["text"]) for c in parts) == count_words(text)


@pytest.mark.parametrize(("lang", "word"), [("en", "Part"), ("uk", "Частина"), (None, "")])
def test_suggest_splits_by_length_into_even_parts(lang, word):
    paras = prose(9000, seed=7)
    text = "\n\n".join(paras)
    out = autoparts.suggest([{"title": "Only", "text": text}], lang)
    assert out["how"] == "length"
    parts = out["chapters"]
    sizes = [count_words(c["text"]) for c in parts]
    assert len(parts) == round(count_words(text) / autoparts.TARGET_WORDS)
    assert min(sizes) >= autoparts.MIN_WORDS
    assert max(sizes) <= 1.4 * sum(sizes) / len(sizes)
    assert "\n\n".join(c["text"] for c in parts) == text
    assert all(c["text"].rstrip()[-1] in ".!?" for c in parts)
    assert [c["title"] for c in parts] == [f"{word} {k}".strip() for k in range(1, len(parts) + 1)]


def test_suggest_names_parts_after_their_chapter_when_there_are_several():
    long = "\n\n".join(prose(5000, seed=3))
    chapters = [{"title": "Intro", "text": "\n\n".join(prose(800, seed=1))}, {"title": "Big", "text": long}]
    out = autoparts.suggest(chapters, "en")
    titles = [c["title"] for c in out["chapters"]]
    assert titles[0] == "Intro"
    assert titles[1:] == ["Big"] + [f"Big ({k})" for k in range(2, len(titles))]
