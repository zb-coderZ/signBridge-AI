from core.sentence_builder import SentenceBuilder, gloss_to_sentence, rule_based_sentence
from core.config import load_config


def test_letters_concatenate():
    sb = SentenceBuilder()
    for c in "CAT":
        sb.add(c)
    assert sb.text == "CAT"


def test_space_and_words():
    sb = SentenceBuilder()
    sb.add("H"); sb.add("I"); sb.space(); sb.add("WAVE")
    assert sb.text == "HI WAVE"
    sb.add("NOD")
    assert sb.text == "HI WAVE NOD"


def test_no_double_or_leading_space():
    sb = SentenceBuilder()
    sb.space()
    assert sb.text == ""
    sb.add("A"); sb.space(); sb.space()
    assert sb.text == "A "


def test_backspace_removes_unit_and_auto_space():
    sb = SentenceBuilder()
    sb.add("A"); sb.add("WAVE")
    assert sb.text == "A WAVE"
    sb.backspace()
    assert sb.text == "A"
    sb.backspace(); sb.backspace()                # extra backspace on empty is safe
    assert sb.text == ""


def test_clear():
    sb = SentenceBuilder()
    sb.add("A"); sb.add("B"); sb.clear()
    assert sb.text == "" and len(sb) == 0


def test_blank_token_ignored():
    sb = SentenceBuilder()
    sb.add("  ")
    assert sb.text == ""


def test_rule_based_cleanup():
    assert rule_based_sentence("ME ME GO STORE") == "I go store."
    assert rule_based_sentence("WHAT YOUR NAME") == "What your name?"
    assert rule_based_sentence("") == ""


def test_gloss_falls_back_to_rules_without_key(monkeypatch):
    cfg = load_config()
    monkeypatch.delenv(cfg["llm"]["env_key"], raising=False)
    s, method = gloss_to_sentence("ME HAPPY", cfg, use_llm=True)
    assert method == "rules" and s == "I happy."
