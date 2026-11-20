import pytest

from chatmodel.preprocessing import normalize_text


@pytest.mark.parametrize(("raw", "expected"), [
    ("  Hello\nWORLD! ", "hello world"),
    ("stup1d 1d10t", "stupid idiot"),
    ("loooooser", "looser"),
    ("İĞÜŞÖÇ ı", "igusoc i"),
    ("Visit https://example.com/test or me@example.com", "visit or"),
    ("ＳＴＵＰＩＤ\u200b!", "stupid"),
    ("game 2026", "game 2026"),
    ("!!! 😀", ""),
])
def test_normalization(raw, expected):
    assert normalize_text(raw) == expected


def test_non_string_rejected():
    with pytest.raises(TypeError):
        normalize_text(None)
