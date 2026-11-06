import pytest

from detention_ai import container

CASE_NUMBERS = ["MSCU4829153", "MAEU7731042", "HLCU5520187",
                "CMAU3301986", "TGHU9004512", "OOLU6612408"]


@pytest.mark.parametrize("raw,expected", [
    ("MSCU 482915-3", "MSCU4829153"),
    ("HLCU 552018-7", "HLCU5520187"),
    ("mscu4829153", "MSCU4829153"),
    ("MSCU-482915-3", "MSCU4829153"),
    ("MSCU482915", None),       # 6 digits
    ("MSC4829153", None),       # 3 letters
    ("MSCU48291534", None),     # 8 digits
])
def test_normalize(raw, expected):
    assert container.normalize(raw) == expected


def test_find_in_free_text_with_turkish_suffix():
    hits = container.find_all("HLCU 552018-7'nin şasisi arızalandı, iade 12 Ekim'e kaldı.")
    assert [h.value for h in hits] == ["HLCU5520187"]


def test_find_ignores_text_without_numbers():
    assert container.find_all("Eylül ayı faturaları sisteme yüklendi") == []


def test_check_digit_known_valid_number():
    # ISO 6346 standard example: CSQU3054383
    assert container.parse("CSQU3054383").check_digit_ok


@pytest.mark.parametrize("no", CASE_NUMBERS)
def test_case_numbers_fail_check_digit_but_are_kept(no):
    c = container.parse(no)
    assert c is not None and c.value == no
    assert not c.check_digit_ok
    assert "kontrol hanesi" in c.warning
