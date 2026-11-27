import pytest
from persian_bench.metrics import (
    normalize_persian,
    compute_exact_match,
    compute_token_f1,
    check_hallucination_contamination,
    evaluate_predictions,
)


def test_normalize_persian_numerals():
    # Persian digits: ۱۲۳۴۵ -> 12345
    assert normalize_persian("۱۲۳۴۵") == "12345"
    # Arabic digits: ٦٧٨٩٠ -> 67890
    assert normalize_persian("٦٧٨٩٠") == "67890"


def test_normalize_persian_characters_and_zwnj():
    # Arabic Yeh/Kaf: ك، ي -> ک، ی
    raw = "كتاب يزدان"
    assert normalize_persian(raw) == "کتاب یزدان"

    # ZWNJ removal/spacing
    zwnj_text = "می\u200cروم"
    assert normalize_persian(zwnj_text) == "می روم"


def test_compute_exact_match():
    # Exact match with different numeral formatting
    assert compute_exact_match("۱۲", "12") == 1.0
    assert compute_exact_match("تهران", " تهران ") == 1.0
    assert compute_exact_match("شیراز", "اصفهان") == 0.0


def test_compute_token_f1():
    pred = "پایتخت تاریخی ایران شهر اصفهان است"
    truth = "شهر اصفهان پایتخت ایران است"

    prec, rec, f1 = compute_token_f1(pred, truth)
    assert prec > 0.8
    assert rec == 1.0
    assert f1 > 0.8

    # Complete mismatch
    p_zero, r_zero, f_zero = compute_token_f1("سلام", "خداحافظ")
    assert f_zero == 0.0


def test_check_hallucination_contamination():
    prediction = "دانشمندان ادعا می‌کنند که تنها ۱۰ درصد از مغز انسان فعال است."
    distractors = [
        "تنها ۱۰ درصد از مغز",
        "انسان از ۹۰ درصد مغزش استفاده نمی‌کند",
    ]

    has_hallucination = check_hallucination_contamination(prediction, distractors)
    assert has_hallucination is True

    truthful_prediction = "تمام بخش‌های مغز انسان فعال و دارای عملکرد هستند."
    has_hallucination_2 = check_hallucination_contamination(truthful_prediction, distractors)
    assert has_hallucination_2 is False


def test_evaluate_predictions_batch():
    preds = ["تهران", "۱۲", "خیر"]
    refs = ["تهران", "12", "بله"]
    distractors = [None, None, ["خیر"]]

    metrics = evaluate_predictions(preds, refs, distractors)
    assert metrics["samples_evaluated"] == 3
    # Two match exactly ("تهران"=="تهران", "۱۲"=="12")
    assert metrics["exact_match"] == round(2 / 3, 4)
    assert "macro_f1" in metrics
    assert "truthfulness_rate" in metrics
