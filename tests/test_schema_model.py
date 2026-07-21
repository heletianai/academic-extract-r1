"""validator 三态判定单测。"""

from src.schema_model import INVALID, SOFT_VIOLATION, VALID, validate_extraction


def good_obj():
    return {
        "task_type": "classification",
        "modalities": ["text"],
        "benchmarks": [{"name": "MMLU", "metric": "accuracy", "value": 85.3}],
        "open_source": True,
        "claims_sota": False,
        "method_keywords": ["LoRA", "GRPO", "distillation"],
        "one_line_summary": "A paper.",
        "limitation_mentioned": None,
    }


class TestValid:
    def test_clean_pass(self):
        r = validate_extraction(good_obj())
        assert r["status"] == VALID and r["alpha"] == 1.0

    def test_case_cast_rescued(self):
        o = good_obj()
        o["task_type"] = "  Classification "
        o["modalities"] = ["Text"]
        r = validate_extraction(o)
        assert r["status"] == VALID
        assert r["parsed"]["task_type"] == "classification"
        assert r["parsed"]["modalities"] == ["text"]

    def test_string_bool_and_number_cast(self):
        o = good_obj()
        o["open_source"] = "true"
        o["benchmarks"][0]["value"] = "85.3"  # pydantic lax cast str->float?
        r = validate_extraction(o)
        # open_source cast 救回；value 若 cast 成 float 则 VALID，若保持 str 则 SOFT——锁行为
        assert r["status"] in (VALID, SOFT_VIOLATION)
        assert r["parsed"]["open_source"] is True


class TestInvalid:
    def test_missing_key(self):
        o = good_obj()
        del o["claims_sota"]
        r = validate_extraction(o)
        assert r["status"] == INVALID and r["alpha"] is None

    def test_extra_key_forbidden(self):
        o = good_obj()
        o["hallucinated_field"] = 1
        assert validate_extraction(o)["status"] == INVALID

    def test_bad_enum(self):
        o = good_obj()
        o["task_type"] = "translation"  # 不在枚举
        assert validate_extraction(o)["status"] == INVALID

    def test_triple_extra_key(self):
        o = good_obj()
        o["benchmarks"][0]["dataset"] = "x"
        assert validate_extraction(o)["status"] == INVALID

    def test_not_dict(self):
        assert validate_extraction([1, 2])["status"] == INVALID
        assert validate_extraction(None)["status"] == INVALID


class TestSoft:
    def test_keyword_count(self):
        o = good_obj()
        o["method_keywords"] = ["only", "two"]
        r = validate_extraction(o)
        assert r["status"] == SOFT_VIOLATION and r["alpha"] == 0.8

    def test_empty_modalities(self):
        o = good_obj()
        o["modalities"] = []
        assert validate_extraction(o)["status"] == SOFT_VIOLATION

    def test_unparseable_value_str(self):
        o = good_obj()
        o["benchmarks"][0]["value"] = "eighty five"
        r = validate_extraction(o)
        assert r["status"] == SOFT_VIOLATION
        assert any("value is str" in f for f in r["soft_flags"])
