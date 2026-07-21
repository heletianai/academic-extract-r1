"""filter_papers 单测（fixture 合成数据）。"""

import json

from src.data.filter_papers import has_numeric_signal, load_and_filter, stratified_sample


def _row(id, cats="cs.CL cs.AI", created="2025-09-01", abstract=None, updated=""):
    return {
        "id": id, "created": created, "updated": updated,
        "title": f"Paper {id}",
        "abstract": abstract if abstract is not None else "A" * 200,
        "categories": cats,
    }


def _write(tmp_path, rows):
    p = tmp_path / "raw.jsonl"
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return p


class TestNumericSignal:
    def test_two_decimals_hit(self):
        assert has_numeric_signal("We achieve 85.3 accuracy and 0.91 F1.")

    def test_percent_counts(self):
        assert has_numeric_signal("gains 12% and 7.5% on two tasks")

    def test_plain_ints_ignored(self):
        # 纯整数（年份/层数）不算数值结果信号
        assert not has_numeric_signal("We use 12 layers trained in 2025.")


class TestLoadAndFilter:
    def test_dedup_keeps_last(self, tmp_path):
        rows = [_row("1", abstract="old " + "A" * 200), _row("1", abstract="new " + "A" * 200)]
        p = _write(tmp_path, rows)
        kept, stats = load_and_filter(p)
        assert stats["unique_ids"] == 1
        assert kept[0]["abstract"].startswith("new")

    def test_primary_category_only(self, tmp_path):
        rows = [
            _row("1", cats="cs.CL cs.AI"),   # 主类命中
            _row("2", cats="cs.AI cs.CL"),   # cs.CL 只是交叉 → 拒
            _row("3", cats="cs.LG stat.ML"), # 主类命中
        ]
        kept, _ = load_and_filter(_write(tmp_path, rows))
        assert {r["id"] for r in kept} == {"1", "3"}

    def test_created_cutoff(self, tmp_path):
        rows = [_row("1", created="2025-06-30"), _row("2", created="2025-07-01")]
        kept, _ = load_and_filter(_write(tmp_path, rows))
        assert {r["id"] for r in kept} == {"2"}

    def test_abstract_length_bounds(self, tmp_path):
        rows = [_row("1", abstract="short"), _row("2", abstract="A" * 3500), _row("3")]
        kept, _ = load_and_filter(_write(tmp_path, rows))
        assert {r["id"] for r in kept} == {"3"}


class TestStratifiedSample:
    def _pool(self):
        pool = []
        for i in range(80):
            r = _row(f"n{i}", abstract="acc 85.3 and 0.91 " + "A" * 200)
            r["numeric_signal"] = True
            pool.append(r)
        for i in range(80):
            r = _row(f"p{i}")
            r["numeric_signal"] = False
            pool.append(r)
        return pool

    def test_ratio_respected(self):
        s = stratified_sample(self._pool(), 100, numeric_ratio=0.5, seed=42)
        n_num = sum(1 for r in s if r["numeric_signal"])
        assert len(s) == 100 and n_num == 50

    def test_shortfall_backfilled(self):
        # 数字组只有 10 条，要 60% → 用 plain 补满
        pool = self._pool()[70:]  # 10 numeric + 80 plain
        s = stratified_sample(pool, 50, numeric_ratio=0.6, seed=1)
        assert len(s) == 50
        assert sum(1 for r in s if r["numeric_signal"]) == 10

    def test_deterministic_by_seed(self):
        a = stratified_sample(self._pool(), 40, 0.5, seed=7)
        b = stratified_sample(self._pool(), 40, 0.5, seed=7)
        assert [r["id"] for r in a] == [r["id"] for r in b]
