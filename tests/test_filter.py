"""filter_papers 单测（fixture 合成数据）。"""

import json

from src.data.filter_papers import has_numeric_signal, load_and_filter, stratified_sample


def _row(id="2509.00001", cats="cs.CL cs.AI", created="2025-09-01", abstract=None, updated=""):
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
        rows = [_row(abstract="old " + "A" * 200), _row(abstract="new " + "A" * 200)]
        p = _write(tmp_path, rows)
        kept, stats = load_and_filter(p)
        assert stats["unique_ids"] == 1
        assert kept[0]["abstract"].startswith("new")

    def test_primary_category_only(self, tmp_path):
        rows = [
            _row("2509.00001", cats="cs.CL cs.AI"),   # 主类命中
            _row("2509.00002", cats="cs.AI cs.CL"),   # cs.CL 只是交叉 → 拒
            _row("2509.00003", cats="cs.LG stat.ML"), # 主类命中
        ]
        kept, _ = load_and_filter(_write(tmp_path, rows))
        assert {r["id"] for r in kept} == {"2509.00001", "2509.00003"}

    def test_id_prefix_window(self, tmp_path):
        # issues-log #002：时间窗按 id 前缀（v1 提交年月），created 不可信不参与筛选
        rows = [
            _row("2506.99999", created="2025-09-01"),  # 2025-06 提交（created 是 replace 日）→ 拒
            _row("2507.00001", created="2018-01-01"),  # 2025-07 提交（created 脏值）→ 收
            _row("1306.1870", created="2025-06-30"),   # 老论文最近 replace → 拒
            _row("cs/0701001", created="2025-08-01"),  # 旧式 id → 拒
        ]
        kept, _ = load_and_filter(_write(tmp_path, rows))
        assert {r["id"] for r in kept} == {"2507.00001"}

    def test_abstract_length_bounds(self, tmp_path):
        rows = [_row("2509.00001", abstract="short"), _row("2509.00002", abstract="A" * 3500), _row("2509.00003")]
        kept, _ = load_and_filter(_write(tmp_path, rows))
        assert {r["id"] for r in kept} == {"2509.00003"}


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
