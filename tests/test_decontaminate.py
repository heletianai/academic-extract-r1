"""13-gram 去污染单测。"""

from src.data.decontaminate import find_contaminated, ngrams, tokenize

LONG = ("we propose a novel method for extracting structured metadata from academic "
        "papers using reinforcement learning with verifiable rewards and schema validation")


def row(id, text):
    return {"id": id, "title": "", "abstract": text}


class TestNgrams:
    def test_short_doc_single_gram(self):
        assert len(ngrams(tokenize("only three words"), 13)) == 1

    def test_empty(self):
        assert ngrams([], 13) == set()


class TestFindContaminated:
    def test_exact_duplicate_caught(self):
        hits = find_contaminated([row("t1", LONG)], [row("h1", LONG)])
        assert len(hits) == 1 and hits[0]["against_id"] == "h1"

    def test_shared_long_span_caught(self):
        # 前半段 13+ 词相同（近重复版本论文），后半不同
        a = LONG + " additional sentences here"
        b = LONG + " completely different ending with other words"
        hits = find_contaminated([row("t1", a)], [row("h1", b)])
        assert len(hits) == 1

    def test_different_docs_clean(self):
        other = ("this paper studies the convergence properties of stochastic gradient "
                 "descent under heavy tailed noise assumptions in deep networks")
        assert find_contaminated([row("t1", LONG)], [row("h1", other)]) == []

    def test_messages_format_supported(self):
        tr = {"id": "t1", "messages": [{"role": "system", "content": "s"},
                                        {"role": "user", "content": f"Title: X\nAbstract: {LONG}"},
                                        {"role": "assistant", "content": "{}"}]}
        assert len(find_contaminated([tr], [row("h1", LONG)])) == 1
