"""claims_sota 白名单正则（checklist 硬化 #2 兑现——7.21 审计抓出的唯一口径走样）。

用途（正则是复核信号，不替代教师语义判断）：
(a) 蒸馏后自动交叉核对：教师 claims_sota 与正则不一致 → 样本带 sota_disagree 标记进 review 队列
(b) 人工黄金集预标注基线（可复现、零成本）
规则=checklist FROZEN：白名单词面命中且非比较引用（达成断言 true / comparable-to 类 false）。
"""

from __future__ import annotations

import re

WHITELIST = re.compile(
    r"\bsota\b"
    r"|state[-\s]of[-\s]the[-\s]art"
    r"|\boutperform\w*\s+all\b"
    r"|\bsurpass\w*\s+all\b"
    r"|\bprevious\s+best\b",
    re.IGNORECASE,
)

# match 前窗口内出现这些 → 该次命中是比较引用或否定，不是达成断言
# （红队 P2 修复：补否定词 not/fails to/without 等；窗口 30→60 且不锚定紧邻，
#   防 "broadly comparable, in some settings, to the SOTA" 类插入语击穿）
EXCLUSION_NEAR = re.compile(
    r"(comparable|close\s+to|match(es|ing)?|approach(es|ing)?|near|on\s+par"
    r"|\bnot\b|\bno\b|does\s+not|do\s+not|fails?\s+to|without|unable\s+to|falls?\s+short)",
    re.IGNORECASE,
)

_PREFIX_WINDOW = 60

# 子句边界：排除词只在 match 所在子句内有效——防宽窗口把
# "comparable to SOTA on X, and achieves SOTA on Y" 的第二个达成断言误排
_CLAUSE_SPLIT = re.compile(r"[;:]|,\s+(and|but|while|whereas|although|though)\b", re.IGNORECASE)


def claims_sota_by_regex(text: str) -> bool:
    """任一白名单命中且其**同子句**前缀窗口无排除/否定词 → true；否则 false。

    定位是 QA 交叉信号非裁决（教师语义为准）：排除从宽（宁可漏报 true），
    减少流向人工抽检队列的假 disagree 噪声。
    """
    if not text:
        return False
    for m in WHITELIST.finditer(text):
        prefix = text[max(0, m.start() - _PREFIX_WINDOW): m.start()]
        last_clause = _CLAUSE_SPLIT.split(prefix)[-1] or ""
        if not EXCLUSION_NEAR.search(last_clause):
            return True
    return False
