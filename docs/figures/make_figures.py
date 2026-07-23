#!/usr/bin/env python3
"""
Generate publication-quality figures for the academic-extract-r1 project.

Pipeline: distilled SFT -> single-turn GRPO -> multi-turn agentic GRPO
Model:    Qwen3-4B-Instruct-2507, LoRA r32, academic paper metadata extraction.

Figures (all -> docs/figures/):
  fig1-sixway-comparison.png  six-way eval bar chart with 95% CI
  fig2-collapse-vs-fix.png    #013 behavioural collapse vs penalty-fixed run
  fig3-mtgrpo-training.png    multi-turn GRPO v2 training monitor (2x3)
  fig4-sft-scaling.png        SFT loss curve + data-scaling curve

Data sources are resolved relative to this file (docs/figures/ -> repo root).
Run:  python3 docs/figures/make_figures.py
"""

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker  # used in fig4 (log-axis minor formatter)

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent          # docs/figures
ROOT = HERE.parent.parent                        # repo root
ART = ROOT / "runs" / "gpu-artifacts"
OUT = HERE

# ----------------------------------------------------------------------------
# Global style: clean, white background, whitegrid, dpi 150
# ----------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10.5,
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.8,
    "grid.color": "#d9d9d9",
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "xtick.color": "#333333",
    "ytick.color": "#333333",
})

# A restrained, colour-blind-friendly palette
C_BASE   = "#9aa0a6"   # grey  - base / untrained baselines
C_SFT    = "#4c72b0"   # blue  - SFT
C_GRPO   = "#55a868"   # green - single-turn GRPO
C_MTPROT = "#c9a227"   # amber - multi-turn protocol (untrained)
C_MTGRPO = "#dd8452"   # orange- multi-turn agentic GRPO
C_TEACH  = "#8172b3"   # purple- teacher upper bound
C_SEARCH = "#c44e52"   # red   - search_rate
C_ANSW   = "#4c72b0"   # blue  - answered_rate
C_WARM   = "#b0b0b0"   # warmup marker


def _read_json(p):
    with open(p) as f:
        return json.load(f)


def _overall(p):
    """Return (mean, lo, hi) from an eval json's overall block."""
    d = _read_json(p)
    o = d["overall"]
    return o["mean"], o["ci95"][0], o["ci95"][1]


def _load_detail(p):
    """Load a reward_detail jsonl -> list of dicts."""
    rows = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _annotate(ax, x, y, text, dy=8, dx=0, color="#222", fs=8, ha="center"):
    ax.annotate(text, (x, y), textcoords="offset points", xytext=(dx, dy),
                ha=ha, fontsize=fs, color=color)


# ============================================================================
# FIGURE 1 -- Six-way comparison, horizontal bars with 95% CI
# ============================================================================
def fig1():
    # (label, color, eval-json path or None, fallback mean, fallback lo, fallback hi)
    specs = [
        ("Base (few-shot)",                       C_BASE,
         ART / "eval-base_fewshot-20260721.json",              0.6713, 0.6376, 0.7021),
        ("SFT",                                   C_SFT,
         ART / "eval-student_zeroshot-20260721.json",          0.9010, 0.8893, 0.9129),
        ("SFT + single-turn GRPO",                C_GRPO,
         ART / "grpo-20260722" / "eval-grpo-v2.json",          0.9044, 0.8923, 0.9166),
        ("SFT + multi-turn protocol (untrained)", C_MTPROT,
         ART / "mt-grpo-0722" / "eval-mt_sft.json",            0.7789, 0.7389, 0.8134),
        ("SFT + multi-turn agentic GRPO",         C_MTGRPO,
         ART / "mt-grpo-0722" / "eval-mt_grpo.json",           0.8932, 0.8747, 0.9092),
        ("API teacher (self-consistency)",        C_TEACH,
         ART / "eval-api_fewshot-20260721.json",               0.9753, 0.9636, 0.9844),
    ]

    labels, colors, means, los, his, is_teacher = [], [], [], [], [], []
    for lab, col, path, fm, flo, fhi in specs:
        if path is not None and Path(path).exists():
            m, lo, hi = _overall(path)
        else:
            m, lo, hi = fm, flo, fhi
        labels.append(lab); colors.append(col)
        means.append(m); los.append(lo); his.append(hi)
        is_teacher.append("teacher" in lab.lower())

    n = len(specs)
    y = list(range(n))[::-1]           # first spec at top

    fig, ax = plt.subplots(figsize=(9.2, 5.2))

    for yi, m, lo, hi, col, teach in zip(y, means, los, his, colors, is_teacher):
        xerr = [[m - lo], [hi - m]]
        if teach:
            # Upper bound: hollow bar, dashed edge, ceiling framing
            ax.barh(yi, m, height=0.62, facecolor="white", edgecolor=col,
                    linewidth=1.6, linestyle="--", hatch="///", zorder=2)
        else:
            ax.barh(yi, m, height=0.62, color=col, edgecolor="white",
                    linewidth=0.5, zorder=2)
        ax.errorbar(m, yi, xerr=xerr, fmt="none", ecolor="#2b2b2b",
                    elinewidth=1.2, capsize=4, capthick=1.2, zorder=3)
        # value label just right of the CI whisker
        ax.text(hi + 0.006, yi, f"{m:.4f}", va="center", ha="left",
                fontsize=9.5, color="#1a1a1a", fontweight="bold")

    # teacher ceiling reference line
    tmean = means[-1]
    ax.axvline(tmean, color=C_TEACH, linestyle=":", linewidth=1.1,
               alpha=0.55, zorder=1)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Overall extraction score  (held-out n=200, mean +/- 95% CI)")
    ax.set_xlim(0.60, 1.075)
    ax.set_xticks([0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00])
    ax.set_title("Six-way comparison: distillation, GRPO, and the agentic protocol")

    # Callout for teacher = ceiling, not an independent capability number.
    # Placed in the open interior above the base bar (which ends at ~0.67).
    ax.annotate(
        "teacher bar = upper bound (dashed/hatched):\n"
        "teacher replay on distilled GT\n"
        "= self-consistency ceiling,\n"
        "not an independent baseline",
        xy=(tmean, y[-1]), xycoords="data",
        xytext=(0.755, y[0] - 0.10), textcoords="data",
        fontsize=8, color=C_TEACH, ha="left", va="top", style="italic",
        arrowprops=dict(arrowstyle="->", color=C_TEACH, lw=0.9,
                        connectionstyle="arc3,rad=-0.25", alpha=0.8))

    # Net-effect brackets: the two headline deltas
    # protocol cost: SFT(0.9010) -> mt protocol untrained (0.7789)
    # RL recovery:   mt protocol (0.7789) -> mt agentic GRPO (0.8932)
    ax.text(0.615, y[3] + 0.42,
            "multi-turn protocol cost  -12.2 pp",
            fontsize=8, color=C_MTPROT, fontweight="bold")
    ax.text(0.615, y[4] - 0.50,
            "agentic GRPO recovery  +11.4 pp (CI-separated)",
            fontsize=8, color=C_MTGRPO, fontweight="bold")

    fig.tight_layout()
    p = OUT / "fig1-sixway-comparison.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


# ============================================================================
# FIGURE 2 -- Behavioural collapse (#013) vs penalty-fixed run (v2)
# ============================================================================
def fig2():
    collapsed = _load_detail(ART / "mt-grpo-0722" / "collapsed-013-detail.jsonl")
    fixed = _load_detail(ART / "mt-grpo-0722" / "v2-final" / "reward_detail.jsonl")

    cx = [r["call"] for r in collapsed]
    c_search = [r["search_rate"] for r in collapsed]
    c_answ = [r["answered_rate"] for r in collapsed]

    fx = [r["call"] for r in fixed]
    f_search = [r["search_rate"] for r in fixed]
    f_answ = [r["answered_rate"] for r in fixed]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.4, 5.0))

    warmup_end = 7  # lr reaches full value; run-log inflection point

    # ---- Left: collapsed run #013 ----
    axL.plot(cx, c_search, "-o", color=C_SEARCH, ms=4, lw=1.8, label="search_rate")
    axL.plot(cx, c_answ, "-s", color=C_ANSW, ms=4, lw=1.8, label="answered_rate")
    axL.axvspan(cx[0], warmup_end, color=C_WARM, alpha=0.16, zorder=0)
    axL.axvline(warmup_end, color="#777", ls="--", lw=1.0)
    _annotate(axL, warmup_end, 0.02, "warmup ends (step 7)\nlr at full value",
              dy=10, dx=4, ha="left", fs=8, color="#555")
    # collapse trough annotation
    axL.annotate("search avoidance:\n0.50 -> 0.03",
                 xy=(18, 0.03), xytext=(20, 0.30),
                 arrowprops=dict(arrowstyle="->", color=C_SEARCH, lw=1.2),
                 fontsize=8.5, color=C_SEARCH, ha="center")
    axL.annotate("answered -> 1.0\n(direct-answer wins in-group)",
                 xy=(14, 1.0), xytext=(9.5, 0.62),
                 arrowprops=dict(arrowstyle="->", color=C_ANSW, lw=1.0),
                 fontsize=8, color=C_ANSW, ha="center")
    axL.set_title("(a) Collapsed run #013  (penalty 0.2 / 0.5 / 0.1)", color="#a33")
    axL.set_xlabel("Training step (= reward-log call)")
    axL.set_ylabel("Rate")
    axL.set_ylim(-0.03, 1.08)
    axL.legend(loc="center right")

    # ---- Right: fixed run v2 ----
    axR.plot(fx, f_search, "-", color=C_SEARCH, lw=1.9, label="search_rate")
    axR.plot(fx, f_answ, "-", color=C_ANSW, lw=1.5, alpha=0.9, label="answered_rate")
    axR.axvspan(fx[0], warmup_end, color=C_WARM, alpha=0.16, zorder=0)
    axR.axvline(warmup_end, color="#777", ls="--", lw=1.0)
    _annotate(axR, warmup_end, 0.02, "warmup ends (step 7)",
              dy=10, dx=4, ha="left", fs=8, color="#555")
    axR.annotate("search learned:\n0.41 -> 1.0",
                 xy=(20, 1.0), xytext=(70, 0.55),
                 arrowprops=dict(arrowstyle="->", color=C_SEARCH, lw=1.2),
                 fontsize=8.5, color=C_SEARCH, ha="center")
    axR.set_title("(b) Fixed run v2  (penalty 0.8 / 0.3 / 0.1)", color="#2a6")
    axR.set_xlabel("Training step (= reward-log call)")
    axR.set_ylabel("Rate")
    axR.set_ylim(-0.03, 1.08)
    axR.legend(loc="center right")

    fig.suptitle(
        "Behavioural collapse vs penalty fix -- the only change is the penalty schedule\n"
        "no_search 0.2 -> 0.8   |   no_answer 0.5 -> 0.3   (no_json unchanged 0.1)",
        fontsize=12.5, fontweight="bold", y=1.02)

    fig.tight_layout()
    p = OUT / "fig2-collapse-vs-fix.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


# ============================================================================
# FIGURE 3 -- Multi-turn GRPO v2 training monitor (2x3)
# ============================================================================
def fig3():
    rows = _load_detail(ART / "mt-grpo-0722" / "v2-final" / "reward_detail.jsonl")
    call = [r["call"] for r in rows]
    reward = [r["reward_mean"] for r in rows]
    dedup = [r["group_dedup_mean"] for r in rows]
    ent = [r["task_type_entropy"] for r in rows]
    gate = [r["gate_rate"] for r in rows]
    search = [r["search_rate"] for r in rows]
    answ = [r["answered_rate"] for r in rows]

    warmup_end = 7

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 7.6))
    (a1, a2, a3), (a4, a5, a6) = axes

    def _warm(ax):
        ax.axvspan(call[0], warmup_end, color=C_WARM, alpha=0.14, zorder=0)
        ax.axvline(warmup_end, color="#888", ls="--", lw=0.8)

    # -- reward mean --
    a1.plot(call, reward, color=C_MTGRPO, lw=1.7)
    _warm(a1)
    a1.axhline(0, color="#bbb", lw=0.8, ls=":")
    a1.set_title("Reward mean")
    a1.set_ylabel("mean group reward")
    a1.set_ylim(-0.25, 1.0)
    _annotate(a1, call[-1], reward[-1], f"{reward[0]:+.2f} -> {reward[-1]:+.2f}",
              dy=-14, dx=-4, ha="right", fs=8, color=C_MTGRPO)

    # -- intra-group diversity (dedup) --
    a2.plot(call, dedup, color="#6a51a3", lw=1.3, alpha=0.85)
    _warm(a2)
    a2.set_title("Intra-group diversity (dedup rate)")
    a2.set_ylabel("unique-completion fraction")
    a2.set_ylim(0.6, 1.03)
    a2.text(0.5, 0.06, "stays > 0.5 all run\n(GRPO advantage signal alive)",
            transform=a2.transAxes, fontsize=8, color="#6a51a3", ha="center")

    # -- entropy --
    a3.plot(call, ent, color="#2b8cbe", lw=1.0, alpha=0.8)
    _warm(a3)
    a3.set_title("task_type entropy")
    a3.set_ylabel("nats")
    a3.set_ylim(0, 2.6)
    a3.text(0.5, 0.06, "no monotone collapse\n(no field-entropy hacking)",
            transform=a3.transAxes, fontsize=8, color="#2b8cbe", ha="center")

    # -- gate rate (no_json dominated) --
    a4.plot(call, gate, color="#d94801", lw=1.3)
    _warm(a4)
    a4.set_title("Gate rate (no_json, behavioural)")
    a4.set_xlabel("Training step (= call)")
    a4.set_ylabel("gated fraction")
    a4.set_ylim(-0.02, 0.30)
    _annotate(a4, call[0], gate[0], "0.25 (learning to close)", dy=6, dx=6,
              ha="left", fs=8, color="#d94801")
    a4.annotate("-> ~0 (converged to close JSON)",
                xy=(120, 0.0), xytext=(120, 0.12),
                arrowprops=dict(arrowstyle="->", color="#d94801", lw=1.0),
                fontsize=8, color="#d94801", ha="center")

    # -- search + answered dual line --
    a5.plot(call, search, color=C_SEARCH, lw=1.7, label="search_rate")
    a5.plot(call, answ, color=C_ANSW, lw=1.3, alpha=0.9, label="answered_rate")
    _warm(a5)
    a5.set_title("Search & answer behaviour")
    a5.set_xlabel("Training step (= call)")
    a5.set_ylabel("rate")
    a5.set_ylim(-0.03, 1.08)
    a5.legend(loc="lower right")
    _annotate(a5, call[-1], 1.0, "both -> 1.0\nsearch-then-commit", dy=-22, dx=-4,
              ha="right", fs=8, color="#444")

    # -- summary panel (text) --
    a6.axis("off")
    summ = (
        "Multi-turn agentic GRPO -- run v2 (final)\n"
        "-------------------------------------------\n"
        "start SFT LoRA  |  native TRL (no unsloth)\n"
        "1000 prompts x 1 epoch = 250 steps\n"
        "batch 4 x accum 8   num_gen 8   temp 1.2\n"
        "max_turns 3   BM25 top-k 3 over 9k pool\n"
        "lr 5e-6   beta 0.05   seed 3407\n"
        "penalty: no_search 0.8 / no_answer 0.3 / no_json 0.1\n"
        "\n"
        "final_loss 0.0299   |   250 steps, 0 incidents\n"
        "5th criterion PASS (LoRA md5 != SFT)\n"
        "\n"
        "eval (held-out 200, greedy):\n"
        "  overall 0.8932  [0.8747 - 0.9092]\n"
        "  answered 1.000  search 1.000\n"
        "  mean_searches = 1.00  (search-then-commit)"
    )
    a6.text(0.02, 0.98, summ, transform=a6.transAxes, fontsize=9,
            va="top", ha="left", family="DejaVu Sans Mono",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#f5f5f5",
                      edgecolor="#cccccc"))

    fig.suptitle("Multi-turn agentic GRPO v2 -- training monitor (reward log, every step; n=32/step)",
                 fontsize=13, fontweight="bold", y=1.005)
    fig.tight_layout()
    p = OUT / "fig3-mtgrpo-training.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


# ============================================================================
# FIGURE 4 -- SFT loss curve (left) + data scaling (right)
# ============================================================================
def fig4():
    # ---- Left: SFT loss curve from log ----
    log = ART / "sft-full-20260721.log"
    steps_per_epoch = 1081  # 2162 total / 2 epochs
    xs, ys = [], []
    with open(log) as f:
        for line in f:
            m = re.search(r"'loss': ([0-9.]+).*'epoch': ([0-9.]+)", line)
            if m:
                loss = float(m.group(1))
                epoch = float(m.group(2))
                xs.append(epoch * steps_per_epoch)
                ys.append(loss)

    # ---- Right: data scaling five points ----
    scaling = [
        (500,  ART / "eval-scale-500.json",  0.8718),
        (1000, ART / "eval-scale-1000.json", 0.8871),
        (1500, ART / "eval-scale-1500.json", 0.8861),
        (4000, ART / "eval-scale-4000.json", 0.8945),
        (8648, ART / "eval-student_zeroshot-20260721.json", 0.9010),
    ]
    sx, sy, slo, shi = [], [], [], []
    for nrec, path, fallback in scaling:
        if path is not None and Path(path).exists():
            m, lo, hi = _overall(path)
        else:
            m, lo, hi = fallback, None, None
        sx.append(nrec); sy.append(m)
        slo.append(lo if lo is not None else m)
        shi.append(hi if hi is not None else m)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.0, 5.0))

    # -- Left panel --
    axL.plot(xs, ys, color=C_SFT, lw=1.4)
    axL.axvline(steps_per_epoch, color="#999", ls="--", lw=0.9)
    _annotate(axL, steps_per_epoch, max(ys) * 0.8, "epoch 1 | 2",
              dy=0, dx=6, ha="left", fs=8, color="#666")
    axL.scatter([xs[0]], [ys[0]], color=C_SFT, zorder=5, s=28)
    axL.scatter([xs[-1]], [ys[-1]], color=C_SFT, zorder=5, s=28)
    _annotate(axL, xs[0], ys[0], f"start {ys[0]:.2f}", dy=6, dx=8,
              ha="left", fs=8.5, color=C_SFT)
    _annotate(axL, xs[-1], ys[-1], "final 0.159", dy=10, dx=-6,
              ha="right", fs=8.5, color=C_SFT)
    axL.set_title("(a) SFT training loss")
    axL.set_xlabel("Training step  (8648 examples x 2 epochs = 2162 steps)")
    axL.set_ylabel("train loss (train_on_responses_only)")
    axL.set_ylim(0, max(ys) * 1.05)

    # -- Right panel --
    yerr = [[m - lo for m, lo in zip(sy, slo)],
            [hi - m for m, hi in zip(sy, shi)]]
    axR.errorbar(sx, sy, yerr=yerr, fmt="-o", color=C_GRPO, ms=7, lw=1.8,
                 capsize=4, capthick=1.1, ecolor="#2b2b2b",
                 markerfacecolor=C_GRPO, markeredgecolor="white", zorder=3)
    for x, y in zip(sx, sy):
        axR.annotate(f"{y:.4f}", (x, y), textcoords="offset points",
                     xytext=(0, 11), ha="center", fontsize=8.5, color="#1a4d2e")
    axR.set_xscale("log")
    axR.set_xticks(sx)
    axR.set_xticklabels([str(x) for x in sx])
    axR.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    axR.set_title("(b) SFT data scaling")
    axR.set_xlabel("Training examples (log scale)")
    axR.set_ylabel("Overall score (held-out n=200)")
    axR.set_ylim(0.855, 0.915)

    # plateau + diminishing-returns annotation
    axR.axhspan(sy[1], sy[2], color="#cccccc", alpha=0.25, zorder=0)
    axR.annotate("diminishing but not saturated",
                 xy=(4000, sy[3]), xytext=(900, 0.905),
                 arrowprops=dict(arrowstyle="->", color="#555", lw=1.0),
                 fontsize=9, color="#333", ha="left", style="italic")
    axR.text(1250, 0.879, "1k-1.5k plateau", fontsize=8, color="#777",
             ha="center", style="italic")

    fig.suptitle("SFT: convergence and data efficiency",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    p = OUT / "fig4-sft-scaling.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


if __name__ == "__main__":
    import matplotlib.ticker  # noqa: F401  (used in fig4)
    outs = []
    for fn in (fig1, fig2, fig3, fig4):
        p = fn()
        print(f"[ok] {fn.__name__:6s} -> {p}")
        outs.append(p)
    print("\nAll figures written to", OUT)
