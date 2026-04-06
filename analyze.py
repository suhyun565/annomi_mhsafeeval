"""
analyze_comparison.py
─────────────────────
v1(orig) → v2(rubric) 방향성 검증

핵심 가설:
  quality=HIGH  → v2 severity가 v1보다 낮아야 함  (↓)
  quality=LOW   → v2 severity가 v1보다 높아야 함  (↑)

Usage:
    python analyze_comparison.py annomi_classified.jsonl annomi_classified2.jsonl
"""

import json, sys
from pathlib import Path
from collections import defaultdict


# ── 로딩 ──────────────────────────────────────────────────────────────────────
def load_turns(path: str) -> list[dict]:
    turns = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj     = json.loads(line)
            quality = (obj.get("quality") or "unknown").lower()
            for turn in obj.get("session", []):
                if turn.get("speaker") == "counselor" and turn.get("severity") is not None:
                    turns.append({
                        "quality":  quality,
                        "category": turn.get("harm_category") or "None",
                        "severity": int(turn.get("severity", 1)),
                    })
    return turns


# ── 집계 ──────────────────────────────────────────────────────────────────────
def aggregate(turns: list[dict]) -> dict:
    # qcs: quality → category → severity → count
    qcs: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for t in turns:
        if t["category"] == "None":   # None 카테고리 제외
            continue
        qcs[t["quality"]][t["category"]][t["severity"]] += 1

    # 카테고리 정렬: quality 내 총합 내림차순
    sorted_qcs = {
        q: dict(sorted(cat_d.items(), key=lambda x: -sum(x[1].values())))
        for q, cat_d in qcs.items()
    }
    return {"total": sum(len(t) for q in sorted_qcs.values() for t in q.values()), "qcs": sorted_qcs}


# ── 헬퍼 ──────────────────────────────────────────────────────────────────────
SEP = "=" * 72

def section(title):
    print(f"\n{SEP}\n  {title}\n{SEP}")

def _fmt(ws):
    return "  ".join(f"{{:<{w}}}" for w in ws)

def pr(fmt, row):
    print(fmt.format(*[str(x) for x in row]))

def pct(n, total):
    return f"{n/total*100:.1f}%" if total else "0.0%"

def avg_sev(sev_d: dict) -> float:
    n = sum(sev_d.values())
    return sum(s * c for s, c in sev_d.items()) / n if n else 0.0

def merged_sev(qcs, q) -> dict:
    """quality q 전체를 severity→count로 합산."""
    d: dict = defaultdict(int)
    for sev_d in qcs.get(q, {}).values():
        for s, c in sev_d.items():
            d[s] += c
    return dict(sorted(d.items()))

def all_sevs(*dicts):
    return sorted({s for d in dicts for s in d})

def all_cats(*qcs_dicts):
    cats = set()
    for qcs in qcs_dicts:
        for q_d in qcs.values():
            cats |= set(q_d.keys())
    return sorted(cats)


# ── 단일 파일: quality × category × severity 테이블 ──────────────────────────
def print_single(label: str, agg: dict):
    section(f"{label}  (전체 turns: {agg['total']})")

    for q in ["high", "low"]:
        cat_d = agg["qcs"].get(q, {})
        q_total = sum(sum(d.values()) for d in cat_d.values())
        all_s = all_sevs(*cat_d.values()) if cat_d else []

        print(f"\n  quality={q.upper()}  (turns={q_total}  |  avg sev={avg_sev(merged_sev(agg['qcs'], q)):.3f})")

        if not cat_d:
            print("  (데이터 없음)")
            continue

        hdrs = ["Category", "total"] + [f"Sev{s}" for s in all_s]
        ws   = [30, 7] + [6] * len(all_s)
        fmt  = _fmt(ws)
        print(fmt.format(*hdrs))
        print("  " + "  ".join("-" * w for w in ws))
        for cat, sev_d in cat_d.items():
            row = [cat[:29], sum(sev_d.values())] + [sev_d.get(s, 0) for s in all_s]
            pr(fmt, row)
        print("  " + "  ".join("-" * w for w in ws))
        tot_row = ["TOTAL", q_total] + [
            sum(cat_d[c].get(s, 0) for c in cat_d) for s in all_s
        ]
        pr(fmt, tot_row)


# ── 비교 핵심 ─────────────────────────────────────────────────────────────────
def print_comparison(label1: str, agg1: dict, label2: str, agg2: dict):
    section(
        f"방향성 검증:  {label1}  →  {label2}\n"
        f"  가설)  quality=HIGH → severity ↓  |  quality=LOW → severity ↑"
    )

    qcs1, qcs2 = agg1["qcs"], agg2["qcs"]
    cats = all_cats(qcs1, qcs2)

    # ── [V0] 한눈에 보는 방향성 판정 ─────────────────────────────────────────
    print("\n[V0]  ★ 방향성 판정 (평균 severity)")
    print(f"\n  {'':25s}  {label1:>10s}  {label2:>10s}  {'Δ(v2-v1)':>10s}  {'기대':>8s}  결과")
    print(f"  {'':25s}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*8}  ────")
    for q, expected in [("high", "↓ 낮아짐"), ("low", "↑ 높아짐")]:
        a1 = avg_sev(merged_sev(qcs1, q))
        a2 = avg_sev(merged_sev(qcs2, q))
        delta = a2 - a1
        ok = ("✓" if delta < 0 else "✗") if q == "high" else ("✓" if delta > 0 else "✗")
        print(f"  {f'quality={q} 평균 sev':25s}  {a1:>10.3f}  {a2:>10.3f}  {delta:>+10.3f}  {expected:>8s}  {ok}")

    # ── [V1] quality=HIGH 상세 ────────────────────────────────────────────────
    for q, expected_dir in [("high", "v2 Δ 음수(↓) 기대"), ("low", "v2 Δ 양수(↑) 기대")]:
        cs1 = qcs1.get(q, {})
        cs2 = qcs2.get(q, {})
        t1  = sum(sum(d.values()) for d in cs1.values())
        t2  = sum(sum(d.values()) for d in cs2.values())
        m1  = merged_sev(qcs1, q)
        m2  = merged_sev(qcs2, q)
        all_s = all_sevs(m1, m2)

        print(f"\n[V{'1' if q=='high' else '2'}]  quality={q.upper()}  ({expected_dir})")
        print(f"       {label1} turns={t1}  avg={avg_sev(m1):.3f}"
              f"  |  {label2} turns={t2}  avg={avg_sev(m2):.3f}"
              f"  |  Δavg={avg_sev(m2)-avg_sev(m1):+.3f}")

        # severity 분포
        print(f"\n  --- severity 분포 ---")
        hdrs = ["Sev", f"{label1}n", f"{label1}%", f"{label2}n", f"{label2}%", "Δn", "Δ%"]
        ws   = [5, 9, 9, 9, 9, 7, 8]
        fmt  = _fmt(ws)
        print(fmt.format(*hdrs))
        print("  " + "  ".join("-" * w for w in ws))
        for s in all_s:
            n1, n2 = m1.get(s, 0), m2.get(s, 0)
            p1 = n1 / t1 * 100 if t1 else 0
            p2 = n2 / t2 * 100 if t2 else 0
            pr(fmt, [f"Sev{s}", n1, f"{p1:.1f}%", n2, f"{p2:.1f}%",
                      f"{n2-n1:+d}", f"{p2-p1:+.1f}%"])

        # 카테고리 × severity 상세
        print(f"\n  --- 카테고리별 avg severity 변화 ---")
        hdrs2 = ["Category", f"{label1}n", f"{label1}avg", f"{label2}n", f"{label2}avg", "Δavg", "방향✓?"]
        ws2   = [30, 9, 10, 9, 10, 8, 7]
        fmt2  = _fmt(ws2)
        print(fmt2.format(*hdrs2))
        print("  " + "  ".join("-" * w for w in ws2))
        for cat in cats:
            d1 = cs1.get(cat, {})
            d2 = cs2.get(cat, {})
            n1, n2 = sum(d1.values()), sum(d2.values())
            if n1 == 0 and n2 == 0:
                continue
            a1, a2 = avg_sev(d1), avg_sev(d2)
            delta = a2 - a1
            ok = ("✓" if delta < 0 else "✗") if q == "high" else ("✓" if delta > 0 else "✗")
            if n1 == 0 or n2 == 0:
                ok = "-"   # 한쪽만 있으면 판정 보류
            pr(fmt2, [cat[:29], n1, f"{a1:.2f}", n2, f"{a2:.2f}", f"{delta:+.2f}", ok])

        print(f"\n  --- 카테고리 × severity count 상세 ---")
        hdrs3 = ["Category", "Sev", f"{label1}n", f"{label1}%", f"{label2}n", f"{label2}%", "Δn"]
        ws3   = [30, 5, 9, 9, 9, 9, 7]
        fmt3  = _fmt(ws3)
        print(fmt3.format(*hdrs3))
        print("  " + "  ".join("-" * w for w in ws3))
        for cat in cats:
            d1 = cs1.get(cat, {})
            d2 = cs2.get(cat, {})
            cat_all_s = all_sevs(d1, d2)
            first = True
            for s in cat_all_s:
                n1, n2 = d1.get(s, 0), d2.get(s, 0)
                if n1 == 0 and n2 == 0:
                    continue
                p1 = n1 / t1 * 100 if t1 else 0
                p2 = n2 / t2 * 100 if t2 else 0
                pr(fmt3, [cat[:29] if first else "", f"S{s}",
                           n1, f"{p1:.1f}%", n2, f"{p2:.1f}%", f"{n2-n1:+d}"])
                first = False
            if not first:
                print("  " + "  ".join("-" * w for w in ws3))


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    f1 = sys.argv[1] if len(sys.argv) > 1 else "annomi_classified.jsonl"
    f2 = sys.argv[2] if len(sys.argv) > 2 else "annomi_classified2.jsonl"

    for f in [f1, f2]:
        if not Path(f).exists():
            print(f"[error] 파일 없음: {f}"); sys.exit(1)

    print(f"\nLoading {f1}…")
    t1 = load_turns(f1)
    print(f"  → {len(t1)} counselor turns")

    print(f"Loading {f2}…")
    t2 = load_turns(f2)
    print(f"  → {len(t2)} counselor turns")

    agg1 = aggregate(t1)
    agg2 = aggregate(t2)
    label1, label2 = "v1(orig)", "v2(rubric)"

    print_single(label1, agg1)
    print_single(label2, agg2)
    print_comparison(label1, agg1, label2, agg2)

    print(f"\n{SEP}\n  Done.\n{SEP}\n")


if __name__ == "__main__":
    main()