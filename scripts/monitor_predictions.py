import argparse, json
from collections import Counter
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="/opt/ticket-router/logs/predictions.jsonl")
    ap.add_argument("--n", type=int, default=2000, help="analyze last N log lines")
    args = ap.parse_args()

    p = Path(args.log)
    if not p.exists():
        print(f"Log not found: {p}")
        return

    lines = p.read_text().splitlines()[-args.n:]
    rows = []
    for ln in lines:
        try:
            rows.append(json.loads(ln))
        except Exception:
            continue

    if not rows:
        print("No valid log rows found.")
        return

    queues = [r.get("predicted_queue", "unknown") for r in rows]
    confs = [float(r.get("confidence", 0.0)) for r in rows]
    lats  = [float(r.get("latency_ms", 0.0)) for r in rows]
    versions = Counter([str(r.get("model_version", "unknown")) for r in rows])

    c = Counter(queues)
    total = sum(c.values())

    def pct(x): 
        return (x / total) * 100.0 if total else 0.0

    confs_sorted = sorted(confs)
    lats_sorted  = sorted(lats)

    def pctl(arr, q):
        if not arr: return 0.0
        idx = int(round((q/100.0) * (len(arr)-1)))
        return arr[max(0, min(idx, len(arr)-1))]

    print("\n=== Ticket Router Monitoring Summary ===")
    print(f"Rows analyzed: {len(rows)} (last {args.n} lines)")
    print(f"Model versions seen: {dict(versions)}")

    print("\nTop queues (by predictions):")
    for q, n in c.most_common(10):
        print(f"  - {q:30s} {n:5d}  ({pct(n):5.1f}%)")

    print("\nConfidence:")
    print(f"  mean: {sum(confs)/len(confs):.3f}")
    print(f"  p50 : {pctl(confs_sorted, 50):.3f}")
    print(f"  p90 : {pctl(confs_sorted, 90):.3f}")

    print("\nLatency (ms):")
    print(f"  mean: {sum(lats)/len(lats):.1f}")
    print(f"  p50 : {pctl(lats_sorted, 50):.1f}")
    print(f"  p90 : {pctl(lats_sorted, 90):.1f}")

if __name__ == "__main__":
    main()