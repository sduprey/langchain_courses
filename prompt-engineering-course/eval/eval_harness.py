"""A tiny evaluation harness. No frameworks -- just enough to make prompting measurable.

    from eval import load_cases, run_eval, print_report, compare, sweep_models, exact, contains
    cases  = load_cases("eval/datasets/sentiment.jsonl")
    report = run_eval(cases, my_prompt_fn, scorer=exact, repeats=5)
    print_report(report)

Why `repeats`: the models are not deterministic even at temperature 0, so a single
pass on a small set moves 1-2 cases run to run. `repeats` runs the whole set k times
and reports the mean and the spread (max - min). A prompt change is only believable
when its effect is bigger than that spread -- `compare()` makes that call for you.

A scorer is any callable (output, expected) -> bool.
"""
import json
import statistics
import time


def load_cases(path):
    cases = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


# --- built-in scorers ---
def exact(output, expected):
    return output.strip().lower() == str(expected).strip().lower()


def contains(output, expected):
    return str(expected).strip().lower() in output.strip().lower()


def judge(judge_fn):
    """Turn an LLM-as-judge callable (output, expected) -> bool into a scorer."""
    return judge_fn


# --- core ---
def _one_pass(cases, prompt_fn, scorer):
    results = []
    for c in cases:
        out = prompt_fn(c["input"])
        results.append(
            {
                "input": c["input"],
                "expected": c["expected"],
                "output": out,
                "pass": bool(scorer(out, c["expected"])),
            }
        )
    return results


def run_eval(cases, prompt_fn, scorer=exact, repeats=1):
    """prompt_fn: (input_str) -> output_str. Runs the set `repeats` times.

    Report keys: n, repeats, accuracies (per run), acc_mean, acc_min, acc_max,
    spread (max-min), accuracy (== acc_mean, back-compat), passed/results (last run),
    seconds.
    """
    t0 = time.time()
    accuracies, last = [], []
    for _ in range(max(1, repeats)):
        last = _one_pass(cases, prompt_fn, scorer)
        accuracies.append(sum(r["pass"] for r in last) / len(last) if last else 0.0)
    mean = statistics.fmean(accuracies)
    return {
        "n": len(cases),
        "repeats": max(1, repeats),
        "accuracies": accuracies,
        "acc_mean": mean,
        "acc_min": min(accuracies),
        "acc_max": max(accuracies),
        "spread": max(accuracies) - min(accuracies),
        "accuracy": mean,
        "passed": sum(r["pass"] for r in last),
        "results": last,
        "seconds": round(time.time() - t0, 2),
    }


def print_report(report, show_failures=True):
    if report["repeats"] > 1:
        print(
            f"accuracy: {report['acc_mean']:.0%} +/- {report['spread'] / 2:.0%}  "
            f"(min {report['acc_min']:.0%}, max {report['acc_max']:.0%}, "
            f"{report['repeats']} runs, n={report['n']})  in {report['seconds']}s"
        )
    else:
        print(
            f"accuracy: {report['acc_mean']:.0%}  "
            f"({report['passed']}/{report['n']})  in {report['seconds']}s"
        )
    if show_failures:
        for r in report["results"]:
            if not r["pass"]:
                print(f"  FAIL  in={r['input']!r}  exp={r['expected']!r}  got={r['output']!r}")


def compare(cases, fn_a, fn_b, labels=("A", "B"), scorer=exact, repeats=3):
    """Score two prompt fns on the same set and say whether the gap beats the noise.

    Verdict is REAL when |mean_b - mean_a| > spread_a + spread_b, else INCONCLUSIVE.
    """
    a = run_eval(cases, fn_a, scorer, repeats)
    b = run_eval(cases, fn_b, scorer, repeats)
    gap = b["acc_mean"] - a["acc_mean"]
    noise = a["spread"] + b["spread"]
    verdict = "REAL" if abs(gap) > noise else "INCONCLUSIVE"
    print(
        f"  {labels[0]:<26} {a['acc_mean']:.0%}   (spread {a['spread']:.0%} over {repeats} runs)"
    )
    print(
        f"  {labels[1]:<26} {b['acc_mean']:.0%}   (spread {b['spread']:.0%} over {repeats} runs)"
    )
    print(f"  gap {gap:+.0%}   vs   run-to-run noise {noise:.0%}   ->   {verdict}")
    return {"a": a, "b": b, "gap": gap, "noise": noise, "verdict": verdict}


def sweep_models(cases, make_fn, models, scorer=exact, repeats=3, sample=None):
    """make_fn(model) -> prompt_fn. Score each model, print an accuracy/latency table.

    `sample` caps the number of cases (use it to keep slow/paid models cheap).
    """
    use = cases if sample is None else cases[:sample]
    rows = {}
    for m in models:
        try:
            rep = run_eval(use, make_fn(m), scorer, repeats)
        except Exception as e:  # noqa: BLE001 - one flaky model should not kill the table
            print(f"  {m:<24} skipped: {type(e).__name__}")
            continue
        rows[m] = rep
        print(
            f"  {m:<24} acc {rep['acc_mean']:.0%}  "
            f"(spread {rep['spread']:.0%}, {rep['repeats']} runs, n={rep['n']})  "
            f"{rep['seconds']:.1f}s"
        )
    return rows
