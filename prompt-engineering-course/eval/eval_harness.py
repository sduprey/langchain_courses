"""A tiny evaluation harness. No frameworks — just enough to make prompting measurable.

    from eval import load_cases, run_eval, print_report, exact, contains
    cases  = load_cases("eval/datasets/sentiment.jsonl")
    report = run_eval(cases, my_prompt_fn, scorer=exact)
    print_report(report)

A scorer is any callable (output, expected) -> bool. `judge()` adapts an
LLM-as-judge function into a scorer (see 03_evaluation/02_llm_as_judge.ipynb).
"""
import json
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


def run_eval(cases, prompt_fn, scorer=exact):
    """prompt_fn: (input_str) -> output_str. Returns a report dict."""
    results, t0 = [], time.time()
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
    passed = sum(r["pass"] for r in results)
    return {
        "n": len(results),
        "passed": passed,
        "accuracy": passed / len(results) if results else 0.0,
        "seconds": round(time.time() - t0, 2),
        "results": results,
    }


def print_report(report, show_failures=True):
    print(
        f"accuracy: {report['accuracy']:.0%}  "
        f"({report['passed']}/{report['n']})  in {report['seconds']}s"
    )
    if show_failures:
        for r in report["results"]:
            if not r["pass"]:
                print(f"  FAIL  in={r['input']!r}  exp={r['expected']!r}  got={r['output']!r}")
