import re
from typing import Dict, Optional


_METRIC_PATTERNS = {
    "ctr":             r"CTR\s*[=:]\s*([\d\.]+)",
    "roi":             r"ROI\s*[=:]\s*([\d\.]+)",
    "conversion_rate": r"conversion[_ ]?rate\s*[=:]\s*([\d\.]+)",
    "impressions":     r"impressions\s*[=:]\s*([\d,]+)",
    "clicks":          r"clicks\s*[=:]\s*([\d,]+)",
}


def extract_metrics_from_text(text: str) -> Dict:
    metrics = {}
    for k, p in _METRIC_PATTERNS.items():
        m = re.search(p, text, re.IGNORECASE)
        if m:
            metrics[k] = float(m.group(1).replace(",", ""))
    return metrics


def analyze_performance(user_input: str, csv_path: Optional[str] = None) -> Dict:
    metrics = extract_metrics_from_text(user_input)

    benchmark = {
        "ctr":             0.045,
        "roi":             3.2,
        "conversion_rate": 0.07,
    }

    comparison = {}
    for m, v in metrics.items():
        b = benchmark.get(m)
        if b:
            comparison[m] = {
                "user_value": v,
                "benchmark":  b,
                "delta":      v - b,
                "verdict":    "good" if v >= b else "poor",
            }

    lines = ["=== PERFORMANCE ANALYSIS ==="]
    for k, v in metrics.items():
        lines.append(f"{k}: {v}")
    lines.append("\nBenchmarks:")
    for k, v in benchmark.items():
        lines.append(f"{k}: {v}")
    if comparison:
        lines.append("\nComparison:")
        for k, c in comparison.items():
            emoji = "✅" if c["verdict"] == "good" else "⚠️"
            lines.append(f"  {emoji} {k}: yours={c['user_value']} vs benchmark={c['benchmark']}")

    return {
        "source":          "text",       # always present — prevents KeyError
        "error":           None,
        "user_metrics":    metrics,
        "benchmark":       benchmark,
        "comparison":      comparison,
        "context_for_llm": "\n".join(lines),
    }