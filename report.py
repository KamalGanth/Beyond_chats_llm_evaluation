import json

def run_report(input_path="output/raw_evaluation.json", out_path="output/final_report.md"):
    d = json.load(open(input_path))

    md = f"""
# LLM Evaluation Report

### Method Used: `{d['method']}`

## Metrics
- Relevance: **{d['relevance']}**
- Completeness: **{d['completeness']}**
- Hallucination Score: **{d['hallucination_score']}**
- Final Score: **{d['final_score']}**

## Latency (ms)
- Relevance: {d['latency_ms']['relevance']}
- Completeness: {d['latency_ms']['completeness']}
- Hallucination: {d['latency_ms']['hallucination']}

## Cost
- Tokens Estimated: {d['cost_estimate']['tokens_estimated']}
- Cost (USD): {d['cost_estimate']['cost_usd']}
"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)

    print("Report Generated:", out_path)
    return out_path
