import json
from utils.scorer import (
    relevance_tfidf,
    relevance_embed,
    relevance_llm_judge,
    completeness_score,
    hallucination_score
)
from utils.timer import time_ms
from utils.cost import estimate_cost


def run_evaluation(method, input_path="output/processed_input.json", out_path="output/raw_evaluation.json"):
    data = json.load(open(input_path))

    user = data["user_message"]
    resp = data["ai_response"]
    ctx = data["context"]

    if method == "tfidf":
        rel, rel_time = time_ms(relevance_tfidf, resp, ctx)

    elif method == "embedding":
        rel, rel_time = time_ms(relevance_embed, resp, ctx)

    elif method == "judge":
        rel, rel_time = time_ms(relevance_llm_judge, resp, ctx)

    else:
        raise ValueError("Invalid method")

    comp, comp_time = time_ms(completeness_score, user, resp)
    hall, hall_time = time_ms(hallucination_score, resp, ctx)

    cost = estimate_cost(resp)

    final_score = round(0.5 * rel + 0.3 * comp + 0.2 * (1 - hall), 4)

    output = {
        "method": method,
        "relevance": rel,
        "completeness": comp,
        "hallucination_score": hall,
        "latency_ms": {
            "relevance": rel_time,
            "completeness": comp_time,
            "hallucination": hall_time
        },
        "cost_estimate": cost,
        "final_score": final_score
    }

    json.dump(output, open(out_path, "w", encoding="utf-8"), indent=2)
    print("Evaluation completed:", out_path)
    return out_path
