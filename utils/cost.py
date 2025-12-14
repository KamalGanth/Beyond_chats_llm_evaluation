def estimate_cost(text, price_per_1k_tokens=0.03):
    token_est = max(1, int(len(text.split()) / 0.75))
    cost = (token_est / 1000) * price_per_1k_tokens
    return {
        "tokens_estimated": token_est,
        "cost_usd": round(cost, 6)
    }
