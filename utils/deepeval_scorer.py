from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from deepeval.models import GeminiModel
import os
import time


def evaluate_with_deepeval(user_query, ai_response, contexts):
    model = GeminiModel(
        model="models/gemini-2.5-flash",
        api_key=os.getenv("GOOGLE_API_KEY") # Replace with your actual API key
    )

    test_case = LLMTestCase(
        input=user_query,
        actual_output=ai_response,
        retrieval_context=contexts
    )

    relevance_metric = AnswerRelevancyMetric(
        model=model,
        threshold=0.5
    )

    faithfulness_metric = FaithfulnessMetric(
        model=model,
        threshold=0.5
    )

    start = time.time()

    relevance_metric.measure(test_case)
    faithfulness_metric.measure(test_case)

    latency_ms = round((time.time() - start) * 1000, 2)

    return {
        "relevance": round(relevance_metric.score, 4),
        "hallucination_score": round(1 - faithfulness_metric.score, 4),
        "latency_ms": latency_ms,
        "reasoning": {
            "relevance_reason": relevance_metric.reason,
            "faithfulness_reason": faithfulness_metric.reason
        }
    }
