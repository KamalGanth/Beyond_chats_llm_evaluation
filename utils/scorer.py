from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import numpy as np
import re


embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def relevance_tfidf(response, contexts):
    texts = [response] + contexts

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform(texts)

    # Convert safely to numpy arrays
    response_vec = tfidf[0].toarray()
    context_vec = tfidf[1:].toarray().mean(axis=0, keepdims=True)

    sim = cosine_similarity(response_vec, context_vec)[0][0]
    return round(float(sim), 4)

def relevance_embed(response, contexts):
    all_text = [response] + contexts
    emb = embed_model.encode(all_text)
    resp_vec = emb[0]
    ctx_vec = emb[1:].mean(axis=0)
    sim = np.dot(resp_vec, ctx_vec) / (np.linalg.norm(resp_vec) * np.linalg.norm(ctx_vec))
    return float(sim)

def relevance_llm_judge(response, contexts):
    """
    Gemini judge: Ask LLM to evaluate relevance score between 0 and 1
    """
    ctx_text = "\n".join(contexts)
    prompt = f"""
    Act as an LLM evaluation judge and 
    Evaluate the relevance of the AI response to the following context on a scale of 0 to 1.
    Context:
    {ctx_text}

    Response:
    {response}

    Return only a number between 0 and 1.
    """

    genai.configure(api_key="xyz")  # Replace with your actual API key

    model = genai.GenerativeModel("gemini-2.5-flash")
    out = model.generate_content(prompt)
    try:
        return float(out.text.strip())
    except:
        return 0.0

def completeness_score(user, response):
    user_tokens = set(re.findall(r"\b\w+\b", user.lower()))
    resp_tokens = set(re.findall(r"\b\w+\b", response.lower()))
    if not user_tokens:
        return 1.0
    return len(user_tokens & resp_tokens) / len(user_tokens)

def hallucination_score(response, contexts):
    ctx = " ".join(contexts).lower()
    resp_sents = re.split(r"[.!?]", response.lower())

    low_overlap = 0
    for s in resp_sents:
        s = s.strip()
        if not s:
            continue
        overlap = sum([1 for w in s.split() if w in ctx])
        ratio = overlap / max(1, len(s.split()))
        if ratio < 0.2:
            low_overlap += 1

    return low_overlap / max(1, len(resp_sents))
