import json, re
from utils.cleaner import clean_json

def run_preprocess(conversation_path, context_path, out_path="output/processed_input.json"):
    conv = clean_json(conversation_path)
    ctx = clean_json(context_path)

    turns = conv.get("conversation_turns", [])
    last_ai, last_user = None, None

    for t in reversed(turns):
        role = t.get("role", "").lower()
        if role.startswith("ai") and not last_ai:
            last_ai = t
        if role.startswith("user") and not last_user:
            last_user = t
        if last_ai and last_user:
            break

    vector_data = ctx.get("data", {}).get("vector_data", [])
    context_texts = [v.get("text", "") for v in vector_data][:6]

    processed = {
        "chat_id": conv.get("chat_id"),
        "user_id": conv.get("user_id"),
        "user_message": last_user.get("message", ""),
        "ai_response": last_ai.get("message", ""),
        "context": context_texts
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(processed, f, indent=2)

    print("Preprocessing completed:", out_path)
    return out_path
