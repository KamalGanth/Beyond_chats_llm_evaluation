import json
import re
import ast

def clean_json(path):
    raw = open(path, "r", encoding="utf-8").read()

    # Direct JSON load attempt
    try:
        return json.loads(raw)
    except Exception:
        pass

    cleaned = raw

    # Remove JS-style comments
    cleaned = re.sub(r'//.*', '', cleaned)
    cleaned = re.sub(r'/\*.*?\*/', '', cleaned, flags=re.DOTALL)

    cleaned = re.sub(r',\s*(\}|\])', r'\1', cleaned)

    cleaned = cleaned.replace("“", '"').replace("”", '"')

    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # object literal eval attempt
    try:
        return ast.literal_eval(cleaned)
    except Exception:
        pass

    
    # final emergency extraction
    print("JSON badly malformed — using emergency extractor")

    return emergency_extract(cleaned)


def emergency_extract(text):
    """
    Extracts usable fields from badly broken JSON
    Ensures pipeline continues instead of crashing
    """
    def extract(pattern):
        m = re.search(pattern, text, re.DOTALL)
        return m.group(1).strip() if m else ""

    return {
        "chat_id": extract(r'"chat_id"\s*:\s*(\d+)'),
        "user_id": extract(r'"user_id"\s*:\s*(\d+)'),
        "conversation_turns": extract_conversation_turns(text)
    }


def extract_conversation_turns(text):
    turns = []

    messages = re.findall(
        r'"role"\s*:\s*"([^"]+)"\s*,\s*"message"\s*:\s*"([^"]+)"',
        text,
        re.DOTALL
    )

    for role, msg in messages:
        turns.append({
            "role": role,
            "message": msg
        })

    return turns
