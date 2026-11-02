"""Extract only explicit ReAct actions from Phoenix LLM span data."""
import json
import re


def _decoded(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (TypeError, ValueError):
            return None
    return value


def _assistant_text(message):
    if not isinstance(message, dict):
        return None
    message = message.get("message", message)
    if not isinstance(message, dict) or message.get("role") != "assistant":
        return None
    content = message.get("content")
    if isinstance(content, str):
        return content
    # OpenInference's actual Responses trace uses nested ``contents`` entries
    # rather than the ``content``/``blocks`` shapes emitted by other clients.
    # Only accept text typed as such, and only after verifying assistant role.
    contents = message.get("contents")
    if isinstance(contents, list):
        texts = []
        for entry in contents:
            if not isinstance(entry, dict):
                continue
            item = entry.get("message_content")
            if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str):
                texts.append(item["text"])
        if texts:
            return "\n".join(texts)
    blocks = message.get("blocks")
    if isinstance(blocks, list):
        return "\n".join(block["text"] for block in blocks
                         if isinstance(block, dict) and isinstance(block.get("text"), str))
    return None


def _parse_action(text):
    if not isinstance(text, str):
        return None
    # Some LlamaIndex spans encode line breaks as literal "\\n" in block text.
    # Only decode the boundaries of ReAct fields, not JSON argument contents.
    text = re.sub(r"\\n(?=[ \t]*Action(?: Input)?:)", "\n", text)
    action = re.search(r"(?m)^[ \t]*Action:[ \t]*([A-Za-z0-9_]+)[ \t]*$", text)
    if not action:
        return None
    arguments = re.search(r"(?m)^[ \t]*Action Input:[ \t]*", text[action.end():])
    if not arguments:
        return None
    try:
        payload, _ = json.JSONDecoder().raw_decode(text[action.end():][arguments.end():].lstrip())
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    return action.group(1), json.dumps(payload, ensure_ascii=False, sort_keys=True)


def extract_traced_action(output_messages, output_value):
    """Read an explicit assistant action; return None for missing or invalid data."""
    messages = _decoded(output_messages)
    candidates = messages if isinstance(messages, list) else []
    fallback = _decoded(output_value)
    if isinstance(fallback, dict):
        candidates = [*candidates, fallback]
    for message in candidates:
        action = _parse_action(_assistant_text(message))
        if action:
            return action
    return None
