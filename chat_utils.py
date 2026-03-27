from __future__ import annotations

from typing import Any


def _normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for message in messages:
        role = str(message["role"]).strip()
        content = str(message["content"]).strip()
        normalized.append({"role": role, "content": content})
    return normalized


def _render_fallback(messages: list[dict[str, Any]], add_generation_prompt: bool) -> str:
    blocks: list[str] = []
    for message in _normalize_messages(messages):
        role = message["role"].capitalize()
        blocks.append(f"{role}: {message['content']}")
    if add_generation_prompt:
        blocks.append("Assistant:")
    return "\n\n".join(blocks)


def render_chat_text(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    add_generation_prompt: bool = False,
) -> str:
    normalized = _normalize_messages(messages)
    chat_template = getattr(tokenizer, "chat_template", None)
    if hasattr(tokenizer, "apply_chat_template") and chat_template:
        return tokenizer.apply_chat_template(
            normalized,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    return _render_fallback(normalized, add_generation_prompt=add_generation_prompt)


def tokenize_conversation(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    max_length: int,
) -> dict[str, list[int]]:
    normalized = _normalize_messages(messages)
    input_ids: list[int] = []
    labels: list[int] = []
    turn_lengths: list[int] = []
    previous_length = 0

    for end in range(len(normalized)):
        rendered = render_chat_text(
            tokenizer,
            normalized[: end + 1],
            add_generation_prompt=False,
        )
        current_ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
        delta = current_ids[previous_length:]
        input_ids.extend(delta)
        if normalized[end]["role"] == "assistant":
            labels.extend(delta)
        else:
            labels.extend([-100] * len(delta))
        turn_lengths.append(len(delta))
        previous_length = len(current_ids)

    original_length = len(input_ids)
    was_truncated = original_length > max_length
    if was_truncated:
        # 截断策略：优先从第一个 turn（通常是最长的 user 问题/上下文）的头部截去多余
        # token，以尽量保留完整的后续对话和监督标签。若第一个 turn 本身不够截，
        # 则回退到保留末尾 max_length 个 token（确保监督标签不丢失）。
        excess = original_length - max_length
        first_turn_len = turn_lengths[0]
        if excess < first_turn_len:
            # 只需截去第一个 turn 的头部，后续所有监督 token 完整保留
            input_ids = input_ids[excess:]
            labels = labels[excess:]
        else:
            # 第一个 turn 全截也不够，保留末尾 max_length 保证监督标签存在
            input_ids = input_ids[-max_length:]
            labels = labels[-max_length:]

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": [1] * len(input_ids),
        "original_length": original_length,
        "was_truncated": was_truncated,
    }
