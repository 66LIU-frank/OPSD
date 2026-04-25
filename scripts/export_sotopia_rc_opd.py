#!/usr/bin/env python
"""Convert SOTOPIA episode exports into RC-OPD social prompt JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return "\n".join(clean_text(item) for item in value if clean_text(item)).strip()
    if isinstance(value, dict):
        parts = []
        for key, val in value.items():
            text = clean_text(val)
            if text:
                parts.append(f"{key}: {text}")
        return "\n".join(parts).strip()
    return str(value).strip()


def clean_utterance(value: Any) -> str:
    text = clean_text(value)
    lowered = text.lower()
    if lowered.startswith("said:"):
        text = text[5:].strip()
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        text = text[1:-1].strip()
    return text


def iter_records(path: Path) -> Iterable[dict[str, Any]]:
    suffix = path.suffix.lower()
    with path.open("r", encoding="utf-8") as f:
        if suffix == ".jsonl":
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
                if isinstance(record, dict):
                    yield record
        elif suffix == ".json":
            data = json.load(f)
            if isinstance(data, list):
                for record in data:
                    if isinstance(record, dict):
                        yield record
            elif isinstance(data, dict):
                records = data.get("episodes") or data.get("data") or data.get("records")
                if isinstance(records, list):
                    for record in records:
                        if isinstance(record, dict):
                            yield record
                else:
                    yield data
        else:
            raise ValueError(f"Unsupported input extension: {suffix}. Use .jsonl or .json.")


def named_entries(value: Any, fallback_names: list[str]) -> list[tuple[str, str]]:
    if isinstance(value, dict):
        entries = [(clean_text(key), clean_text(val)) for key, val in value.items()]
        return [(key or f"Agent {idx + 1}", val) for idx, (key, val) in enumerate(entries) if val]
    if isinstance(value, list):
        entries = []
        for idx, val in enumerate(value):
            name = fallback_names[idx] if idx < len(fallback_names) else f"Agent {idx + 1}"
            text = clean_text(val)
            if text:
                entries.append((name, text))
        return entries
    text = clean_text(value)
    return [(fallback_names[0] if fallback_names else "Agent 1", text)] if text else []


def fallback_agent_names(record: dict[str, Any]) -> list[str]:
    for key in ("agent_names", "agents", "agent_ids"):
        value = record.get(key)
        if isinstance(value, list) and value:
            return [clean_text(item) or f"Agent {idx + 1}" for idx, item in enumerate(value)]
    return ["Agent 1", "Agent 2"]


def render_actor_message(message: Any) -> tuple[str, str] | None:
    if not isinstance(message, (list, tuple)) or len(message) < 3:
        return None
    sender, receiver, content = message[:3]
    sender_text = clean_text(sender)
    receiver_text = clean_text(receiver)
    content_text = clean_utterance(content)
    if (
        not sender_text
        or sender_text == "Environment"
        or receiver_text != "Environment"
        or not content_text
        or "did nothing" in content_text.lower()
        or "left the conversation" in content_text.lower()
    ):
        return None
    return sender_text, content_text


def render_dialogue_from_raw_messages(raw_messages: Any) -> str:
    return "\n".join(
        f"{sender}: {content}"
        for sender, content, _, _ in iter_actor_messages(raw_messages)
    ).strip()


def iter_actor_messages(raw_messages: Any) -> Iterable[tuple[str, str, int, int]]:
    if not isinstance(raw_messages, list):
        return
    for turn_index, turn in enumerate(raw_messages):
        if not isinstance(turn, list):
            continue
        for message_index, message in enumerate(turn):
            rendered = render_actor_message(message)
            if rendered is None:
                continue
            sender, content = rendered
            yield sender, content, turn_index, message_index


def first_text(record: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        text = clean_text(record.get(key))
        if text:
            return text
    return ""


def build_full_dialogue_examples(record: dict[str, Any], agent_indices: list[int]) -> list[dict[str, Any]]:
    names = fallback_agent_names(record)

    personas = named_entries(
        record.get("agents_background")
        or record.get("agent_backgrounds")
        or [record.get("agent_persona"), record.get("opponent_persona")],
        names,
    )
    goals = named_entries(
        record.get("social_goals")
        or record.get("agent_goals")
        or [record.get("agent_goal"), record.get("opponent_goal")],
        [name for name, _ in personas] or names,
    )

    if len(personas) < 2:
        return []

    scenario = first_text(record, ("scenario", "environment", "background"))
    dialogue_history = first_text(record, ("social_interactions", "dialogue_history", "conversation", "history"))
    if not dialogue_history:
        dialogue_history = render_dialogue_from_raw_messages(record.get("raw_messages") or record.get("messages"))

    examples: list[dict[str, Any]] = []
    for agent_index in agent_indices:
        if agent_index >= len(personas):
            continue
        opponent_index = 1 - agent_index if len(personas) == 2 else next(
            idx for idx in range(len(personas)) if idx != agent_index
        )
        agent_name, agent_persona = personas[agent_index]
        opponent_name, opponent_persona = personas[opponent_index]
        goal = goals[agent_index][1] if agent_index < len(goals) else ""

        example = {
            "scenario": scenario,
            "agent_persona": agent_persona,
            "opponent_persona": opponent_persona,
            "agent_goal": goal,
            "dialogue_history": dialogue_history,
            "instruction": (
                f"Write the next message for {agent_name} to {opponent_name}. "
                "Respond with the message only, staying consistent with the persona and private goal."
            ),
            "source": record.get("source") or "sotopia-pi",
            "source_episode_id": record.get("episode_id") or record.get("pk") or record.get("id"),
            "agent_index": agent_index,
            "agent_name": agent_name,
            "opponent_name": opponent_name,
        }
        examples.append(example)
    return examples


def build_prefix_examples(
    record: dict[str, Any],
    max_prefix_examples_per_episode: int,
    max_history_chars: int,
) -> list[dict[str, Any]]:
    names = fallback_agent_names(record)
    personas = dict(
        named_entries(
            record.get("agents_background")
            or record.get("agent_backgrounds")
            or [record.get("agent_persona"), record.get("opponent_persona")],
            names,
        )
    )
    goals = dict(
        named_entries(
            record.get("social_goals")
            or record.get("agent_goals")
            or [record.get("agent_goal"), record.get("opponent_goal")],
            list(personas.keys()) or names,
        )
    )
    if len(personas) < 2:
        return []

    scenario = first_text(record, ("scenario", "environment", "background"))
    raw_messages = record.get("raw_messages") or record.get("messages")
    actor_messages = list(iter_actor_messages(raw_messages))
    if not actor_messages:
        return []

    examples: list[dict[str, Any]] = []
    history: list[str] = []
    for sender, target_response, turn_index, message_index in actor_messages:
        if sender not in personas:
            history.append(f"{sender}: {target_response}")
            continue

        opponent_names = [name for name in personas if name != sender]
        opponent_name = opponent_names[0] if opponent_names else "the other participant"
        dialogue_history = "\n".join(history).strip()
        if max_history_chars > 0 and len(dialogue_history) > max_history_chars:
            dialogue_history = dialogue_history[-max_history_chars:].lstrip()

        examples.append(
            {
                "scenario": scenario,
                "agent_persona": personas[sender],
                "opponent_persona": personas.get(opponent_name, ""),
                "agent_goal": goals.get(sender, ""),
                "dialogue_history": dialogue_history or "(No prior dialogue.)",
                "instruction": (
                    f"Write the next message for {sender} to {opponent_name}. "
                    "Respond with the message only, staying consistent with the persona and private goal."
                ),
                "target_response": target_response,
                "source": record.get("source") or "sotopia-pi",
                "source_episode_id": record.get("episode_id") or record.get("pk") or record.get("id"),
                "turn_index": turn_index,
                "message_index": message_index,
                "agent_name": sender,
                "opponent_name": opponent_name,
            }
        )

        history.append(f"{sender}: {target_response}")
        if max_prefix_examples_per_episode > 0 and len(examples) >= max_prefix_examples_per_episode:
            break

    return examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes-jsonl", type=Path, required=True, help="SOTOPIA episode .jsonl/.json file.")
    parser.add_argument("--output-file", type=Path, required=True, help="Output RC-OPD JSONL path.")
    parser.add_argument(
        "--agent-index",
        choices=("0", "1", "both"),
        default="both",
        help="Which agent perspective to export in full-dialogue mode.",
    )
    parser.add_argument(
        "--history-mode",
        choices=("prefix", "full"),
        default="prefix",
        help="prefix exports dialogue-prefix next-message samples; full exports one final prompt per agent.",
    )
    parser.add_argument(
        "--max-prefix-examples-per-episode",
        type=int,
        default=4,
        help="Maximum prefix samples per episode in prefix mode. 0 means no limit.",
    )
    parser.add_argument(
        "--max-history-chars",
        type=int,
        default=4000,
        help="Keep only the last N dialogue-history characters in prefix mode. 0 means no truncation.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=0,
        help="Maximum output examples. 0 means no limit.",
    )
    parser.add_argument(
        "--require-dialogue",
        action="store_true",
        help="Skip records without a rendered dialogue history.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.episodes_jsonl.exists():
        raise FileNotFoundError(args.episodes_jsonl)

    agent_indices = [0, 1] if args.agent_index == "both" else [int(args.agent_index)]
    args.output_file.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    with args.output_file.open("w", encoding="utf-8") as f:
        for record in iter_records(args.episodes_jsonl):
            if args.history_mode == "prefix":
                examples = build_prefix_examples(
                    record,
                    max_prefix_examples_per_episode=args.max_prefix_examples_per_episode,
                    max_history_chars=args.max_history_chars,
                )
            else:
                examples = build_full_dialogue_examples(record, agent_indices)
            for example in examples:
                if args.require_dialogue and not example["dialogue_history"]:
                    skipped += 1
                    continue
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
                written += 1
                if args.max_examples and written >= args.max_examples:
                    print(f"Wrote {written} examples to {args.output_file}; skipped {skipped}.")
                    return
            if not examples:
                skipped += 1
    print(f"Wrote {written} examples to {args.output_file}; skipped {skipped}.")


if __name__ == "__main__":
    main()
