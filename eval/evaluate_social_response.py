#!/usr/bin/env python
"""Evaluate social next-message generation on RC-OPD SOTOPIA prefix samples."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


STOP_WORDS = {
    "about",
    "after",
    "again",
    "also",
    "because",
    "being",
    "could",
    "from",
    "have",
    "into",
    "more",
    "that",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "with",
    "would",
    "your",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def text_words(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[A-Za-z0-9']+", text.lower())
        if len(token) > 3 and token not in STOP_WORDS
    ]


def token_f1(prediction: str, reference: str) -> float:
    pred = text_words(prediction)
    ref = text_words(reference)
    if not pred or not ref:
        return 0.0
    pred_counts = Counter(pred)
    ref_counts = Counter(ref)
    overlap = sum((pred_counts & ref_counts).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def keyword_overlap(prediction: str, source: str) -> float:
    source_words = set(text_words(source))
    if not source_words:
        return 0.0
    pred_words = set(text_words(prediction))
    return len(source_words & pred_words) / len(source_words)


def has_role_prefix(text: str, agent_name: str, opponent_name: str) -> bool:
    head = text.strip().splitlines()[0][:80] if text.strip() else ""
    names = [re.escape(name) for name in [agent_name, opponent_name] if name]
    if names and re.match(rf"^({'|'.join(names)}|Agent\s*\d+|Assistant|User|You)\s*:", head):
        return True
    return bool(re.match(r"^[A-Z][A-Za-z .'-]{1,40}\s*:", head))


def has_disclaimer(text: str) -> bool:
    lowered = text.lower()
    patterns = [
        "as an ai",
        "i am an ai",
        "i'm an ai",
        "as a language model",
        "i cannot roleplay",
        "i can't roleplay",
    ]
    return any(pattern in lowered for pattern in patterns)


def build_prompt(record: dict[str, Any]) -> str:
    sections = [
        ("Scenario", record.get("scenario", "")),
        ("Agent persona", record.get("agent_persona", "")),
        ("Opponent persona", record.get("opponent_persona", "")),
        ("Private goal", record.get("agent_goal", "")),
        ("Conversation so far", record.get("dialogue_history", "")),
        ("Task", record.get("instruction", "Write the next message.")),
    ]
    return "\n\n".join(f"{label}:\n{value}" for label, value in sections if str(value).strip())


def batched(items: list[Any], batch_size: int) -> list[list[Any]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name_or_path", type=str, default="/data/lsg/models/Qwen3-1.7B")
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument(
        "--dataset_file",
        type=Path,
        default=Path("/data/lsg/work/OPSD/.sotopia_data/rc_opd/sotopia_pi_rc_opd_eval.jsonl"),
    )
    parser.add_argument("--output_file", type=Path, required=True)
    parser.add_argument("--summary_csv", type=Path, default=Path("eval_results/social_response_summary.csv"))
    parser.add_argument("--num_samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--torch_dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_jsonl(args.dataset_file)
    random.Random(args.seed).shuffle(records)
    if args.num_samples > 0:
        records = records[: args.num_samples]

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.torch_dtype]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
    )
    if args.checkpoint_dir:
        model = PeftModel.from_pretrained(model, args.checkpoint_dir)
    model.eval()

    results: list[dict[str, Any]] = []
    for batch in tqdm(batched(records, args.batch_size), desc="Generating"):
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": build_prompt(record)}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for record in batch
        ]
        encoded = tokenizer(prompts, padding=True, return_tensors="pt").to(model.device)
        input_width = encoded["input_ids"].shape[1]

        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        for row, sequence in zip(batch, generated):
            completion_ids = sequence[input_width:]
            prediction = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
            target = str(row.get("target_response", "")).strip()
            result = {
                "source_episode_id": row.get("source_episode_id"),
                "turn_index": row.get("turn_index"),
                "agent_name": row.get("agent_name"),
                "opponent_name": row.get("opponent_name"),
                "scenario": row.get("scenario"),
                "agent_goal": row.get("agent_goal"),
                "dialogue_history": row.get("dialogue_history"),
                "target_response": target,
                "prediction": prediction,
                "token_f1": token_f1(prediction, target),
                "goal_overlap": keyword_overlap(prediction, str(row.get("agent_goal", ""))),
                "persona_overlap": keyword_overlap(prediction, str(row.get("agent_persona", ""))),
                "word_count": len(text_words(prediction)),
                "empty": not bool(prediction.strip()),
                "role_prefix": has_role_prefix(
                    prediction,
                    str(row.get("agent_name", "")),
                    str(row.get("opponent_name", "")),
                ),
                "disclaimer": has_disclaimer(prediction),
            }
            results.append(result)

    def avg(key: str) -> float:
        return sum(float(row[key]) for row in results) / max(len(results), 1)

    summary = {
        "model_name_or_path": args.model_name_or_path,
        "checkpoint_dir": args.checkpoint_dir,
        "dataset_file": str(args.dataset_file),
        "num_samples": len(results),
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "mean_token_f1": avg("token_f1"),
        "mean_goal_overlap": avg("goal_overlap"),
        "mean_persona_overlap": avg("persona_overlap"),
        "mean_word_count": avg("word_count"),
        "empty_rate": avg("empty"),
        "role_prefix_rate": avg("role_prefix"),
        "disclaimer_rate": avg("disclaimer"),
    }

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2, ensure_ascii=False)

    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not args.summary_csv.exists()
    with args.summary_csv.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(summary)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved detailed results to {args.output_file}")
    print(f"Appended summary to {args.summary_csv}")


if __name__ == "__main__":
    main()
