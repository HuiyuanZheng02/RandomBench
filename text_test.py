#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RandomBench Text Test Script

Evaluates MLLMs on logic-neutral text-based random choice tasks.
Supports language selection (zh/en), option label replacement (greek/geometry/random),
and question subset testing.

Usage:
  python text_test.py --settings ../settings.txt --question-ids 1-100 --prompt-language zh
  python text_test.py --settings ../settings.txt --question-ids 26-32 --replace-labels greek
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import string
from collections import Counter, OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from openai import AsyncOpenAI
from tqdm import tqdm

# Concurrency and retry configuration
MAX_RETRIES = 5
RETRY_BASE_SLEEP = 30.0
SUCCESS_DELAY_MIN = 0.5
SUCCESS_DELAY_MAX = 1.5
INNER_CONCURRENCY = 10
OUTER_CONCURRENCY = 5


@dataclass(frozen=True)
class Settings:
    api_key: str
    base_url: str
    timeout_s: float = 60.0
    models: Tuple[str, ...] = ("gpt-5.1",)                    
    repeats_per_question: int = 50
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 100


# ---------- System prompts  ----------
SYSTEM_PROMPT_ZH = "你是一个严谨的助手。请根据下面的问题做随机选择。禁止生成图片。"
SYSTEM_PROMPT_EN = "You are a rigorous assistant. Randomly choose based on the question below. Do not generate images."


def read_settings(path: Path) -> Settings:
    """Read API keys and model list from a KEY=VALUE file."""
    raw: Dict[str, str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#") or "=" not in s:
                continue
            k, v = s.split("=", 1)
            raw[k.strip()] = v.strip().strip('"').strip("'")

    api_key = raw.get("API_KEY") or os.environ.get("API_KEY") or ""
    base_url = raw.get("BASE_URL") or os.environ.get("BASE_URL") or ""
    if not api_key or not base_url:
        raise ValueError("Missing API_KEY or BASE_URL in settings file or environment")

    models_str = raw.get("MODELS") or os.environ.get("MODELS")
    if models_str:
        models = tuple(m.strip() for m in re.split(r"[,\s]+", models_str) if m.strip())
    else:
        models = Settings(api_key=api_key, base_url=base_url).models

    repeats = int(raw.get("REPEATS") or os.environ.get("REPEATS") or 50)
    timeout_s = float(raw.get("TIMEOUT_S") or os.environ.get("TIMEOUT_S") or 60.0)
    temperature = float(raw.get("TEMPERATURE") or os.environ.get("TEMPERATURE") or 1.0)
    top_p = float(raw.get("TOP_P") or os.environ.get("TOP_P") or 1.0)
    max_tokens = int(raw.get("MAX_TOKENS") or os.environ.get("MAX_TOKENS") or 100)

    return Settings(
        api_key=api_key,
        base_url=base_url.rstrip("/"),
        timeout_s=timeout_s,
        models=models,
        repeats_per_question=repeats,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )


def read_questions_json(path: Path) -> List[Dict[str, Any]]:
    """Read RB_Text.json and return a list of question dictionaries."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not data:
        raise ValueError(f"Question bank is empty: {path}")
    return data


def parse_question_ids(spec: str, total: int) -> List[int]:
    """Parse a comma-separated list of question IDs (1-based)."""
    ids: List[int] = []
    for token in re.split(r"[,\s]+", spec.strip()):
        if not token:
            continue
        qid = int(token)
        if qid < 1 or qid > total:
            raise ValueError(f"Question ID {qid} out of range 1..{total}")
        ids.append(qid)
    if not ids:
        raise ValueError("question-ids is empty")
    return ids


def parse_question_line(line: str) -> Tuple[str, "OrderedDict[str, str]"]:
    """Extract stem and option labels from a formatted prompt line."""
    matches = list(re.finditer(r"\(([A-Z])\)\s*", line))
    if not matches:
        return line.strip(), OrderedDict()

    stem = line[: matches[0].start()].strip()
    options: "OrderedDict[str, str]" = OrderedDict()
    for i, m in enumerate(matches):
        label = m.group(1)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(line)
        text = line[start:end].strip()
        text = re.sub(r"\s+$", "", text)
        options[label] = text
    if not stem:
        stem = line.strip()
    return stem, options


def make_replacement_labels(mode: str, count: int, seed: int) -> List[str]:
    """Generate alternative option symbols."""
    if mode == "none":
        return [chr(ord("A") + i) for i in range(count)]
    if mode == "greek":
        base = ["α", "β", "γ", "δ", "ε", "ζ"]
        return [base[i] for i in range(count)]
    if mode == "geometry":
        base = ["▲", "■", "●", "◆", "◇", "△"]
        return [base[i] for i in range(count)]
    if mode == "random":
        rng = random.Random(seed)
        labels = []
        for _ in range(count):
            token = "".join(rng.choice(string.ascii_lowercase + string.digits) for _ in range(6))
            labels.append(f"tag_{token}")
        return labels
    raise ValueError(f"Unknown replace-labels mode: {mode}")


def build_user_prompt(
    stem: str,
    options: Mapping[str, str],
    *,
    prompt_language: str,
    replace_labels_mode: str,
    replacement_seed: int,
) -> Tuple[str, Dict[str, str], Dict[str, str]]:
    """
    Construct the final user prompt with strict output format instructions.
    Returns the prompt text and two mapping dicts for label normalisation.
    """
    if not options:
        if prompt_language == "en":
            return (
                f"{stem}\n\nPlease output your final random choice directly in plain text only.",
                {},
                {},
            )
        return (
            f"{stem}\n\n请直接输出你的最终选择。注意：必须只输出纯文本。",
            {},
            {},
        )

    canonical_labels = list(options.keys())
    shown_labels = make_replacement_labels(replace_labels_mode, len(canonical_labels), replacement_seed)
    canonical_to_shown = {c: s for c, s in zip(canonical_labels, shown_labels)}
    shown_to_canonical = {s: c for c, s in canonical_to_shown.items()}

    if replace_labels_mode == "greek":
        example = "α"
    elif replace_labels_mode == "geometry":
        example = "▲"
    elif replace_labels_mode == "random":
        example = "tag_xxxxxx"
    else:
        example = "A"

    if prompt_language == "en":
        lines = [
            stem,
            "",
            "Randomly select one option from the list below. Your response must strictly follow these rules:",
            f"Output only the option label (e.g., {example}). Do not include any other text, punctuation, spaces, or line breaks.",
            "Do not explain your choice, do not repeat the question, do not use sentences.",
        ]
    else:
        lines = [
            stem,
            "",
            "从下列选项中完全随机地选择一个。你的回答必须严格遵循以下规则：",
            f"只输出一个选项标识符（例如 {example}），不要有任何其他文字、标点、空格或换行。",
            "不要解释你的选择，不要重复题目，不要使用句子。",
        ]

    for c in canonical_labels:
        lines.append(f"({canonical_to_shown[c]}) {options[c]}")

    return "\n".join(lines), canonical_to_shown, shown_to_canonical


async def async_chat_completions(
    *,
    client: AsyncOpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> str:
    """Send a request with retry and return the stripped response."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            content = resp.choices[0].message.content
            if content is None or content.strip() == "":
                raise ValueError("Empty response")
            await asyncio.sleep(random.uniform(SUCCESS_DELAY_MIN, SUCCESS_DELAY_MAX))
            return content.strip()
        except Exception:
            if attempt == MAX_RETRIES:
                raise
            await asyncio.sleep(RETRY_BASE_SLEEP * attempt)


def extract_choice(
    text: str,
    valid_labels: Iterable[str],
    shown_to_canonical: Mapping[str, str],
) -> Optional[str]:
    """
    Normalise a model's answer back to the canonical label (A/B/C/D).
    Supports both the new symbols and fallback to original uppercase letters.
    """
    cleaned = text.strip()
    if not cleaned:
        return None

    upper_map = {k.upper(): v for k, v in shown_to_canonical.items()}
    token = re.sub(r"^[\(\[\{<\s]+|[\)\]\}>\s\.,;:!]+$", "", cleaned)
    token_upper = token.upper()
    if token_upper in upper_map:
        return upper_map[token_upper]

    valid = set(valid_labels)
    m = re.search(r"\b([A-Z])\b", cleaned.upper())
    if m and m.group(1) in valid:
        return m.group(1)
    for lab in valid:
        if re.search(rf"(\({lab}\))|(\b{lab}\b)|([选答答案：:\s]{lab})", cleaned.upper()):
            return lab
    return None


async def process_one_question(
    client: AsyncOpenAI,
    model: str,
    stem: str,
    options: "OrderedDict[str, str]",
    settings: Settings,
    *,
    prompt_language: str,
    replace_labels_mode: str,
    replacement_seed: int,
) -> Dict[str, Any]:
    """Run all repetitions for a single question and aggregate results."""
    user_prompt, canonical_to_shown, shown_to_canonical = build_user_prompt(
        stem, options,
        prompt_language=prompt_language,
        replace_labels_mode=replace_labels_mode,
        replacement_seed=replacement_seed,
    )
    inner_semaphore = asyncio.Semaphore(INNER_CONCURRENCY)
    system_prompt = SYSTEM_PROMPT_EN if prompt_language == "en" else SYSTEM_PROMPT_ZH

    async def single_request():
        async with inner_semaphore:
            return await async_chat_completions(
                client=client,
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=settings.temperature,
                top_p=settings.top_p,
                max_tokens=settings.max_tokens,
            )

    tasks = [single_request() for _ in range(settings.repeats_per_question)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    answers: List[str] = []
    counts: Counter[str] = Counter()
    invalid = 0
    use_option_stats = bool(options)

    for res in results:
        if isinstance(res, Exception):
            answers.append("")
            invalid += 1
            continue
        answer = str(res)
        answers.append(answer)

        if use_option_stats:
            canonical = extract_choice(answer, options.keys(), shown_to_canonical)
            if canonical is None:
                invalid += 1
            else:
                counts[canonical] += 1
        else:
            cleaned = answer.strip()
            if cleaned:
                counts[cleaned] += 1
            else:
                invalid += 1

    analysis = {
        "summary": "option_frequency_based" if use_option_stats else "unrestricted_frequency",
        "total_calls": settings.repeats_per_question,
        "counts": dict(counts),
        "invalid": invalid,
    }

    return {
        "stem": stem,
        "options": dict(options),
        "display_labels": canonical_to_shown,
        "answers": answers,
        "raw_counts": dict(Counter(a.strip() for a in answers if a.strip())),
        "analysis": analysis,
        "total_calls": settings.repeats_per_question,
    }


def build_output_path(out_dir: Path, model: str, run_ts: str, language: str) -> Path:
    """Generate an output filename inside the given directory."""
    safe_model = model.replace("/", "_").replace("\\", "_")
    return out_dir / f"text_{language}_{safe_model}_{run_ts}.json"


async def run_benchmark_async(
    settings: Settings,
    selected_questions: List[Tuple[int, str]],
    out_dir: Path,
    language: str,
    *,
    replace_labels_mode: str,
    replacement_seed: int,
) -> None:
    """Main async workflow: iterate over models and questions, save results incrementally."""
    started_at = datetime.now(timezone.utc).isoformat()
    run_ts = datetime.now().strftime("%m%d%H%M")

    client = AsyncOpenAI(
        base_url=settings.base_url,
        api_key=settings.api_key,
        timeout=settings.timeout_s,
    )
    outer_semaphore = asyncio.Semaphore(OUTER_CONCURRENCY)

    async def process_with_semaphore(idx: int, line: str, model: str):
        async with outer_semaphore:
            stem, options = parse_question_line(line)
            q_item = await process_one_question(
                client, model, stem, options, settings,
                prompt_language=language,
                replace_labels_mode=replace_labels_mode,
                replacement_seed=replacement_seed + idx,
            )
            q_item["index"] = idx
            q_item["raw_line"] = line
            return idx, q_item

    for model in settings.models:
        model_results: Dict[str, Any] = {
            "meta": {
                "started_at": started_at,
                "model": model,
                "repeats_per_question": settings.repeats_per_question,
                "temperature": settings.temperature,
                "top_p": settings.top_p,
                "max_tokens": settings.max_tokens,
                "prompt_language": language,
                "replace_labels_mode": replace_labels_mode,
                "replacement_seed": replacement_seed,
            },
            "questions": [],
        }

        tasks = [process_with_semaphore(idx, line, model) for idx, line in selected_questions]
        with tqdm(total=len(tasks), desc=f"Text {model} ({language})", unit="q") as bar:
            for coro in asyncio.as_completed(tasks):
                _, q_item = await coro
                model_results["questions"].append(q_item)
                model_results["questions"].sort(key=lambda x: x["index"])

                out_file = build_output_path(out_dir, model, run_ts, language)
                out_file.parent.mkdir(parents=True, exist_ok=True)
                out_file.write_text(json.dumps(model_results, ensure_ascii=False, indent=2), encoding="utf-8")
                bar.update(1)

        model_results["meta"]["finished_at"] = datetime.now(timezone.utc).isoformat()
        out_file = build_output_path(out_dir, model, run_ts, language)
        out_file.write_text(json.dumps(model_results, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="RandomBench Text Test")
    parser.add_argument("--settings", default="../settings.txt", help="Path to settings file")
    parser.add_argument("--questions", default="RB_Text.json", help="Path to RB_Text.json")
    parser.add_argument("--question-ids", required=True, help="Comma-separated question IDs, e.g. 4,10,25")
    parser.add_argument("--prompt-language", choices=["zh", "en"], default="zh", help="Prompt language")
    parser.add_argument("--replace-labels", choices=["none", "greek", "geometry", "random"], default="none",
                        help="Replace A/B/C/D with other symbols")
    parser.add_argument("--replacement-seed", type=int, default=20260428, help="Seed for random label generation")
    parser.add_argument("--out", default="", help="Output directory (default: Results/text_<lang>)")
    args = parser.parse_args()

    settings = read_settings(Path(args.settings))
    all_questions = read_questions_json(Path(args.questions))
    ids = parse_question_ids(args.question_ids, len(all_questions))

    selected = []
    for qid in ids:
        item = all_questions[qid - 1]
        prompt = item["prompt_en"] if args.prompt_language == "en" else item["prompt_zh"]
        selected.append((qid, prompt))

    if args.out:
        out_dir = Path(args.out)
    else:
        out_dir = Path("Results") / f"text_{args.prompt_language}"

    asyncio.run(
        run_benchmark_async(
            settings=settings,
            selected_questions=selected,
            out_dir=out_dir,
            language=args.prompt_language,
            replace_labels_mode=args.replace_labels,
            replacement_seed=args.replacement_seed,
        )
    )

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())