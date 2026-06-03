#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RandomBench image question test script (photo_test.py)

Used to perform randomness tests on images from the RB_Vision question bank,
supports both Chinese and English prompts. Output results are stored in
language-specific subdirectories: Results/photo_zh and Results/photo_en.

Usage:
  python photo_test.py --settings settings.txt --language zh --question-ids 1,3,5
  python photo_test.py --settings settings.txt --language en --question-ids all

Command-line arguments:
  --settings       Path to configuration file (default: RandomBench/settings.txt)
  --vision_json    Path to RB_Vision.json (default: RandomBench/RB_Vision.json)
  --photos_dir     Directory containing images (default: RandomBench/RB_Vision)
  --question-ids   Comma-separated list of question IDs, or 'all'
  --language       Prompt language: 'zh' or 'en' (default: zh)
  --out            Custom output file prefix (default: auto-generated)
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import mimetypes
import os
import random
import re
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from openai import AsyncOpenAI
from tqdm import tqdm

# ---------- Runtime parameters ----------
MAX_RETRIES = 5
RETRY_BASE_SLEEP = 2.0
SUCCESS_DELAY_MIN = 0.5
SUCCESS_DELAY_MAX = 2.0
INNER_CONCURRENCY = 5
OUTER_CONCURRENCY = 3

# ---------- System prompts ----------
SYSTEM_PROMPT_EN = (
    "You are a rigorous assistant. "
    "Please make a random choice based on the question and image below. "
    "Use plain text only. Do not generate any images."
)
SYSTEM_PROMPT_ZH = (
    "你是一个严谨的助手。"
    "请根据下面的问题和图片做随机选择。"
    "请用纯文本回答，禁止生成图片。"
)

LOGGER_NAME = "random_bench.photo"
logger = logging.getLogger(LOGGER_NAME)


@dataclass(frozen=True)
class Settings:
    api_key: str
    base_url: str
    timeout_s: float = 60.0
    models: Tuple[str, ...] = ("gpt-5.1",)
    analysis_model: str = "gpt-5.1"
    repeats_per_question: int = 50
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 200
    analysis_api_key: str = ""
    analysis_base_url: str = ""


def setup_run_logging(*, run_name: str, logs_dir: Path) -> Path:
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{run_name}.log"

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)

    logging.Formatter.converter = time.gmtime
    fmt = logging.Formatter(
        fmt="%(asctime)s.%(msecs)03dZ [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    root.addHandler(sh)
    root.addHandler(fh)
    return log_path


def read_settings(path: Path) -> Settings:
    raw: Dict[str, str] = {}
    text = path.read_text(encoding="utf-8").splitlines()
    for line in text:
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        raw[k.strip()] = v.strip().strip('"').strip("'")

    api_key = raw.get("API_KEY") or os.environ.get("API_KEY") or ""
    base_url = raw.get("BASE_URL") or os.environ.get("BASE_URL") or ""
    if not api_key:
        raise ValueError("Missing API_KEY")
    if not base_url:
        raise ValueError("Missing BASE_URL")

    models_str = raw.get("MODELS") or os.environ.get("MODELS")
    if models_str:
        models = tuple(m.strip() for m in re.split(r"[,\s]+", models_str) if m.strip())
    else:
        models = Settings(api_key=api_key, base_url=base_url).models

    analysis_model = raw.get("ANALYSIS_MODEL") or os.environ.get("ANALYSIS_MODEL")
    if not analysis_model:
        analysis_model = Settings(api_key=api_key, base_url=base_url).analysis_model

    repeats = raw.get("REPEATS") or os.environ.get("REPEATS")
    repeats_i = int(repeats) if repeats else Settings(api_key=api_key, base_url=base_url).repeats_per_question

    timeout_s = float(raw.get("TIMEOUT_S") or os.environ.get("TIMEOUT_S") or 60.0)
    temperature = float(raw.get("TEMPERATURE") or os.environ.get("TEMPERATURE") or 1.0)
    top_p = float(raw.get("TOP_P") or os.environ.get("TOP_P") or 1.0)
    max_tokens = int(raw.get("MAX_TOKENS") or os.environ.get("MAX_TOKENS") or 200)

    analysis_api_key = raw.get("ANALYSIS_API_KEY") or os.environ.get("ANALYSIS_API_KEY") or ""
    analysis_base_url = raw.get("ANALYSIS_BASE_URL") or os.environ.get("ANALYSIS_BASE_URL") or ""

    return Settings(
        api_key=api_key,
        base_url=base_url.rstrip("/"),
        timeout_s=timeout_s,
        models=models,
        analysis_model=analysis_model,
        repeats_per_question=repeats_i,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        analysis_api_key=analysis_api_key.rstrip("/") if analysis_base_url else "",
        analysis_base_url=analysis_base_url.rstrip("/") if analysis_base_url else "",
    )


def read_photo_questions_json(path: Path) -> Dict[int, Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    qmap: Dict[int, Dict[str, Any]] = {}
    for item in data:
        qid = int(item["id"])
        qmap[qid] = item
    return qmap


def parse_question_ids(spec: str) -> List[int]:
    ids: List[int] = []
    for token in re.split(r"[,\s]+", spec.strip()):
        if not token:
            continue
        ids.append(int(token))
    if not ids:
        raise ValueError("question-ids is empty")
    return ids


def build_user_prompt(stem: str, options: Mapping[str, str], language: str) -> str:
    lines = [stem]
    if options:
        lines.append("")
        if language == "en":
            lines.append("Please randomly select one option based on the image.")
            lines.append("Output requirement: output only one uppercase letter (e.g., A), and nothing else.")
        else:
            lines.append("请根据图片随机选择一个选项。")
            lines.append("输出要求：只输出一个大写字母（例如 A），不要输出其它任何内容。")
        lines.append("")
        for k, v in options.items():
            lines.append(f"({k}) {v}")
    else:
        lines.append("")
        if language == "en":
            lines.append("Please provide a short, intuitive answer based on the image, without explaining your reasoning.")
        else:
            lines.append("请结合图片内容，给出一个简短直观的回答，不要解释你的思考过程。")
    return "\n".join(lines)


def image_to_data_url(path: Path) -> str:
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    mime, _ = mimetypes.guess_type(path.name)
    if not mime:
        mime = "image/png"
    return f"data:{mime};base64,{b64}"


async def async_chat_completions_with_image(
    *,
    client: AsyncOpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    image_data_url: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> str:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {"type": "image_url", "image_url": {"url": image_data_url}},
                        ],
                    },
                ],
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            content = resp.choices[0].message.content
            if content is None or content.strip() == "":
                raise ValueError("Empty response or None")
            await asyncio.sleep(random.uniform(SUCCESS_DELAY_MIN, SUCCESS_DELAY_MAX))
            return content.strip()
        except Exception as e:
            logger.warning("Request failed (attempt %s/%s): %s", attempt, MAX_RETRIES, e)
            if attempt < MAX_RETRIES:
                await asyncio.sleep(RETRY_BASE_SLEEP * attempt)
            else:
                logger.error("Still failing after multiple retries, giving up this request")
                return ""


def extract_choice(text: str, valid_labels: Iterable[str]) -> Optional[str]:
    valid = set(valid_labels)
    if not valid:
        return None
    m = re.search(r"\b([A-Z])\b", text.strip().upper())
    if m and m.group(1) in valid:
        return m.group(1)
    for lab in valid:
        if re.search(rf"(\({lab}\))|(\b{lab}\b)|([选答答案：:\s]{lab})", text.upper()):
            return lab
    return None


async def analyze_responses_with_vlm_async(
    *,
    client: AsyncOpenAI,
    model: str,
    question: str,
    raw_counts: Dict[str, int],
    image_data_url: str,
) -> Dict[str, Any]:
    if not raw_counts:
        return {"summary": "No responses", "total_valid": 0}

    unique_answers = list(raw_counts.keys())
    numbered_answers = "\n".join(f"- {a}" for a in unique_answers)

    system_prompt = (
        "You are a meticulous data annotator. Observe the image, read the test question, "
        "then classify the free-text responses.\n\n"
        "Steps:\n"
        "1. Determine how many distinct visual candidates (K) fit the question.\n"
        "2. Define concise standard names for these K candidates.\n"
        "3. Map each unique response to a standard name.\n\n"
        "CRITICAL RULES for building classification_map:\n"
        "- The keys MUST BE THE EXACT, COMPLETE strings from the provided list of unique responses.\n"
        "- Do NOT add any prefix like '- ', '* ', or any other character.\n"
        "- Do NOT split a response at line breaks. Keep the entire multi-line response as one key.\n"
        "- Do NOT remove or modify any character, including punctuation, spaces, line breaks, or markdown.\n"
        "- The key must be a verbatim copy of the response as it appears in the list.\n"
        "- If a response contains special formatting (like `**bold**`), keep it exactly as is.\n"
        "- If a response is vague, refuses to answer, or cannot be matched, map it to \"Invalid\".\n\n"
        "Required JSON format:\n"
        "{\n"
        '  "K": 4,\n'
        '  "standard_options": ["Option A", "Option B", "Option C", "Option D"],\n'
        '  "classification_map": {\n'
        '    "exact original response 1": "Option A",\n'
        '    "exact original response 2": "Option B"\n'
        "  }\n"
        "}"
    )
    user_prompt = f"Test question: {question}\n\nUnique responses to classify:\n{numbered_answers}"

    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": image_data_url}},
                    ],
                },
            ],
            temperature=0.1,
            top_p=1.0,
            max_tokens=1024,
            response_format={"type": "json_object"},
        )
        text = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return {"summary": f"VLM analysis call failed: {e}", "error": str(e)}

    try:
        parsed = json.loads(text)
        classification_map = parsed.get("classification_map", {})
        standard_options = parsed.get("standard_options", [])

        mapped_counts = {opt: 0 for opt in standard_options}
        invalid_count = 0
        for raw_ans, count in raw_counts.items():
            std_opt = classification_map.get(raw_ans, "Invalid")
            if std_opt in mapped_counts:
                mapped_counts[std_opt] += count
            elif std_opt == "Invalid":
                invalid_count += count
            else:
                mapped_counts[std_opt] = mapped_counts.get(std_opt, 0) + count

        parsed["mapped_counts"] = mapped_counts
        parsed["invalid_count"] = invalid_count
        parsed["total_valid"] = sum(mapped_counts.values())
        parsed["summary"] = f"Mapped responses to {parsed.get('K', len(standard_options))} image options"
        return parsed
    except Exception as e:
        return {"summary": "Failed to parse VLM output as JSON", "raw_output": text, "error": str(e)}


async def process_one_question(
    client: AsyncOpenAI,
    model: str,
    stem: str,
    options: "OrderedDict[str, str]",
    image_path: Path,
    settings: Settings,
    language: str,
    analysis_client: AsyncOpenAI = None,
) -> Dict[str, Any]:
    system_prompt = SYSTEM_PROMPT_EN if language == "en" else SYSTEM_PROMPT_ZH
    user_prompt = build_user_prompt(stem, options, language)
    image_data_url = image_to_data_url(image_path)
    inner_semaphore = asyncio.Semaphore(INNER_CONCURRENCY)

    async def single_request_with_limit() -> str:
        async with inner_semaphore:
            return await async_chat_completions_with_image(
                client=client,
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                image_data_url=image_data_url,
                temperature=settings.temperature,
                top_p=settings.top_p,
                max_tokens=settings.max_tokens,
            )

    tasks = [single_request_with_limit() for _ in range(settings.repeats_per_question)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    answers: List[str] = []
    for res in results:
        if isinstance(res, Exception):
            logger.warning("Single request failed: %s", res)
            answers.append("")
        else:
            answers.append(str(res))

    use_option_stats = bool(options)
    counts: Counter[str] = Counter()
    invalid = 0
    if use_option_stats:
        for text in answers:
            choice = extract_choice(text, options.keys())
            if choice is None:
                invalid += 1
            else:
                counts[choice] += 1
        analysis = {
            "summary": "option_frequency_based",
            "total_calls": settings.repeats_per_question,
            "counts": {k: int(counts.get(k, 0)) for k in options.keys()},
            "invalid": invalid,
        }
        raw_counts: Counter[str] = counts
    else:
        raw_counts = Counter(a.strip() for a in answers if a.strip())
        analysis = await analyze_responses_with_vlm_async(
            client=analysis_client or client,
            model=settings.analysis_model,
            question=stem,
            raw_counts=dict(raw_counts),
            image_data_url=image_data_url,
        )

    return {
        "stem": stem,
        "options": dict(options),
        "image_path": str(image_path),
        "answers": answers,
        "raw_counts": dict(raw_counts),
        "analysis": analysis,
        "total_calls": settings.repeats_per_question,
    }


def build_default_output_prefix() -> Path:
    return Path("RandomBench/Results/photo")


def build_model_output_path(base_path: Path, model: str, run_ts: str) -> Path:
    safe_model = model.replace("/", "_").replace("\\", "_")
    if base_path.suffix == "":
        return base_path.with_name(f"{base_path.name}_{safe_model}_{run_ts}.json")
    return base_path.with_name(f"{base_path.stem}_{safe_model}_{run_ts}{base_path.suffix}")


async def run_photo_benchmark_async(
    *,
    settings: Settings,
    question_items: List[Dict[str, Any]],
    photos_dir: Path,
    out_prefix: Path,
    language: str,
) -> None:
    started_at = datetime.now(timezone.utc).isoformat()
    run_ts = datetime.now().strftime("%m%d%H%M")
    run_name = f"photo_{language}_run_{run_ts}"
    logs_dir = Path("RandomBench/logs")
    log_path = setup_run_logging(run_name=run_name, logs_dir=logs_dir)

    client = AsyncOpenAI(
        base_url=settings.base_url,
        api_key=settings.api_key,
        timeout=settings.timeout_s,
    )

    analysis_api_key = settings.analysis_api_key or settings.api_key
    analysis_base_url = settings.analysis_base_url or settings.base_url
    analysis_client = AsyncOpenAI(
        base_url=analysis_base_url,
        api_key=analysis_api_key,
        timeout=settings.timeout_s,
    )

    outer_semaphore = asyncio.Semaphore(OUTER_CONCURRENCY)

    async def process_one_question_with_semaphore(item: Dict[str, Any], model: str) -> Tuple[int, Dict[str, Any]]:
        async with outer_semaphore:
            qid = int(item["id"])
            stem = item["prompt_EN"] if language == "en" else item["prompt_ZH"]
            options: "OrderedDict[str, str]" = OrderedDict()
            image_path = photos_dir / item["image_path"]
            if not image_path.exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")
            q_item = await process_one_question(
                client=client,
                model=model,
                stem=stem,
                options=options,
                image_path=image_path,
                settings=settings,
                language=language,
                analysis_client=analysis_client,
            )
            q_item["index"] = qid
            q_item["raw_line"] = stem
            q_item["prompt_language"] = language
            q_item["image_hash_name"] = item["image_path"]
            return qid, q_item

    for model in settings.models:
        logger.info("Starting model (async): %s", model)
        model_results: Dict[str, Any] = {
            "meta": {
                "started_at": started_at,
                "run_name": run_name,
                "log_path": str(log_path),
                "system_prompt": SYSTEM_PROMPT_EN if language == "en" else SYSTEM_PROMPT_ZH,
                "model": model,
                "repeats_per_question": settings.repeats_per_question,
                "temperature": settings.temperature,
                "top_p": settings.top_p,
                "max_tokens": settings.max_tokens,
                "language": language,
            },
            "questions": [],
        }
        tasks = [process_one_question_with_semaphore(item, model) for item in question_items]
        with tqdm(total=len(tasks), desc=f"Model (image) {model}", unit="q") as bar:
            for coro in asyncio.as_completed(tasks):
                idx, q_item = await coro
                model_results["questions"].append(q_item)
                model_results["questions"].sort(key=lambda x: x["index"])
                out_path = build_model_output_path(out_prefix, model, run_ts)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(json.dumps(model_results, ensure_ascii=False, indent=2), encoding="utf-8")
                bar.update(1)
                logger.info("Completed question: model=%s idx=%s", model, idx)

        model_results["meta"]["finished_at"] = datetime.now(timezone.utc).isoformat()
        out_path = build_model_output_path(out_prefix, model, run_ts)
        out_path.write_text(json.dumps(model_results, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Final result written: model=%s path=%s", model, str(out_path))


def main() -> int:
    parser = argparse.ArgumentParser(description="RandomBench image question test")
    parser.add_argument("--settings", default="RandomBench/settings.txt", help="Path to configuration file")
    parser.add_argument("--vision_json", default="RandomBench/RB_Vision.json", help="Path to RB_Vision.json")
    parser.add_argument("--photos_dir", default="RandomBench/RB_Vision", help="Directory containing images")
    parser.add_argument("--question-ids", required=True, help="Comma-separated list of question IDs")
    parser.add_argument("--language", choices=["zh", "en"], default="zh", help="Prompt language")
    parser.add_argument("--out", default="", help="Custom output file prefix (default: Results/photo_<lang>)")
    args = parser.parse_args()

    settings = read_settings(Path(args.settings))
    qmap = read_photo_questions_json(Path(args.vision_json))
    wanted_ids = parse_question_ids(args.question_ids)

    missing = [qid for qid in wanted_ids if qid not in qmap]
    if missing:
        raise ValueError(f"Question IDs not found in question bank: {missing}")

    question_items = [qmap[qid] for qid in wanted_ids]
    if not args.out:
        out_prefix = Path(f"RandomBench/Results/photo_{args.language}/photo")
    else:
        out_prefix = Path(args.out)

    asyncio.run(
        run_photo_benchmark_async(
            settings=settings,
            question_items=question_items,
            photos_dir=Path(args.photos_dir),
            out_prefix=out_prefix,
            language=args.language,
        )
    )

    print(f"Done. Results saved under prefix: {out_prefix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())