# -*- coding: utf-8 -*-
import json
import random

ALL_KEYWORDS = ["谢谢", "不客气", "再见", "拜拜", "好的", "嗯嗯", "哦哦", "是呀"]


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return data


def save_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def is_valid_round(doctor_reply):
    if not doctor_reply:
        return False
    has_keyword = any(keyword in doctor_reply for keyword in ALL_KEYWORDS)
    if has_keyword and len(doctor_reply.strip()) < 15:
        return False
    return True


def process_and_sample_dialogues(records, seed=42):
    random.seed(seed)
    new_records = []
    skipped_count = 0

    for rec in records:
        dial = rec.get("dialogue", [])
        raw_rounds = [dial[i:i + 2] for i in range(0, len(dial), 2) if i + 1 < len(dial)]
        if not raw_rounds:
            skipped_count += 1
            continue
        valid_rounds = [
            r for r in raw_rounds
            if is_valid_round(r[1])
        ]
        if len(valid_rounds) < 3:
            skipped_count += 1
            continue
        random_picks = random.sample(valid_rounds[1:], 2)
        sampled_rounds = [valid_rounds[0]] + random_picks
        new_dial = [utt for r in sampled_rounds for utt in r]
        rec_new = dict(rec)
        rec_new["dialogue"] = new_dial
        new_records.append(rec_new)
    return new_records


if __name__ == "__main__":
    input_file = "./data/ReMeDi-large-0-converted.jsonl"
    output_file = "./results/ReMeDi_raw_sampled.jsonl"
    records = load_jsonl(input_file)
    sampled = process_and_sample_dialogues(records, seed=42)
    save_jsonl(sampled, output_file)
