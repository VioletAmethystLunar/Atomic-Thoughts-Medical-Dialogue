# -*- coding: utf-8 -*-
import json

input_path = "./results/medical_thought_chains_ReMeDi_qwen3_80b.jsonl"
output_path = "./results/filtered_struct_ReMeDi_qwen3_80b.jsonl"

EXCLUDE_LEVEL2 = {"请求开药", "请求开病假单", "请求开检查单"}
ALL_KEYWORDS = ["谢谢", "不客气", "再见", "拜拜", "好的", "嗯嗯", "哦哦", "是呀"]


def is_non_clinical_turn(chain_item):
    structured = chain_item.get("structured_annotation", {})
    if not structured:
        return True
    thought_chain = structured.get("thought_chain", [])
    if not thought_chain:
        return True
    doctor_reply = chain_item.get("doctor_reply", "")
    if doctor_reply:
        if any(keyword in doctor_reply for keyword in ALL_KEYWORDS):
            if len(doctor_reply.strip()) < 15:
                return True
    return False


total = 0
removed_whole = 0
modified_partial = 0
kept = 0

with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
    for line in fin:
        line = line.strip()
        if not line:
            continue
        total += 1

        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            print(f"第 {total} 行 JSON 格式错误，已跳过。")
            continue

        thought_chains = obj.get("thought_chains", [])
        total_turns = len(thought_chains)

        # ------------------------------------------------------
        # Layer 1: Filter by EXCLUDE_LEVEL2
        # ------------------------------------------------------
        should_remove_case = False
        for chain in thought_chains:
            structured = chain.get("structured_annotation", {})
            thought_list = structured.get("thought_chain", [])
            for tc in thought_list:
                if tc.get("level2") in EXCLUDE_LEVEL2:
                    should_remove_case = True
                    break
            if should_remove_case:
                break

        if should_remove_case:
            removed_whole += 1
            continue
        # ------------------------------------------------------
        # Layer 2: Non-clinical dialogue filtering
        # ------------------------------------------------------
        bad_indices = []
        for idx, chain in enumerate(thought_chains):
            if is_non_clinical_turn(chain):
                bad_indices.append(idx)

        if bad_indices:
            if total_turns < 3:
                removed_whole += 1
                continue
            else:
                new_chains = [
                    chain for i, chain in enumerate(thought_chains)
                    if i not in bad_indices
                ]
                if not new_chains:
                    removed_whole += 1
                    continue
                obj["thought_chains"] = new_chains
                modified_partial += 1
        else:
            kept += 1
        fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

