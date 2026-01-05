import json

filtered_path = "./results/filtered_struct_ReMeDi_qwen3_80b.jsonl"
raw_path = "./results/ReMeDi_raw_sampled.jsonl"
output_path = "./results/ReMeDi_filtered_by_struct_qwen3_80b.jsonl"

# Extract all case_ids
case_ids = set()
with open(filtered_path, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        if "case_id" in data:
            case_ids.add(data["case_id"])

# Filter raw data
matched = []
with open(raw_path, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        if data.get("id") in case_ids:
            matched.append(data)

# Output results
with open(output_path, "w", encoding="utf-8") as f:
    for item in matched:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

