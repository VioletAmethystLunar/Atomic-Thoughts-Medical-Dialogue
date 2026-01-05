# -*- coding: utf-8 -*-
import os
import json
import logging
import time
import re
from tqdm import tqdm
from typing import Dict, List, Optional
from openai import OpenAI

import base64
import threading
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from openai import APIError
import collections
from openpyxl import Workbook

# Initialize Logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Global Variables and Locks
token_lock = Lock()
request_lock = Lock()
rate_limit_semaphore = threading.Semaphore(5)  # 初始并发数

# Performance Monitoring Variables
total_tokens = 0
request_count = 0
start_time = time.time()
failed_requests = 0

# Sliding Window Record
request_timestamps = collections.deque(maxlen=100)
window_lock = Lock()

# Token Bucket
token_buckets = {'start_time': time.time(), 'count': 0}


class Config:
    MAX_WORKERS = 10
    REQUEST_INTERVAL = 1.0
    MAX_QPM = 58
    MAX_TPM = 95000
    MAX_RETRIES = 3
    RETRY_DELAY = 5
    RATE_LIMIT_WINDOW = 60
    SAFETY_FACTOR = 0.9


def check_qpm_limit():
    current_time = time.time()
    with window_lock:
        while request_timestamps and current_time - request_timestamps[0] > Config.RATE_LIMIT_WINDOW:
            request_timestamps.popleft()

        if len(request_timestamps) >= Config.MAX_QPM * Config.SAFETY_FACTOR:
            oldest = request_timestamps[0]
            sleep_time = max(0, Config.RATE_LIMIT_WINDOW - (current_time - oldest)) + 0.5  # 缓冲时间
            time.sleep(sleep_time)
            return False
        return True


def check_tpm_limit(used_tokens):
    current_time = time.time()
    with token_lock:
        elapsed = current_time - token_buckets['start_time']

        if elapsed > Config.RATE_LIMIT_WINDOW:
            token_buckets.update({
                'start_time': current_time,
                'count': used_tokens
            })
        else:
            token_buckets['count'] += used_tokens
            if token_buckets['count'] > Config.MAX_TPM * Config.SAFETY_FACTOR:
                sleep_time = Config.RATE_LIMIT_WINDOW - elapsed + 1
                time.sleep(sleep_time)
                token_buckets.update({
                    'start_time': time.time(),
                    'count': 0
                })


write_lock = threading.Lock()


class MedicalThoughtChainGenerator:
    def __init__(self, api_key: str, model: str = "qwen3-next-80b-a3b-instruct"):
        self.api_key = api_key
        self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.client = OpenAI(api_key=api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.model = model
        logger.info(f"MedicalThoughtChainGenerator initialized with model: {model}")

    def _ensure_think_answer_tags(self, text: str, doctor_reply: str) -> str:
        text = text.strip()
        if re.search(r"<\s*think\s*>", text, re.IGNORECASE) and re.search(r"<\s*/\s*think\s*>", text, re.IGNORECASE) \
                and re.search(r"<\s*answer\s*>", text, re.IGNORECASE) and re.search(r"<\s*/\s*answer\s*>", text,
                                                                                    re.IGNORECASE):
            text = re.sub(r"<\s*think\s*>", "<think>", text, flags=re.IGNORECASE)
            text = re.sub(r"<\s*/\s*think\s*>", "</think>", text, flags=re.IGNORECASE)
            text = re.sub(r"<\s*answer\s*>", "<answer>", text, flags=re.IGNORECASE)
            text = re.sub(r"<\s*/\s*answer\s*>", "</answer>", text, flags=re.IGNORECASE)
            return text
        answer_part = doctor_reply.strip() if doctor_reply and len(
            doctor_reply.strip()) > 0 else "您好，请您提供更多信息。"
        wrapped = f"<think>{text}</think><answer>{answer_part}</answer>"
        return wrapped

    def _call_model_with_retry(self, messages: List[Dict], max_retries: int = 3, backoff: float = 1.0) -> str:
        last_exc = None
        for attempt in range(1, max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=1024,
                    top_p=0.95,
                    extra_body={"enable_thinking": False}
                )
                content = ""
                if hasattr(resp, "choices") and len(resp.choices) > 0:
                    choice = resp.choices[0]
                    if hasattr(choice, "message") and getattr(choice.message, "content", None) is not None:
                        content = choice.message.content
                    elif getattr(choice, "text", None) is not None:
                        content = choice.text
                elif getattr(resp, "message", None) is not None:
                    content = resp.message.get("content", "")
                content = (content or "").strip()
                return content
            except Exception as e:
                last_exc = e
                sleep_t = backoff * (2 ** (attempt - 1))
                logger.warning(
                    f"Model call failed (attempt {attempt}/{max_retries}). Retrying after {sleep_t}s. Error: {e}")
                time.sleep(sleep_t)
        logger.error("Model call failed after retries.")
        raise last_exc

    def _create_thread_client(self):
        return OpenAI(api_key=self.api_key, base_url=self.base_url)

    def generate_thought_chains(self, cases: list, output_file: str) -> None:
        logger.info(f"Starting parallel processing of {len(cases)} cases")
        processed_ids = set()

        if os.path.exists(output_file):
            with open(output_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        processed_ids.add(item.get("id"))
                    except Exception:
                        continue
            logger.info(f"Resuming from checkpoint, already processed {len(processed_ids)} cases")

        def task_wrapper(case):
            global total_tokens, request_count, failed_requests
            case_id = case.get("id")
            if case_id in processed_ids:
                return None
            if not check_qpm_limit():
                logger.debug(f"[{case_id}] QPM rate limit triggered, waiting to retry")

            try:
                self.client = self._create_thread_client()
                result = self._process_case(case)
                with write_lock, open(output_file, "a", encoding="utf-8") as fout:
                    for sample in result.get("samples", []):
                        sample["id"] = case_id
                        fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
                        fout.flush()

                with request_lock:
                    request_count += 1
                    request_timestamps.append(time.time())
                logger.info(f"[Success] Case {case_id} | Current request count: {request_count}")
                return case_id

            except Exception as e:
                with request_lock:
                    failed_requests += 1
                logger.error(f"[Fail] Case {case_id}: {e}")
                return None

        with ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
            futures = {executor.submit(task_wrapper, case): case for case in cases}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing cases"):
                _ = future.result()

        logger.info(f"Save in {output_file}")


    def _process_case(self, case: Dict) -> Dict:
        # print(case)
        case_id = case.get("id")
        dialogue = case.get("dialogue", []) or []
        full_dialogue = case.get("full_dialogue", []) or []  #
        samples = []
        struct_data = case.get("structured_data", {}) or {}

        thought_chains = (
                case.get("thought_chains")
                or case.get("struct_obj", {}).get("thought_chains")
                or []
        )

        level1 = level2 = ""
        level3 = ""
        try:
            thought_chain = struct_data.get("thought_chain") or []
            if isinstance(thought_chain, list) and len(thought_chain) > 0:
                first = thought_chain[0] or {}
                level1 = first.get("level1", "") or ""
                level2 = first.get("level2", "") or ""
                l3 = first.get("level3", []) or []
                if isinstance(l3, list):
                    level3 = "；".join([str(x) for x in l3])
                else:
                    level3 = str(l3)
        except Exception:
            logger.debug("structured_data parsing issue for case: %s", case_id, exc_info=True)

        samples = []
        for sampled_turn in dialogue:
            if not sampled_turn.startswith("患者："):
                continue
            try:
                idx = full_dialogue.index(sampled_turn)
            except ValueError:
                idx = -1
                for k, sentence in enumerate(full_dialogue):
                    if sampled_turn in sentence:
                        idx = k
                        break
                if idx == -1:
                    continue

            history = "\n".join(full_dialogue[:idx])
            patient_input = sampled_turn.replace("患者：", "").strip()
            doctor_reply = ""
            if idx + 1 < len(full_dialogue) and full_dialogue[idx + 1].startswith("医生："):
                doctor_reply = full_dialogue[idx + 1].replace("医生：", "").strip()

            instruction_text = (
                "你是一名医生，能够用清晰精炼的方式和患者对话。请先进行思考，再做出回答，"
                "思考放入<think></think>，回答放入<answer></answer>。\n"
                "<任务>\n"
                "# 任务要求\n"
                "思考：结合<对话历史>，快速判断当前<本轮患者发言>所属的场景\n"
                "对于每个归属的场景，分别按照该场景的医生思考步骤进行独立分析；\n"
                "回复：回答时应面向患者清晰解释你的建议背后的依据，可以适当提及症状、检查目的、治疗逻辑和因果关系，让患者理解你的建议。\n\n"
                "# 任务注意\n"
                "1. 你的回复应该面向本轮<本轮患者发言>的内容；\n"
                "2. 在推荐药物或推荐检查时，应该告知具体的名字；\n"
                "3. 如果不能做出明确的诊断、治疗或者其他建议，你需要进行追问; \n"
                "4. 你不能逃避患者的问题；\n"
                "</任务>\n"
            )

            thought_prompt = \
                f"""医疗思维链生成： 基于以下信息生成专业思维链： 
                === 历史对话 === 
                {history} 
                === 当前患者对话 === 
                {patient_input} 
                === 医生回复 === 
                {doctor_reply} 
                === 患者意图场景 === 
                {level1} 
                === 患者子意图 === 
                {level2} 
                === 医生原子动作 === 
                {level3} 
                要求： 
                1. 请你以医生第一人称进行思考，但不要以“作为医生，我会这样分析”类似开头 
                2. 因果连贯性：使用"首先→进而→最终"等逻辑表达，但最后不要写出逻辑链 
                3. 医学专业性：融入相关医学知识 
                4. 临床思维逻辑：保留医生的临床逻辑 
                5. 语言流畅度：采用自然推理口吻 输出：纯文本描述 """

            input_text = (
                f"<对话历史>\n{history}\n</对话历史>\n"
                f"<本轮患者发言>\n{patient_input}\n</本轮患者发言>\n"
            )

            messages = [
                {"role": "system", "content": "你是一位专业的临床医生助手，擅长进行医学思维链推理与写作。"},
                {"role": "user", "content": thought_prompt}
            ]
            generated = self._call_model_with_retry(messages=messages, max_retries=3, backoff=1.0)
            final_output = self._ensure_think_answer_tags(generated, doctor_reply)

            sample = {
                "instruction": instruction_text,
                "input": input_text,
                "output": final_output
            }
            samples.append(sample)

        return {"id": case_id, "samples": samples}


def print_stats():
    while True:
        elapsed = time.time() - start_time
        current_qpm = (request_count / elapsed) * 60 if elapsed > 0 else 0
        print(
            f"\n[Stats] Real-time QPM: {current_qpm:.1f} | "
            f"Total Requests: {request_count} | Failed: {failed_requests}\n"
        )
        time.sleep(10)


if __name__ == "__main__":
    stats_thread = threading.Thread(target=print_stats, daemon=True)
    stats_thread.start()
    struct_file = "./results/medical_thought_chains_ReMeDi_qwen3_80b.jsonl"
    raw_file = "./results/ReMeDi_filtered_by_struct_qwen3_80b.jsonl"
    output_file = "./results/ReMeDi_thoughtchain_generated_qwen3_80b.jsonl"

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        logger.error("Error: DASHSCOPE_API_KEY environment variable is not set.")
        exit(1)

    full_dialogue_map = {}
    with open("./data/ReMeDi-large-0-converted.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            full_dialogue_map[obj["id"]] = obj["dialogue"]

    # --- Step 1: Load Structures ---
    struct_map = {}
    with open(struct_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            struct_map[obj.get("case_id")] = obj

    # --- Step 2: Merge Data ---
    cases = []
    with open(raw_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            raw = json.loads(line)
            cid = raw.get("id")
            struct_obj = struct_map.get(cid, {}) or {}

            structured_annotation = {}
            if "thought_chains" in struct_obj and isinstance(struct_obj["thought_chains"], list) and len(
                    struct_obj["thought_chains"]) > 0:
                try:
                    structured_annotation = struct_obj["thought_chains"][0].get("structured_annotation", {}) or {}
                except Exception:
                    structured_annotation = {}
            else:
                structured_annotation = struct_obj.get("structured_annotation", {}) or {}

            merged = {
                "id": cid,
                "dialogue": raw.get("dialogue", []),
                "full_dialogue": full_dialogue_map.get(cid, []),
                "struct_obj": struct_obj,
                "structured_data": structured_annotation
            }

            cases.append(merged)

    # --- Step 3: Generate ---
    generator = MedicalThoughtChainGenerator(api_key)
    generator.generate_thought_chains(cases, output_file)

    logger.info(f"Save in {output_file}")
