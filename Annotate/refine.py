import os
import json
import time
import threading
import collections
import re
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from tqdm import tqdm
from openai import OpenAI, APIError

# Initialize Logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Global Variables and Locks
token_lock = threading.Lock()
request_lock = threading.Lock()
rate_limit_semaphore = threading.Semaphore(5)
write_lock = threading.Lock()
window_lock = threading.Lock()

# Performance Monitoring Variables
total_tokens = 0
request_count = 0
failed_requests = 0
start_time = time.time()

request_timestamps = collections.deque(maxlen=200)
token_buckets = {'start_time': time.time(), 'count': 0}


class Config:
    MAX_WORKERS = 64
    MAX_QPM = 250
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
            sleep_time = max(0, Config.RATE_LIMIT_WINDOW - (current_time - oldest)) + 0.5
            print(f"[{datetime.now().strftime('%H:%M:%S')}] QPM approaching the limit, waiting {sleep_time:.1f}s")
            time.sleep(sleep_time)
            return False
        return True


def check_tpm_limit(used_tokens):
    current_time = time.time()
    with token_lock:
        elapsed = current_time - token_buckets['start_time']
        if elapsed > Config.RATE_LIMIT_WINDOW:
            token_buckets.update({'start_time': current_time, 'count': used_tokens})
        else:
            token_buckets['count'] += used_tokens
            if token_buckets['count'] > Config.MAX_TPM * Config.SAFETY_FACTOR:
                sleep_time = Config.RATE_LIMIT_WINDOW - elapsed + 1
                print(f"[{datetime.now().strftime('%H:%M:%S')}] 达到TPM上限，等待{sleep_time:.1f}s")
                time.sleep(sleep_time)
                token_buckets.update({'start_time': time.time(), 'count': 0})


def extract_think_and_answer(text: str):
    think_match = re.search(r"<think>(.*?)</think>", text, re.S)
    answer_match = re.search(r"<answer>(.*?)</answer>", text, re.S)
    think_text = think_match.group(1).strip() if think_match else ""
    answer_text = answer_match.group(1).strip() if answer_match else ""
    return think_text, answer_text


def process_case(data, simulator, output_file):
    global total_tokens, request_count, failed_requests

    case_id = data.get("id")
    think_data, answer_data = extract_think_and_answer(data["output"])

    for attempt in range(Config.MAX_RETRIES):
        try:
            if not check_qpm_limit():
                continue

            with rate_limit_semaphore:
                client = OpenAI(api_key=simulator.api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

                user_prompt = (
                    f"以下是医生的思考内容（<think>）和原始回复（<answer>）。\n"
                    f"请你基于这些内容，生成一条更自然、更有同理心、但医学表达准确的医生回复。\n\n"
                    f"<think>\n{think_data}\n</think>\n\n"
                    f"<answer>\n{answer_data}\n</answer>\n\n"
                    f"请直接输出格式为：医生回复：xxxx"
                )

                completion = client.chat.completions.create(
                    model=simulator.model,
                    messages=[
                        {"role": "system", "content": simulator.system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.2,
                    max_tokens=1024,
                    top_p=0.95,
                    extra_body={"enable_thinking": False},
                    timeout=60
                )

                reply = completion.choices[0].message.content.strip()
                if not reply.startswith("医生回复"):
                    reply = "医生回复：" + reply

                with request_lock:
                    request_timestamps.append(time.time())
                    request_count += 1
                check_tpm_limit(completion.usage.total_tokens)
                total_tokens += completion.usage.total_tokens

                data["revised_doctor_reply"] = reply
                with write_lock, open(output_file, "a", encoding="utf-8") as fout:
                    fout.write(json.dumps(data, ensure_ascii=False) + "\n")

                print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Case {case_id} 完成 | QPM:{len(request_timestamps)} | Tokens:{total_tokens}")
                return reply

        except APIError as e:
            failed_requests += 1
            backoff = min(Config.RETRY_DELAY * (2 ** attempt), 30)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 限流/错误，第{attempt+1}次重试等待{backoff}s: {str(e)}")
            time.sleep(backoff)
        except Exception as e:
            failed_requests += 1
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Case {case_id} 失败: {str(e)}")
            time.sleep(Config.RETRY_DELAY)

    data["revised_doctor_reply"] = f"医生回复：生成失败"
    with write_lock, open(output_file, "a", encoding="utf-8") as fout:
        fout.write(json.dumps(data, ensure_ascii=False) + "\n")
    return None


class DoctorReplySimulator:
    def __init__(self, api_key, model, system_prompt):
        self.api_key = api_key
        self.model = model
        self.system_prompt = system_prompt


def print_stats():
    while not stop_event.is_set():
        elapsed = time.time() - start_time
        current_qpm = (request_count / elapsed) * 60 if elapsed > 0 else 0
        print(
            f"\n[Stats] Real-time QPM: {current_qpm:.1f} | "
            f"Total Requests: {request_count} | Failed: {failed_requests}\n"
        )
        stop_event.wait(10)


if __name__ == "__main__":
    input_file = "./results/ReMeDi_thoughtchain_generated_qwen3_80b.jsonl"
    output_file = "./results/ReMeDi_thoughtchain_refined_qwen3_80b.jsonl"

    system_prompt = (
        "你是一位医生，你需要根据对话的历史信息，来回答患者，具体来说，你需要做以下的思考：\n"
        "要求模型基于思维链摘要和原始医生回复，输出改写医生回复。该回复应该保留重要的信息和适合的关怀语气，不需要保留思维链中冗余的内容。"
        "开具病假单不要自己扩展具体天数。不要每句话结尾都一样：比如“祝您早日康复！”、“如有任何不适，请随时联系我。”、“祝您健康！”。"
        "不要说病情复杂。"
        "当患者没有提供足够信息时，可以用“如果xxx，则xxx”来引导患者。"
        "输出格式为:医生回复："
    )
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        logger.error("Error: DASHSCOPE_API_KEY environment variable is not set.")
        exit(1)
    simulator = DoctorReplySimulator(api_key=api_key, model="qwen3-next-80b-a3b-instruct", system_prompt=system_prompt)
    with open(input_file, "r", encoding="utf-8") as fin:
        cases = [json.loads(line) for line in fin if line.strip()]
    logger.info(f"Load {len(cases)} cases")
    processed_ids = set()
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        with open(output_file, "r", encoding="utf-8") as fout:
            for line in fout:
                try:
                    record = json.loads(line)
                    print(record["id"])
                    if "id" in record:
                        processed_ids.add(record["id"])
                except json.JSONDecodeError:
                    continue
        logger.info(f"Detected {len(processed_ids)} processed cases; they will be skipped")

    remaining_cases = [case for case in cases if case.get("id") not in processed_ids]
    logger.info(f"Remaining: {len(remaining_cases)}")

    if not remaining_cases:
        logger.info("Done.")
        exit(0)

    stop_event = threading.Event()
    threading.Thread(target=print_stats, daemon=True).start()

    with ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
        futures = {executor.submit(process_case, case, simulator, output_file): case for case in remaining_cases}

        for future in tqdm(as_completed(futures), total=len(futures), desc="并行生成医生回复"):
            try:
                _ = future.result(timeout=120)
            except Exception as e:
                print(f"[警告] 某线程超时或卡死: {e}")

    stop_event.set()
    logger.info(f"Done.")

