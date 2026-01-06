# -*- coding: utf-8 -*-
import json
import os
import copy
import re
import time
import collections
import threading
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import pandas as pd
import numpy as np
from tqdm import tqdm
from openai import OpenAI, APIError, APIConnectionError, APIStatusError

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('medical_thought_chain.log'),
        logging.StreamHandler()
    ]
)

# Initialize Logger
logger = logging.getLogger(__name__)

# Global Variables and Locks
token_lock = Lock()
request_lock = Lock()
rate_limit_semaphore = threading.Semaphore(10)
stop_event = threading.Event()
print_lock = threading.Lock()

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
    MAX_WORKERS = 8
    REQUEST_INTERVAL = 1.0
    MAX_QPM = 13500
    MAX_TPM = 1080000
    MAX_RETRIES = 5
    RETRY_DELAY = 2
    RATE_LIMIT_WINDOW = 60
    SAFETY_FACTOR = 0.9


# ========== Safe Print Function ==========
def safe_print(*args, **kwargs):
    with print_lock:
        print(*args, **kwargs)


def check_qpm_limit():
    current_time = time.time()
    with window_lock:
        while request_timestamps and current_time - request_timestamps[0] > Config.RATE_LIMIT_WINDOW:
            request_timestamps.popleft()

        if len(request_timestamps) >= Config.MAX_QPM * Config.SAFETY_FACTOR:
            oldest = request_timestamps[0]
            sleep_time = max(0, Config.RATE_LIMIT_WINDOW - (current_time - oldest)) + 0.5
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


def write_jsonl(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as file:
        for json_obj in data:
            json_line = json.dumps(json_obj, ensure_ascii=False)
            file.write(json_line + '\n')


def write_json(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                json_obj = json.loads(line.strip())
            except json.JSONDecodeError as e:
                print(line)
                print(e)
                continue
            data.append(json_obj)
    return data


class MedicalThoughtChainGenerator:
    def __init__(self, api_key: str, model: str = "qwen3-next-80b-a3b-instruct"):
        self.client = OpenAI(api_key=api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.model = model

        self.case_cache = {}
        self.action_library = self._load_action_library()

        self.embedding_cache = {}
        self.embedding_lock = Lock()
        self.embedding_similarity_threshold = 0.65

        logger.info(f"MedicalThoughtChainGenerator initialized with model: {model}")

    def _get_embedding(self, text: str) -> List[float]:
        if not text:
            return [0.0] * 256

        with self.embedding_lock:
            if text in self.embedding_cache:
                return self.embedding_cache[text]

        try:
            resp = self.client.embeddings.create(
                model="text-embedding-v4",
                input=[text],
                dimensions=256
            )
            embedding = resp.data[0].embedding

            with self.embedding_lock:
                self.embedding_cache[text] = embedding

            return embedding
        except Exception as e:
            logger.error(f"Failed to get embedding for text: {text[:20]}... Error: {e}")
            return [0.0] * 256

    def _calculate_cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return np.dot(v1, v2) / (norm1 * norm2)

    def _check_semantic_similarity_with_llm(self, action1: str, action2: str) -> bool:
        prompt = f"""请判断以下两个医疗对话中的原子动作是否表达了相同的核心语义（即是否可以合并为同一个动作）：

        动作A：{action1}
        动作B：{action2}

        只需回答"是"或"否"。
        """
        try:
            response = self._call_llm(prompt, max_tokens=10)
            return "是" in response or "Yes" in response or "yes" in response
        except Exception as e:
            logger.warning(f"LLM similarity check failed: {e}")
            return False

    def generate_thought_chains(self, cases: List[Dict]) -> List[Dict]:
        logger.info(f"Starting to process {len(cases)} cases")
        results = []

        for case in tqdm(cases, desc="Processing cases", unit="case"):
            try:
                result = self._process_case(case)
                results.append(result)
                logger.debug(f"Successfully processed case: {case.get('id', 'unknown')}")
            except Exception as e:
                logger.error(f"Error processing case {case.get('id', 'unknown')}: {str(e)}")
                continue

        logger.info(f"Completed processing {len(results)}/{len(cases)} cases successfully")
        return results

    def _load_action_library(self):
        try:
            with open("./library/action_library.json", "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def _save_action_library(self):
        with open("./library/action_library.json", "w", encoding="utf-8") as f:
            json.dump(self.action_library, f, indent=2, ensure_ascii=False)

    def _process_case(self, case: Dict) -> Dict:
        case_id = case.get("id", f"case_{hash(str(case))}")
        dialogue = case.get("dialogue", [])

        logger.info(f"开始处理病例 {case_id}，共 {len(dialogue)} 轮对话")

        # 初始化缓存
        self.case_cache[case_id] = {
            "dialogue": dialogue,
            "context": {},
        }

        output_results = {
            "case_id": case_id,
            "thought_chains": [],
        }

        last_doctor_turn_idx = max(
            (i for i, t in enumerate(dialogue) if t.startswith("医生：")),
            default=-1,
        )

        try:
            for turn_idx, turn in enumerate(dialogue):
                if getattr(self, "stop_event", None) and self.stop_event.is_set():
                    break

                if turn.startswith("医生："):
                    thought_chain = self._generate_doctor_thought_chain(
                        case_id,
                        turn[3:],
                        turn_idx,
                        is_last_doctor_turn=(turn_idx == last_doctor_turn_idx),
                        case_data=case,
                    )
                    output_results["thought_chains"].append(
                        {
                            "turn_index": turn_idx,
                            "doctor_reply": turn[3:],
                            **thought_chain,
                        }
                    )
        except Exception as e:
            logger.error(f"病例 {case_id} 处理出错: {e}", exc_info=True)
        finally:
            if case_id in self.case_cache:
                del self.case_cache[case_id]

        return output_results

    def _generate_doctor_thought_chain(self, case_id: str, doctor_reply: str, turn_idx: int,
                                       is_last_doctor_turn: bool = False, case_data: dict = None) -> Dict:
        # 获取对话历史
        history = self._get_recent_dialogue_history(case_id, turn_idx, limit=3)

        # 获取上一句患者的发言
        patient_input = ""
        full_dialogue = self.case_cache[case_id]["dialogue"]
        if turn_idx > 0:
            last_turn = full_dialogue[turn_idx - 1]
            if last_turn.startswith("患者："):
                patient_input = last_turn[3:]
            else:
                patient_input = last_turn

        # 生成标注
        annotation = self._generate_annotation(
            doctor_reply,
            history,
            case_data,
            doctor_turn_idx=turn_idx,
            patient_input=patient_input
        )

        results = {"structured_annotation": annotation}
        return results

    def _generate_action_definition(self, level1: str, action_name: str) -> dict:
        prompt = f"""你是一个医疗对话标注系统，请为以下原子动作生成描述。
                一级意图场景类型（level1）: {level1}
                原子动作（level3）: {action_name}
                请输出格式为 JSON：
                {{ "definition": "该原子动作的定义" }}"""
        try:
            response = self._call_llm(prompt)
            match = re.search(r'{.*}', response, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except Exception:
            pass
        return {"definition": f"{action_name} 的定义待补充。"}

    def _detect_intent(self, doctor_reply: str, history: list,
                       case_data: dict = None, doctor_turn_idx: int = None,
                       patient_input: str = "") -> dict:
        default_label = "无可参考患者意图"
        label_intent = ""
        try:
            if case_data and isinstance(case_data, dict) and "intent" in case_data:
                intents_all = case_data["intent"]
                if doctor_turn_idx is None:
                    chosen_intent = ""
                else:
                    idx = doctor_turn_idx - 1
                    while idx >= 0 and (idx % 2 == 1):
                        idx -= 1
                    if idx >= 0 and idx < len(intents_all):
                        intent_entry = intents_all[idx]
                        if isinstance(intent_entry, list) and intent_entry:
                            chosen_intent = intent_entry[0]
                        elif isinstance(intent_entry, str) and intent_entry.strip():
                            chosen_intent = intent_entry.strip()
                        else:
                            chosen_intent = ""
                    else:
                        chosen_intent = ""
                if chosen_intent:
                    label_intent = chosen_intent
                else:
                    label_intent = ""
        except Exception:
            label_intent = ""

        if not label_intent:
            label_intent = default_label

        level1_definitions = {
            "疾病诊断": "患者通过在线平台描述症状，希望医生做出判断",
            "治疗建议": "患者关心如何治疗，包含方案选择、用药指导等",
            "检查咨询": "患者希望通过检查支持诊断，包含必要性、结果解读等",
            "预后评估": "患者希望了解疾病长期影响和发展趋势",
            "健康指导": "患者希望得到生活方式、预防复发等指导"
        }

        level2_definitions = {
            "疾病诊断": ["疾病诊断需求", "症状解释需求", "严重程度评估"],
            "治疗建议": ["治疗方案选择", "用药指导需求", "治疗效果评估", "请求开药", "请求开病假单"],
            "检查咨询": ["检查结果解读", "检查项目必要性", "检查流程咨询", "检查方案选择", "请求开检查单"],
            "预后评估": ["疾病转归预测", "并发症风险评估", "功能恢复预期"],
            "健康指导": ["生活方式指导", "预防复发指导", "自我管理技能"]
        }

        allowed_level1 = set(level1_definitions.keys())
        level1_prompt = "\n".join([f"{k}：{v}" for k, v in level1_definitions.items()])
        level2_prompt = "\n".join([f"{k} -> {', '.join(v)}" for k, v in level2_definitions.items()])

        base_prompt = f"""
        你是医学问答系统中的标签助手，请根据当前患者的提问，结合上下文判断其核心意图类别。
        需要同时输出一级意图和二级子意图。

        要求：
        1. 一级意图必须严格从以下五个中选择（最多2个）：{list(allowed_level1)}。
        2. 二级意图必须从对应的子意图集合中选择（最多2个），不能越界。

        === 可参考的患者意图 ===
        {label_intent}

        === 一级意图场景定义 ===
        {level1_prompt}

        === 二级子意图列表 ===
        {level2_prompt}

        === 对话历史 ===
        {json.dumps(history, indent=2, ensure_ascii=False)}

        === 患者当前发言 ===
        {patient_input}

        输出格式示例：
        {{
        "intents": [
            {{"level1": "意图1", "level2": ["子意图1"]}},
            {{"level1": "意图2", "level2": ["子意图2"]}}
        ]
        }}
        """

        for _ in range(3):
            try:
                response = self._call_llm(base_prompt)
                match = re.search(r'{.*}', response, re.DOTALL)
                if match:
                    result = json.loads(match.group(0))
                    intents = []
                    for item in result.get("intents", []):
                        l1 = item.get("level1")
                        if l1 not in allowed_level1:
                            continue
                        intents.append(item)
                    if intents:
                        return {"intents": intents}
            except Exception:
                pass
        return {"intents": [{"level1": "其他", "level2": []}]}

    def _generate_filtered_action_prompt(self, intent_result) -> str:
        if isinstance(intent_result, dict):
            level_list = intent_result.get("level1", [])
            if not level_list and 'intents' in intent_result:
                level_list = [i.get('level1') for i in intent_result['intents']]
        elif isinstance(intent_result, list):
            level_list = intent_result
        else:
            level_list = []

        filtered = {}
        for level1 in level_list:
            if level1 in self.action_library:
                actions = self.action_library[level1]
                if isinstance(actions, dict):
                    cleaned = {k: v for k, v in actions.items() if k != "患者子意图"}
                    filtered[level1] = cleaned
                else:
                    filtered[level1] = actions
        return json.dumps(filtered, ensure_ascii=False, indent=2)

    def _generate_annotation(
            self,
            doctor_reply: str,
            history: List[str],
            case_data: dict = None,
            doctor_turn_idx: Optional[int] = None,
            patient_input: str = ""
    ) -> dict:
        # Step 1: 识别一级意图
        level_list = self._detect_intent(
            doctor_reply,
            history,
            case_data=case_data,
            doctor_turn_idx=doctor_turn_idx,
            patient_input=patient_input
        )
        intents = level_list.get("intents", [])

        # Step 2: 生成 Action Prompt
        action_prompt = self._generate_filtered_action_prompt(level_list)

        # 公共原子动作
        common_actions = {
            "采集病史": {
                "definition": "病史采集主要包括以下几个方面的内容："
                              "1. 主诉：指患者就诊的主要原因或最突出的症状及其持续时间。"
                              "2. 现病史：详细记录患者此次发病的时间、起因、症状的发展变化情况，以及已经采取的诊断和治疗措施等信息。"
                              "3. 既往史：包括患者的过去健康状况、曾经患过的各种疾病、手术外伤经历、预防接种记录及药物过敏等情况。"
                              "4. 家族史：询问患者家族中是否有遗传性疾病或相似病症的发生情况。"
                              "5. 个人史：涉及患者的生活习惯、职业环境、婚姻生育状况等因素，这些都可能与某些疾病的发生有关联。"
                              "6. 系统回顾：对身体各系统进行全面的询问，以发现潜在的问题。 "
            },
            "解读诊断性检查": {
                "definition": "内容应包括："
                              "- 结果解释：对信号/曲线的定性或定量结论；功能状态或异常模式的临床解释；说明关键检查结果（如心电图、肺活量测定）的意义，分析其对患者健康状况的暗示。"
                              "- 诊断意义：将检查结果与潜在疾病关联，说明其如何支持或排除某些诊断。 "
            },
            "解读实验室检查结果": {
                "definition": "根据实验室提供的数值型检测结果， 将具体数值与参考区间的比较，判断是否异常、异常程度，与临床相关的解释或提示。"
            },
            "检测影像学发现": {
                "definition": "根据影像所见的结构/形态学描述，描述异常影像特征的定位与性质，初步诊断或鉴别诊断提示。"
            }
        }

        if "公共原子动作" not in self.action_library:
            self.action_library["公共原子动作"] = {}
        for k, v in common_actions.items():
            if k not in self.action_library["公共原子动作"]:
                self.action_library["公共原子动作"][k] = v

        try:
            prompt_dict = json.loads(action_prompt)
        except Exception:
            prompt_dict = {}
        prompt_dict["公共原子动作"] = self.action_library["公共原子动作"]

        flattened = {}
        for level1, actions in prompt_dict.items():
            if isinstance(actions, dict):
                for k, v in actions.items():
                    flattened[k] = v
        action_prompt_str = json.dumps(flattened, ensure_ascii=False, indent=2)

        level1_list = [item["level1"] for item in intents]
        level2_list = [sub for item in intents for sub in item["level2"]]
        level1_str = json.dumps(level1_list, ensure_ascii=False, indent=2)
        level2_str = json.dumps(level2_list, ensure_ascii=False, indent=2)

        prompt = f"""医疗思维链标注任务：
        请为医生的回复进行标注：

        === 当前患者发言 ===
        {patient_input}

        === 相关对话历史 ===
        {json.dumps(history, indent=2, ensure_ascii=False)}

        === 医生回复 ===
        {doctor_reply}

        请执行以下任务：
        1. 思维动作分解：请确保遍历所有识别出的一级意图场景和二级子意图，识别场景下的所有原子医疗动作，可以生成新的原子动作，不一定必须从现有的里面去选择。

         === 识别出的一级意图场景level1 ===
        {level1_str}
        === 识别出的二级患者子意图level2 ===
        {level2_str}
        === 请根据原子动作的定义在以下原子动作进行选择 ===
        {action_prompt_str}

        **重要约束：**
        - 每个原子动作必须是一个简短的动词短语，不得超过8个汉字；
        - 禁止生成句子或解释性描述；
        - 若动作超过8个字，请自动压缩为更简短的核心动词表达；
        - 例如：“建议患者去医院进一步检查” → “建议就诊”。

        输出格式：
        {{
             "thought_chain": [
                {{"level1": "", "level2": "","level3": ["原子动作1", "原子动作2", "原子动作3"]}}
            ]
        }}"""
        response = self._call_llm(prompt)
        annotation = self._parse_annotation_response(response)

        # === Auto-update Atomic Action Library ===
        thought_chain = annotation.get("thought_chain", [])
        new_action_added = False

        candidate_actions = []
        if "公共原子动作" in self.action_library:
            candidate_actions.extend(list(self.action_library["公共原子动作"].keys()))

        valid_level1s = {"疾病诊断", "治疗建议", "检查咨询", "预后评估", "健康指导"}
        relevant_level1s = set()
        for step in thought_chain:
            l1 = step.get("level1")
            if isinstance(l1, list) and l1: l1 = l1[0]
            if l1 in valid_level1s:
                relevant_level1s.add(l1)

        for l1 in relevant_level1s:
            if l1 not in self.action_library:
                self.action_library[l1] = {}
            candidate_actions.extend([k for k in self.action_library[l1].keys() if k != "患者子意图"])

        candidate_embeddings = {}
        for act in candidate_actions:
            candidate_embeddings[act] = self._get_embedding(act)

        for step in thought_chain:
            level1 = step.get("level1")
            if isinstance(level1, list):
                level1 = level1[0] if level1 else None

            if level1 not in valid_level1s:
                continue

            actions = step.get("level3", [])
            if not isinstance(actions, list): actions = []

            for idx, original_action in enumerate(list(actions)):
                action_name = original_action.strip()
                if not action_name: continue

                if action_name in candidate_embeddings:
                    continue

                new_emb = self._get_embedding(action_name)
                best_match_action = None
                best_match_score = -1.0

                current_scope_candidates = list(self.action_library.get("公共原子动作", {}).keys())
                if level1 in self.action_library:
                    current_scope_candidates.extend([k for k in self.action_library[level1].keys() if k != "患者子意图"])

                current_scope_candidates = list(set(current_scope_candidates))

                for cand in current_scope_candidates:
                    cand_emb = candidate_embeddings.get(cand)
                    if cand_emb is None:
                        cand_emb = self._get_embedding(cand)

                    score = self._calculate_cosine_similarity(new_emb, cand_emb)
                    if score > best_match_score:
                        best_match_score = score
                        best_match_action = cand

                final_action = action_name

                if best_match_action:
                    if best_match_score > 0.95:
                        final_action = best_match_action
                        logger.info(f"Action auto-normalized (High Sim {best_match_score:.2f}): {action_name} -> {best_match_action}")
                    elif best_match_score > self.embedding_similarity_threshold:
                        is_same = self._check_semantic_similarity_with_llm(action_name, best_match_action)
                        if is_same:
                            final_action = best_match_action
                            logger.info(
                                f"Action auto-normalized (LLM Confirmed, Sim {best_match_score:.2f}): {action_name} -> {best_match_action}")
                        else:
                            logger.info(
                                f"Action confirmed as new (LLM Denied, Sim {best_match_score:.2f}): {action_name} != {best_match_action}")
                    else:
                        logger.info(f"Action confirmed as new (Low Sim {best_match_score:.2f}): {action_name}")

                if final_action == action_name and action_name not in candidate_embeddings:
                    logger.info(f"New atomic action added to library: {level1} -> {action_name}")
                    definition = self._generate_action_definition(level1, action_name)
                    if level1 not in self.action_library: self.action_library[level1] = {}
                    self.action_library[level1][action_name] = definition

                    candidate_embeddings[action_name] = new_emb
                    new_action_added = True

                actions[idx] = final_action

        if new_action_added:
            self._save_action_library()

        annotation["_new_action_added"] = new_action_added
        return annotation

    def _get_recent_dialogue_history(
            self, case_id: str, current_idx: int, speaker: Optional[str] = None, limit: int = 3
    ) -> List[str]:
        full_dialogue = self.case_cache[case_id]["dialogue"][:current_idx]
        if speaker:
            return [t for t in full_dialogue if t.startswith(speaker)][-limit:]
        return full_dialogue[-limit:]

    def _call_llm(self, prompt: str, max_tokens: int = 1024, max_retries: int = 3) -> str:
        retry_count = 0
        while retry_count <= max_retries:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=max_tokens,
                    top_p=0.95,
                    extra_body={"enable_thinking": False}
                )
                return response.choices[0].message.content.strip()

            except APIError as e:
                status_code = getattr(e, 'status_code', None)
                logger.error(f"[API Error] Request Failed. Status: {status_code}")
                logger.error(f"Error Message: {e.message}")

                if status_code == 400:
                    logger.error("Stop retrying due to 400 Bad Request.")
                    return f"API Error: 400 - {e.message}"

                if isinstance(e, APIConnectionError):
                    logger.warning("Network Connection Error. Will retry...")

                if retry_count < max_retries:
                    sleep_time = 2 ** retry_count
                    logger.warning(f"Retrying in {sleep_time}s...")
                    time.sleep(sleep_time)
                    retry_count += 1
                else:
                    logger.error("Max retries reached.")
                    return "API Error"

            except Exception as e:
                logger.error(f"Unknown Exception: {str(e)}")
                if retry_count < max_retries:
                    time.sleep(2 ** retry_count)
                    retry_count += 1
                else:
                    return f"API Error"

        return "API Call Failed"

    def _parse_annotation_response(self, response: str) -> Dict:
        try:
            if response.startswith("{") and response.endswith("}"):
                return json.loads(response)
            for wrapper in ["```json", "```"]:
                if wrapper in response:
                    json_str = response.split(wrapper)[1].split("```")[0]
                    return json.loads(json_str)
            return {}
        except Exception as e:
            return {"error": f"Parsing Exception: {str(e)}"}


def print_stats():
    while True:
        elapsed = time.time() - start_time
        current_qpm = (request_count / elapsed) * 60 if elapsed > 0 else 0
        print(
            f"\n[Stats] Real-time QPM: {current_qpm:.1f} | "
            f"Total Requests: {request_count} | Failed: {failed_requests}\n"
        )
        time.sleep(10)


def load_processed_ids(output_path):
    processed_ids = set()
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        for k in ['case_id', 'caseId', 'id']:
                            if k in obj:
                                processed_ids.add(str(obj[k]))
                                break
                except json.JSONDecodeError:
                    continue
    return processed_ids


def handle_case_result(case, future, output_path):
    global request_count, failed_requests
    try:
        if not check_qpm_limit():
            return

        result = future.result()
        if result:
            with write_lock:
                with open(output_path, "a", encoding="utf-8") as f:
                    if isinstance(result, dict):
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    elif isinstance(result, list):
                        for r in result:
                            f.write(json.dumps(r, ensure_ascii=False) + "\n")

            with request_lock:
                request_timestamps.append(time.time())
                request_count += 1

            logger.info(f"Completed Case: {case.get('id', 'unknown')}")
        else:
            logger.warning(f"Empty Result: {case.get('id', 'unknown')}")

    except Exception as e:
        logger.error(f"Processing Failed {case.get('id', 'unknown')}: {e}")
        failed_requests += 1


if __name__ == "__main__":
    try:
        logger.info("Starting generation process")

        raw_data_path = './data/ReMeDi-large-0-converted.jsonl'  
        output_path = "./results/medical_thought_chains_ReMeDi_qwen3_80b.jsonl"

        raw_data = read_jsonl(raw_data_path)
        logger.info(f"Loaded {len(raw_data)} records")

        processed_ids = load_processed_ids(output_path)
        logger.info(f"Processed samples: {len(processed_ids)}")

        pending_cases = [case for case in raw_data if str(case.get('id', case.get('case_id'))) not in processed_ids]
        logger.info(f"Remaining cases to process: {len(pending_cases)}")

        stats_thread = threading.Thread(target=print_stats, daemon=True)
        stats_thread.start()

        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            logger.error("Error: DASHSCOPE_API_KEY environment variable is not set.")
            exit(1)
        generator = MedicalThoughtChainGenerator(api_key=api_key, model="qwen3-next-80b-a3b-instruct")

        with ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
            futures = {executor.submit(generator._process_case, case): case for case in pending_cases}
            for future in as_completed(futures):
                case = futures[future]
                handle_case_result(case, future, output_path)
    except KeyboardInterrupt:
        logger.warning("User interrupted, saving in progress")
    finally:
        logger.info("Success")

