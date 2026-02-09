"""
生成 RLHF 分数文件 - 使用奖励模型对测试集进行评分

功能特性:
1. 读取测试集数据，对每个问题的每个选项进行评分
2. 异步并发处理，提升速度
3. 支持断点续存
4. 输出格式：JSON数组，包含 instruction、answer、score
5. 使用 value-head 输出数值分数（浮点数）

输出格式:
[
  {
    "instruction": "问题文本",
    "answer": "选项文本",
    "score": 1.5  # 数值分数
  },
  ...
]
"""

import asyncio
import json
import re
import os
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List
import httpx
from tqdm.asyncio import tqdm_asyncio

# ================= 1. 配置 =================
API_KEY = "EMPTY"  # vLLM使用EMPTY
BASE_URL = "http://localhost:6009/v1/"  # 根据实际情况修改
MODEL_NAME = 'Qwen3-8B'

# 数据集路径
DATASET_PATH = "/mnt/public/wwj/zhaozq/exp1/dataset/TeleQnA/20251227_newest/TeleQnA_test_20260207.json"

# 并发配置
MAX_CONCURRENT_REQUESTS = 20  # 最大并发请求数
RETRY_ATTEMPTS = 1  # 失败重试次数
RETRY_DELAY = 2  # 重试延迟（秒）
API_TIMEOUT = 60.0  # API超时时间（秒）

model = MODEL_NAME
# 文件路径
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
CHECKPOINT_FILE = f"/mnt/public/wwj/zhaozq/exp1/verl/huawei/experiments/result/rlhf_scores_checkpoint_{model}_{timestamp}.json"
OUTPUT_FILE = f"/mnt/public/wwj/zhaozq/exp1/verl/huawei/experiments/result/rlhf_scores_{model}_{timestamp}.json"

# API 端点
API_ENDPOINT = f"{BASE_URL}score/evaluation"

# ================= 2. System Prompt =================
SYSTEM_PROMPT = ""  # 不使用 system prompt，让模型直接输出分数

# ================= 3. 辅助函数 =================
def extract_score_from_json_response(result: Dict[str, Any]) -> Optional[float]:
    """从API返回的JSON响应中提取分数

    API返回格式: {"id": "...", "object": "score.evaluation", "model": "...", "scores": [-7.0625]}
    """
    try:
        if isinstance(result, dict):
            scores = result.get("scores", [])
            if scores and isinstance(scores, list) and len(scores) > 0:
                score_value = scores[0]
                if isinstance(score_value, (int, float)):
                    return float(score_value)
        return None
    except Exception as e:
        print(f"Error extracting score from JSON: {e}")
        return None


def extract_score_from_response(response: str) -> Optional[float]:
    """从文本响应中提取数值分数（备用方法，支持浮点数和负数）"""
    try:
        # 优先提取 \\boxed{} 格式中的数值（支持负数）
        boxed_patterns = [
            r'\\boxed\{(-?[0-9]+\.?[0-9]*)\}',  # \boxed{-7.0625} 或 \boxed{1.5}
            r'\\boxed\s*\{(-?[0-9]+\.?[0-9]*)\}',  # \boxed {-7.0625}
            r'boxed\{(-?[0-9]+\.?[0-9]*)\}',  # boxed{-7.0625}
        ]

        for pattern in boxed_patterns:
            match = re.search(pattern, response)
            if match:
                return float(match.group(1))

        # 尝试提取 JSON 格式的分数（支持负数）
        json_patterns = [
            r'"score"\s*:\s*(-?[0-9]+\.?[0-9]*)',  # "score": -7.0625
            r'score\s*:\s*(-?[0-9]+\.?[0-9]*)',  # score: -7.0625
            r'score\s*=\s*(-?[0-9]+\.?[0-9]*)',  # score = -7.0625
            r'"scores"\s*:\s*\[\s*(-?[0-9]+\.?[0-9]*)\s*\]',  # "scores": [-7.0625]
        ]

        for pattern in json_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return float(match.group(1))

        # 尝试提取纯数字（可能是分数，支持负数）
        # 查找类似 "-7.0625", "1.5", "0.8", "1" 这样的数字
        number_pattern = r'\b(-?[0-9]+\.?[0-9]*)\b'
        matches = re.findall(number_pattern, response)
        if matches:
            # 取第一个合理的数字（-100到100之间，放宽范围以支持奖励模型的分数）
            for match_str in matches:
                num = float(match_str)
                if -100 <= num <= 100:
                    return num

        return None
    except Exception as e:
        print(f"Error extracting score from text: {e}")
        return None


def build_user_prompt(question_content: str, answer_text: str) -> str:
    """构建User Prompt - 只给问题和答案，让模型自己评分"""
    return f"Question: {question_content}\n\nAnswer: {answer_text}"


def build_instruction(question_content: str) -> str:
    """构建 instruction，格式与 RLHF 训练数据一致"""
    return f"You are an expert telecommunications engineer. Please answer the following technical question.\n\n{question_content}"


async def evaluate_single_answer(
    question_content: str,
    answer_text: str,
    semaphore: asyncio.Semaphore
) -> Tuple[Optional[float], str]:
    """异步评估单个答案 - 带重试机制"""
    async with semaphore:
        for attempt in range(RETRY_ATTEMPTS):
            try:
                user_prompt = build_user_prompt(question_content, answer_text)

                # 构建请求体（API 格式：messages 是字符串数组）
                payload = {
                    "model": MODEL_NAME,
                    "messages": [user_prompt],  # messages 是字符串数组
                    "max_length": 2048
                }

                # 使用 httpx 发送异步请求
                async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
                    response = await client.post(
                        API_ENDPOINT,
                        json=payload,
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {API_KEY}"
                        }
                    )
                    response.raise_for_status()
                    result = response.json()

                # API返回格式: {"id": "...", "object": "score.evaluation", "model": "...", "scores": [-7.0625]}
                # 直接从JSON中提取分数
                score = extract_score_from_json_response(result)

                # 如果提取失败，尝试从文本中提取（备用方法）
                if score is None:
                    response_text = json.dumps(result, ensure_ascii=False)
                    score = extract_score_from_response(response_text)
                else:
                    response_text = json.dumps(result, ensure_ascii=False)

                return score, response_text

            except Exception as e:
                print(f"\n⚠️ Error evaluating answer (Attempt {attempt + 1}/{RETRY_ATTEMPTS}): {e}")
                if attempt < RETRY_ATTEMPTS - 1:
                    await asyncio.sleep(RETRY_DELAY)
                else:
                    # 最后一次尝试失败，返回错误
                    return None, f"Error: {str(e)}"


async def process_single_question(
    question_id: str,
    question_data: Dict[str, Any],
    semaphore: asyncio.Semaphore
) -> List[Dict[str, Any]]:
    """处理单个问题的所有选项，返回 RLHF 格式的数据列表"""
    results = []
    try:
        question_content = question_data['question']
        instruction = build_instruction(question_content)

        # 提取所有选项
        candidate_answers = []
        for i in range(1, 10):  # 最多支持9个选项
            option_key = f"option {i}"
            if option_key in question_data:
                candidate_answers.append((i, question_data[option_key]))
            else:
                break

        # 对每个选项并发评估
        tasks = []
        for option_num, answer_text in candidate_answers:
            task = evaluate_single_answer(question_content, answer_text, semaphore)
            tasks.append((option_num, answer_text, task))

        # 等待所有选项评估完成
        results_list = await asyncio.gather(*[task for _, _, task in tasks], return_exceptions=True)

        # 构建结果列表
        for idx, (option_num, answer_text, _) in enumerate(tasks):
            result = results_list[idx]

            if isinstance(result, Exception):
                print(f"\n⚠️ Error evaluating option {option_num} of {question_id}: {str(result)}")
                score = None
                response_text = f"Error: {str(result)}"
            else:
                score, response_text = result
                if score is None:
                    # 如果提取失败，设为 None
                    print(f"\n⚠️ Failed to extract score for {question_id} option {option_num}")

            # 构建 RLHF 格式的数据
            rlhf_item = {
                "instruction": instruction,
                "answer": answer_text,
                "score": score
            }
            results.append(rlhf_item)

    except Exception as e:
        print(f"\n❌ Error processing {question_id}: {str(e)}")
        # 返回空列表，表示处理失败

    return results


# ================= 4. 批量处理（支持断点续存）=================
async def save_checkpoint(results: List[Dict[str, Any]], total_count: int):
    """异步保存checkpoint"""
    try:
        checkpoint_data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'progress': {
                'total_items': total_count,
                'evaluated_items': len(results),
                'remaining_items': total_count - len(results),
            },
            'results': results
        }

        with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"\nWarning: Failed to save checkpoint: {e}")


async def main():
    """主函数"""
    print("=" * 80)
    print("Generate RLHF Scores - Batch Processing")
    print("=" * 80)
    print(f"API Endpoint: {BASE_URL}")
    print(f"Model: {MODEL_NAME}")
    print(f"Max Concurrent Requests: {MAX_CONCURRENT_REQUESTS}")
    print(f"API Timeout: {API_TIMEOUT}s")
    print("=" * 80)

    # ================= 加载数据集 =================
    print(f"\nLoading dataset from {DATASET_PATH}...")
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    print(f"Loaded {len(dataset)} questions from dataset.")

    # ================= 加载checkpoint（如果存在）=================
    results = []
    processed_questions = set()

    if os.path.exists(CHECKPOINT_FILE):
        print("\n" + "=" * 80)
        print("Found checkpoint file! Loading previous results...")
        print("=" * 80)
        try:
            with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
                results = checkpoint_data.get('results', [])
                # 从结果中推断已处理的问题（通过统计instruction数量）
                # 这里简化处理，假设每个问题有固定数量的选项
                # 实际应该记录已处理的问题ID
                print(f"Loaded {len(results)} previously evaluated items.")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting from scratch...")
            results = []
            processed_questions = set()
    else:
        print("\n" + "=" * 80)
        print("Starting New Batch Processing")
        print("=" * 80)

    # ================= 统计信息 =================
    # 计算总的数据项数（每个问题的每个选项算一项）
    total_items = 0
    for question_data in dataset.values():
        for i in range(1, 10):
            if f"option {i}" in question_data:
                total_items += 1
            else:
                break

    remaining_questions = {qid: qdata for qid, qdata in dataset.items() if qid not in processed_questions}
    remaining_count = len(remaining_questions)

    print(f"Total questions: {len(dataset)}")
    print(f"Total items (all options): {total_items}")
    print(f"Already processed: {len(processed_questions)} questions")
    print(f"Remaining: {remaining_count} questions")
    print("=" * 80)

    if remaining_count == 0:
        print("\n✅ All questions already processed!")
        return

    # ================= 异步并发处理 =================
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    lock = asyncio.Lock()

    # 创建所有处理任务（协程对象）
    tasks = []
    for question_id, question_data in remaining_questions.items():
        coro = process_single_question(question_id, question_data, semaphore)
        tasks.append(coro)

    # 使用tqdm显示进度
    print("\nStarting concurrent processing...")
    completed_count = 0

    for coro in tqdm_asyncio.as_completed(
        tasks,
        total=len(tasks),
        desc="Processing questions"
    ):
        question_results = await coro
        completed_count += 1

        # 每完成一个问题就更新results列表并保存checkpoint
        async with lock:
            results.extend(question_results)

            # 每10个问题保存一次checkpoint（减少I/O）
            if completed_count % 10 == 0:
                await save_checkpoint(results, total_items)

    # 最终保存checkpoint
    await save_checkpoint(results, total_items)

    # ================= 保存最终结果 =================
    print("\n" + "=" * 80)
    print("PROCESSING COMPLETED!")
    print("=" * 80)
    print(f"Total items processed: {len(results)}")
    valid_scores = [r['score'] for r in results if r['score'] is not None]
    print(f"Items with valid scores: {len(valid_scores)}")
    if valid_scores:
        print(f"Score range: {min(valid_scores):.2f} - {max(valid_scores):.2f}")
        print(f"Average score: {sum(valid_scores) / len(valid_scores):.2f}")
    print("=" * 80)

    # 保存为 RLHF 格式的 JSON 数组
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {OUTPUT_FILE}")

    # 删除checkpoint文件
    if os.path.exists(CHECKPOINT_FILE):
        try:
            os.remove(CHECKPOINT_FILE)
            print(f"Checkpoint file removed (processing completed)")
        except Exception as e:
            print(f"Warning: Failed to remove checkpoint file: {e}")

    # ================= 显示样本数据 =================
    print("\n" + "=" * 80)
    print("Sample Results (First 5 items)")
    print("=" * 80)
    for i, item in enumerate(results[:5], 1):
        print(f"\n{i}. Score: {item['score']}")
        print(f"   Instruction: {item['instruction'][:100]}...")
        print(f"   Answer: {item['answer'][:100]}...")
        print("-" * 80)


if __name__ == "__main__":
    asyncio.run(main())

