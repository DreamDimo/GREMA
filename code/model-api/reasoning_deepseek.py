import json
import os
import asyncio
import aiohttp
from collections import defaultdict
import time
from typing import Dict, Any, List, Tuple

# ================= 配置区域 =================
# API 配置
# API_KEY = "AIzaSyA3e1ICK-De_o9RQUJ5BR1IJxmwaKLMhCA"
# BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"  # 根据实际情况修改
# MODEL_NAME = 'gemini-3-flash-preview'
API_KEY = "yhzy9VtrPip__J4S_ySzZP7-LCMbjYji1WPjki_c_BYOd3en1-a7nN7BMSSjT-Py8-jYNYjEWOsHKiaUuFqptA"
BASE_URL = "https://api.modelarts-maas.com/v2/chat/completions"  # 根据实际情况修改
MODEL_NAME = 'deepseek-v3.2'

# 文件路径
DATA_FILE = "E:/project/verl/huawei/experiments/result/Medical_train_with_options.json"
OUTPUT_FILE = "E:/project/verl/huawei/experiments/result/Medical_train_with_ds_reason_deepseek.json"

# 并发配置
# MAX_CONCURRENT_REQUESTS: 同时发送的请求数量
#
# 当前vllm服务器配置（H200双卡）：
#   - GPU: 2x H200 (141GB显存/卡)
#   - 模型: Qwen3-Next-80B-A3B-Instruct (80B参数)
#   - tensor_parallel_size: 2
#   - gpu_memory_utilization: 0.7
#   - max_model_len: 8192
#
# 针对H200双卡+80B模型的建议：
#   - 保守值: 15-20（稳定，适合长时间运行）
#   - 推荐值: 20-30（充分利用GPU资源）
#   - 激进值: 30-40（需要监控OOM和响应时间）
#
# 调整策略：
#   1. 从20开始，观察GPU利用率和响应时间
#   2. 如果GPU利用率<80%，可以逐步增加到25-30
#   3. 如果出现OOM错误，降低到15-18
#   4. 如果响应时间>10秒/请求，考虑降低并发数
MAX_CONCURRENT_REQUESTS = 10  # H200双卡推荐起始值，可根据实际情况调整

# BATCH_SIZE: 每处理多少条数据显示一次进度更新
BATCH_SIZE = 50  # 每处理多少条显示一次进度


# ================= 核心函数 =================

async def call_api(messages: List[Dict[str, str]], temperature: float = 0.1, max_tokens: int = 1200) -> str:
    """
    使用 aiohttp 调用 ModelArts MAAS API
    格式类似于 requests，但保持异步性能
    """
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {API_KEY}'
    }
    data = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            BASE_URL, 
            headers=headers, 
            json=data,
            ssl=False  # 对应 requests 的 verify=False
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"API request failed with status {response.status}: {error_text}")
            
            result = await response.json()
            return result['choices'][0]['message']['content']


def load_data(filepath: str) -> Dict[str, Any]:
    """加载训练数据"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def save_data(data: Dict[str, Any], filepath: str):
    """保存数据到文件"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


async def save_data_safe(data: Dict[str, Any], filepath: str, lock: asyncio.Lock):
    """
    线程安全地保存数据到文件
    使用锁保护文件写入操作
    """
    async with lock:
        # 深拷贝数据，避免在写入过程中数据被修改
        data_copy = json.loads(json.dumps(data))
        # 使用临时文件，然后原子性地替换
        temp_filepath = filepath + ".tmp"
        with open(temp_filepath, 'w', encoding='utf-8') as f:
            json.dump(data_copy, f, indent=4, ensure_ascii=False)
        # 原子性替换
        os.replace(temp_filepath, filepath)


def extract_answer_option(item: Dict[str, Any]) -> Tuple[str, int]:
    """
    提取正确答案的选项内容和索引
    返回: (答案文本, 选项索引 1-based)
    """
    answer_key = item.get('answer', '')
    if not answer_key.startswith('option '):
        return None, -1

    option_num = int(answer_key.replace('option ', ''))
    answer_text = item.get(answer_key, '')
    return answer_text, option_num


def build_initial_prompt(question: str, option_text: str) -> Tuple[str, str]:
    """
    第一阶段提示词：让模型自己判断答案是否正确
    不告诉模型正确答案，让它自主评分
    返回: (system_prompt, user_prompt)
    """
    system_prompt = """You are an expert telecommunications engineer. You will evaluate whether an answer to a technical question is correct or incorrect. Provide clear reasoning for your evaluation, then output a score in JSON format.

Score Meaning:
- Score 1: The answer is CORRECT and fully addresses the question
- Score 0: The answer is INCORRECT or does not properly address the question"""

    user_prompt = f"""#### Question
{question}

#### Answer to Evaluate
{option_text}

#### Evaluation Task
Please evaluate whether this answer is correct or incorrect.

**Your Analysis:**
1. Analyze what the question is asking for
2. Evaluate the technical accuracy of the answer
3. Check if the answer properly addresses the question
4. Determine if this answer is correct or incorrect

**Requirements:**
- Be precise and direct - every sentence must add value
- Focus on technical accuracy and relevance
- Use clear, step-by-step reasoning
- Provide your honest evaluation

**Output Format:**
First provide your reasoning analysis, then end with your score in JSON format:
```json
{{
  "score": <0 or 1>
}}
```

Where:
- **1** means the answer is Correct (factually accurate and properly addresses the question)
- **0** means the answer is Incorrect (contains errors, is factually wrong, or doesn't address the question)"""

    return system_prompt, user_prompt


def build_correction_prompt(question: str, option_text: str, wrong_reasoning: str, wrong_score: int, correct_score: int) -> str:
    """
    第二阶段提示词：纠正模型的错误判断
    告诉模型它的判断是错误的，让它重新生成正确的推理
    返回: correction_prompt (作为user消息)
    """
    correction_prompt = f"""Your previous evaluation was **INCORRECT**.

**Correct Evaluation:**
The correct score for this answer should be **{correct_score}**, not {wrong_score}.

**Your Previous (Incorrect) Reasoning:**
{wrong_reasoning}

**Task:**
Now that you know the correct score is **{correct_score}**, please provide a NEW reasoning analysis that correctly explains why this answer deserves a score of {correct_score}.

**Requirements:**
1. Re-analyze the question and answer with the correct perspective
2. Identify what you missed or misunderstood in your previous evaluation
3. Provide clear, accurate reasoning for why the score should be {correct_score}
4. Be precise and direct - every sentence must add value

**Output Format:**
First provide your corrected reasoning analysis, then end with the correct score in JSON format:
```json
{{
  "score": {correct_score}
}}
```

Remember:
- **1** means the answer is Correct (factually accurate)
- **0** means the answer is Incorrect (contains errors or is factually wrong)"""

    return correction_prompt


def extract_score_from_response(response_text: str) -> int:
    """
    从模型响应中提取分数
    返回: 1 或 0，如果提取失败返回 -1
    """
    import re
    # 尝试从JSON格式中提取
    json_match = re.search(r'"score"\s*:\s*(\d+)', response_text)
    if json_match:
        return int(json_match.group(1))
    
    # 尝试其他格式
    score_match = re.search(r'score[:\s]+(\d+)', response_text, re.IGNORECASE)
    if score_match:
        return int(score_match.group(1))
    
    return -1


async def generate_reasoning_for_option(
        question: str,
        option_key: str,
        option_text: str,
        correct_score: int,
        semaphore: asyncio.Semaphore
) -> Tuple[str, str, int, bool]:
    """
    两阶段生成打分推理轨迹：
    1. 第一阶段：让模型自己判断并打分
    2. 第二阶段：如果打分错误，纠正并重新生成
    
    返回: (option_key, reasoning_text, score, was_corrected)
    - was_corrected: True表示经过纠正，False表示第一次就正确
    """
    async with semaphore:
        try:
            # ===== 第一阶段：让模型自己判断 =====
            system_prompt, user_prompt = build_initial_prompt(question, option_text)

            # 调用API - 第一次评估
            first_reasoning = await call_api(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=1200,
            )
            first_reasoning = first_reasoning.strip()
            predicted_score = extract_score_from_response(first_reasoning)

            # 检查是否成功提取分数
            if predicted_score == -1:
                print(f"  Warning: Could not extract score from response for {option_key}")
                return option_key, first_reasoning, correct_score, False

            # ===== 检查是否需要纠正 =====
            if predicted_score == correct_score:
                # 第一次就正确，直接返回
                return option_key, first_reasoning, correct_score, False
            
            # ===== 第二阶段：模型判断错误，进行纠正 =====
            print(f"  Correcting {option_key}: predicted {predicted_score}, correct {correct_score}")
            
            correction_prompt = build_correction_prompt(
                question, option_text, first_reasoning, predicted_score, correct_score
            )

            # 调用API - 纠正评估（多轮对话）
            corrected_reasoning = await call_api(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": first_reasoning},
                    {"role": "user", "content": correction_prompt},
                ],
                temperature=0.1,
                max_tokens=1200,
            )
            corrected_reasoning = corrected_reasoning.strip()
            
            # 验证纠正后的分数
            corrected_score = extract_score_from_response(corrected_reasoning)
            if corrected_score != correct_score:
                print(f"  Warning: Correction failed for {option_key}, score still wrong")
            
            return option_key, corrected_reasoning, correct_score, True

        except Exception as e:
            print(f"  Error generating reasoning for {option_key}: {e}")
            return option_key, None, correct_score, False


async def generate_reasoning(item: Dict[str, Any], question_key: str, semaphore: asyncio.Semaphore) -> Tuple[
    str, Dict[str, Dict[str, Any]], Any]:
    """
    为单个问题的所有选项生成打分推理轨迹
    返回: (question_key, {option_key: {"reasoning": ..., "score": ..., "option_text": ..., "was_corrected": ...}, ...}, error_info)
    """
    try:
        # 提取问题和答案
        question = item.get('question', '')
        correct_answer_key = item.get('answer', '')

        if not correct_answer_key.startswith('option '):
            return question_key, {}, "Invalid answer format"

        # 提取所有选项
        choices = {}
        for i in range(1, 6):
            key = f"option {i}"
            if key in item:
                choices[key] = item[key]

        # 为每个选项生成打分推理轨迹（正确答案1分，错误答案0分）
        reasoning_tasks = []
        for option_key, option_text in choices.items():
            correct_score = 1 if (option_key == correct_answer_key) else 0
            task = generate_reasoning_for_option(
                question, option_key, option_text, correct_score, semaphore
            )
            reasoning_tasks.append(task)

        # 并发执行所有选项的推理生成
        results = await asyncio.gather(*reasoning_tasks, return_exceptions=True)

        # 收集结果
        reasoning_dict = {}
        correction_stats = {"corrected": 0, "first_try_correct": 0}
        
        for result in results:
            if isinstance(result, Exception):
                print(f"  Exception in reasoning generation: {result}")
                continue
            opt_key, reasoning, score, was_corrected = result
            if reasoning:
                reasoning_dict[opt_key] = {
                    "reasoning": reasoning,
                    "score": score,
                    "option_text": choices[opt_key],
                    "was_corrected": was_corrected
                }
                if was_corrected:
                    correction_stats["corrected"] += 1
                else:
                    correction_stats["first_try_correct"] += 1

        # 在结果中包含统计信息
        if correction_stats["corrected"] > 0:
            print(f"  {question_key}: {correction_stats['corrected']} corrected, {correction_stats['first_try_correct']} correct on first try")

        return question_key, reasoning_dict, None

    except Exception as e:
        return question_key, {}, str(e)


async def process_batch(
        items: List[Tuple[str, Dict[str, Any]]],
        semaphore: asyncio.Semaphore,
        start_idx: int
) -> List[Tuple[str, Dict[str, Dict[str, Any]], Any]]:
    """
    并发处理一批问题
    """
    tasks = [
        generate_reasoning(item, key, semaphore)
        for key, item in items
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # 处理异常
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            key, _ = items[i]
            processed_results.append((key, {}, str(result)))
        else:
            processed_results.append(result)

    return processed_results


def format_output_item(original_item: Dict[str, Any], reasoning_dict: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    格式化输出项，为每个选项添加打分推理轨迹和分数
    输出格式适合训练奖励模型：
    - 每个选项包含：原始文本、打分推理轨迹、分数（1或0）
    格式: <reasoning>...</reasoning> <score>1/0</score> <answer>...</answer>
    """
    output_item = original_item.copy()

    # 为每个有推理的选项添加打分推理轨迹和分数
    for option_key, reasoning_data in reasoning_dict.items():
        if option_key in output_item:
            reasoning_text = reasoning_data["reasoning"]
            score = reasoning_data["score"]
            option_text = reasoning_data["option_text"]

            output_item[option_key] = f"<reasoning>{reasoning_text}</reasoning> <score>{score}</score> <answer>{option_text}</answer>"

    return output_item


# ================= 主逻辑 =================

async def main():
    # 1. 加载数据
    dataset = load_data(DATA_FILE)

    # 2. 加载或创建输出文件
    if os.path.exists(OUTPUT_FILE):
        output_data = load_data(OUTPUT_FILE)
        items_to_process = [(key, item) for key, item in dataset.items() if key not in output_data]
    else:
        output_data = {}
        items_to_process = list(dataset.items())

    total_to_process = len(items_to_process)
    print(f"Questions to process: {total_to_process}")

    if total_to_process == 0:
        print("All questions have been processed!")
        return
    # 3. 创建信号量控制并发和文件写入锁
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    file_lock = asyncio.Lock()

    # 4. 分批处理
    start_time = time.time()
    processed_count = 0
    success_count = 0
    error_count = 0
    total_corrections = 0  # 总纠正次数
    total_first_try_correct = 0  # 第一次就正确的次数

    for batch_start in range(0, total_to_process, MAX_CONCURRENT_REQUESTS):
        batch_end = min(batch_start + MAX_CONCURRENT_REQUESTS, total_to_process)
        batch = items_to_process[batch_start:batch_end]

        print(f"\nProcessing batch {batch_start // MAX_CONCURRENT_REQUESTS + 1} "
              f"(items {batch_start + 1}-{batch_end} of {total_to_process})...")

        # 并发处理当前批次
        results = await process_batch(batch, semaphore, batch_start)

        # 处理结果并更新内存数据
        batch_updates = {}  # 本批次的所有更新
        for key, reasoning_dict, error in results:
            processed_count += 1

            if reasoning_dict and len(reasoning_dict) > 0:
                # 成功生成思考轨迹（至少部分选项成功）
                original_item = dataset[key]
                formatted_item = format_output_item(original_item, reasoning_dict)

                # 更新内存中的数据
                output_data[key] = formatted_item
                batch_updates[key] = formatted_item

                # 统计成功生成的选项数和纠正统计
                num_options = len([k for k in original_item.keys() if k.startswith('option ')])
                num_reasoning = len(reasoning_dict)
                
                # 统计纠正情况
                corrections = sum(1 for v in reasoning_dict.values() if v.get('was_corrected', False))
                first_try = num_reasoning - corrections
                total_corrections += corrections
                total_first_try_correct += first_try

                if num_reasoning == num_options:
                    success_count += 1
                    if processed_count % 10 == 0:
                        correction_info = f" ({corrections} corrected)" if corrections > 0 else ""
                        print(f"  [{processed_count}/{total_to_process}] ✓ {key[:20]}... (all {num_reasoning} options{correction_info})")
                else:
                    # 部分成功
                    correction_info = f" ({corrections} corrected)" if corrections > 0 else ""
                    print(
                        f"  [{processed_count}/{total_to_process}] ⚠ {key[:20]}... ({num_reasoning}/{num_options} options{correction_info})")
                    success_count += 1  # 仍然算作成功，因为至少部分完成
            else:
                # 处理失败
                error_count += 1
                print(f"  [{processed_count}/{total_to_process}] ✗ {key[:20]}... Error: {error}")

        # 批量写入文件（线程安全）
        if batch_updates:
            await save_data_safe(output_data, OUTPUT_FILE, file_lock)

        # 定期显示进度
        if processed_count % BATCH_SIZE == 0 or batch_end == total_to_process:
            elapsed = time.time() - start_time
            avg_time = elapsed / processed_count if processed_count > 0 else 0
            remaining = (total_to_process - processed_count) * avg_time

            total_evaluated = total_corrections + total_first_try_correct
            correction_rate = (total_corrections / total_evaluated * 100) if total_evaluated > 0 else 0
            
            print(f"\n--- Progress Update ---")
            print(f"Processed: {processed_count}/{total_to_process} ({processed_count * 100 / total_to_process:.1f}%)")
            print(f"Success: {success_count} | Errors: {error_count}")
            print(f"First-try correct: {total_first_try_correct} | Corrected: {total_corrections} (correction rate: {correction_rate:.1f}%)")
            print(f"Elapsed: {elapsed:.1f}s | Avg: {avg_time:.2f}s/item")
            print(f"Estimated remaining: {remaining / 60:.1f} minutes")
            print(f"Output saved to {OUTPUT_FILE}\n")

    # 5. 最终保存（确保所有数据都写入）
    await save_data_safe(output_data, OUTPUT_FILE, file_lock)

    total_time = time.time() - start_time
    total_evaluated = total_corrections + total_first_try_correct
    correction_rate = (total_corrections / total_evaluated * 100) if total_evaluated > 0 else 0
    
    print(f"\n{'=' * 60}")
    print("Processing Complete!")
    print(f"{'=' * 60}")
    print(f"Total processed: {processed_count}")
    print(f"Success: {success_count}")
    print(f"Errors: {error_count}")
    print(f"\nCorrection Statistics:")
    print(f"  First-try correct: {total_first_try_correct} ({100-correction_rate:.1f}%)")
    print(f"  Required correction: {total_corrections} ({correction_rate:.1f}%)")
    print(f"  Total evaluated: {total_evaluated}")
    print(f"\nTotal time: {total_time / 60:.1f} minutes")
    print(f"Average time per item: {total_time / processed_count:.2f} seconds")
    print(f"\nOutput saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
