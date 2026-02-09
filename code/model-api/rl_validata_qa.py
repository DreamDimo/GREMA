"""
答案评价脚本 - 使用DeepSeek评价模型生成的答案是否正确

功能特性:
1. 读取包含question, model_answer的数据集（可选ground_truth字段）
2. 将问题、答案和ground_truth（如果存在）发送给DeepSeek进行评价
3. 开启思考模式，让模型基于问题本身判断答案的正确性
4. ground_truth作为参考信息，不要求答案完全匹配，只要事实正确即可
5. 返回0（错误）或1（正确）
6. 为每个问题添加evaluation字段
7. 计算并显示正确率
"""

import json
import os
import asyncio
import aiohttp
import re
import time
from typing import Dict, Any, List, Tuple, Optional

# ================= 配置区域 =================
# API 配置
API_KEY = "sk-32c6d293585d44718d2b1dde5fd02fdc"
BASE_URL = "https://api.deepseek.com/v1/chat/completions"
MODEL_NAME = 'deepseek-chat'

# 文件路径
# DATA_FILE = "E:/project/verl/huawei/dataset/dataset_Qwen3-8B_20260110_121524.json"  # 输入文件路径
# OUTPUT_FILE = "E:/project/verl/huawei/experiments/result/evaluated_dataset_Qwen3-8B_20260110_121524.json"  # 输出文件路径

DATA_FILE = r"E:\project\verl\huawei\dataset_Qwen3-8B-rm_20260208_111420.json"  # 输入文件路径
OUTPUT_FILE = r"E:\project\verl\huawei\experiments\result\evaluated_dataset_Qwen3-8B-rm_20260208_111420.json" 
# 并发配置
MAX_CONCURRENT_REQUESTS = 20  # 最大并发请求数
RETRY_ATTEMPTS = 1  # 失败重试次数
RETRY_DELAY = 2  # 重试延迟（秒）
API_TIMEOUT = 120.0  # API超时时间（秒）

# BATCH_SIZE: 每处理多少条数据显示一次进度更新
BATCH_SIZE = 50  # 每处理多少条显示一次进度

# ================= System Prompt =================
SYSTEM_PROMPT = """You are an expert telecommunications engineer. You will evaluate whether an answer to a technical question is correct or incorrect based on your expertise. Please reason step by step, and put your final answer within \\boxed{}."""


# ================= 核心函数 =================

async def call_api(messages: List[Dict[str, str]], temperature: float = 0.1, max_tokens: int = 2048) -> str:
    """
    使用 aiohttp 调用 ModelArts MAAS API（DeepSeek）
    开启思考模式
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
    
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=API_TIMEOUT)) as session:
        async with session.post(
            BASE_URL, 
            headers=headers, 
            json=data,
            ssl=False
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"API request failed with status {response.status}: {error_text}")
            
            result = await response.json()
            return result['choices'][0]['message']['content']


def load_data(filepath: str) -> List[Dict[str, Any]]:
    """加载数据集（数组格式）"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 如果是新格式（包含statistics和data字段）
    if isinstance(data, dict) and 'data' in data:
        return data['data']
    
    # 如果是字典格式，转换为数组
    if isinstance(data, dict):
        data = list(data.values())
    return data


def calculate_statistics(dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    """计算统计信息"""
    evaluated_items = [item for item in dataset if 'evaluation' in item]
    total_evaluated = len(evaluated_items)
    correct_count = sum(1 for item in evaluated_items if item.get('evaluation') == 1)
    incorrect_count = total_evaluated - correct_count
    accuracy = correct_count / total_evaluated * 100 if total_evaluated > 0 else 0.0
    
    return {
        "total_items": len(dataset),
        "total_evaluated": total_evaluated,
        "correct": correct_count,
        "incorrect": incorrect_count,
        "accuracy": round(accuracy, 2)
    }


def save_data(data: List[Dict[str, Any]], filepath: str):
    """保存数据到文件（包含统计信息）"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    statistics = calculate_statistics(data)
    output = {
        "statistics": statistics,
        "data": data
    }
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


async def save_data_safe(data: List[Dict[str, Any]], filepath: str, lock: asyncio.Lock):
    """
    线程安全地保存数据到文件（包含动态更新的统计信息）
    使用锁保护文件写入操作
    """
    async with lock:
        # 深拷贝数据，避免在写入过程中数据被修改
        data_copy = json.loads(json.dumps(data))
        # 计算统计信息
        statistics = calculate_statistics(data_copy)
        # 构建输出结构（统计信息在最上面）
        output = {
            "statistics": statistics,
            "data": data_copy
        }
        # 使用临时文件，然后原子性地替换
        temp_filepath = filepath + ".tmp"
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(temp_filepath, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        # 原子性替换
        os.replace(temp_filepath, filepath)


def build_evaluation_prompt(question: str, model_answer: str, ground_truth: str) -> str:
    """
    构建评价提示词，匹配 TeleQnA_reward.py 格式
    ground_truth 作为参考信息
    返回: user_prompt
    """
    user_prompt = f"""Score Meaning:
- Score 1: The answer is CORRECT and fully addresses the question
- Score 0: The answer is INCORRECT or does not properly address the question

#### Question
{question}

#### Answer to Evaluate
{model_answer}

#### Reference Answer (Ground Truth)
{ground_truth}

**Important Evaluation Guidelines:**
1. **If you are CERTAIN** that the answer to evaluate is correct or incorrect based on your expertise, evaluate accordingly.
2. **If you are UNCERTAIN** about the correctness of the answer, you MUST refer to the reference answer (ground truth) above to make your judgment.
3. A question may have multiple correct answers, and the answer to evaluate does not need to match the reference answer exactly in wording. However, if you are uncertain, use the reference answer as the authoritative source.
4. When in doubt, prioritize the reference answer for evaluation."""
    
    return user_prompt


def extract_evaluation_score(response_text: str) -> Optional[int]:
    """
    从模型响应中提取评价分数（0或1），优先提取 \\boxed{} 格式
    返回: 0, 1, 或 None（如果提取失败）
    """
    if not response_text:
        return None
    
    try:
        # 优先提取 \\boxed{0} 或 \\boxed{1} 格式（匹配 TeleQnA_reward.py 格式）
        boxed_patterns = [
            r'\\boxed\{([01])\}',  # \boxed{0} 或 \boxed{1}
            r'\\boxed\s*\{([01])\}',  # \boxed {0} 或 \boxed {1}（带空格）
            r'boxed\{([01])\}',  # boxed{0} 或 boxed{1}（无反斜杠）
        ]
        
        for pattern in boxed_patterns:
            match = re.search(pattern, response_text)
            if match:
                return int(match.group(1))
        
        # 如果找不到 boxed 格式，尝试其他模式
        patterns = [
            r'"evaluation"\s*:\s*([01])',  # "evaluation": 0 或 1
            r'"score"\s*:\s*([01])',  # "score": 0 或 1
            r'evaluation[:\s]+([01])',
            r'score[:\s]+([01])',
            r'\bscore\s*=?\s*([01])\b',  # score = 0 或 1
            r'\b([01])\s*\((?:Correct|Incorrect)\)',  # 1 (Correct) 或 0 (Incorrect)
            r'(?:Correct|Incorrect)\s*[:\-]?\s*([01])',  # Correct: 1 或 Incorrect: 0
            r'```json\s*\{\s*"evaluation"\s*:\s*([01])\s*\}\s*```',  # JSON 块
            r'```json\s*\{\s*"score"\s*:\s*([01])\s*\}\s*```',  # JSON 块
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                return int(match.group(1))
        
        # 如果找不到明确的分数，尝试从文本中推断
        response_lower = response_text.lower()
        if 'incorrect' in response_lower or 'wrong' in response_lower or 'false' in response_lower:
            if 'correct' not in response_lower[:response_lower.index('incorrect')] if 'incorrect' in response_lower else True:
                return 0
        if 'correct' in response_lower or 'accurate' in response_lower or 'true' in response_lower:
            return 1
        
        return None
    except Exception as e:
        print(f"Error extracting evaluation score: {e}")
        return None


async def evaluate_answer(
    item: Dict[str, Any],
    index: int,
    semaphore: asyncio.Semaphore
) -> Tuple[int, Dict[str, Any], Optional[str]]:
    """
    评价单个问题的答案
    返回: (index, updated_item, error_message)
    """
    async with semaphore:
        for attempt in range(RETRY_ATTEMPTS):
            try:
                question = item.get('question', '')
                model_answer = item.get('model_answer', '')
                ground_truth = item.get('ground_truth', '')
                
                if not question or not model_answer or not ground_truth:
                    return index, item, "Missing required fields (question, model_answer, or ground_truth)"
                
                # 构建提示词（提供 ground_truth 作为参考）
                user_prompt = build_evaluation_prompt(question, model_answer, ground_truth)
                
                # 调用API
                response_text = await call_api(
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1,
                    max_tokens=2048
                )
                
                # 提取评价分数
                evaluation_score = extract_evaluation_score(response_text)
                
                if evaluation_score is None:
                    return index, item, f"Failed to extract evaluation score from response"
                
                # 更新item，添加evaluation字段
                updated_item = item.copy()
                updated_item['evaluation'] = evaluation_score
                updated_item['evaluation_reasoning'] = response_text  # 保留推理过程用于调试
                
                return index, updated_item, None
                
            except Exception as e:
                error_msg = f"Attempt {attempt + 1}/{RETRY_ATTEMPTS}: {str(e)}"
                if attempt < RETRY_ATTEMPTS - 1:
                    await asyncio.sleep(RETRY_DELAY)
                else:
                    return index, item, error_msg
        
        return index, item, "Failed after all retries"


async def process_batch(
    items: List[Dict[str, Any]],
    indices: List[int],
    semaphore: asyncio.Semaphore
) -> List[Tuple[int, Dict[str, Any], Optional[str]]]:
    """
    并发处理一批问题
    """
    tasks = [
        evaluate_answer(item, idx, semaphore)
        for item, idx in zip(items, indices)
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # 处理异常
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            processed_results.append((indices[i], items[i], str(result)))
        else:
            processed_results.append(result)
    
    return processed_results


# ================= 主逻辑 =================

async def main():
    print("=" * 80)
    print("Answer Evaluation Script - DeepSeek")
    print("=" * 80)
    print(f"API Endpoint: {BASE_URL}")
    print(f"Model: {MODEL_NAME}")
    print(f"Max Concurrent Requests: {MAX_CONCURRENT_REQUESTS}")
    print(f"Input File: {DATA_FILE}")
    print(f"Output File: {OUTPUT_FILE}")
    print("=" * 80)
    
    # 1. 加载数据
    print(f"\nLoading dataset from {DATA_FILE}...")
    dataset = load_data(DATA_FILE)
    print(f"Loaded {len(dataset)} items from dataset.")
    
    # 2. 检查输出文件，支持断点续存
    if os.path.exists(OUTPUT_FILE):
        print(f"\nFound existing output file: {OUTPUT_FILE}")
        print("Loading existing results...")
        output_data = load_data(OUTPUT_FILE)
        # 找出还没有评价的项
        evaluated_indices = {i for i, item in enumerate(output_data) if 'evaluation' in item}
        items_to_process = [(i, item) for i, item in enumerate(dataset) if i not in evaluated_indices]
        # 合并已评价的结果
        for i in evaluated_indices:
            if i < len(output_data) and i < len(dataset):
                dataset[i] = output_data[i]
    else:
        output_data = dataset.copy()
        items_to_process = list(enumerate(dataset))
    
    total_to_process = len(items_to_process)
    print(f"Items to process: {total_to_process}")
    
    if total_to_process == 0:
        print("All items have been evaluated!")
        # 计算最终统计
        evaluated_items = [item for item in dataset if 'evaluation' in item]
        if evaluated_items:
            correct_count = sum(1 for item in evaluated_items if item.get('evaluation') == 1)
            total_count = len(evaluated_items)
            accuracy = correct_count / total_count * 100 if total_count > 0 else 0
            print(f"\nFinal Statistics:")
            print(f"Total evaluated: {total_count}")
            print(f"Correct (1): {correct_count}")
            print(f"Incorrect (0): {total_count - correct_count}")
            print(f"Accuracy: {accuracy:.2f}%")
        return
    
    # 3. 创建信号量控制并发和文件写入锁
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    file_lock = asyncio.Lock()
    
    # 4. 分批处理
    start_time = time.time()
    processed_count = 0
    success_count = 0
    error_count = 0
    correct_count = 0  # 评价为1的数量
    
    for batch_start in range(0, total_to_process, MAX_CONCURRENT_REQUESTS):
        batch_end = min(batch_start + MAX_CONCURRENT_REQUESTS, total_to_process)
        batch_items = [item for _, item in items_to_process[batch_start:batch_end]]
        batch_indices = [idx for idx, _ in items_to_process[batch_start:batch_end]]
        
        print(f"\nProcessing batch {batch_start // MAX_CONCURRENT_REQUESTS + 1} "
              f"(items {batch_start + 1}-{batch_end} of {total_to_process})...")
        
        # 并发处理当前批次
        results = await process_batch(batch_items, batch_indices, semaphore)
        
        # 处理结果并更新数据
        batch_updates = False
        for idx, updated_item, error in results:
            processed_count += 1
            
            if error is None:
                # 成功评价
                dataset[idx] = updated_item
                batch_updates = True
                success_count += 1
                
                if updated_item.get('evaluation') == 1:
                    correct_count += 1
                
                if processed_count % 10 == 0:
                    eval_score = updated_item.get('evaluation', 'N/A')
                    print(f"  [{processed_count}/{total_to_process}] ✓ Item {idx} - Evaluation: {eval_score}")
            else:
                # 处理失败
                error_count += 1
                print(f"  [{processed_count}/{total_to_process}] ✗ Item {idx} - Error: {error}")
        
        # 批量写入文件（线程安全）
        if batch_updates:
            await save_data_safe(dataset, OUTPUT_FILE, file_lock)
        
        # 定期显示进度
        if processed_count % BATCH_SIZE == 0 or batch_end == total_to_process:
            elapsed = time.time() - start_time
            avg_time = elapsed / processed_count if processed_count > 0 else 0
            remaining = (total_to_process - processed_count) * avg_time
            
            current_accuracy = correct_count / success_count * 100 if success_count > 0 else 0
            
            print(f"\n--- Progress Update ---")
            print(f"Processed: {processed_count}/{total_to_process} ({processed_count * 100 / total_to_process:.1f}%)")
            print(f"Success: {success_count} | Errors: {error_count}")
            print(f"Correct (1): {correct_count} | Incorrect (0): {success_count - correct_count}")
            print(f"Current Accuracy: {current_accuracy:.2f}%")
            print(f"Elapsed: {elapsed:.1f}s | Avg: {avg_time:.2f}s/item")
            print(f"Estimated remaining: {remaining / 60:.1f} minutes")
            print(f"Output saved to {OUTPUT_FILE}\n")
    
    # 5. 最终保存（确保所有数据都写入）
    await save_data_safe(dataset, OUTPUT_FILE, file_lock)
    
    # 6. 计算最终统计
    evaluated_items = [item for item in dataset if 'evaluation' in item]
    total_evaluated = len(evaluated_items)
    final_correct_count = sum(1 for item in evaluated_items if item.get('evaluation') == 1)
    final_incorrect_count = total_evaluated - final_correct_count
    final_accuracy = final_correct_count / total_evaluated * 100 if total_evaluated > 0 else 0
    
    total_time = time.time() - start_time
    
    print(f"\n{'=' * 80}")
    print("Evaluation Complete!")
    print(f"{'=' * 80}")
    print(f"Total processed: {processed_count}")
    print(f"Success: {success_count}")
    print(f"Errors: {error_count}")
    print(f"\nFinal Statistics:")
    print(f"  Total evaluated: {total_evaluated}")
    print(f"  Correct (1): {final_correct_count}")
    print(f"  Incorrect (0): {final_incorrect_count}")
    print(f"  Accuracy: {final_accuracy:.2f}%")
    print(f"\nTotal time: {total_time / 60:.1f} minutes")
    print(f"Average time per item: {total_time / processed_count:.2f} seconds" if processed_count > 0 else "")
    print(f"\nOutput saved to: {OUTPUT_FILE}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    asyncio.run(main())
