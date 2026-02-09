"""
数据集生成脚本 - 从TeleQnA_test_filtered.json生成包含模型答案的数据集（简答题形式）

功能特性:
1. 读取问题数据集
2. 以简答题形式调用模型生成答案（不显示选项）
3. 去除think标签，只保留答案
4. 提取ground_truth（正确答案对应的选项文本内容）
5. 支持断点续存和并发请求
6. 实时保存checkpoint

提示词格式:
"{question} Let's think step by step and provide a clear and accurate answer."

输出格式:
{
  "question_id": {
    "question": "问题文本",
    "model_answer": "模型生成的答案（已去除think标签）",
    "ground_truth": "正确答案的选项文本内容（如'Non-Seamless WLAN Offload'）"
  }
}
"""

import asyncio
import json
import re
import os
from datetime import datetime
from typing import Dict, Any, Optional
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

# ================= 1. 配置 =================
API_KEY = "EMPTY"  # vLLM使用EMPTY
BASE_URL = "http://localhost:6009/v1/"  # 根据实际情况修改
MODEL_NAME = 'Qwen3-8B-rm'

# 数据集路径
DATASET_PATH = "/mnt/public/wwj/zhaozq/exp1/dataset/TeleQnA/20251227_newest/TeleQnA_test_filtered.json"

# 并发配置
MAX_CONCURRENT_REQUESTS = 20  # 最大并发请求数
RETRY_ATTEMPTS = 1  # 失败重试次数
RETRY_DELAY = 2  # 重试延迟（秒）
API_TIMEOUT = 120.0  # API超时时间（秒）

# 文件路径
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
CHECKPOINT_FILE = f"huawei/experiments/result/dataset_generation_checkpoint_{MODEL_NAME}_{timestamp}.json"
OUTPUT_FILE = f"huawei/experiments/result/dataset_{MODEL_NAME}_{timestamp}.json"

# 创建异步OpenAI客户端
client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=API_TIMEOUT)

# ================= 2. System Prompt =================
SYSTEM_PROMPT = """You are an expert telecommunications engineer. Answer the question with only the most essential one sentence. Your answer must not exceed 30 words. You should answer in English."""

# ================= 3. 辅助函数 =================
def remove_think_tags(response: str) -> str:
    """
    去除think标签（<think>...</think> 或 <think>...</think>），只保留答案部分
    """
    if not response:
        return ""

    answer_text = response

    # 移除 <think>...</think> 标签及其内容（支持多种格式）
    answer_text = re.sub(r'<think>.*?</think>', '', answer_text, flags=re.DOTALL | re.IGNORECASE)
    answer_text = re.sub(r'<think>.*?</think>', '', answer_text, flags=re.DOTALL | re.IGNORECASE)

    # 移除 <answer>...</answer> 标签，但保留内容
    answer_text = re.sub(r'<answer>(.*?)</answer>', r'\1', answer_text, flags=re.DOTALL | re.IGNORECASE)

    # 清理多余空白
    answer_text = answer_text.strip()

    return answer_text


def get_ground_truth_text(question_data: Dict[str, Any]) -> str:
    """
    根据answer字段（如"option 2"）获取对应的选项文本作为ground_truth
    """
    answer_key = question_data.get('answer', '')
    if not answer_key:
        return ""

    # answer_key格式为 "option 2"，需要转换为 "option 2" 作为key
    if answer_key in question_data:
        return question_data[answer_key]

    return ""


def build_user_prompt(question_content: str) -> str:
    """构建User Prompt - 要求只输出最关键的一句话，不超过30个词"""
    return f"""{question_content}

Provide only the most essential one-sentence answer. Maximum 30 words."""


async def generate_answer(
    question_id: str,
    question_data: Dict[str, Any],
    semaphore: asyncio.Semaphore
) -> Dict[str, Any]:
    """异步生成单个问题的答案 - 带重试机制"""
    async with semaphore:
        for attempt in range(RETRY_ATTEMPTS):
            try:
                question_content = question_data['question']

                # 简答题形式，不显示选项
                user_prompt = build_user_prompt(question_content)

                # 调用API生成答案
                response = await client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0,
                    max_tokens=2048,
                )

                response_text = response.choices[0].message.content.strip()

                # 去除think标签，只保留答案
                model_answer = remove_think_tags(response_text)

                # 获取ground_truth
                ground_truth = get_ground_truth_text(question_data)

                return {
                    'question_id': question_id,
                    'question': question_content,
                    'model_answer': model_answer,
                    'ground_truth': ground_truth,
                    'raw_response': response_text,  # 保留原始响应用于调试
                    'error': None
                }

            except Exception as e:
                print(f"\n⚠️ Error generating answer for {question_id} (Attempt {attempt + 1}/{RETRY_ATTEMPTS}): {e}")
                if attempt < RETRY_ATTEMPTS - 1:
                    await asyncio.sleep(RETRY_DELAY)
                else:
                    # 最后一次尝试失败，返回错误
                    return {
                        'question_id': question_id,
                        'question': question_data.get('question', ''),
                        'model_answer': '',
                        'ground_truth': get_ground_truth_text(question_data),
                        'raw_response': '',
                        'error': str(e)
                    }


# ================= 4. 批量生成（支持断点续存）=================
async def save_checkpoint(results: list, total_count: int):
    """异步保存checkpoint"""
    try:
        checkpoint_data = {
            'mode': 'dataset_generation',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'progress': {
                'total_questions': total_count,
                'generated_questions': len(results),
                'remaining_questions': total_count - len(results),
                'failed_generations': sum(1 for r in results if r.get('error'))
            },
            'results': results
        }

        # 确保目录存在
        os.makedirs(os.path.dirname(CHECKPOINT_FILE), exist_ok=True)

        with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"\nWarning: Failed to save checkpoint: {e}")


async def main():
    """主函数"""
    print("=" * 80)
    print("Dataset Generation Script")
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
    generated_questions = set()

    if os.path.exists(CHECKPOINT_FILE):
        print("\n" + "=" * 80)
        print("Found checkpoint file! Loading previous results...")
        print("=" * 80)
        try:
            with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
                results = checkpoint_data.get('results', [])
                generated_questions = set(r['question_id'] for r in results)
            print(f"Loaded {len(results)} previously generated questions.")
            print(f"Resuming from question #{len(results) + 1}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting from scratch...")
            results = []
            generated_questions = set()
    else:
        print("\n" + "=" * 80)
        print("Starting New Dataset Generation")
        print("=" * 80)

    # ================= 统计信息 =================
    total_count = len(dataset)
    remaining_questions = {qid: qdata for qid, qdata in dataset.items() if qid not in generated_questions}
    remaining_count = len(remaining_questions)

    print(f"Total questions: {total_count}")
    print(f"Already generated: {len(generated_questions)}")
    print(f"Remaining: {remaining_count}")
    print("=" * 80)

    if remaining_count == 0:
        print("\n✅ All questions already generated!")
        # 直接保存最终结果（只包含三个字段）
        final_dataset = [{
            'question': r['question'],
            'model_answer': r['model_answer'],
            'ground_truth': r['ground_truth']
        } for r in results]

        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(final_dataset, f, indent=2, ensure_ascii=False)
        print(f"Final dataset saved to: {OUTPUT_FILE}")
        return

    # ================= 异步并发生成 =================
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    lock = asyncio.Lock()

    # 创建所有生成任务
    tasks = []
    for question_id, question_data in remaining_questions.items():
        task = generate_answer(question_id, question_data, semaphore)
        tasks.append(task)

    # 使用tqdm显示进度
    print("\nStarting concurrent generation...")
    completed_results = []

    for coro in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="Generating"):
        result = await coro
        completed_results.append(result)

        # 每完成一个就更新results列表并保存checkpoint
        async with lock:
            results.append(result)

            # 每10个保存一次checkpoint（减少I/O）
            if len(completed_results) % 10 == 0:
                await save_checkpoint(results, total_count)

    # 最终保存checkpoint
    await save_checkpoint(results, total_count)

    # ================= 计算最终统计 =================
    failed_generations = sum(1 for r in results if r.get('error'))

    print("\n" + "=" * 80)
    print("DATASET GENERATION COMPLETED!")
    print("=" * 80)
    print(f"Total Questions:      {total_count}")
    print(f"Generated:            {len(results)}")
    print(f"Failed Generations:     {failed_generations}")
    print("=" * 80)

    # ================= 保存最终结果 =================
    # 转换为最终数据集格式（只包含三个字段）
    final_dataset = [{
        'question': r['question'],
        'model_answer': r['model_answer'],
        'ground_truth': r['ground_truth']
    } for r in results]

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_dataset, f, indent=2, ensure_ascii=False)

    print(f"\nFinal dataset saved to: {OUTPUT_FILE}")

    # 删除checkpoint文件
    if os.path.exists(CHECKPOINT_FILE):
        try:
            os.remove(CHECKPOINT_FILE)
            print(f"Checkpoint file removed (generation completed)")
        except Exception as e:
            print(f"Warning: Failed to remove checkpoint file: {e}")

    # ================= 显示样本 =================
    print("\n" + "=" * 80)
    print("Sample Generated Data (First 3)")
    print("=" * 80)

    for i, r in enumerate(results[:3], 1):
        print(f"\n{i}. Question ID: {r['question_id']}")
        print(f"   Question: {r['question'][:100]}...")
        print(f"   Model Answer: {r['model_answer'][:100]}...")
        print(f"   Ground Truth: {r['ground_truth'][:100]}...")
        if r.get('error'):
            print(f"   Error: {r['error']}")
        print("-" * 80)


if __name__ == "__main__":
    asyncio.run(main())
