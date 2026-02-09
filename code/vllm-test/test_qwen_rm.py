"""
Qwen RM Pointwise 异步批量评估脚本 - 支持断点续存和并发请求

功能特性:
1. Pointwise评估: 每个选项单独输入模型评估，然后汇总
2. 异步并发: 同时处理多个请求，大幅提升速度
3. 断点续存: 程序中断后可以从上次停止的地方继续
4. 实时保存: 每评估完一个问题立即保存到checkpoint文件
5. 异常处理: 单个问题出错不会影响整个评估流程
6. 并发控制: 限制同时进行的请求数量，避免过载

使用说明:
- 首次运行: 直接运行脚本，会创建checkpoint文件
- 中断后继续: 再次运行脚本，会自动加载checkpoint并继续评估
- 完成评估: checkpoint文件会被自动删除，最终结果保存到带时间戳的文件
"""

import asyncio
import json
import re
import os
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

# ================= 1. 配置 =================
API_KEY = "EMPTY"  # vLLM使用EMPTY
BASE_URL = "http://localhost:6008/v1/"  # 根据实际情况修改
MODEL_NAME = 'Qwen3-8b-rm-origin'

# 数据集路径
DATASET_PATH = "/mnt/public/wwj/zhaozq/exp1/dataset/TeleQnA/20251227_newest/TeleQnA_test.json"

# 并发配置
MAX_CONCURRENT_REQUESTS = 20  # 最大并发请求数
RETRY_ATTEMPTS = 1  # 失败重试次数
RETRY_DELAY = 2  # 重试延迟（秒）
API_TIMEOUT = 60.0  # API超时时间（秒）

model = MODEL_NAME
# 文件路径
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
CHECKPOINT_FILE = f"/mnt/public/wwj/zhaozq/exp1/verl/huawei/experiments/result/evaluation_checkpoint_{model}_{timestamp}.json"
OUTPUT_FILE = f"/mnt/public/wwj/zhaozq/exp1/verl/huawei/experiments/result/evaluation_results_pointwise_async_{model}_{timestamp}.json"

# 创建异步OpenAI客户端
client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=API_TIMEOUT)

# ================= 2. System Prompt (Pointwise模式) =================
SYSTEM_PROMPT = """You are an expert telecommunications engineer. You will evaluate whether an answer to a technical question is correct or incorrect. Please reason step by step, and put your final answer within \\boxed{}."""

# ================= 3. 辅助函数 =================
def extract_single_score_from_response(response: str) -> Optional[int]:
    """从模型响应中提取单个分数 (0 或 1)，优先提取 \\boxed{} 格式"""
    try:
        # 优先提取 \\boxed{0} 或 \\boxed{1} 格式（匹配 JSON 数据格式）
        boxed_patterns = [
            r'\\boxed\{([01])\}',  # \boxed{0} 或 \boxed{1}
            r'\\boxed\s*\{([01])\}',  # \boxed {0} 或 \boxed {1}（带空格）
            r'boxed\{([01])\}',  # boxed{0} 或 boxed{1}（无反斜杠）
        ]
        
        for pattern in boxed_patterns:
            match = re.search(pattern, response)
            if match:
                return int(match.group(1))
        
        # 如果找不到 boxed 格式，尝试其他模式
        patterns = [
            r'"score"\s*:\s*([01])',           # "score": 0 或 1
            r'score\s*:\s*([01])',             # score: 0 或 1
            r'\bscore\s*=?\s*([01])\b',        # score = 0 或 1
            r'\b([01])\s*\((?:Correct|Incorrect)\)',  # 1 (Correct) 或 0 (Incorrect)
            r'(?:Correct|Incorrect)\s*[:\-]?\s*([01])',  # Correct: 1 或 Incorrect: 0
            r'```json\s*\{\s*"score"\s*:\s*([01])\s*\}\s*```',  # JSON 块
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return int(match.group(1))

        # 如果找不到明确的分数，尝试从文本中推断
        response_lower = response.lower()
        if 'incorrect' in response_lower or 'wrong' in response_lower or 'false' in response_lower:
            if 'correct' not in response_lower[:response_lower.index('incorrect')] if 'incorrect' in response_lower else True:
                return 0
        if 'correct' in response_lower or 'accurate' in response_lower or 'true' in response_lower:
            return 1

        return None
    except Exception as e:
        print(f"Error extracting score: {e}")
        return None


def compare_scores(predicted_scores: Optional[Dict[str, int]], ground_truth_answer: Dict[str, int]) -> Tuple[bool, Dict[str, bool], int, int]:
    """
    比较预测分数和真实答案，按选项级别统计

    Returns:
        Tuple[question_correct, option_correctness, total_options, correct_options]
        - question_correct: 整道题是否完全正确
        - option_correctness: 每个选项的正确性 {"Option 1": True/False, ...}
        - total_options: 总选项数
        - correct_options: 正确选项数
    """
    if predicted_scores is None:
        return False, {}, 0, 0

    # 统计每个选项的正确性
    option_correctness = {}
    total_options = len(ground_truth_answer)
    correct_options = 0
    question_correct = True

    for key in ground_truth_answer.keys():
        if key in predicted_scores and predicted_scores[key] == ground_truth_answer[key]:
            option_correctness[key] = True
            correct_options += 1
        else:
            option_correctness[key] = False
            question_correct = False

    return question_correct, option_correctness, total_options, correct_options


def build_user_prompt(question_content: str, option_text: str) -> str:
    """构建User Prompt - 只评估一个选项，匹配 JSON 数据格式"""
    return f"""Score Meaning:
- Score 1: The answer is CORRECT and fully addresses the question
- Score 0: The answer is INCORRECT or does not properly address the question

#### Question
{question_content}

#### Answer to Evaluate
{option_text}"""


async def evaluate_single_option(
    question_content: str,
    option_text: str,
    option_number: int,
    semaphore: asyncio.Semaphore
) -> Tuple[Optional[int], str]:
    """异步评估单个选项 (Pointwise) - 带重试机制"""
    async with semaphore:
        for attempt in range(RETRY_ATTEMPTS):
            try:
                user_prompt = build_user_prompt(question_content, option_text)

                # 调用API
                response = await client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0,
                    max_tokens=2048,  # 单个选项评估不需要太长
                )

                response_text = response.choices[0].message.content.strip()

                # 提取分数
                score = extract_single_score_from_response(response_text)

                return score, response_text

            except Exception as e:
                print(f"\n⚠️ Error evaluating option {option_number} (Attempt {attempt + 1}/{RETRY_ATTEMPTS}): {e}")
                if attempt < RETRY_ATTEMPTS - 1:
                    await asyncio.sleep(RETRY_DELAY)
                else:
                    # 最后一次尝试失败，返回错误
                    return None, f"Error: {str(e)}"


async def evaluate_single_question_pointwise(
    question_id: str,
    question_data: Dict[str, Any],
    semaphore: asyncio.Semaphore
) -> Dict[str, Any]:
    """异步评估单个问题 - Pointwise模式：每个选项单独评估（带异常处理）"""
    try:
        question_content = question_data['question']

        # 提取所有选项
        candidate_answers = []
        for i in range(1, 10):  # 最多支持9个选项
            option_key = f"option {i}"
            if option_key in question_data:
                candidate_answers.append(question_data[option_key])
            else:
                break

        # 对每个选项并发评估
        tasks = []
        for i, option_text in enumerate(candidate_answers, 1):
            task = evaluate_single_option(question_content, option_text, i, semaphore)
            tasks.append((i, task))

        # 等待所有选项评估完成
        results_list = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)

        predicted_scores = {}
        all_responses = []
        failed_extraction = False

        for idx, (i, _) in enumerate(tasks):
            option_key = f"option {i}"
            option_text = candidate_answers[i - 1]
            result = results_list[idx]

            if isinstance(result, Exception):
                print(f"\n⚠️ Error evaluating option {i} of {question_id}: {str(result)}")
                failed_extraction = True
                score = 0
                response = f"Error: {str(result)}"
            else:
                score, response = result
                if score is None:
                    failed_extraction = True
                    # 如果提取失败，设为0（保守策略）
                    score = 0

            predicted_scores[option_key] = score
            all_responses.append({
                'option': option_key,
                'text': option_text,
                'score': score,
                'response': response
            })

        ground_truth_answer = question_data['answer']

        # 比较结果（新版：按选项级别统计）
        question_correct, option_correctness, total_options, correct_options = compare_scores(predicted_scores, ground_truth_answer)

        return {
            'question_id': question_id,
            'predicted_scores': predicted_scores,
            'ground_truth': ground_truth_answer,
            'is_correct': question_correct,  # 整道题是否完全正确
            'option_correctness': option_correctness,  # 每个选项的正确性
            'total_options': total_options,  # 总选项数
            'correct_options': correct_options,  # 正确选项数
            'failed_extraction': failed_extraction,
            'all_responses': all_responses,
            'error': None
        }

    except Exception as e:
        # 如果整个问题评估出错，返回错误信息但不中断程序
        print(f"\n❌ Error evaluating {question_id}: {str(e)}")
        ground_truth = question_data.get('answer', {})
        return {
            'question_id': question_id,
            'predicted_scores': None,
            'ground_truth': ground_truth,
            'is_correct': False,
            'option_correctness': {},
            'total_options': len(ground_truth),
            'correct_options': 0,
            'failed_extraction': True,
            'all_responses': [],
            'error': str(e)
        }

# ================= 4. 批量评估（支持断点续存）- Pointwise模式 =================
async def save_checkpoint(results: list, total_count: int):
    """异步保存checkpoint"""
    try:
        # Question级别统计
        question_correct_count = sum(1 for r in results if r['is_correct'])
        failed_extractions = sum(1 for r in results if r['predicted_scores'] is None)

        # Option级别统计
        total_options = sum(r.get('total_options', 0) for r in results)
        correct_options = sum(r.get('correct_options', 0) for r in results)
        option_accuracy = correct_options / total_options if total_options > 0 else 0

        checkpoint_data = {
            'mode': 'pointwise',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'progress': {
                # Question级别
                'total_questions': total_count,
                'evaluated_questions': len(results),
                'remaining_questions': total_count - len(results),
                'correct_questions': question_correct_count,
                'incorrect_questions': len(results) - question_correct_count - failed_extractions,
                'failed_extractions': failed_extractions,
                'question_accuracy': question_correct_count / len(results) if len(results) > 0 else 0,
                # Option级别（新增）
                'total_options': total_options,
                'correct_options': correct_options,
                'incorrect_options': total_options - correct_options,
                'option_accuracy': option_accuracy
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
    print("Qwen RM Pointwise Async Batch Evaluation")
    print("=" * 80)
    print(f"API Endpoint: {BASE_URL}")
    print(f"Model: {MODEL_NAME}")
    print(f"Max Concurrent Requests: {MAX_CONCURRENT_REQUESTS}")
    print(f"API Timeout: {API_TIMEOUT}s")
    print(f"Mode: Pointwise (each option evaluated separately)")
    print("=" * 80)

    # ================= 加载数据集 =================
    print(f"\nLoading dataset from {DATASET_PATH}...")
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    print(f"Loaded {len(dataset)} questions from dataset.")

    # ================= 加载checkpoint（如果存在）=================
    results = []
    evaluated_questions = set()

    if os.path.exists(CHECKPOINT_FILE):
        print("\n" + "=" * 80)
        print("Found checkpoint file! Loading previous results...")
        print("=" * 80)
        try:
            with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
                results = checkpoint_data.get('results', [])
                evaluated_questions = set(r['question_id'] for r in results)
            print(f"Loaded {len(results)} previously evaluated questions.")
            print(f"Resuming from question #{len(results) + 1}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting from scratch...")
            results = []
            evaluated_questions = set()
    else:
        print("\n" + "=" * 80)
        print("Starting New Batch Evaluation - POINTWISE MODE")
        print("(Each option is evaluated separately)")
        print("=" * 80)

    # ================= 统计信息 =================
    total_count = len(dataset)
    remaining_questions = {qid: qdata for qid, qdata in dataset.items() if qid not in evaluated_questions}
    remaining_count = len(remaining_questions)

    print(f"Total questions: {total_count}")
    print(f"Already evaluated: {len(evaluated_questions)}")
    print(f"Remaining: {remaining_count}")
    print("=" * 80)

    if remaining_count == 0:
        print("\n✅ All questions already evaluated!")
        return

    # ================= 异步并发评估 =================
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    lock = asyncio.Lock()

    # 创建所有评估任务
    tasks = []
    for question_id, question_data in remaining_questions.items():
        task = evaluate_single_question_pointwise(question_id, question_data, semaphore)
        tasks.append(task)

    # 使用tqdm显示进度
    print("\nStarting concurrent evaluation...")
    completed_results = []

    for coro in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="Evaluating"):
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
    # Question级别统计
    question_correct_count = sum(1 for r in results if r['is_correct'])
    failed_extractions = sum(1 for r in results if r['predicted_scores'] is None)

    # Option级别统计
    total_options = sum(r.get('total_options', 0) for r in results)
    correct_options = sum(r.get('correct_options', 0) for r in results)
    incorrect_options = total_options - correct_options

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETED - POINTWISE MODE!")
    print("=" * 80)
    print(f"Mode: Pointwise (each option evaluated separately)")
    print("\n📊 Question-Level Statistics:")
    print(f"  Total Questions:      {total_count}")
    print(f"  Evaluated:            {len(results)}")
    print(f"  Correct Questions:    {question_correct_count}")
    print(f"  Incorrect Questions:  {len(results) - question_correct_count - failed_extractions}")
    print(f"  Failed Extractions:   {failed_extractions}")
    print(f"  Question Accuracy:    {question_correct_count}/{len(results)} ({question_correct_count / len(results) * 100:.2f}%)")

    print("\n📊 Option-Level Statistics (Main Metric):")
    print(f"  Total Options:        {total_options}")
    print(f"  Correct Options:      {correct_options}")
    print(f"  Incorrect Options:    {incorrect_options}")
    print(f"  Option Accuracy:      {correct_options}/{total_options} ({correct_options / total_options * 100:.2f}%)")
    print("=" * 80)

    # ================= 保存最终结果 =================
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump({
            'mode': 'pointwise',
            'description': 'Each option is evaluated separately by the model, then scores are aggregated',
            'summary': {
                # Question级别统计
                'question_level': {
                    'total_questions': total_count,
                    'evaluated_questions': len(results),
                    'correct_questions': question_correct_count,
                    'incorrect_questions': len(results) - question_correct_count - failed_extractions,
                    'failed_extractions': failed_extractions,
                    'question_accuracy': question_correct_count / len(results) if len(results) > 0 else 0
                },
                # Option级别统计（主要指标）
                'option_level': {
                    'total_options': total_options,
                    'correct_options': correct_options,
                    'incorrect_options': incorrect_options,
                    'option_accuracy': correct_options / total_options if total_options > 0 else 0
                }
            },
            'results': results
        }, f, indent=2, ensure_ascii=False)

    print(f"\nDetailed results saved to: {OUTPUT_FILE}")

    # 删除checkpoint文件
    if os.path.exists(CHECKPOINT_FILE):
        try:
            os.remove(CHECKPOINT_FILE)
            print(f"Checkpoint file removed (evaluation completed)")
        except Exception as e:
            print(f"Warning: Failed to remove checkpoint file: {e}")

    # ================= 显示错误案例 =================
    print("\n" + "=" * 80)
    print("Sample Incorrect Questions (First 5)")
    print("=" * 80)

    incorrect_samples = [r for r in results if not r['is_correct'] and r.get('predicted_scores') is not None][:5]
    for i, sample in enumerate(incorrect_samples, 1):
        print(f"\n{i}. Question ID: {sample['question_id']}")
        print(f"   Ground Truth:  {sample['ground_truth']}")
        print(f"   Predicted:     {sample['predicted_scores']}")
        print(f"   Option Stats:  {sample['correct_options']}/{sample['total_options']} correct")

        # 显示每个选项的正确性
        if sample.get('option_correctness'):
            print(f"   Option Details:")
            for opt, is_correct in sample['option_correctness'].items():
                status = "✓" if is_correct else "✗"
                print(f"     {status} {opt}: GT={sample['ground_truth'].get(opt)}, Pred={sample['predicted_scores'].get(opt)}")
        if sample.get('error'):
            print(f"   Error: {sample['error']}")
        print("-" * 80)


if __name__ == "__main__":
    asyncio.run(main())