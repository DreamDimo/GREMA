# -*- coding: utf-8 -*-
"""
使用 RM-R1 模型对 TeleQnA 两选项数据集进行 pairwise 评估（vLLM 异步并发）。

功能特性:
- Pairwise 评估: 每道题一次请求，模型输出 [[A]] 或 [[B]]
- vLLM 异步并发: 多题同时推理，加速评估
- 断点续存: 中断后再次运行从检查点恢复
- 实时保存: 每评估完一题即写入检查点文件

数据集格式: { "question id": { "question", "option 1", "option 2", "answer": { "Option 1": 0/1, "Option 2": 0/1 } } }
"""
import asyncio
import json
import os
import re
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

from tqdm import tqdm
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm import SamplingParams
from transformers import AutoTokenizer

# ================= 1. 配置 =================
# DATASET_PATH = "/seu_share/home/220256455/dataset/TeleQnA/20251227_newest/TeleQnA_test_two_options.json"
DATASET_PATH = "/seu_share/home/220256455/dataset/Medical/Medical_test_two_options.json"
MODEL_PATH = "/seu_share/home/220256455/model/RM-R1"

# 检查点：固定路径，中断后再次运行会从此恢复
CHECKPOINT_FILE = "/seu_share/home/220256455/verl/huawei/experiments/result/RM-R1_Medical.json"

# 最终结果输出：文件名
OUTPUT_FILENAME = "/seu_share/home/220256455/verl/huawei/experiments/result/scores_RM-R1_Medical.json"

# 并发与生成参数
MAX_CONCURRENT_REQUESTS = 8   # 同时进行中的请求数（RM 单条较长，不宜过大）
MAX_NEW_TOKENS = 8192
RETRY_ATTEMPTS = 1
RETRY_DELAY = 2

# 全局
llm_engine = None
tokenizer = None


def compute_accuracy(results: list) -> Tuple[int, int, float]:
    """根据当前 results 计算实时准确率。返回 (correct, total_evaluated, accuracy)。"""
    n = len(results)
    if n == 0:
        return 0, 0, 0.0
    correct = sum(1 for r in results if r.get("is_correct"))
    return correct, n, correct / n

# ================= 2. System & User Prompt (Pairwise) =================
INSTRUCT_SYSTEM_PROMPT = (
    "Please act as an impartial judge and evaluate the quality of the responses provided by two AI Chatbots to the Client's question displayed below.\n\n"
    "First, classify the task into one of two categories: <type>Reasoning</type> or <type>Chat</type>.\n"
    "- Use <type>Reasoning</type> for tasks that involve math, coding, or require domain knowledge, multi-step inference, logical deduction, or combining information to reach a conclusion.\n"
    "- Use <type>Chat</type> for tasks that involve open-ended or factual conversation, stylistic rewrites, safety questions, or general helpfulness requests without deep reasoning.\n\n"

    "If the task is Reasoning:\n"
    "1. Solve the Client's question yourself and present your final answer within <solution>...</solution> tags.\n"
    "2. Evaluate the two Chatbot responses based on correctness, completeness, and reasoning quality, referencing your own solution.\n"
    "3. Include your evaluation inside <eval>...</eval> tags, quoting or summarizing the Chatbots using the following tags:\n"
    "   - <quote_A>...</quote_A> for direct quotes from Chatbot A\n"
    "   - <summary_A>...</summary_A> for paraphrases of Chatbot A\n"
    "   - <quote_B>...</quote_B> for direct quotes from Chatbot B\n"
    "   - <summary_B>...</summary_B> for paraphrases of Chatbot B\n"
    "4. End with your final judgment in the format: <answer>[[A]]</answer> or <answer>[[B]]</answer>\n\n"

    "If the task is Chat:\n"
    "1. Generate evaluation criteria (rubric) tailored to the Client's question and context, enclosed in <rubric>...</rubric> tags.\n"
    "2. Assign weights to each rubric item based on their relative importance.\n"
    "3. Inside <rubric>, include a <justify>...</justify> section explaining why you chose those rubric criteria and weights.\n"
    "4. Compare both Chatbot responses according to the rubric.\n"
    "5. Provide your evaluation inside <eval>...</eval> tags, using <quote_A>, <summary_A>, <quote_B>, and <summary_B> as described above.\n"
    "6. End with your final judgment in the format: <answer>[[A]]</answer> or <answer>[[B]]</answer>\n\n"

    "Important Notes:\n"
    "- Be objective and base your evaluation only on the content of the responses.\n"
    "- Do not let response order, length, or Chatbot names affect your judgment.\n"
    "- Follow the response format strictly depending on the task type.\n\n"

    "Your output must follow one of the two formats below:\n\n"
    "For Reasoning:\n"
    "<type>Reasoning</type>\n\n"
    "<solution> your own solution for the problem </solution>\n\n"
    "<eval>\n"
    "  include direct comparisons supported by <quote_A>...</quote_A> or <summary_A>...</summary_A>, and <quote_B>...</quote_B>, or <summary_B>...</summary_B>\n"
    "</eval>\n\n"
    "<answer>[[A/B]]</answer>\n\n"

    "For Chat:\n"
    "<type>Chat</type>\n\n"
    "<rubric>\n"
    "  detailed rubric items\n"
    "  <justify> justification for the rubric </justify>\n"
    "</rubric>\n\n"
    "<eval>\n"
    "  include direct comparisons supported by <quote_A>...</quote_A> or <summary_A>...</summary_A>, and <quote_B>...</quote_B>, or <summary_B>...</summary_B> tags\n"
    "</eval>\n\n"
    "<answer>[[A/B]]</answer>"
)

INSTRUCT_SINGLE_USER_PROMPT_TEMPLATE = (
    "[Client Question]\n{question}\n\n[The Start of Chatbot A's Response]\n{answer_a}\n[The End of Chatbot A's Response]\n\n"
    "[The Start of Chatbot B's Response]\n{answer_b}\n[The End of Chatbot B's Response]"
)

# ================= 3. 辅助函数 =================
def parse_rm_answer(completion: str) -> Optional[str]:
    """从模型输出中解析 <answer>[[A]]</answer> 或 <answer>[[B]]</answer>，返回 'A' 或 'B'。"""
    if not completion:
        return None
    m = re.search(r"<answer>\s*\[\[([AB])\]\]\s*</answer>", completion, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).upper()
    m = re.search(r"\[\[([AB])\]\]", completion)
    if m:
        return m.group(1).upper()
    return None


def get_correct_option(answer_dict: dict) -> Optional[str]:
    """从 answer 字典中得到正确选项（值为 1 的键）。"""
    for k, v in answer_dict.items():
        if v == 1:
            return k
    return None


async def evaluate_single_question_pairwise(
    question_id: str,
    question_data: Dict[str, Any],
    semaphore: asyncio.Semaphore,
) -> Dict[str, Any]:
    """异步评估单题：pairwise 一次请求，解析 [[A]]/[[B]]。"""
    question = question_data["question"]
    opt1 = question_data.get("option 1", "")
    opt2 = question_data.get("option 2", "")
    answer = question_data.get("answer", {})
    gt_option = get_correct_option(answer)

    async with semaphore:
        for attempt in range(RETRY_ATTEMPTS):
            try:
                user_prompt = INSTRUCT_SINGLE_USER_PROMPT_TEMPLATE.format(
                    question=question,
                    answer_a=opt1,
                    answer_b=opt2,
                )
                messages = [
                    {"role": "system", "content": INSTRUCT_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ]
                prompt_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                sampling_params = SamplingParams(
                    temperature=0,
                    top_p=1.0,
                    max_tokens=MAX_NEW_TOKENS,
                )
                request_id = str(uuid.uuid4())
                results_generator = llm_engine.generate(
                    prompt_text, sampling_params, request_id
                )
                final_output = None
                async for request_output in results_generator:
                    final_output = request_output
                completion = final_output.outputs[0].text.strip()
                chosen_ab = parse_rm_answer(completion)
                chosen_option = (
                    "option 1"
                    if chosen_ab == "A"
                    else ("option 2" if chosen_ab == "B" else None)
                )
                is_correct = chosen_option == gt_option if chosen_option else False
                return {
                    "question_id": question_id,
                    "predicted": chosen_ab,
                    "predicted_option": chosen_option,
                    "ground_truth_option": gt_option,
                    "is_correct": is_correct,
                    "completion_preview": completion[:500] if completion else "",
                }
            except Exception as e:
                if attempt < RETRY_ATTEMPTS - 1:
                    await asyncio.sleep(RETRY_DELAY)
                else:
                    return {
                        "question_id": question_id,
                        "predicted": None,
                        "predicted_option": None,
                        "ground_truth_option": gt_option,
                        "is_correct": False,
                        "completion_preview": f"Error: {e}",
                    }
    return {}


async def initialize_engine():
    """初始化 vLLM 引擎和 tokenizer"""
    global llm_engine, tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, use_fast=False, trust_remote_code=True
    )
    print("Loading vLLM Async Engine...")
    engine_args = AsyncEngineArgs(
        model=MODEL_PATH,
        tokenizer=MODEL_PATH,
        max_model_len=16384,
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85,
        enable_chunked_prefill=True,
    )
    llm_engine = AsyncLLMEngine.from_engine_args(engine_args)
    print("Engine loaded.\n")


async def shutdown_engine():
    """退出前显式关闭 vLLM 引擎，避免 __del__ 时触发 'NoneType' object is not callable。"""
    global llm_engine
    if llm_engine is None:
        return
    try:
        if hasattr(llm_engine, "shutdown") and callable(getattr(llm_engine, "shutdown")):
            s = llm_engine.shutdown()
            if asyncio.iscoroutine(s):
                await s
        if hasattr(llm_engine, "close") and callable(getattr(llm_engine, "close")):
            c = llm_engine.close()
            if asyncio.iscoroutine(c):
                await c
    except Exception:
        pass
    finally:
        llm_engine = None


def save_checkpoint_sync(results: list, evaluated_ids: list):
    """同步写入检查点；文件最上方写入当前正确率统计。"""
    try:
        correct_cur, n_cur, acc_cur = compute_accuracy(results)
        checkpoint_data = {
            "correct": correct_cur,
            "total_evaluated": n_cur,
            "accuracy": round(acc_cur, 4),
            "evaluated_ids": list(evaluated_ids),
            "results": results,
        }
        with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
            json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"\nWarning: 保存检查点失败: {e}")


async def main():
    print("=" * 60)
    print("RM-R1 Pairwise 评估 (vLLM 异步)")
    print("=" * 60)
    print(f"Dataset: {DATASET_PATH}")
    print(f"Model:   {MODEL_PATH}")
    print(f"Concurrent: {MAX_CONCURRENT_REQUESTS}")
    print("=" * 60)

    try:
        await initialize_engine()

        print(f"Loading dataset from {DATASET_PATH}...")
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        total = len(dataset)
        print(f"Loaded {total} questions.\n")

        # 加载检查点
        results = []
        evaluated_ids = set()
        if os.path.exists(CHECKPOINT_FILE):
            print("=" * 60)
            print(f"发现检查点: {CHECKPOINT_FILE}，从上次进度恢复")
            print("=" * 60)
            try:
                with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                    ckpt = json.load(f)
                results = ckpt.get("results", [])
                evaluated_ids = set(ckpt.get("evaluated_ids", []))
                if not evaluated_ids and results:
                    evaluated_ids = {r["question_id"] for r in results}
                acc_ckpt = ckpt.get("accuracy")
                c_ckpt = ckpt.get("correct")
                n_ckpt = ckpt.get("total_evaluated", len(results))
                if acc_ckpt is not None and c_ckpt is not None and n_ckpt:
                    print(f"当前检查点准确率: {acc_ckpt:.4f} ({c_ckpt}/{n_ckpt})")
                print(f"已评估: {len(evaluated_ids)} 题，剩余: {total - len(evaluated_ids)} 题\n")
            except Exception as e:
                print(f"加载检查点失败: {e}，从头开始")
                results = []
                evaluated_ids = set()

        remaining = {
            qid: qdata
            for qid, qdata in dataset.items()
            if qid not in evaluated_ids
        }
        if not remaining:
            print("✅ 已全部评估完毕。")
            correct = sum(1 for r in results if r.get("is_correct"))
            acc = correct / total if total else 0
            print(f"Accuracy: {acc:.4f} ({correct}/{total})")
            out_path = OUTPUT_FILENAME
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(
                    {"accuracy": acc, "correct": correct, "total": total, "results": results},
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            print(f"Results saved to {out_path}")
            return

        semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        lock = asyncio.Lock()

        tasks = [
            evaluate_single_question_pairwise(qid, qdata, semaphore)
            for qid, qdata in remaining.items()
        ]

        print("Starting concurrent evaluation... (实时准确率显示在进度条)")
        pbar = tqdm(total=len(tasks), desc="Evaluating", unit="q")
        for coro in asyncio.as_completed(tasks):
            result = await coro
            async with lock:
                results.append(result)
                evaluated_ids.add(result["question_id"])
                save_checkpoint_sync(results, list(evaluated_ids))
                correct_cur, n_cur, acc_cur = compute_accuracy(results)
                pbar.update(1)
                pbar.set_postfix(acc=f"{acc_cur:.2%}", correct=correct_cur, n=n_cur)
        pbar.close()

        correct = sum(1 for r in results if r.get("is_correct"))
        failed_parse = sum(1 for r in results if r.get("predicted_option") is None)
        acc = correct / total if total else 0

        print("\n" + "=" * 60)
        print(f"Total: {total}, Correct: {correct}, Failed parse: {failed_parse}")
        print(f"Accuracy: {acc:.4f} ({correct}/{total})")
        print("=" * 60)

        out_path = OUTPUT_FILENAME
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                {"accuracy": acc, "correct": correct, "total": total, "results": results},
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"Results saved to {out_path}")
    finally:
        await shutdown_engine()


if __name__ == "__main__":
    asyncio.run(main())
