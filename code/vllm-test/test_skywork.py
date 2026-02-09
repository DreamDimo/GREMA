# -*- coding: utf-8 -*-
"""
使用 Skywork-Reward 模型对两选项数据集进行评估（逐题打分比较）。

与 test_rm-r1 同结构：同一数据集格式、检查点、输出文件。
Skywork-Reward 为打分模型：对 Option 1 / Option 2 各打一次分，取分高者为预测选项。

数据集格式: { "question id": { "question", "option 1", "option 2", "answer": { "Option 1": 0/1, "Option 2": 0/1 } } }
"""
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ================= 1. 配置 =================
# DATASET_PATH = "/seu_share/home/220256455/dataset/TeleQnA/20251227_newest/TeleQnA_test_two_options.json"
DATASET_PATH = "/seu_share/home/220256455/dataset/Medical/Medical_test_two_options.json"
MODEL_PATH = "/seu_share/home/220256455/model/Skywork"  # 或本地路径

CHECKPOINT_FILE = "/seu_share/home/220256455/verl/huawei/experiments/result/Skywork-Reward_Medical.json"
OUTPUT_FILENAME = "/seu_share/home/220256455/verl/huawei/experiments/result/scores_Skywork-Reward_Medical.json"

RETRY_ATTEMPTS = 1

# 全局（main 里加载）
rm_model = None
rm_tokenizer = None


def compute_accuracy(results: list) -> Tuple[int, int, float]:
    """根据当前 results 计算实时准确率。返回 (correct, total_evaluated, accuracy)。"""
    n = len(results)
    if n == 0:
        return 0, 0, 0.0
    correct = sum(1 for r in results if r.get("is_correct"))
    return correct, n, correct / n


def get_correct_option(answer_dict: dict) -> Optional[str]:
    """从 answer 字典中得到正确选项（值为 1 的键）。"""
    for k, v in answer_dict.items():
        if v == 1:
            return k
    return None


def score_single(question: str, response: str) -> Optional[float]:
    """对单条 (question, response) 打一个 reward 分。输入会放到模型所在设备。"""
    conv = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": response},
    ]
    try:
        inputs = rm_tokenizer.apply_chat_template(
            conv, tokenize=True, return_tensors="pt"
        )
        # 放到模型所在设备（device_map="auto" 时由 transformers 自动选的 GPU）
        inputs = inputs.to(next(rm_model.parameters()).device)
        with torch.no_grad():
            out = rm_model(inputs).logits[0][0].item()
        return out
    except Exception:
        return None


def evaluate_single_question(
    question_id: str,
    question_data: Dict[str, Any],
) -> Dict[str, Any]:
    """评估单题：对 option 1 / option 2 各打一次分，分高者作为预测选项。"""
    question = question_data["question"]
    opt1 = question_data.get("option 1", "")
    opt2 = question_data.get("option 2", "")
    answer = question_data.get("answer", {})
    gt_option = get_correct_option(answer)

    for _ in range(RETRY_ATTEMPTS):
        try:
            score1 = score_single(question, opt1)
            score2 = score_single(question, opt2)
            if score1 is None and score2 is None:
                chosen_option = None
            elif score1 is None:
                chosen_option = "Option 2"
            elif score2 is None:
                chosen_option = "Option 1"
            else:
                chosen_option = "Option 1" if score1 > score2 else "Option 2"
            is_correct = chosen_option == gt_option if chosen_option else False
            return {
                "question_id": question_id,
                "predicted_option": chosen_option,
                "ground_truth_option": gt_option,
                "is_correct": is_correct,
                "score_option1": score1,
                "score_option2": score2,
            }
        except Exception as e:
            continue
    return {
        "question_id": question_id,
        "predicted_option": None,
        "ground_truth_option": gt_option,
        "is_correct": False,
        "score_option1": None,
        "score_option2": None,
    }


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


def main():
    global rm_model, rm_tokenizer

    print("=" * 60)
    print("Skywork-Reward 两选项评估")
    print("=" * 60)
    print(f"Dataset: {DATASET_PATH}")
    print(f"Model:   {MODEL_PATH}")
    print("=" * 60)

    print("Loading model and tokenizer...")
    rm_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",  # 自动使用当前可见的 GPU，不写死 cuda:0
        num_labels=1,
    )
    rm_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    print("Model loaded.\n")

    print(f"Loading dataset from {DATASET_PATH}...")
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    total = len(dataset)
    print(f"Loaded {total} questions.\n")

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
        with open(OUTPUT_FILENAME, "w", encoding="utf-8") as f:
            json.dump(
                {"accuracy": acc, "correct": correct, "total": total, "results": results},
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"Results saved to {OUTPUT_FILENAME}")
        return

    print("Starting evaluation... (实时准确率显示在进度条)")
    pbar = tqdm(remaining.items(), desc="Evaluating", unit="q")
    for qid, qdata in pbar:
        result = evaluate_single_question(qid, qdata)
        results.append(result)
        evaluated_ids.add(qid)
        save_checkpoint_sync(results, list(evaluated_ids))
        correct_cur, n_cur, acc_cur = compute_accuracy(results)
        pbar.set_postfix(acc=f"{acc_cur:.2%}", correct=correct_cur, n=n_cur)

    correct = sum(1 for r in results if r.get("is_correct"))
    failed_parse = sum(1 for r in results if r.get("predicted_option") is None)
    acc = correct / total if total else 0

    print("\n" + "=" * 60)
    print(f"Total: {total}, Correct: {correct}, Failed: {failed_parse}")
    print(f"Accuracy: {acc:.4f} ({correct}/{total})")
    print("=" * 60)

    with open(OUTPUT_FILENAME, "w", encoding="utf-8") as f:
        json.dump(
            {"accuracy": acc, "correct": correct, "total": total, "results": results},
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Results saved to {OUTPUT_FILENAME}")


if __name__ == "__main__":
    main()
