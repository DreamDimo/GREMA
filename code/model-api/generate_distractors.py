"""
干扰项生成脚本 - 基于推理错误分析生成高质量的干扰选项

核心改进:
1. **推理分析**: 模型先分析正确答案的生成过程
2. **错误识别**: 识别推理过程中容易出错的关键点
3. **针对性生成**: 基于具体错误点生成干扰项
4. **质量保证**: 
   - 所有干扰项各不相同（自动去重）
   - 长度风格与正确答案相似（0.5x ~ 2x）
   - 每个干扰项对应一个具体的推理错误

功能特性:
1. 自动生成：根据问题和正确答案自动生成干扰项
2. 异步并发：同时处理多个请求，提升速度
3. 断点续存：程序中断后可以从上次停止的地方继续
4. 实时保存：每处理完一定数量立即保存
5. 智能去重：自动过滤重复和不合格的干扰项

使用说明:
- 输入格式: {"q1": {"question": "...", "answer": "..."}, ...}
- 输出格式: {"q1": {"question": "...", "answer": "...", "distractors": ["...", "...", ...]}, ...}

干扰项生成策略:
- Step 1: 分析正确答案的推理过程
- Step 2: 识别可能的错误点（概念混淆、计算错误、不完整推理等）
- Step 3: 针对每个错误点生成一个干扰项
- Step 4: 验证去重，确保质量
"""

import json
import os
import asyncio
import time
from typing import Dict, Any, List, Tuple
from openai import AsyncOpenAI

# ================= 配置区域 =================
# API 配置 - OpenAI 格式
# 标准 OpenAI API
# API_KEY = "sk-..."
# BASE_URL = "https://api.openai.com/v1"
# MODEL_NAME = "gpt-4o"

# ModelArts MAAS API (兼容 OpenAI 格式)
API_KEY = "yhzy9VtrPip__J4S_ySzZP7-LCMbjYji1WPjki_c_BYOd3en1-a7nN7BMSSjT-Py8-jYNYjEWOsHKiaUuFqptA"
BASE_URL = "https://api.modelarts-maas.com/v2/"  # 与 reasoning.py 保持一致
MODEL_NAME = 'deepseek-v3.2'

# vLLM 本地服务
# API_KEY = "EMPTY"
# BASE_URL = "http://localhost:6009/v1"
# MODEL_NAME = "Qwen3-8B"

# API 超时配置
API_TIMEOUT = 150.0  # 生成干扰项可能需要更长时间

# 文件路径
INPUT_FILE = "E:/project/verl/huawei/dataset/TeleQnA_test_filtered_sft_formatted.json"  # 输入的 SFT 格式数据
OUTPUT_FILE = "E:/project/verl/huawei/experiments/result/TeleQnA_test_filtered_with_options.json"  # 输出的多选题格式

# 干扰项配置
NUM_DISTRACTORS = 3  # 每个问题生成3个干扰项（加上正确答案共4个选项）
DISTRACTOR_TYPES = [
    "partially_correct",  # 部分正确但不完整
    "common_misconception",  # 常见误解
    "related_concept",  # 相关但不正确的概念
    "technical_error"  # 技术上的错误
]

# 并发配置
MAX_CONCURRENT_REQUESTS = 10  # 并发数
BATCH_SIZE = 20  # 每处理多少条显示一次进度

# 创建异步OpenAI客户端
client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=API_TIMEOUT)


# ================= 核心函数 =================

async def call_api(messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 2000) -> str:
    """
    调用 OpenAI 兼容 API
    注意：生成干扰项使用较高的 temperature 以增加多样性
    """
    try:
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise Exception(f"API request failed: {str(e)}")


def load_data(filepath: str) -> Dict[str, Any]:
    """
    加载 SFT 格式数据并转换为 QA 对
    
    输入格式: [{"instruction": "...", "input": "...", "output": "..."}, ...]
    输出格式: {"q_0": {"question": "...", "answer": "..."}, ...}
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data_list = json.load(f)
    
    # 转换为字典格式
    qa_dict = {}
    for idx, item in enumerate(data_list):
        question_id = f"q_{idx}"
        qa_dict[question_id] = {
            "question": item.get("input", ""),
            "answer": item.get("output", ""),
            "instruction": item.get("instruction", "")  # 保留 instruction 以备后用
        }
    
    return qa_dict


async def save_data_safe(data: Dict[str, Any], filepath: str, lock: asyncio.Lock):
    """线程安全地保存数据到文件"""
    async with lock:
        data_copy = json.loads(json.dumps(data))
        temp_filepath = filepath + ".tmp"
        with open(temp_filepath, 'w', encoding='utf-8') as f:
            json.dump(data_copy, f, indent=4, ensure_ascii=False)
        os.replace(temp_filepath, filepath)


def format_to_multiple_choice(question_id: str, qa_data: Dict[str, Any], distractors: List[str]) -> Dict[str, Any]:
    """
    将 QA 对和干扰项格式化为多选题格式
    
    输入: 
      - question_id: "q_123"
      - qa_data: {"question": "...", "answer": "..."}
      - distractors: ["distractor1", "distractor2", "distractor3"]
    
    输出格式:
      {
        "question": "...",
        "option 1": "...",
        "option 2": "...",
        "option 3": "...",
        "option 4": "...",
        "answer": "option X"
      }
    """
    import random
    
    question = qa_data.get("question", "")
    correct_answer = qa_data.get("answer", "")
    
    # 组合所有选项（正确答案 + 干扰项）
    all_options = [correct_answer] + distractors
    
    # 随机打乱顺序
    random.shuffle(all_options)
    
    # 找到正确答案的位置
    correct_index = all_options.index(correct_answer) + 1  # 1-based index
    
    # 构建输出格式
    result = {"question": question}
    for i, option in enumerate(all_options, 1):
        result[f"option {i}"] = option
    result["answer"] = f"option {correct_index}"
    
    return result


def build_distractor_generation_prompt(question: str, correct_answer: str, num_distractors: int = 4) -> Tuple[str, str]:
    """
    构建生成干扰项的提示词
    
    改进设计原则：
    1. 先分析正确答案的生成过程（推理链）
    2. 识别推理过程中容易出错的关键点
    3. 基于这些错误点生成针对性的干扰项
    4. 确保干扰项各不相同且长度风格一致
    """
    system_prompt = """You are an expert medical professional with extensive knowledge in medicine, healthcare, and clinical practice. Your task is to generate high-quality distractors for medical questions by analyzing potential reasoning errors."""

    # 计算正确答案的长度范围
    correct_len = len(correct_answer)
    min_len = int(correct_len * 0.5)
    max_len = int(correct_len * 2.0)

    user_prompt = f"""# Medical Question
{question}

# Correct Answer (Length: {correct_len} characters)
{correct_answer}

# Task
Generate {num_distractors} distinct, high-quality distractors (incorrect answer options) for this medical question.

# CRITICAL Requirements (MUST FOLLOW)

## 1. Length Requirement ⚠️ IMPORTANT
- Each distractor MUST be between {min_len} and {max_len} characters long
- This is approximately 50%-200% of the correct answer's length
- **If a distractor is too short or too long, it will be automatically rejected**
- Aim for similar length as the correct answer when possible

## 2. Uniqueness Requirement ⚠️ IMPORTANT
- All {num_distractors} distractors MUST be completely different from each other
- No two distractors should have the same meaning or be paraphrases
- Each distractor must represent a DIFFERENT type of error
- **Duplicate or similar distractors will be rejected**

## 3. Different from Correct Answer ⚠️ IMPORTANT
- Each distractor MUST be clearly different from the correct answer
- Don't just slightly modify the correct answer
- Ensure substantive differences in meaning and content

# Step-by-Step Approach

## Step 1: Analyze the Correct Answer Generation Process
First, think about how the correct answer was derived:
- What medical knowledge or clinical reasoning steps are needed to arrive at this answer?
- What are the key medical concepts, conditions, or facts involved?
- What diagnostic steps, treatment protocols, or clinical guidelines are required?
- What anatomical, physiological, or pathological principles are involved?

## Step 2: Identify {num_distractors} DIFFERENT Potential Error Points
Consider where someone (medical students, healthcare professionals) might make mistakes. Identify {num_distractors} DISTINCT error types:
- **Conceptual confusion**: Mixing up related but different diseases, symptoms, or treatments
- **Incomplete reasoning**: Stopping halfway through the differential diagnosis or treatment plan
- **Dosage/calculation errors**: Making errors in drug dosages, lab value interpretations, or statistical data
- **Common misconceptions**: Typical misunderstandings about diseases, medications, or procedures
- **Outdated knowledge**: Using old treatment guidelines, deprecated medications, or obsolete diagnostic criteria
- **Overgeneralization**: Applying treatment protocols too broadly without considering patient-specific factors
- **Confusing similar conditions**: Mixing up conditions with similar presentations (e.g., different types of diabetes, hypertension stages)
- **Mechanism confusion**: Misunderstanding disease mechanisms, drug actions, or physiological processes
- **Symptom misinterpretation**: Confusing primary vs secondary symptoms
- **Treatment contraindication**: Suggesting inappropriate treatments for specific conditions

## Step 3: Generate {num_distractors} Distractors - Each from a DIFFERENT Error Point
For each distractor:
1. Pick a DIFFERENT error type from Step 2
2. Create a complete, detailed answer that reflects that specific error
3. Ensure the length is within {min_len}-{max_len} characters
4. Make it medically plausible but clearly incorrect

# Quality Checklist (Verify Before Submitting)
✓ All {num_distractors} distractors are between {min_len} and {max_len} characters
✓ Each distractor is completely unique (no duplicates or similar ones)
✓ Each distractor represents a DIFFERENT type of error
✓ All distractors are different from the correct answer
✓ All distractors are medically plausible but incorrect
✓ Proper medical terminology is used throughout

# Output Format
Provide your analysis and exactly {num_distractors} distractors in the following JSON format.
**IMPORTANT**: Make sure each "text" field is properly formatted and has the correct length.

```json
{{
  "reasoning_analysis": "Brief explanation of how the correct medical answer is derived and key clinical error points",
  "distractor_1": {{
    "error_type": "What type of medical/clinical error this represents",
    "text": "The complete distractor text (must be {min_len}-{max_len} characters)"
  }},
  "distractor_2": {{
    "error_type": "A DIFFERENT type of medical/clinical error",
    "text": "The complete distractor text (must be {min_len}-{max_len} characters, DIFFERENT from distractor_1)"
  }},
  "distractor_3": {{
    "error_type": "A DIFFERENT type of medical/clinical error",
    "text": "The complete distractor text (must be {min_len}-{max_len} characters, DIFFERENT from distractor_1 and distractor_2)"
  }},
  "distractor_4": {{
    "error_type": "A DIFFERENT type of medical/clinical error",
    "text": "The complete distractor text (must be {min_len}-{max_len} characters, DIFFERENT from all previous distractors)"
  }}
}}
```

Generate the analysis and distractors now:"""

    return system_prompt, user_prompt


def parse_distractors_from_response(response_text: str, num_expected: int = 4) -> List[str]:
    """
    从模型响应中提取干扰项
    支持新格式（带 error_type 和 text）和旧格式
    """
    import re
    distractors = []
    
    # 尝试提取新的 JSON 格式（带 error_type 和 text）
    try:
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            distractor_dict = json.loads(json_str)
            
            # 新格式：distractor_1: {"error_type": "...", "text": "..."}
            for i in range(1, num_expected + 1):
                key = f"distractor_{i}"
                if key in distractor_dict:
                    item = distractor_dict[key]
                    if isinstance(item, dict) and 'text' in item:
                        distractors.append(item['text'])
                    elif isinstance(item, str):
                        # 兼容旧格式
                        distractors.append(item)
        
        # 如果 JSON 提取成功且数量足够，返回
        if len(distractors) >= num_expected:
            # 去重：确保所有干扰项各不相同
            unique_distractors = []
            seen = set()
            for d in distractors:
                d_lower = d.lower().strip()
                if d_lower not in seen:
                    unique_distractors.append(d)
                    seen.add(d_lower)
            return unique_distractors[:num_expected]
    except Exception as e:
        print(f"  JSON parsing error: {e}")
        pass
    
    # 尝试提取带 "text": 的格式
    try:
        text_matches = re.findall(r'"text"\s*:\s*"([^"]+)"', response_text)
        if text_matches and len(text_matches) >= num_expected:
            # 去重
            unique_distractors = []
            seen = set()
            for d in text_matches:
                d_lower = d.lower().strip()
                if d_lower not in seen:
                    unique_distractors.append(d)
                    seen.add(d_lower)
            return unique_distractors[:num_expected]
    except:
        pass
    
    # 尝试提取编号列表格式
    patterns = [
        r'(?:distractor[_\s]*|option[_\s]*)(\d+)[:\s]*["\']?(.*?)["\']?(?=\n|$)',
        r'(\d+)\.\s*["\']?(.*?)["\']?(?=\n\d+\.|$)',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, response_text, re.IGNORECASE | re.MULTILINE)
        if matches:
            potential = [match[1].strip() for match in matches if match[1].strip() and len(match[1].strip()) > 10]
            if len(potential) >= num_expected:
                # 去重
                unique_distractors = []
                seen = set()
                for d in potential:
                    d_lower = d.lower().strip()
                    if d_lower not in seen:
                        unique_distractors.append(d)
                        seen.add(d_lower)
                return unique_distractors[:num_expected]
    
    # 如果都失败了，按行分割
    lines = response_text.strip().split('\n')
    potential_distractors = [line.strip().strip('"\'').strip() 
                            for line in lines 
                            if line.strip() and len(line.strip()) > 10]
    
    if len(potential_distractors) >= num_expected:
        # 去重
        unique_distractors = []
        seen = set()
        for d in potential_distractors:
            d_lower = d.lower().strip()
            if d_lower not in seen:
                unique_distractors.append(d)
                seen.add(d_lower)
        return unique_distractors[:num_expected]
    
    return distractors if distractors else []


def validate_and_deduplicate_distractors(
        distractors: List[str], 
        correct_answer: str, 
        min_length_ratio: float = 0.5,
        max_length_ratio: float = 2.0,
        debug: bool = False
) -> List[str]:
    """
    验证和去重干扰项
    
    要求：
    1. 干扰项各不相同（不区分大小写）
    2. 干扰项与正确答案不同
    3. 长度与正确答案相近（0.5x ~ 2x）
    """
    if not distractors:
        return []
    
    correct_len = len(correct_answer)
    min_len = int(correct_len * min_length_ratio)
    max_len = int(correct_len * max_length_ratio)
    
    validated = []
    seen = set()
    correct_lower = correct_answer.lower().strip()
    
    rejection_stats = {
        "duplicate": 0,
        "same_as_correct": 0,
        "too_short": 0,
        "too_long": 0
    }
    
    for idx, d in enumerate(distractors, 1):
        d_stripped = d.strip()
        d_lower = d_stripped.lower()
        d_len = len(d_stripped)
        
        # 检查是否与已有的重复
        if d_lower in seen:
            rejection_stats["duplicate"] += 1
            if debug:
                print(f"    Rejected distractor {idx}: DUPLICATE")
            continue
        
        # 检查是否与正确答案相同
        if d_lower == correct_lower:
            rejection_stats["same_as_correct"] += 1
            if debug:
                print(f"    Rejected distractor {idx}: SAME AS CORRECT ANSWER")
            continue
        
        # 检查长度是否合理
        if d_len < min_len:
            rejection_stats["too_short"] += 1
            if debug:
                print(f"    Rejected distractor {idx}: TOO SHORT ({d_len} < {min_len})")
            continue
        
        if d_len > max_len:
            rejection_stats["too_long"] += 1
            if debug:
                print(f"    Rejected distractor {idx}: TOO LONG ({d_len} > {max_len})")
            continue
        
        validated.append(d_stripped)
        seen.add(d_lower)
    
    # 如果有拒绝的，打印统计
    if debug and any(rejection_stats.values()):
        print(f"    Rejection stats: {rejection_stats}")
    
    return validated


async def generate_distractors_for_question(
        question_id: str,
        qa_data: Dict[str, Any],
        num_distractors: int,
        semaphore: asyncio.Semaphore
) -> Tuple[str, List[str], Any]:
    """
    为单个问题生成干扰项
    
    返回: (question_id, distractors_list, error_info)
    """
    async with semaphore:
        try:
            question = qa_data.get('question', '')
            correct_answer = qa_data.get('answer', '')
            
            if not question or not correct_answer:
                return question_id, [], "Missing question or answer"
            
            # 构建提示词
            system_prompt, user_prompt = build_distractor_generation_prompt(
                question, correct_answer, num_distractors
            )
            
            # 调用 API
            response = await call_api(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,  # 较高温度以增加多样性
                max_tokens=2000,
            )
            
            # 提取干扰项
            distractors = parse_distractors_from_response(response, num_distractors)
            
            # 记录提取到的原始干扰项数量
            raw_count = len(distractors)
            
            # 验证和去重
            validated_distractors = validate_and_deduplicate_distractors(
                distractors, correct_answer
            )
            
            # 详细的反馈信息
            if len(validated_distractors) < num_distractors:
                rejected_count = raw_count - len(validated_distractors)
                correct_len = len(correct_answer)
                min_len = int(correct_len * 0.5)
                max_len = int(correct_len * 2.0)
                
                reasons = []
                if rejected_count > 0:
                    reasons.append(f"{rejected_count} rejected (length/duplicate/same as correct)")
                
                feedback = f"Only {len(validated_distractors)}/{num_distractors} valid " \
                          f"(raw: {raw_count}, {', '.join(reasons) if reasons else 'unknown reason'}). " \
                          f"Required length: {min_len}-{max_len} chars"
                print(f"  Warning [{question_id}]: {feedback}")
            
            return question_id, validated_distractors, None
            
        except Exception as e:
            print(f"  Error generating distractors for {question_id}: {e}")
            return question_id, [], str(e)


async def process_batch(
        items: List[Tuple[str, Dict[str, Any]]],
        num_distractors: int,
        semaphore: asyncio.Semaphore
) -> List[Tuple[str, List[str], Any]]:
    """并发处理一批问题"""
    tasks = [
        generate_distractors_for_question(key, item, num_distractors, semaphore)
        for key, item in items
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # 处理异常
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            key, _ = items[i]
            processed_results.append((key, [], str(result)))
        else:
            processed_results.append(result)
    
    return processed_results


# ================= 主逻辑 =================

async def main():
    print("=" * 80)
    print("Multiple Choice Question Generator - OpenAI Compatible API")
    print("=" * 80)
    print(f"API Endpoint: {BASE_URL}")
    print(f"Model: {MODEL_NAME}")
    print(f"Number of distractors: {NUM_DISTRACTORS} (Total 4 options)")
    print(f"Max Concurrent Requests: {MAX_CONCURRENT_REQUESTS}")
    print(f"\nInput Format: SFT format (instruction, input, output)")
    print(f"Output Format: Multiple choice (question, option 1-4, answer)")
    print("=" * 80)
    
    # 1. 加载数据
    print(f"\nLoading SFT data from {INPUT_FILE}...")
    dataset = load_data(INPUT_FILE)
    print(f"Loaded {len(dataset)} QA pairs")
    
    # 2. 加载或创建输出文件（断点续训）
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            output_data = json.load(f)
        # 从输出文件中提取已处理的原始 key (q_XXX)
        processed_keys = set()
        for out_key in output_data.keys():
            # 从 "question 123" 反推原始 key "q_123"
            if out_key.startswith("question "):
                idx = out_key.split()[1]
                processed_keys.add(f"q_{idx}")
        
        items_to_process = [(key, item) for key, item in dataset.items() 
                           if key not in processed_keys]
        print(f"Found existing output file with {len(output_data)} items")
    else:
        output_data = {}
        items_to_process = list(dataset.items())
    
    total_to_process = len(items_to_process)
    print(f"Questions to process: {total_to_process}")
    
    if total_to_process == 0:
        print("All questions have been processed!")
        return
    
    # 3. 创建信号量和文件锁
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    file_lock = asyncio.Lock()
    
    # 4. 分批处理
    start_time = time.time()
    processed_count = 0
    success_count = 0
    error_count = 0
    partial_success_count = 0
    
    for batch_start in range(0, total_to_process, MAX_CONCURRENT_REQUESTS):
        batch_end = min(batch_start + MAX_CONCURRENT_REQUESTS, total_to_process)
        batch = items_to_process[batch_start:batch_end]
        
        print(f"\nProcessing batch {batch_start // MAX_CONCURRENT_REQUESTS + 1} "
              f"(items {batch_start + 1}-{batch_end} of {total_to_process})...")
        
        # 并发处理当前批次
        results = await process_batch(batch, NUM_DISTRACTORS, semaphore)
        
        # 处理结果并更新数据
        batch_updates = {}
        for key, distractors, error in results:
            processed_count += 1
            
            if error:
                error_count += 1
                print(f"  [{processed_count}/{total_to_process}] ✗ {key[:30]}... Error: {error}")
            elif len(distractors) >= NUM_DISTRACTORS:
                # 完全成功：有足够的干扰项
                success_count += 1
                
                # 使用前 NUM_DISTRACTORS 个干扰项
                selected_distractors = distractors[:NUM_DISTRACTORS]
                
                # 格式化为多选题格式
                formatted_item = format_to_multiple_choice(key, dataset[key], selected_distractors)
                
                # 生成输出 key: "question XXX" (从 "q_123" 提取数字)
                if key.startswith("q_"):
                    question_num = key[2:]  # 提取数字部分
                    output_key = f"question {question_num}"
                else:
                    output_key = f"question {processed_count}"
                
                output_data[output_key] = formatted_item
                batch_updates[output_key] = formatted_item
                
                if processed_count % 5 == 0:
                    print(f"  [{processed_count}/{total_to_process}] ✓ {output_key}... (4 options generated)")
            else:
                # 部分成功：干扰项不足
                partial_success_count += 1
                if len(distractors) > 0:
                    # 如果至少有一些干扰项，仍然保存
                    formatted_item = format_to_multiple_choice(key, dataset[key], distractors)
                    
                    if key.startswith("q_"):
                        question_num = key[2:]
                        output_key = f"question {question_num}"
                    else:
                        output_key = f"question {processed_count}"
                    
                    output_data[output_key] = formatted_item
                    batch_updates[output_key] = formatted_item
                    print(f"  [{processed_count}/{total_to_process}] ⚠ {output_key}... (only {len(distractors)+1} options)")
                else:
                    print(f"  [{processed_count}/{total_to_process}] ✗ {key}... (no valid distractors)")
        
        # 批量写入文件
        if batch_updates:
            await save_data_safe(output_data, OUTPUT_FILE, file_lock)
        
        # 定期显示进度
        if processed_count % BATCH_SIZE == 0 or batch_end == total_to_process:
            elapsed = time.time() - start_time
            avg_time = elapsed / processed_count if processed_count > 0 else 0
            remaining = (total_to_process - processed_count) * avg_time
            
            print(f"\n--- Progress Update ---")
            print(f"Processed: {processed_count}/{total_to_process} ({processed_count * 100 / total_to_process:.1f}%)")
            print(f"Success: {success_count} | Partial: {partial_success_count} | Errors: {error_count}")
            print(f"Elapsed: {elapsed:.1f}s | Avg: {avg_time:.2f}s/item")
            print(f"Estimated remaining: {remaining / 60:.1f} minutes")
            print(f"Output saved to {OUTPUT_FILE}\n")
    
    # 5. 最终保存
    await save_data_safe(output_data, OUTPUT_FILE, file_lock)
    
    total_time = time.time() - start_time
    
    print(f"\n{'=' * 80}")
    print("Distractor Generation Complete!")
    print(f"{'=' * 80}")
    print(f"Total processed: {processed_count}")
    print(f"Full success: {success_count} (4 options)")
    print(f"Partial success: {partial_success_count} (< 4 options)")
    print(f"Errors: {error_count}")
    print(f"\nTotal time: {total_time / 60:.1f} minutes")
    print(f"Average time per item: {total_time / processed_count:.2f} seconds")
    print(f"\nOutput saved to: {OUTPUT_FILE}")
    print(f"\nOutput format:")
    print(f"  - Key: 'question XXX'")
    print(f"  - Fields: 'question', 'option 1-4', 'answer' (option X)")


if __name__ == "__main__":
    asyncio.run(main())
