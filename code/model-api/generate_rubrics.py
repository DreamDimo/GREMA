import json
import os
import asyncio
import time
from typing import Dict, Any, List, Tuple
from openai import AsyncOpenAI

# ================= 配置区域 =================
# API 配置 - OpenAI 格式
# ModelArts MAAS API (兼容 OpenAI 格式)
API_KEY = "yhzy9VtrPip__J4S_ySzZP7-LCMbjYji1WPjki_c_BYOd3en1-a7nN7BMSSjT-Py8-jYNYjEWOsHKiaUuFqptA"
BASE_URL = "https://api.modelarts-maas.com/v2/"
MODEL_NAME = 'deepseek-v3.2'

# API 超时配置
API_TIMEOUT = 120.0  # API超时时间（秒）

# 文件路径
DATA_FILE = "E:/project/verl/huawei/dataset/TeleQnA_test_filtered_sft_formatted.json"
OUTPUT_FILE = "E:/project/verl/huawei/experiments/result/Medical_test_with_rubrics.json"

# 并发配置
MAX_CONCURRENT_REQUESTS = 15  # 同时发送的请求数量
BATCH_SIZE = 50  # 每处理多少条显示一次进度

# 创建异步OpenAI客户端
client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=API_TIMEOUT)


# ================= 核心函数 =================

async def call_api(messages: List[Dict[str, str]], temperature: float = 0.3, max_tokens: int = 2000) -> Tuple[str, str]:
    """
    使用 AsyncOpenAI 调用 OpenAI 兼容 API
    
    返回: (reasoning_content, final_content)
    - reasoning_content: DeepSeek 的 <think> 思考过程（如果有）
    - final_content: 最终的回答内容
    """
    try:
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        message = response.choices[0].message
        
        # 获取思考过程
        reasoning_content = ""
        if hasattr(message, 'reasoning_content') and message.reasoning_content:
            reasoning_content = message.reasoning_content.strip()
        
        # 获取最终回答
        final_content = message.content.strip() if message.content else ""
        
        return reasoning_content, final_content
    except Exception as e:
        raise Exception(f"API request failed: {str(e)}")


def load_data(filepath: str) -> Dict[str, Any]:
    """加载训练数据"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


async def save_data_safe(data: Dict[str, Any], filepath: str, lock: asyncio.Lock):
    """
    线程安全地保存数据到文件
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


def build_rubrics_prompt(question: str) -> Tuple[str, str]:
    """
    构建生成评判规则（rubrics）的提示词
    要求模型根据问题生成5条具体的评判规则
    
    返回: (system_prompt, user_prompt)
    """
    system_prompt = """You are an expert medical educator and assessment specialist. Your task is to create clear, specific, and actionable evaluation rubrics for medical questions."""

    user_prompt = f"""# Medical Question
{question}

# Task
Create exactly 5 evaluation rubrics (rules) that can be used to judge whether an answer to this question is correct or incorrect. Each rubric should be a specific criterion that a correct answer MUST satisfy.

# Requirements for Each Rubric
1. Be specific and directly related to the question
2. Be measurable and objective (not vague or subjective)
3. Focus on key medical facts, concepts, or principles
4. Be clear enough that anyone can use it to evaluate an answer
5. Cover different important aspects of a complete answer

# Output Format
Provide exactly 5 rubrics in the following format:

**Rubric 1:** [Clear, specific criterion that a correct answer must meet]

**Rubric 2:** [Clear, specific criterion that a correct answer must meet]

**Rubric 3:** [Clear, specific criterion that a correct answer must meet]

**Rubric 4:** [Clear, specific criterion that a correct answer must meet]

**Rubric 5:** [Clear, specific criterion that a correct answer must meet]

# Example Structure (for illustration only, adapt to the actual question)
If the question was "What is diabetes?", good rubrics might include:
- Correctly identifies diabetes as a metabolic disorder
- Mentions elevated blood glucose/sugar levels
- References insulin dysfunction (production or utilization)
- Describes at least one major symptom or complication
- Distinguishes between Type 1 and Type 2 or mentions different types

Now, create 5 specific rubrics for the given medical question above."""

    return system_prompt, user_prompt


def parse_rubrics(response_text: str) -> List[str]:
    """
    从模型响应中提取5条评判规则
    返回: 包含5条规则的列表，如果提取失败返回空列表
    """
    import re
    
    # 尝试多种格式提取规则
    rubrics = []
    
    # 格式1: **Rubric N:** 内容
    pattern1 = r'\*\*Rubric\s+\d+:\*\*\s*(.+?)(?=\*\*Rubric\s+\d+:|\Z)'
    matches1 = re.findall(pattern1, response_text, re.DOTALL | re.IGNORECASE)
    if matches1:
        rubrics = [m.strip() for m in matches1]
    
    # 格式2: Rubric N: 内容 (没有粗体标记)
    if not rubrics:
        pattern2 = r'Rubric\s+\d+:\s*(.+?)(?=Rubric\s+\d+:|\Z)'
        matches2 = re.findall(pattern2, response_text, re.DOTALL | re.IGNORECASE)
        if matches2:
            rubrics = [m.strip() for m in matches2]
    
    # 格式3: 数字编号 1. 2. 3. 等
    if not rubrics:
        pattern3 = r'(?:^|\n)\d+\.\s*(.+?)(?=\n\d+\.|\Z)'
        matches3 = re.findall(pattern3, response_text, re.DOTALL)
        if matches3:
            rubrics = [m.strip() for m in matches3]
    
    # 清理每条规则（去除多余的换行和空格）
    rubrics = [' '.join(r.split()) for r in rubrics if r.strip()]
    
    return rubrics


async def generate_rubrics_for_question(
        question: str,
        question_key: str,
        semaphore: asyncio.Semaphore
) -> Tuple[str, List[str], str, Any]:
    """
    为单个问题生成5条评判规则
    
    返回: (question_key, rubrics_list, full_response, error_info)
    - rubrics_list: 包含5条规则的列表
    - full_response: 模型的完整响应（包含thinking过程）
    """
    async with semaphore:
        try:
            # 构建提示词
            system_prompt, user_prompt = build_rubrics_prompt(question)
            
            # 调用API生成规则
            think_content, final_content = await call_api(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,  # 稍高的温度以获得更多样化的规则
                max_tokens=2000,
            )
            
            # 组合 think 和 final content
            if think_content:
                full_response = f"<think>{think_content}</think>\n{final_content}"
            else:
                full_response = final_content
            
            full_response = full_response.strip()
            
            # 解析规则
            rubrics = parse_rubrics(final_content)
            
            return question_key, rubrics, full_response, None
            
        except Exception as e:
            print(f"  Error generating rubrics for {question_key}: {e}")
            return question_key, [], "", str(e)




async def process_batch(
        items: List[Tuple[str, Dict[str, Any]]],
        semaphore: asyncio.Semaphore
) -> List[Tuple[str, List[str], str, Any]]:
    """
    并发处理一批问题
    """
    tasks = [
        generate_rubrics_for_question(item["question"], key, semaphore)
        for key, item in items
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # 处理异常
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            key, _ = items[i]
            processed_results.append((key, [], "", str(result)))
        else:
            processed_results.append(result)
    
    return processed_results


def format_output_item(original_item: Dict[str, Any], rubrics: List[str], full_response: str) -> Dict[str, Any]:
    """
    格式化输出项，添加生成的评判规则
    """
    output_item = original_item.copy()
    
    # 添加评判规则
    output_item["rubrics"] = rubrics
    
    # 添加完整的模型响应（包含thinking过程）
    output_item["rubrics_generation_response"] = full_response
    
    # 添加规则数量统计
    output_item["num_rubrics"] = len(rubrics)
    
    return output_item


# ================= 主逻辑 =================

async def main():
    # 1. 加载数据
    print(f"Loading data from {DATA_FILE}...")
    dataset = load_data(DATA_FILE)
    print(f"Loaded {len(dataset)} questions")
    
    # 2. 加载或创建输出文件
    if os.path.exists(OUTPUT_FILE):
        output_data = load_data(OUTPUT_FILE)
        items_to_process = [(key, item) for key, item in dataset.items() if key not in output_data]
        print(f"Found existing output file with {len(output_data)} processed questions")
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
    partial_success_count = 0  # 生成了规则但数量不是5条
    error_count = 0
    
    for batch_start in range(0, total_to_process, MAX_CONCURRENT_REQUESTS):
        batch_end = min(batch_start + MAX_CONCURRENT_REQUESTS, total_to_process)
        batch = items_to_process[batch_start:batch_end]
        
        print(f"\nProcessing batch {batch_start // MAX_CONCURRENT_REQUESTS + 1} "
              f"(items {batch_start + 1}-{batch_end} of {total_to_process})...")
        
        # 并发处理当前批次
        results = await process_batch(batch, semaphore)
        
        # 处理结果
        batch_updates = {}
        
        for key, rubrics, full_response, error in results:
            processed_count += 1
            
            if len(rubrics) > 0:
                # 成功生成了规则（无论数量是否为5）
                original_item = dataset[key]
                formatted_item = format_output_item(original_item, rubrics, full_response)
                
                output_data[key] = formatted_item
                batch_updates[key] = formatted_item
                
                if len(rubrics) == 5:
                    success_count += 1
                    if processed_count % 10 == 0:
                        print(f"  [{processed_count}/{total_to_process}] ✓ {key[:30]}... (5 rubrics)")
                else:
                    partial_success_count += 1
                    print(f"  [{processed_count}/{total_to_process}] ⚠ {key[:30]}... (got {len(rubrics)} rubrics)")
            
            else:
                # 完全失败
                error_count += 1
                print(f"  [{processed_count}/{total_to_process}] ✗ {key[:30]}... Error: {error}")
        
        # 批量写入文件（线程安全）
        if batch_updates:
            await save_data_safe(output_data, OUTPUT_FILE, file_lock)
        
        # 定期显示进度
        if processed_count % BATCH_SIZE == 0 or batch_end == total_to_process:
            elapsed = time.time() - start_time
            avg_time = elapsed / processed_count if processed_count > 0 else 0
            remaining = (total_to_process - processed_count) * avg_time
            
            print(f"\n--- Progress Update ---")
            print(f"Processed: {processed_count}/{total_to_process} ({processed_count * 100 / total_to_process:.1f}%)")
            print(f"Success (5 rubrics): {success_count}")
            print(f"Partial (≠5 rubrics): {partial_success_count}")
            print(f"Errors: {error_count}")
            print(f"Elapsed: {elapsed:.1f}s | Avg: {avg_time:.2f}s/item")
            print(f"Estimated remaining: {remaining / 60:.1f} minutes")
            print(f"Output saved to {OUTPUT_FILE}\n")
    
    # 5. 最终保存
    await save_data_safe(output_data, OUTPUT_FILE, file_lock)
    
    total_time = time.time() - start_time
    
    print(f"\n{'=' * 60}")
    print("Processing Complete!")
    print(f"{'=' * 60}")
    print(f"Total processed: {processed_count}")
    if processed_count > 0:
        print(f"Success (5 rubrics): {success_count} ({success_count / processed_count * 100:.1f}%)")
        print(f"Partial (≠5 rubrics): {partial_success_count} ({partial_success_count / processed_count * 100:.1f}%)")
        print(f"Errors: {error_count} ({error_count / processed_count * 100:.1f}%)")
        print(f"\nTotal time: {total_time / 60:.1f} minutes")
        print(f"Average time per item: {total_time / processed_count:.2f} seconds")
    print(f"\nOutput saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
