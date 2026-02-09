import re
import json
import asyncio
from typing import Optional, Dict, Any
from openai import AsyncOpenAI

# ================= API 配置 =================
API_KEY = "EMPTY"  # vLLM使用EMPTY
BASE_URL = "http://localhost:6009/v1/"  # 根据实际情况修改
MODEL_NAME = 'Qwen3-8B-rm-ot'
API_TIMEOUT = 60.0  # API超时时间（秒）
RETRY_ATTEMPTS = 1  # 失败重试次数
RETRY_DELAY = 2  # 重试延迟（秒）

# 创建异步OpenAI客户端
client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=API_TIMEOUT)

# ================= System Prompt =================
SYSTEM_PROMPT = """You are an expert telecommunications engineer. You will evaluate whether an answer to a technical question is correct or incorrect. Please reason step by step, and put your final answer within \\boxed{}."""


def extract_score_from_response(response: str) -> Optional[int]:
    """从模型响应中提取分数 (0 或 1)，优先提取 \\boxed{} 格式"""
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
            r'"score"\s*:\s*([01])',  # "score": 0 或 1
            r'score\s*:\s*([01])',  # score: 0 或 1
            r'\bscore\s*=?\s*([01])\b',  # score = 0 或 1
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
            if 'correct' not in response_lower[
                                :response_lower.index('incorrect')] if 'incorrect' in response_lower else True:
                return 0
        if 'correct' in response_lower or 'accurate' in response_lower or 'true' in response_lower:
            return 1

        return None
    except Exception as e:
        print(f"Error extracting score: {e}")
        return None


def build_evaluation_prompt(question: str, answer: str) -> str:
    """构建评估提示词，匹配 JSON 数据格式"""
    return f"""Score Meaning:
- Score 1: The answer is CORRECT and fully addresses the question
- Score 0: The answer is INCORRECT or does not properly address the question

#### Question
{question}

#### Answer to Evaluate
{answer}"""


async def evaluate_answer_async(question: str, answer: str) -> Optional[int]:
    """异步评估答案"""
    for attempt in range(RETRY_ATTEMPTS):
        try:
            user_prompt = build_evaluation_prompt(question, answer)

            # 调用API
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

            # 提取分数
            score = extract_score_from_response(response_text)
            return score

        except Exception as e:
            print(f"【API Error】Attempt {attempt + 1}/{RETRY_ATTEMPTS}: {e}")
            if attempt < RETRY_ATTEMPTS - 1:
                await asyncio.sleep(RETRY_DELAY)
            else:
                print(f"【API Error】Failed after {RETRY_ATTEMPTS} attempts")
                return None


async def compute_score_async(question: str, answer_text: str) -> float:
    """异步计算分数"""
    api_score = await evaluate_answer_async(question, answer_text)
    if api_score is None:
        return 0.0
    return float(api_score)


def get_or_create_loop():
    """获取或创建事件循环，支持并行调用"""
    try:
        # 尝试获取当前事件循环
        loop = asyncio.get_running_loop()
        return loop
    except RuntimeError:
        # 如果没有运行中的事件循环，尝试获取或创建
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                raise RuntimeError("Event loop is closed")
            return loop
        except RuntimeError:
            # 创建新的事件循环
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop


def compute_score(data_source, extra_info, prompt_str, solution_str, ground_truth, method="strict", format_score=0.0,
                       score=1.0):
    """
    计算分数 - 使用API评估答案正确性（支持并行调用）

    Args:
        data_source: 数据源（应包含问题信息）
        extra_info: 额外信息（可能包含问题）
        solution_str: 模型生成的答案
        ground_truth: 真实答案
        method: 评估方法
        format_score: 格式分数（保留兼容性）
        score: 基础分数（保留兼容性）

    Returns:
        评估分数（0-1之间，或更高分数如果包含推理和工具调用）
    """
    # 提取问题
    question = prompt_str

    # 提取答案文本（去除标签）
    answer_text = solution_str
    # 移除推理标签，只保留内容
    answer_text = re.sub(r'<think>.*?</think>', '', answer_text, flags=re.DOTALL)
    # 移除answer标签，只保留内容
    answer_text = re.sub(r'<answer>(.*?)</answer>', r'\1', answer_text, flags=re.DOTALL)
    answer_text = answer_text.strip()

    # 使用API评估答案
    try:
        try:
            # 尝试获取运行中的事件循环
            loop = asyncio.get_running_loop()
            # 如果事件循环正在运行，使用 run_coroutine_threadsafe
            import concurrent.futures
            future = asyncio.run_coroutine_threadsafe(
                compute_score_async(question, answer_text), loop
            )
            api_score = future.result(timeout=API_TIMEOUT + 10)
        except RuntimeError:
            # 如果没有运行中的事件循环，创建新的事件循环并运行
            loop = get_or_create_loop()
            api_score = loop.run_until_complete(compute_score_async(question, answer_text))

        if api_score is None:
            # API调用失败，回退到格式检查
            api_score = 0.0
        else:
            # API评估成功，基础分数为api_score（0或1）
            score = api_score

        return score

    except Exception as e:
        print(f"【Error】API evaluation failed: {e}")
        # 出错时回退到原来的格式检查方式
        score = 0.0
        is_think = re.search(r'<think>(.*?)</think>', solution_str, re.DOTALL)
        if is_think and len(is_think.group(1)) > 10:
            score += 0.2
        return score