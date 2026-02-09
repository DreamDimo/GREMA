from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm import SamplingParams
from transformers import AutoTokenizer
import uvicorn
import os
import uuid
import time
import asyncio
import json

# === Configuration ===
# Use ModelScope
os.environ['VLLM_USE_MODELSCOPE'] = 'True'
# MODEL_PATH = '/mnt/public/wwj/zhaozq/exp1/model/Qwen/Qwen3-8B'
# MODEL_PATH = '/mnt/public/wwj/zhaozq/exp1/LLaMA-Factory/saves/qwen3-8b/full/sft_rl'
# MODEL_PATH = '/mnt/public/wwj/zhaozq/exp1/LLaMA-Factory/saves/qwen3-8b/full/sft_rm_boxed'
# MODEL_PATH = '/mnt/public/wwj/zhaozq/exp1/model/Qwen/Qwen3-8B-RL'
# MODEL_PATH = '/mnt/public/wwj/zhaozq/exp1/model/Qwen/Qwen3-8B-RL-listwise'
# MODEL_PATH = '/mnt/public/wwj/zhaozq/exp1/LLaMA-Factory/saves/qwen3-8b/full/sft_base'
# MODEL_PATH = '/mnt/public/wwj/zhaozq/exp1/model/Qwen/Qwen3-8B-RL–new'
# MODEL_PATH = '/mnt/public/wwj/zhaozq/exp1/LLaMA-Factory/saves/qwen3-8b/full/sft_rm_single'
# MODEL_PATH = '/mnt/public/wwj/zhaozq/exp1/model/Qwen/Qwen3-8B-RL–listwise-shortanswer'
# MODEL_PATH = '/mnt/public/wwj/zhaozq/exp1/LLaMA-Factory/saves/qwen3-8b/full/sft_base'
# MODEL_PATH = '/mnt/public/wwj/zhaozq/exp2/qwen3-32b/Qwen/Qwen3-32B'
# MODEL_PATH = '/mnt/public/wwj/zhaozq/exp1/LLaMA-Factory/saves/qwen3-8b/full/sft_rm_boxed_ot'
# MODEL_PATH = '/mnt/public/wwj/zhaozq/exp1/LLaMA-Factory/output/qwen3_lora_sft'
# MODEL_PATH = '/mnt/public/wwj/zhaozq/exp1/LLaMA-Factory/saves/qwen3-8b/full/sft_rm_boxed'
# MODEL_PATH = '/mnt/public/wwj/zhaozq/exp1/model/Qwen/Qwen3-8b-rl-constant'
# MODEL_PATH = '/mnt/public/wwj/zhaozq/exp1/model/Qwen/Qwen3-8b-rl-Qwen3-8b'
# MODEL_PATH = '/mnt/public/wwj/zhaozq/exp1/model/Qwen/sft_rm_single'
# MODEL_PATH = '/mnt/public/wwj/zhaozq/exp1/model/Qwen/Qwen3-8b-rm-dgen'
# MODEL_PATH = '/mnt/public/wwj/zhaozq/exp1/model/Qwen/Qwen3-8b-rl-ds-ot'
# MODEL_PATH = '/mnt/public/wwj/zhaozq/exp1/model/Qwen/Qwen3-8b-rm-ot-step5'
# MODEL_PATH = '/mnt/public/wwj/zhaozq/exp1/LLaMA-Factory/saves/qwen3-8b/full/sft_rm_boxed_medical'
# MODEL_PATH = '/mnt/public/wwj/zhaozq/exp1/LLaMA-Factory/saves/qwen3-8b/full/sft_rm_boxed_TeleQnA'
# MODEL_PATH = '/mnt/public/wwj/zhaozq/exp1/LLaMA-Factory/saves/qwen3-8b/full/sft_rm_boxed_TeleQnA_all'
# MODEL_PATH = '/mnt/public/wwj/zhaozq/exp1/LLaMA-Factory/saves/qwen3-8b/full/sft_rm_boxed_TeleQnA_all/checkpoint-4000'
MODEL_PATH = '/mnt/public/wwj/zhaozq/exp1/LLaMA-Factory/saves/qwen3-8b/full/sft_rm_boxed_TeleQnA_all'
# MODEL_PATH = '/mnt/public/wwj/zhaozq/exp1/model/Qwen/Qwen3-8b-rm-20260208'


app = FastAPI()

# Global variables
llm_engine = None
tokenizer = None


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if llm_engine is None:
        return {"status": "initializing"}
    return {
        "status": "healthy",
        "model": "Qwen3-8B",
        "tensor_parallel_size": 1
    }


@app.on_event("startup")
async def startup_event():
    """
    Initialize the engine and tokenizer on app startup.
    This ensures everything is ready before the first request.
    """
    global llm_engine, tokenizer

    print("Loading Tokenizer...")
    # Tokenizer for manual template application
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False, trust_remote_code=True)

    print("Loading vLLM Async Engine...")
    # Define arguments for AsyncLLMEngine with optimized concurrent settings
    engine_args = AsyncEngineArgs(
        model=MODEL_PATH,
        tokenizer=MODEL_PATH,
        max_model_len=16384,
        trust_remote_code=True,
        tensor_parallel_size=1,  # Single GPU
        gpu_memory_utilization=0.5,  # Increased for single GPU
        enable_chunked_prefill=True,  # Better concurrent request handling
    )

    # Create the Async Engine
    llm_engine = AsyncLLMEngine.from_engine_args(engine_args)
    print("Engine loaded successfully. Ready for concurrent requests.")


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """
    OpenAI-compatible endpoint using vLLM AsyncBackend with concurrent support.
    """
    # 1. Parse request
    json_data = await request.json()
    messages = json_data.get("messages", [])
    temperature = json_data.get("temperature", 0)
    top_p = json_data.get("top_p", 0.8)
    stream = json_data.get("stream", False)

    # Generate a unique request ID (vLLM needs this to track the stream)
    request_id = str(uuid.uuid4())
    print(f"[{request_id}] New request received")

    # === Logic: Apply Template ===
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    # === Logic: Sampling Parameters ===
    stop_token_ids = [151645, 151643]
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=20,
        min_p=0,
        max_tokens=16384,
        stop_token_ids=stop_token_ids
    )

    # 2. Generate (Async) - vLLM automatically handles concurrent requests
    results_generator = llm_engine.generate(prompt_text, sampling_params, request_id)

    # Handle streaming vs non-streaming
    if stream:
        # Streaming response for better concurrency
        async def stream_results():
            previous_text = ""
            async for request_output in results_generator:
                if await request.is_disconnected():
                    await llm_engine.abort(request_id)
                    break

                current_text = request_output.outputs[0].text
                delta_text = current_text[len(previous_text):]
                previous_text = current_text

                chunk = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "model": "Qwen3-8B",
                    "created": int(time.time()),
                    "choices": [{
                        "index": 0,
                        "delta": {"content": delta_text},
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(chunk)}\n\n"

            # Final chunk
            final_chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "model": "Qwen3-8B",
                "created": int(time.time()),
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }]
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"
            print(f"[{request_id}] Request completed (streaming)")

        return StreamingResponse(stream_results(), media_type="text/event-stream")

    else:
        # Non-streaming mode: wait for final result
        final_output = None
        async for request_output in results_generator:
            final_output = request_output
            if await request.is_disconnected():
                await llm_engine.abort(request_id)
                return {"error": "Client disconnected"}

        # Extract the generated text from the final output
        generated_text = final_output.outputs[0].text

        print(f"[{request_id}] Request completed (non-streaming)")

        # 3. Format response
        return {
            "id": request_id,
            "object": "chat.completion",
            "model": "Qwen3-8B",
            "created": int(time.time()),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": generated_text,
                    },
                    "finish_reason": final_output.outputs[0].finish_reason
                }
            ],
            "usage": {
                "prompt_tokens": len(final_output.prompt_token_ids),
                "completion_tokens": len(final_output.outputs[0].token_ids),
                "total_tokens": len(final_output.prompt_token_ids) + len(final_output.outputs[0].token_ids)
            }
        }


if __name__ == "__main__":
    print("Starting API Server on port 6009...")
    # Note: We don't initialize LLM here anymore, it's done in the @app.on_event("startup")
    uvicorn.run(app, host='0.0.0.0', port=6009, workers=1)
