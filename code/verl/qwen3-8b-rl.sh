#!/usr/bin/env bash
set -xeuo pipefail

## !!!!!!!important!!!!!!
## set the following environment variables on all your nodes
# env_vars:
#   CUDA_DEVICE_MAX_CONNECTIONS: "1"
#   NCCL_NVLS_ENABLE: "0"
#   VLLM_USE_V1: 1
# install mbridge=0.1.13 on all your node with the following command:
# pip3 install git+https://github.com/ISEEKYAN/mbridge




enable_overlong_buffer=True
overlong_buffer_len=$((1024 * 1))
overlong_penalty_factor=1.0

loss_agg_mode="token-mean"

train_prompt_bsz=${TRAIN_BS:-32}
n_resp_per_prompt=8
train_prompt_mini_bsz=16

NNODES=${NNODES:-1}

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
val_top_p=0.7
# Performance Related Parameter (3-GPU setup)
use_dynamic_bsz=False  # Disable dynamic batch size for stability
actor_ppo_max_token_len=$((2048))
infer_ppo_max_token_len=$((2048))
offload=True
OPTIM_OFFLOAD=${OPTIM_OFFLOAD:-False}
gen_tp=1  # 3-GPU setup: use 1 for rollout, 3 for training (FSDP)
train_tp=${TP:-1}  # FSDP doesn't need TP
train_pp=${PP:-1}  # No pipeline parallel

EP=${EP:-1}  # No expert parallelism
ETP=1
CP=1
optimizer_offload_fraction=${OFFLOAD_FRACTION:-0.5}
last_layer=${LAST_LAYER:-10}

project_name='verl-qwen3'
exp_name="qwen3-8b-rm-2gpu-${NNODES}"
CKPTS_DIR="/mnt/public/wwj/zhaozq/exp1/verl/huawei/checkpoint/Qwen3-8B-rm-ot"

# TODO: support cuda graph for rollout by setting the following config
    # actor_rollout_ref.rollout.cudagraph_capture_sizes=[1,2,4,8,16,32]
    # actor_rollout_ref.rollout.enforce_eager=False

python3 -m verl.trainer.main_ppo \
    data.train_files="/mnt/public/wwj/zhaozq/exp1/dataset/TeleQnA/parquet_rl/TeleQnA_train_grpo_format_modified.parquet" \
    data.val_files="/mnt/public/wwj/zhaozq/exp1/dataset/TeleQnA/parquet_rl/TeleQnA_test_grpo_format_10percent_modified.parquet" \
    custom_reward_function.path=/mnt/public/wwj/zhaozq/exp1/verl/huawei/experiments/code/TeleQnA_reward.py \
    custom_reward_function.name=compute_score \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.train_batch_size=12 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.enforce_eager=True \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=2048 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=2048 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=2048 \
    actor_rollout_ref.model.path="/mnt/public/wwj/zhaozq/exp1/model/Qwen/Qwen3-8B" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=12 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.optim.clip_grad=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.max_num_batched_tokens=2048 \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.max_model_len=2048 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=sync \
    actor_rollout_ref.rollout.ignore_eos=False \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.enforce_eager=true\
    actor_rollout_ref.nccl_timeout=1200 \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    reward_model.reward_manager=naive \
    trainer.logger=['console','wandb'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=3 \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=False \
    trainer.test_freq=100 \
    trainer.save_freq=100 \
    trainer.total_epochs=1 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    trainer.log_val_generations=10 $@