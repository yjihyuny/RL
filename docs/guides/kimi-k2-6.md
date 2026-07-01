# Kimi K2.6 Megatron Smoke Guide

This guide documents the current NeMo RL smoke-test path for `moonshotai/Kimi-K2.6` with the Megatron backend and vLLM rollout backend.

Kimi K2.6 is a very large MLA/MoE model. The commands below are intended to validate configuration, tokenizer compatibility, rollout plumbing, and one-step training paths. They are not full training recipes.

## Scope

The current path covers:

- Hugging Face tokenizer and chat-template loading with `trust_remote_code=True`
- Megatron Bridge provider/config resolution through the existing Kimi/Kimi-VL path
- SFT smoke configuration with the Megatron backend
- GRPO smoke configuration with Megatron policy workers and vLLM rollout workers
- vLLM rollout smoke checks for prompt formatting, `max_model_len`, greedy generation, and logprobs

The current path does not claim:

- Full 1T-parameter training validation
- Native INT4 training support
- Multimodal MoonViT training support
- 256K context validation
- Throughput-optimized TP/PP/EP/CP settings

## Validated Results (HSG GB200)

The dummy-start live-refit path (vLLM `load_format=dummy`, Megatron TP8/PP8/EP8/ETP1
policy → vLLM TP8/EP64 generation, selected full model-weight refit) has been run on
HSG GB200 for GRPO smoke:

| Run | Steps | Reward | Notes |
|-----|-------|--------|-------|
| OpenMath `new512` (hard) | 10/10 | avg 0.2062 (33/160 correct), 9/10 nonzero-reward steps | Exit 0, ~65 min |
| OpenMath `new512`, 30-step + TensorBoard | 21/30 | avg 0.1964 (66/336) | Stopped by short QoS walltime, not a refit/model crash — proved stability past 10 steps |
| Easy-math `new128` | 10/10 | avg 0.869 (139/160 correct) | Confirms nonzero advantages/losses, backward, optimizer step, and repeated refit + generation |

These validate config/tokenizer resolution, rollout plumbing, repeated live refit, and
end-to-end training steps. They are smoke runs, not full training recipes.

## Environment

Kimi K2.6 uses `trust_remote_code=True` for the tokenizer/model path. Use a trusted revision when running in production.

On bare-metal or custom containers, CUDA extension builds may need explicit CUDA, CCCL, and cuDNN paths. The following block was sufficient for the smoke-test environment:

```bash
source "$HOME/.local/bin/env"

export CUDA_HOME=/usr/local/cuda
export CUDA_PATH=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:/home/nvidia/repo/RL/.venv/lib/python3.13/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH"
export CPATH="$CUDA_HOME/targets/x86_64-linux/include/cccl:/home/nvidia/repo/RL/.venv/lib/python3.13/site-packages/nvidia/cudnn/include:$CPATH"
export CPLUS_INCLUDE_PATH="$CUDA_HOME/targets/x86_64-linux/include/cccl:/home/nvidia/repo/RL/.venv/lib/python3.13/site-packages/nvidia/cudnn/include:$CPLUS_INCLUDE_PATH"
```

If you see errors such as `cuda/std/utility` or `cuda/std/tuple` not found, the CCCL include path is missing. If you see `cudnn.h` or cuDNN runtime loading errors, the cuDNN include or library path is missing.

For Megatron workers, `TORCH_CUDA_ARCH_LIST` must also be set for the GPUs in the environment. For example:

```bash
export TORCH_CUDA_ARCH_LIST="9.0"
```

Use the architecture value appropriate for the target GPU.

## Hardware

The 2-GPU diagnostic node used during initial validation had approximately 95 GiB per GPU. It was enough to validate tokenizer/config/data paths, but not enough to complete SFT, GRPO, or vLLM generation with Kimi K2.6.

Recommended validation targets:

| Purpose | Hardware |
|---|---|
| Tokenizer/config checks | CPU or small GPU |
| Substitute-model plumbing | Small GPU node |
| Real Kimi K2.6 smoke | 1 node with 8x H100 80GB or equivalent |
| Lower-risk real Kimi validation | 8x H200, B200, B300, or other larger-memory GPUs |
| 256K context or throughput tuning | Larger-memory and/or multi-node resources |

For real Kimi smoke runs, use 8-way tensor/expert parallelism as the starting point. The checked-in configs may be temporarily set to 2-GPU diagnostic values; override them or edit them before running on an 8-GPU node.

## Prompt And Tokenizer Artifact

Create a tokenizer/prompt artifact before rollout testing:

```bash
uv run python scripts/phase10_kimi26_prompt_artifact.py \
  --model moonshotai/Kimi-K2.6 \
  --output phase10_kimi26_prompt_artifact.json
```

This verifies that the Hugging Face tokenizer path works and records the exact prompt text and token IDs. A successful run prints output similar to:

```text
tokenizer_class=TikTokenTokenizer
num_prompt_tokens=29
prompt_text='<|im_user|>user<|im_middle|>Solve: ... <think>'
```

The vLLM smoke script uses this artifact to check that vLLM receives the same `prompt_token_ids` as the HF tokenizer path.

## SFT Smoke

Use:

```bash
uv run python examples/run_sft.py \
  --config examples/configs/sft_kimi_k2_6_megatron.yaml
```

For an 8-GPU smoke target, use overrides equivalent to:

```bash
uv run python examples/run_sft.py \
  --config examples/configs/sft_kimi_k2_6_megatron.yaml \
  cluster.gpus_per_node=8 \
  policy.train_global_batch_size=8 \
  policy.megatron_cfg.tensor_model_parallel_size=8 \
  policy.megatron_cfg.expert_model_parallel_size=8
```

The smoke config intentionally uses:

- `max_num_steps: 1`
- `train_micro_batch_size: 1`
- `max_total_sequence_length: 4096`
- checkpointing disabled
- validation disabled
- OpenMathInstruct-2 data

On a 2x95 GiB node, SFT reached data loading, Ray initialization, Megatron worker initialization, and Kimi model construction, then failed with CUDA OOM during MoE layer construction. Treat that result as a resource limit, not as proof of a config or tokenizer failure.

## GRPO Smoke

Use:

```bash
uv run python examples/run_grpo.py \
  --config examples/configs/grpo_kimi_k2_6_megatron.yaml
```

For an 8-GPU smoke target, use overrides equivalent to:

```bash
uv run python examples/run_grpo.py \
  --config examples/configs/grpo_kimi_k2_6_megatron.yaml \
  cluster.gpus_per_node=8 \
  policy.train_global_batch_size=8 \
  policy.megatron_cfg.tensor_model_parallel_size=8 \
  policy.megatron_cfg.expert_model_parallel_size=8 \
  policy.generation.vllm_cfg.tensor_parallel_size=8 \
  policy.generation.vllm_cfg.gpu_memory_utilization=0.5
```

The smoke config intentionally uses:

- `num_prompts_per_step: 1`
- `num_generations_per_prompt: 2`
- `max_num_steps: 1`
- `max_new_tokens: 64`
- `generation_batch_size: 1`
- `logprob_batch_size: 1`
- validation and checkpointing disabled

On a 2x95 GiB node, GRPO reached config loading, dataset setup, vLLM worker environment setup, and Kimi architecture resolution, then failed with CUDA OOM during vLLM model construction. Full rollout/training validation requires larger hardware.

## vLLM Rollout Smoke

The rollout smoke script checks:

- vLLM loads Kimi K2.6 config/tokenizer with `trust_remote_code=True`
- `max_model_len` is the intended smoke value
- vLLM prompt token IDs match the HF tokenizer artifact
- greedy generation works
- prompt and generation logprobs are returned

Run on the real 8-GPU validation node:

```bash
uv run --extra vllm python scripts/phase10_kimi26_vllm_smoke.py \
  --model moonshotai/Kimi-K2.6 \
  --prompt-artifact phase10_kimi26_prompt_artifact.json \
  --output phase10_kimi26_vllm_real_smoke.json \
  --tensor-parallel-size 8 \
  --max-model-len 4096 \
  --max-new-tokens 32 \
  --gpu-memory-utilization 0.5 \
  --load-format auto
```

If memory is tight, try:

```bash
uv run --extra vllm python scripts/phase10_kimi26_vllm_smoke.py \
  --model moonshotai/Kimi-K2.6 \
  --prompt-artifact phase10_kimi26_prompt_artifact.json \
  --output phase10_kimi26_vllm_real_smoke_2048.json \
  --tensor-parallel-size 8 \
  --max-model-len 2048 \
  --max-new-tokens 16 \
  --gpu-memory-utilization 0.4 \
  --load-format auto \
  --enforce-eager
```

The dummy-weight path can validate config/model construction without downloading real weights, but it still instantiates full Kimi K2.6 module shapes:

```bash
uv run --extra vllm python scripts/phase10_kimi26_vllm_smoke.py \
  --model moonshotai/Kimi-K2.6 \
  --prompt-artifact phase10_kimi26_prompt_artifact.json \
  --output phase10_kimi26_vllm_dummy_smoke.json \
  --tensor-parallel-size 8 \
  --max-model-len 4096 \
  --max-new-tokens 32 \
  --gpu-memory-utilization 0.5 \
  --load-format dummy
```

On a 2x95 GiB node, the dummy path resolved `KimiK25ForConditionalGeneration`, accepted `max_model_len=4096`, initialized TP/NCCL, selected the MLA backend, and selected the WNA16 Marlin MoE path, then failed with CUDA OOM before generation/logprobs.

## Checkpoint Notes

The inspected Hugging Face repository exposes BF16 safetensors metadata with 64 shards and approximately 554 GiB of weight files. Ensure the target machine has enough local or shared storage for HF cache and Megatron converted checkpoints before running full import/loading tests.

Native INT4 training is not validated. Use the BF16 path for training/checkpoint conversion unless a future validation explicitly adds INT4/QAT support.

## Known Limitations

- 2x95 GiB GPUs are insufficient for full Kimi K2.6 SFT, GRPO, or vLLM generation smoke.
- Full SFT/GRPO acceptance is backlogged to 8-GPU high-memory hardware.
- Full checkpoint import/loading was not completed on the 2-GPU diagnostic node.
- Native INT4 training is not validated.
- MoonViT/multimodal training is not validated.
- 256K context validation requires larger resources and is not covered by the 4K smoke configs.
- vLLM behavior depends on compatible vLLM and `transformers` versions and requires `trust_remote_code=True`.

## Minimum Result To Record

For any Kimi K2.6 smoke run, record:

```text
hardware:
command:
config:
dtype:
max_model_len:
tensor_parallel_size:
expert_parallel_size:
trust_remote_code:
result:
failure, if any:
```

Do not mark Kimi K2.6 training support as fully validated from config-only or OOM-blocked runs. Mark those as partial validation with the exact point reached.
