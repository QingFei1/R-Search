CUDA_VISIBLE_DEVICES=0 \
VLLM_RPC_TIMEOUT=100000 \
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --served-model-name Llama-3.2-3B-Instruct \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.11 \
    --enforce-eager \
    --port 8004 \
    --max-model-len 8192