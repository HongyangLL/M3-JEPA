# Check required environment variables
: "${MLP_WORKER_0_HOST:?Missing MLP_WORKER_0_HOST}"
: "${MLP_WORKER_0_PORT:?Missing MLP_WORKER_0_PORT}"
: "${MLP_ROLE_INDEX:?Missing MLP_ROLE_INDEX}"
: "${MLP_WORKER_NUM:?Missing MLP_WORKER_NUM}"
: "${MLP_WORKER_GPU:?Missing MLP_WORKER_GPU}"

# Run distributed training
torchrun --nnodes="$MLP_WORKER_NUM" \
         --node_rank="$MLP_ROLE_INDEX" \
         --nproc_per_node="$MLP_WORKER_GPU" \
         --master_addr="$MLP_WORKER_0_HOST" \
         --master_port="$MLP_WORKER_0_PORT" \
         main.py --fname configs/mlp_moe_k4.yaml
