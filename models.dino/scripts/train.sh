#!/bin/bash

# ECG Asymmetric DINO Training Script
# Usage: ./scripts/train.sh [pretrain|finetune] [OPTIONS]

set -e  # Exit on error

# Default values
PHASE="pretrain"
CONFIG=""
RESUME=""
GPUS=1

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        pretrain|finetune)
            PHASE=$1
            shift
            ;;
        -c|--config)
            CONFIG=$2
            shift 2
            ;;
        -r|--resume)
            RESUME=$2
            shift 2
            ;;
        -g|--gpus)
            GPUS=$2
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set default config if not provided
if [ -z "$CONFIG" ]; then
    CONFIG="configs/${PHASE}.yaml"
fi

# Build command
CMD="poetry run python train/train.py --config $CONFIG"
if [ -n "$RESUME" ]; then
    CMD="$CMD --resume $RESUME"
fi

# Launch training
if [ $GPUS -gt 1 ]; then
    poetry run torchrun --nproc_per_node=$GPUS $CMD
else
    $CMD
fi
