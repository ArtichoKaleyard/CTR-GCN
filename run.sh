#!/bin/bash
# CTR-GCN 训练入口 —— 注册模型 + 调用 Foundry skeleton train
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "用法: bash run.sh <config-name>"
    echo ""
    echo "可用配置:"
    for f in conf/experiments/ntu60_xsub*.yaml; do
        name=$(basename "$f" .yaml)
        echo "  $name"
    done
    exit 1
fi

CONFIG="conf/experiments/${1}.yaml"
if [ ! -f "$CONFIG" ]; then
    echo "配置文件不存在: $CONFIG"
    exit 1
fi

exec uv run python -c "
from foundry_entry import register_ctrgcn; register_ctrgcn()
from foundry.projects.skeleton.train import main
import sys; sys.argv = ['train', '--config', '${CONFIG}']
main()
"
