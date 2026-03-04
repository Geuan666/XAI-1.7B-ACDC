# Repository Guidelines

## 代码结构树（重点）
```text
XAI-1.7B-ACDC/
├── AGENTS.md
├── pair/                              # 提示词与元数据（数据目录）
└── Automatic-Circuit-Discovery/
    ├── acdc/                          # ACDC 核心实现与任务逻辑
    │   ├── induction/ ioi/ greaterthan/ docstring/ tracr_task/
    ├── subnetwork_probing/            # 子网络探测与 transformer_lens 相关代码
    ├── tests/
    │   ├── acdc/
    │   └── subnetwork_probing/
    ├── experiments/                   # 复现实验脚本与 results/
    ├── notebooks/                     # 分析与演示
    ├── assets/                        # 图像与静态资源
    └── pyproject.toml                 # 依赖与工具配置
```

## 核心命令（在 `Automatic-Circuit-Discovery/` 下执行）
- `python -m pip install -e . --no-deps`
作用：以可编辑模式安装项目本体（不强制改动现有依赖）。
- `pytest -q -m "not slow" tests --maxfail=8`
作用：快速回归测试（默认推荐先跑这个）。
- `pytest -q -m slow tests`
作用：运行慢测，做完整验证时使用。
- `python acdc/main.py`
作用：运行 ACDC 主流程入口。
- `python experiments/launch_induction.py`
作用：启动常用 induction 实验脚本。

## 项目核心思路（ACDC）
- 目标：从 Transformer 中自动发现“真正起作用”的电路（关键节点与边）。
- 方法：比较 clean/corrupted 运行，按边做因果干预与打分，逐步剪枝。
- 输出：得到更稀疏、可解释的子图，用于分析具体任务（如 induction、IOI、tracr）。

## 运行环境说明
- 当前统一使用 `base` 环境（Python 3.10）。
- GPU 目标环境：`NVIDIA RTX 4090 24GB`。
- 目前依赖与项目已做兼容处理，按上述命令可直接运行与测试。
