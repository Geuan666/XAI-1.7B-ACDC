# Overnight Round 7 Log (2026-03-09)

## Hypothesis
在 `user_json_pad` shift 下，full-core 功能可能稳定，但必要性主干会从原先 head 集群向部分 MLP 节点迁移。

## Experimental Design
- 在 shift 样本上完成 node-level ablation（`augment_mode=user_json_pad`, `n=139`）。
- 输出 node summary + bootstrap CI，并与 orig 进行配对 delta bootstrap。
- 新增可复现脚本：
  - `experiments/analyze_toolcall_node_reallocation.py`
- 关键指标：`drop_full_nec`（节点在 full-circuit 中的必要性增量）。

## Results
1. 功能层面仍强：`full_core` 在 user_json 下继续保持高 suff/nec（与 Round 6 一致）。

2. 节点级机制重配显著（user_json - orig, n=139）：
- `MLP27`: delta median `+0.1408`, 95% CI `[+0.1275, +0.1546]`
- `MLP20`: delta median `+0.1315`, 95% CI `[+0.1165, +0.1358]`
- `L21H12`: delta median `-0.0769`, 95% CI `[-0.0827, -0.0722]`
- `L24H6`: delta median `-0.0492`, 95% CI `[-0.0593, -0.0423]`

3. 分层结构重写（user_json stratification）：
- `stable_necessary_backbone`: `MLP27`, `L23H6`
- `stable_but_weak_or_redundant`: `L17H8`, `L21H1`, `L21H12`, `L20H5`, `MLP22`, `L24H6`
- `unstable_but_necessary`: `MLP20`

## Interpretation
- “电路有效”与“电路实现路径固定”必须分开叙事。
- 在 user-side JSON shift 下，关键必要性由部分 head 转向高层 MLP（尤其 `MLP27/MLP20`），是机制层非平稳性的直接证据。

## Decision
- 将节点重分配图（paired bootstrap CI）作为主证据图之一。
- 下一轮必须加入强对照：同 payload 注入 system 位置，区分“payload 结构效应”与“user 通道特异效应”。
