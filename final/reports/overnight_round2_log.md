# Overnight Round 2 Log (2026-03-08)

## Hypothesis
Core 结构若真实存在，应在聚合超参数变化下保持较高稳定性；若不稳定，则“共性电路”结论需降级。

## Experimental Design
- 对聚合参数做 27 组 sweep：
  - `gap_min ∈ {0.4, 0.5, 0.6}`
  - `core_node_th ∈ {0.45, 0.50, 0.55}`
  - `core_edge_th ∈ {0.30, 0.35, 0.40}`
- 每组重建 global core，计算跨组节点/边 Jaccard 与频率。

## Results
- 27 组变体的 pairwise Jaccard：
  - node: mean `0.870`, median `0.909`
  - edge: mean `0.853`, median `0.852`
- 稳定节点（频率 >= 0.70）共 8 个：
  - `L17H8,L20H5,L21H1,L21H12,MLP22,L23H6,L24H6,MLP27`
- 节点频率 = 1.0（所有 27 组都出现）的 8 个节点同上。

## Interpretation
- “共性结构存在”获得更强支持（超参数扰动下结构稳健）。
- 但“结构稳定”不等于“全部节点都必要”：与 Round 1 节点必要性结果对照，`MLP22/MLP27` 结构上稳定但必要性偏弱或负，提示它们更像冗余/并行支路。

## Decision For Next Round
- 进入“结构稳定 vs 因果必要”双轴结论框架：
  - 轴 1：结构频率（稳定出现）
  - 轴 2：drop-necessity（不可替代性）
- 下一轮优先构建二维分层图并形成论文可用结论分级：
  - `Stable + Necessary`（主干）
  - `Stable + Weak`（辅助/冗余）
  - `Unstable + Weak`（边缘）
