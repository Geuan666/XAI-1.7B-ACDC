# Overnight Round 6 Log (2026-03-09)

## Hypothesis
在更强分布偏移下，full-core 可能仍有效，但内部关键组（head/MLP/backbone）的必要性排序可能发生重排。

## Experimental Design
- 新增 `evaluate_toolcall_shift_robustness.py`，对 clean/corrupt 同步注入中性上下文后重新做因果评估。
- 主要模式（v3）：
  - `orig`
  - `user_pad_short`（自然语言注入）
  - `user_json_pad`（结构化 JSON 注入）
- 评估组：
  - `full_core`
  - `stable_necessary_backbone`
  - `stable_but_weak_or_redundant`
  - `all_heads`
  - `all_mlps`
- 在三模式共同可用的 `n=119` 样本上做 bootstrap 对比（1000 次）。

## Results
1. `full_core` 在 shift 下仍保持高水平（甚至更高）：
- `orig`: suff/nec 中位约 `0.90/0.92`
- `user_json_pad`: `0.97/0.95`
- `user_pad_short`: `1.09/1.07`

2. 但内部机制显著变化（共同样本 n=119）：
- `all_heads` 组内必要性下降（ratio 与 abs 都下降）：
  - `user_json_pad - orig` 的 `drop_full_nec` 95% CI 约 `[-0.220, -0.186]`
- `all_mlps` 组内必要性显著上升：
  - `user_json_pad - orig` 的 `drop_full_nec` 95% CI 约 `[+0.194, +0.250]`
  - 绝对量 `drop_full_nec_abs` 95% CI 约 `[+1.50, +2.06]`

3. `stable_necessary_backbone` 在 shift 下仍为正必要，但贡献幅度降低：
- `user_json_pad - orig` 的 `drop_full_nec` 95% CI 约 `[-0.178, -0.157]`

## Interpretation
- 新证据支持“功能稳定、机制可重配”结论：
  - 外部行为层面（full_core suff/nec）稳健；
  - 内部因果路由会随分布偏移从 head 主导转向更高 MLP 参与。
- 这说明我们的最终叙事必须区分：
  - `是否有效`（function-level robustness）
  - `如何实现`（mechanism-level allocation）

## Decision
- 将“机制重配性”纳入正式结论，而不是只报告单一静态主干。
- 下一轮建议：对 shift 样本重跑 node-level ablation，验证具体迁移到哪些 MLP/head 节点。
