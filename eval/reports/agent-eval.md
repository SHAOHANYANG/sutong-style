主臂是 agent-balanced-k2-followup，对照是 retrieval-balanced-k2（同样 59 条，逐条配对，agent − 对照）。规则见 SPEC 4.8。
其余各臂是探索性的，不替换 README 里主臂的那一行。style_win_rate 本轮不跑。

校验器与评估用的是同一套规则：agent 按「违规最少」挑选最终版本，而违规就是从计算 entity_recall、numeral_recall、hallucination_rate 的同一份抽取结果里导出的。这三项的改善在一定程度上是构造使然，不能单独作为事实漂移被治好的证据；规则抓不到的语义漂移 agent 同样看不见。必须连同「没有优化的」那张表一起读。

人工抽查：pending（由仓库所有者进行，不得由模型代填）。

**主臂 − 对照（retrieval-balanced-k2）：agent 直接优化的指标**

| 指标 | 均值差 | 95% 区间 | 结论 |
|---|---|---|---|
| entity_recall | 0.000000 | [0.000000, 0.000000] | 未检出差异 |
| numeral_recall | 0.028249 | [0.005650, 0.059322] | 改善 |
| hallucination_rate | 0.000000 | [0.000000, 0.000000] | 未检出差异 |

**主臂 − 对照（retrieval-balanced-k2）：agent 没有优化的指标**

| 指标 | 均值差 | 95% 区间 | 结论 |
|---|---|---|---|
| profile_gap_per_case | -0.000471 | [-0.009576, 0.007578] | 未检出差异 |
| copy_ratio | 0.008086 | [0.001485, 0.017025] | 只报告 |
| style_distance | 0.003087 | [-0.008065, 0.015029] | 只报告 |
| length_ratio | 0.002629 | [-0.000865, 0.007245] | 只报告 |
| cjk_numeral_z | -0.019791 | [-0.059373, 0.000000] | 只报告 |
| profile_gap | -0.003420 | — | 只报点估计 |
| cjk_numeral_ratio_profile_mean_delta | agent 0.176430，基线 0.196221 | — | 只报告 |

**主臂 − k0（累计）（retrieval-k0）：agent 直接优化的指标**

| 指标 | 均值差 | 95% 区间 | 结论 |
|---|---|---|---|
| entity_recall | 0.000000 | [0.000000, 0.000000] | 未检出差异 |
| numeral_recall | 0.058757 | [0.019774, 0.101695] | 改善 |
| hallucination_rate | -0.009887 | [-0.025424, 0.000000] | 未检出差异 |

**主臂 − k0（累计）（retrieval-k0）：agent 没有优化的指标**

| 指标 | 均值差 | 95% 区间 | 结论 |
|---|---|---|---|
| profile_gap_per_case | -0.029200 | [-0.061495, 0.003069] | 未检出差异 |
| copy_ratio | 0.040846 | [0.026105, 0.057235] | 只报告 |
| style_distance | 0.010097 | [-0.049270, 0.069199] | 只报告 |
| length_ratio | 0.031290 | [0.004571, 0.074011] | 只报告 |
| cjk_numeral_z | 0.181228 | [0.005223, 0.474460] | 只报告 |
| profile_gap | 0.007051 | — | 只报点估计 |
| cjk_numeral_ratio_profile_mean_delta | agent 0.176430，基线 -0.004799 | — | 只报告 |

| 组合 | 角色 | style_distance | profile_gap | profile_gap_per_case | entity_recall | numeral_recall | hallucination_rate | copy_ratio | 字数比 | cjk_numeral_ratio 逐维均值差 |
|---|---|---|---|---|---|---|---|---|---|---|
| agent-balanced-k2-followup | 主臂 | 0.932609 | 0.210867 | 0.484792 | 1.000000 | 0.919774 | 0.004237 | 0.750093 | 1.036982 | 0.176430 |
| agent-balanced-k2-restate | 探索性 | 0.936193 | 0.211124 | 0.521786 | 1.000000 | 0.985876 | 0.004237 | 0.735459 | 1.077686 | 0.199794 |
| agent-k0-followup | 探索性 | 0.918620 | 0.199842 | 0.514732 | 1.000000 | 0.875141 | 0.014124 | 0.712926 | 1.010783 | -0.004799 |
| agent-k0-restate | 探索性 | 0.921601 | 0.198957 | 0.539434 | 1.000000 | 0.940113 | 0.014124 | 0.704787 | 1.042600 | 0.005207 |
| retrieval-balanced-k2 | 对照 | 0.929522 | 0.214287 | 0.485262 | 1.000000 | 0.891525 | 0.004237 | 0.742007 | 1.034353 | 0.196221 |
| retrieval-k0 | 对照 | 0.922513 | 0.203816 | 0.513992 | 1.000000 | 0.861017 | 0.014124 | 0.709247 | 1.005692 | -0.004799 |

| 臂 | 1 轮 | 2 轮 | 3 轮 | accepted | max_rounds | recursion_limit | fallback | 选中第 1 轮 | 选中第 2 轮 | 选中第 3 轮 | 未选中 | 第一轮有违规的样本 | 最终输出不同于基线的样本 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| agent-balanced-k2-followup | 46 | 0 | 13 | 51 | 8 | 0 | 0 | 51 | 0 | 8 | 0 | 13 | 8 |
| agent-balanced-k2-restate | 46 | 9 | 4 | 56 | 3 | 0 | 0 | 48 | 10 | 1 | 0 | 13 | 11 |
| agent-k0-followup | 41 | 2 | 16 | 44 | 15 | 0 | 0 | 54 | 4 | 1 | 0 | 18 | 5 |
| agent-k0-restate | 41 | 10 | 8 | 51 | 8 | 0 | 0 | 47 | 12 | 0 | 0 | 18 | 12 |

| 臂 | 违规类型 | 第一轮条数 | 最终条数 | 修掉 | 新引入 | 第一轮涉及样本 | 最终涉及样本 |
|---|---|---|---|---|---|---|---|
| agent-balanced-k2-followup | entity_missing | 0 | 0 | 0 | 0 | 0 | 0 |
| agent-balanced-k2-followup | numeral_missing | 11 | 7 | 4 | 0 | 10 | 6 |
| agent-balanced-k2-followup | entity_hallucination | 1 | 1 | 0 | 0 | 1 | 1 |
| agent-balanced-k2-followup | title | 3 | 2 | 1 | 0 | 2 | 1 |
| agent-balanced-k2-restate | entity_missing | 0 | 0 | 0 | 0 | 0 | 0 |
| agent-balanced-k2-restate | numeral_missing | 11 | 2 | 9 | 0 | 10 | 2 |
| agent-balanced-k2-restate | entity_hallucination | 1 | 1 | 0 | 0 | 1 | 1 |
| agent-balanced-k2-restate | title | 3 | 0 | 3 | 0 | 2 | 0 |
| agent-k0-followup | entity_missing | 0 | 0 | 0 | 0 | 0 | 0 |
| agent-k0-followup | numeral_missing | 14 | 12 | 2 | 0 | 14 | 12 |
| agent-k0-followup | entity_hallucination | 3 | 3 | 0 | 0 | 3 | 3 |
| agent-k0-followup | title | 3 | 2 | 1 | 0 | 2 | 1 |
| agent-k0-restate | entity_missing | 0 | 0 | 0 | 0 | 0 | 0 |
| agent-k0-restate | numeral_missing | 14 | 6 | 8 | 0 | 14 | 6 |
| agent-k0-restate | entity_hallucination | 3 | 3 | 0 | 0 | 3 | 3 |
| agent-k0-restate | title | 3 | 0 | 3 | 0 | 2 | 0 |

兜底样本：无。

**反馈回显**（SPEC 4.8 的事后诊断，不在那一轮的预注册里；4.9 起是预注册的报告项）

修订反馈里用「」引用了缺失的片段。输出若把反馈抄回去，片段就出现在输出里，校验器判为已修复。下表数的是仍含反馈引导语或「片段」的输出；「被防护切过的轮」只在打开回显防护时非零，切掉之后的文本不再计入前几列。

| 臂 | 最终输出改动的样本 | 其中最终输出含反馈回显 | 修订轮含回显 | 第 2 轮含回显 | 被防护切过的轮 |
|---|---|---|---|---|---|
| agent-balanced-k2-followup | 8 | 0 | 6 / 26 | 6 / 13 | 0 |
| agent-balanced-k2-restate | 11 | 10 | 12 / 17 | 10 / 13 | 0 |
| agent-k0-followup | 5 | 1 | 5 / 34 | 5 / 18 | 0 |
| agent-k0-restate | 12 | 9 | 13 / 26 | 11 / 18 | 0 |

**探索臂（各自对第一轮所取的文件，未做多重比较校正）**

| 臂 | 基线 | 指标 | 均值差 | 95% 区间 | 结论 |
|---|---|---|---|---|---|
| agent-balanced-k2-restate | retrieval-balanced-k2 | entity_recall | 0.000000 | [0.000000, 0.000000] | 未检出差异 |
| agent-balanced-k2-restate | retrieval-balanced-k2 | numeral_recall | 0.094350 | [0.033898, 0.168927] | 改善 |
| agent-balanced-k2-restate | retrieval-balanced-k2 | hallucination_rate | 0.000000 | [0.000000, 0.000000] | 未检出差异 |
| agent-balanced-k2-restate | retrieval-balanced-k2 | profile_gap_per_case | 0.036523 | [0.015578, 0.060458] | 恶化 |
| agent-balanced-k2-restate | retrieval-balanced-k2 | copy_ratio | -0.006548 | [-0.015568, 0.002032] | 只报告 |
| agent-balanced-k2-restate | retrieval-balanced-k2 | style_distance | 0.006671 | [-0.011913, 0.030375] | 只报告 |
| agent-balanced-k2-restate | retrieval-balanced-k2 | length_ratio | 0.043333 | [0.020108, 0.069513] | 只报告 |
| agent-balanced-k2-restate | retrieval-balanced-k2 | cjk_numeral_z | 0.003573 | [0.000000, 0.010720] | 只报告 |
| agent-k0-followup | retrieval-k0 | entity_recall | 0.000000 | [0.000000, 0.000000] | 未检出差异 |
| agent-k0-followup | retrieval-k0 | numeral_recall | 0.014124 | [0.000000, 0.036723] | 未检出差异 |
| agent-k0-followup | retrieval-k0 | hallucination_rate | 0.000000 | [0.000000, 0.000000] | 未检出差异 |
| agent-k0-followup | retrieval-k0 | profile_gap_per_case | 0.000740 | [-0.003741, 0.006685] | 未检出差异 |
| agent-k0-followup | retrieval-k0 | copy_ratio | 0.003680 | [-0.001506, 0.011026] | 只报告 |
| agent-k0-followup | retrieval-k0 | style_distance | -0.003893 | [-0.009978, 0.000051] | 只报告 |
| agent-k0-followup | retrieval-k0 | length_ratio | 0.005091 | [0.000323, 0.013276] | 只报告 |
| agent-k0-followup | retrieval-k0 | cjk_numeral_z | 0.000000 | [0.000000, 0.000000] | 只报告 |
| agent-k0-restate | retrieval-k0 | entity_recall | 0.000000 | [0.000000, 0.000000] | 未检出差异 |
| agent-k0-restate | retrieval-k0 | numeral_recall | 0.079096 | [0.028249, 0.141243] | 改善 |
| agent-k0-restate | retrieval-k0 | hallucination_rate | 0.000000 | [0.000000, 0.000000] | 未检出差异 |
| agent-k0-restate | retrieval-k0 | profile_gap_per_case | 0.025443 | [0.003662, 0.050273] | 恶化 |
| agent-k0-restate | retrieval-k0 | copy_ratio | -0.004460 | [-0.011693, 0.002758] | 只报告 |
| agent-k0-restate | retrieval-k0 | style_distance | -0.000912 | [-0.015625, 0.011964] | 只报告 |
| agent-k0-restate | retrieval-k0 | length_ratio | 0.036908 | [0.015636, 0.060932] | 只报告 |
| agent-k0-restate | retrieval-k0 | cjk_numeral_z | 0.010005 | [0.000000, 0.026443] | 只报告 |

| 目标 | 阈值 | 主臂 | 达到 |
|---|---|---|---|
| numeral_recall | >= 0.92 | 0.919774 | 未达到 |
| hallucination_rate | <= 0.005 | 0.004237 | 达到 |
| profile_gap | < 0.204 | 0.210867 | 未达到 |

PLAN 出口条件（hallucination_rate 明显下降）：未达到（结论：未检出差异）。
