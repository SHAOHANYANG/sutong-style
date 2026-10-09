主臂是 agent2-balanced-k2-system，对照是 retrieval-balanced-k2（同样 59 条，逐条配对，agent − 对照）。规则见 SPEC 4.9。
其余各臂是探索性的，不替换 README 里主臂的那一行。style_win_rate 本轮不跑。

校验器与评估用的是同一套规则：agent 按「违规最少」挑选最终版本，而违规就是从计算 entity_recall、numeral_recall、hallucination_rate 的同一份抽取结果里导出的。这三项的改善在一定程度上是构造使然，不能单独作为事实漂移被治好的证据；规则抓不到的语义漂移 agent 同样看不见。必须连同「没有优化的」那张表一起读。

人工抽查：pending（由仓库所有者进行，不得由模型代填）。

**主臂 − 对照（retrieval-balanced-k2）：agent 直接优化的指标**

| 指标 | 均值差 | 95% 区间 | 结论 |
|---|---|---|---|
| entity_recall | 0.000000 | [0.000000, 0.000000] | 未检出差异 |
| numeral_recall | 0.006780 | [0.000000, 0.020339] | 未检出差异 |
| hallucination_rate | -0.004237 | [-0.012712, 0.000000] | 未检出差异 |

**主臂 − 对照（retrieval-balanced-k2）：agent 没有优化的指标**

| 指标 | 均值差 | 95% 区间 | 结论 |
|---|---|---|---|
| profile_gap_per_case | 0.000866 | [-0.002907, 0.006040] | 未检出差异 |
| copy_ratio | -0.000351 | [-0.001879, 0.000605] | 只报告 |
| style_distance | -0.000755 | [-0.004095, 0.002256] | 只报告 |
| length_ratio | -0.000352 | [-0.002395, 0.000947] | 只报告 |
| cjk_numeral_z | 0.000000 | [0.000000, 0.000000] | 只报告 |
| profile_gap | -0.005780 | — | 只报点估计 |
| cjk_numeral_ratio_profile_mean_delta | agent 0.196221，基线 0.196221 | — | 只报告 |

**主臂 − k0（累计）（retrieval-k0）：agent 直接优化的指标**

| 指标 | 均值差 | 95% 区间 | 结论 |
|---|---|---|---|
| entity_recall | 0.000000 | [0.000000, 0.000000] | 未检出差异 |
| numeral_recall | 0.037288 | [0.003390, 0.076271] | 改善 |
| hallucination_rate | -0.014124 | [-0.032486, 0.000000] | 未检出差异 |

**主臂 − k0（累计）（retrieval-k0）：agent 没有优化的指标**

| 指标 | 均值差 | 95% 区间 | 结论 |
|---|---|---|---|
| profile_gap_per_case | -0.027863 | [-0.060568, 0.005190] | 未检出差异 |
| copy_ratio | 0.032409 | [0.018861, 0.047255] | 只报告 |
| style_distance | 0.006254 | [-0.052955, 0.065653] | 只报告 |
| length_ratio | 0.028309 | [0.001449, 0.070646] | 只报告 |
| cjk_numeral_z | 0.201019 | [0.018777, 0.506597] | 只报告 |
| profile_gap | 0.004691 | — | 只报点估计 |
| cjk_numeral_ratio_profile_mean_delta | agent 0.196221，基线 -0.004799 | — | 只报告 |

| 组合 | 角色 | style_distance | profile_gap | profile_gap_per_case | entity_recall | numeral_recall | hallucination_rate | copy_ratio | 字数比 | cjk_numeral_ratio 逐维均值差 |
|---|---|---|---|---|---|---|---|---|---|---|
| agent2-balanced-k2-followup | 探索性 | 0.928811 | 0.216992 | 0.484911 | 1.000000 | 0.908475 | 0.004237 | 0.745806 | 1.035879 | 0.196221 |
| agent2-balanced-k2-restate | 探索性 | 0.930080 | 0.210698 | 0.483986 | 1.000000 | 0.911864 | 0.004237 | 0.746911 | 1.035792 | 0.196221 |
| agent2-balanced-k2-system | 主臂 | 0.928767 | 0.208507 | 0.486128 | 1.000000 | 0.898305 | 0.000000 | 0.741656 | 1.034001 | 0.196221 |
| agent2-k0-followup | 探索性 | 0.922420 | 0.202212 | 0.513044 | 1.000000 | 0.866667 | 0.014124 | 0.710732 | 1.006560 | -0.004799 |
| agent2-k0-restate | 探索性 | 0.922918 | 0.196719 | 0.510836 | 1.000000 | 0.892090 | 0.014124 | 0.714413 | 1.007249 | -0.001225 |
| agent2-k0-system | 探索性 | 0.919296 | 0.201072 | 0.507164 | 1.000000 | 0.900565 | 0.004237 | 0.710848 | 1.007252 | -0.001225 |
| retrieval-balanced-k2 | 对照 | 0.929522 | 0.214287 | 0.485262 | 1.000000 | 0.891525 | 0.004237 | 0.742007 | 1.034353 | 0.196221 |
| retrieval-k0 | 对照 | 0.922513 | 0.203816 | 0.513992 | 1.000000 | 0.861017 | 0.014124 | 0.709247 | 1.005692 | -0.004799 |

| 臂 | 1 轮 | 2 轮 | 3 轮 | accepted | max_rounds | recursion_limit | fallback | 选中第 1 轮 | 选中第 2 轮 | 选中第 3 轮 | 未选中 | 第一轮有违规的样本 | 最终输出不同于基线的样本 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| agent2-balanced-k2-followup | 46 | 0 | 13 | 48 | 11 | 0 | 0 | 56 | 0 | 3 | 0 | 13 | 3 |
| agent2-balanced-k2-restate | 46 | 2 | 11 | 48 | 11 | 0 | 0 | 53 | 6 | 0 | 0 | 13 | 6 |
| agent2-balanced-k2-system | 46 | 2 | 11 | 48 | 11 | 0 | 0 | 52 | 7 | 0 | 0 | 13 | 7 |
| agent2-k0-followup | 41 | 1 | 17 | 42 | 17 | 0 | 0 | 56 | 3 | 0 | 0 | 18 | 3 |
| agent2-k0-restate | 41 | 5 | 13 | 46 | 13 | 0 | 0 | 48 | 11 | 0 | 0 | 18 | 11 |
| agent2-k0-system | 41 | 6 | 12 | 47 | 12 | 0 | 0 | 46 | 12 | 1 | 0 | 18 | 13 |

| 臂 | 违规类型 | 第一轮条数 | 最终条数 | 修掉 | 新引入 | 第一轮涉及样本 | 最终涉及样本 |
|---|---|---|---|---|---|---|---|
| agent2-balanced-k2-followup | entity_missing | 0 | 0 | 0 | 0 | 0 | 0 |
| agent2-balanced-k2-followup | numeral_missing | 11 | 9 | 2 | 0 | 10 | 8 |
| agent2-balanced-k2-followup | entity_hallucination | 1 | 1 | 0 | 0 | 1 | 1 |
| agent2-balanced-k2-followup | title | 3 | 3 | 0 | 0 | 2 | 2 |
| agent2-balanced-k2-restate | entity_missing | 0 | 0 | 0 | 0 | 0 | 0 |
| agent2-balanced-k2-restate | numeral_missing | 11 | 9 | 2 | 0 | 10 | 9 |
| agent2-balanced-k2-restate | entity_hallucination | 1 | 1 | 0 | 0 | 1 | 1 |
| agent2-balanced-k2-restate | title | 3 | 2 | 1 | 0 | 2 | 1 |
| agent2-balanced-k2-system | entity_missing | 0 | 0 | 0 | 0 | 0 | 0 |
| agent2-balanced-k2-system | numeral_missing | 11 | 9 | 2 | 0 | 10 | 9 |
| agent2-balanced-k2-system | entity_hallucination | 1 | 0 | 1 | 0 | 1 | 0 |
| agent2-balanced-k2-system | title | 3 | 3 | 0 | 0 | 2 | 2 |
| agent2-k0-followup | entity_missing | 0 | 0 | 0 | 0 | 0 | 0 |
| agent2-k0-followup | numeral_missing | 14 | 13 | 1 | 0 | 14 | 13 |
| agent2-k0-followup | entity_hallucination | 3 | 3 | 0 | 0 | 3 | 3 |
| agent2-k0-followup | title | 3 | 3 | 0 | 0 | 2 | 2 |
| agent2-k0-restate | entity_missing | 0 | 0 | 0 | 0 | 0 | 0 |
| agent2-k0-restate | numeral_missing | 14 | 10 | 4 | 0 | 14 | 10 |
| agent2-k0-restate | entity_hallucination | 3 | 3 | 0 | 0 | 3 | 3 |
| agent2-k0-restate | title | 3 | 2 | 1 | 0 | 2 | 1 |
| agent2-k0-system | entity_missing | 0 | 0 | 0 | 0 | 0 | 0 |
| agent2-k0-system | numeral_missing | 14 | 10 | 4 | 0 | 14 | 10 |
| agent2-k0-system | entity_hallucination | 3 | 1 | 2 | 0 | 3 | 1 |
| agent2-k0-system | title | 3 | 2 | 1 | 0 | 2 | 1 |

兜底样本：无。

**反馈回显**（SPEC 4.8 的事后诊断，不在那一轮的预注册里；4.9 起是预注册的报告项）

修订反馈里用「」引用了缺失的片段。输出若把反馈抄回去，片段就出现在输出里，校验器判为已修复。下表数的是仍含反馈引导语或「片段」的输出；「被防护切过的轮」只在打开回显防护时非零，切掉之后的文本不再计入前几列。

| 臂 | 最终输出改动的样本 | 其中最终输出含反馈回显 | 修订轮含回显 | 第 2 轮含回显 | 被防护切过的轮 |
|---|---|---|---|---|---|
| agent2-balanced-k2-followup | 3 | 0 | 0 / 26 | 0 / 13 | 6 |
| agent2-balanced-k2-restate | 6 | 0 | 0 / 24 | 0 / 13 | 19 |
| agent2-balanced-k2-system | 7 | 0 | 0 / 24 | 0 / 13 | 0 |
| agent2-k0-followup | 3 | 0 | 0 / 35 | 0 / 18 | 5 |
| agent2-k0-restate | 11 | 0 | 0 / 31 | 0 / 18 | 18 |
| agent2-k0-system | 13 | 0 | 0 / 30 | 0 / 18 | 0 |

**探索臂（各自对第一轮所取的文件，未做多重比较校正）**

| 臂 | 基线 | 指标 | 均值差 | 95% 区间 | 结论 |
|---|---|---|---|---|---|
| agent2-balanced-k2-followup | retrieval-balanced-k2 | entity_recall | 0.000000 | [0.000000, 0.000000] | 未检出差异 |
| agent2-balanced-k2-followup | retrieval-balanced-k2 | numeral_recall | 0.016949 | [0.000000, 0.042373] | 未检出差异 |
| agent2-balanced-k2-followup | retrieval-balanced-k2 | hallucination_rate | 0.000000 | [0.000000, 0.000000] | 未检出差异 |
| agent2-balanced-k2-followup | retrieval-balanced-k2 | profile_gap_per_case | -0.000351 | [-0.005773, 0.004044] | 未检出差异 |
| agent2-balanced-k2-followup | retrieval-balanced-k2 | copy_ratio | 0.003800 | [0.000000, 0.008472] | 只报告 |
| agent2-balanced-k2-followup | retrieval-balanced-k2 | style_distance | -0.000711 | [-0.005709, 0.003926] | 只报告 |
| agent2-balanced-k2-followup | retrieval-balanced-k2 | length_ratio | 0.001526 | [0.000000, 0.003751] | 只报告 |
| agent2-balanced-k2-followup | retrieval-balanced-k2 | cjk_numeral_z | 0.000000 | [0.000000, 0.000000] | 只报告 |
| agent2-balanced-k2-restate | retrieval-balanced-k2 | entity_recall | 0.000000 | [0.000000, 0.000000] | 未检出差异 |
| agent2-balanced-k2-restate | retrieval-balanced-k2 | numeral_recall | 0.020339 | [0.000000, 0.057627] | 未检出差异 |
| agent2-balanced-k2-restate | retrieval-balanced-k2 | hallucination_rate | 0.000000 | [0.000000, 0.000000] | 未检出差异 |
| agent2-balanced-k2-restate | retrieval-balanced-k2 | profile_gap_per_case | -0.001277 | [-0.005208, 0.001425] | 未检出差异 |
| agent2-balanced-k2-restate | retrieval-balanced-k2 | copy_ratio | 0.004904 | [-0.000293, 0.013940] | 只报告 |
| agent2-balanced-k2-restate | retrieval-balanced-k2 | style_distance | 0.000557 | [-0.003541, 0.005186] | 只报告 |
| agent2-balanced-k2-restate | retrieval-balanced-k2 | length_ratio | 0.001439 | [-0.001890, 0.006503] | 只报告 |
| agent2-balanced-k2-restate | retrieval-balanced-k2 | cjk_numeral_z | 0.000000 | [0.000000, 0.000000] | 只报告 |
| agent2-k0-followup | retrieval-k0 | entity_recall | 0.000000 | [0.000000, 0.000000] | 未检出差异 |
| agent2-k0-followup | retrieval-k0 | numeral_recall | 0.005650 | [0.000000, 0.016949] | 未检出差异 |
| agent2-k0-followup | retrieval-k0 | hallucination_rate | 0.000000 | [0.000000, 0.000000] | 未检出差异 |
| agent2-k0-followup | retrieval-k0 | profile_gap_per_case | -0.000948 | [-0.003450, 0.000501] | 未检出差异 |
| agent2-k0-followup | retrieval-k0 | copy_ratio | 0.001485 | [0.000000, 0.004184] | 只报告 |
| agent2-k0-followup | retrieval-k0 | style_distance | -0.000093 | [-0.000465, 0.000220] | 只报告 |
| agent2-k0-followup | retrieval-k0 | length_ratio | 0.000868 | [0.000000, 0.002219] | 只报告 |
| agent2-k0-followup | retrieval-k0 | cjk_numeral_z | 0.000000 | [0.000000, 0.000000] | 只报告 |
| agent2-k0-restate | retrieval-k0 | entity_recall | 0.000000 | [0.000000, 0.000000] | 未检出差异 |
| agent2-k0-restate | retrieval-k0 | numeral_recall | 0.031073 | [0.005650, 0.064972] | 改善 |
| agent2-k0-restate | retrieval-k0 | hallucination_rate | 0.000000 | [0.000000, 0.000000] | 未检出差异 |
| agent2-k0-restate | retrieval-k0 | profile_gap_per_case | -0.003155 | [-0.018030, 0.016419] | 未检出差异 |
| agent2-k0-restate | retrieval-k0 | copy_ratio | 0.005167 | [0.001595, 0.010067] | 只报告 |
| agent2-k0-restate | retrieval-k0 | style_distance | 0.000405 | [-0.011892, 0.012086] | 只报告 |
| agent2-k0-restate | retrieval-k0 | length_ratio | 0.001557 | [-0.001957, 0.006468] | 只报告 |
| agent2-k0-restate | retrieval-k0 | cjk_numeral_z | 0.003573 | [0.000000, 0.010720] | 只报告 |
| agent2-k0-system | retrieval-k0 | entity_recall | 0.000000 | [0.000000, 0.000000] | 未检出差异 |
| agent2-k0-system | retrieval-k0 | numeral_recall | 0.039548 | [0.005650, 0.087571] | 改善 |
| agent2-k0-system | retrieval-k0 | hallucination_rate | -0.009887 | [-0.025424, 0.000000] | 未检出差异 |
| agent2-k0-system | retrieval-k0 | profile_gap_per_case | -0.006827 | [-0.020387, 0.005220] | 未检出差异 |
| agent2-k0-system | retrieval-k0 | copy_ratio | 0.001601 | [-0.001373, 0.005137] | 只报告 |
| agent2-k0-system | retrieval-k0 | style_distance | -0.003217 | [-0.014316, 0.007708] | 只报告 |
| agent2-k0-system | retrieval-k0 | length_ratio | 0.001560 | [-0.001161, 0.005304] | 只报告 |
| agent2-k0-system | retrieval-k0 | cjk_numeral_z | 0.003573 | [0.000000, 0.010720] | 只报告 |

| 目标 | 阈值 | 主臂 | 达到 |
|---|---|---|---|
| numeral_recall | >= 0.92 | 0.898305 | 未达到 |
| hallucination_rate | <= 0.005 | 0.000000 | 达到 |
| profile_gap | < 0.204 | 0.208507 | 未达到 |

PLAN 出口条件（hallucination_rate 明显下降）：未达到（结论：未检出差异）。
