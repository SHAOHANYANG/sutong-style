# CHANGELOG

按 Phase 记录：做了什么、指标变化多少、踩了什么坑。

指标没改善也照实写。负结果也是结果。

## 2026-10-09 EDT T4.1：演示页面

`web/` 下新增单页前端（Vite、React、TypeScript、Tailwind）。上方是白话输入和流式输出，下方的面板把每个节点的 trace 事件翻译成一行说明：检索到哪些范例、每轮写了多少字、校验发现的每一处事实改动、路由的决定。修订后的结果可以标出相对第 1 轮的改动。

- SSE 用 `fetch` 加 `ReadableStream` 读取；增量解析、字符级 diff、事件归并、trace 到文字的转换都是纯函数，有 10 项单测。
- 轮询 `/healthz` 区分三种后端状态；后端不可用时页面照常加载。
- 在真实模型上验证：提交示例后面板随事件逐步更新；400px 宽度无横向溢出。
- 与 PLAN 的出入（没用 shadcn/ui、没做 base 与微调并排对比、检索行没有相似度）见 QUESTIONS Q34。
- 坑：前端默认的接口地址起初写的是 `localhost`，浏览器把它解析到 IPv6，而接口服务只监听 127.0.0.1，结果健康检查一直失败。默认值改成了 `127.0.0.1`。

## 2026-10-09 EDT 服务接入真实模型（T3.1 之后、T3.7 之前）

`/v1/transform` 现在跑的是真实流水线：三路检索、微调模型生成、保真校验与文体打分、修订循环。此前接口只有骨架，依赖是占位实现。

- 新增 `scripts/serve_model.py`：GPU 进程，加载 sutong-v2 和 bge-m3，提供 OpenAI 兼容的三个路由。解码复用 `greedy_decode`。
- 新增 `infra/model_server_client.py` 与 `api/assembly.py`：接口进程用 `openai` SDK 调模型服务，按 SPEC §4.6 主配置装配检索，不 import torch。
- 设置环境变量 `SUTONG_MODEL_BASE_URL` 才接入真实流水线；不设时行为与之前相同。
- 实测：顺序 10 个请求全部成功、无兜底；一轮通过的 8 个均值 2.46 秒，触发修订的 2 个为 7.60 秒和 7.88 秒，中位数 2.24 秒，显存约 5.1 GB。单机顺序请求，不是并发压测。
- 没有做的：vLLM、并发与吞吐测试、缓存（T3.2 起）。偏离 SPEC §5「走 vLLM」的理由见 QUESTIONS Q33。
- 观察（未做系统评估）：这 10 条是真人撰写的短白话，输出与输入很接近，改动多在个别用词。这与项目一开始记录的「域外输入风格化不足」一致，需要 `human_eval` 才能下结论。

## 2026-10-09 EDT T2.5 第二轮：回显防护与 system 反馈格式

规则是 SPEC §4.9，在写代码和跑 GPU 之前提交（`d66d84f`）。生成在 WSL2（2026-10-09T18:28Z，代码 `7ff1cf8`，sutong-v2，贪心解码，seed 42），六个臂共约 19 分钟，评估在 Windows。数字取自 `eval/reports/agent2-eval.*`、六份 `agent2-*.json` 和 `agent2-run-manifest.json`。§4.8 的规则、结论和 README 那一行没有动。

### 做了什么

- **回显防护**：生成之后、校验之前，切掉输出里抄回来的修订反馈，后面的校验、打分、挑选都只看切过的文本。
- **第三种反馈格式 system**：反馈接在 system 消息后面，user 轮保持纯输入。动机是第一轮看到模型把 user 轮的全部文字当成要改写的内容。
- 主臂预先定为 `agent2-balanced-k2-system`：三种格式里只有它的结果无法从第一轮的数据推出来。

### 主臂结论：未检出差异

| 指标 | 检索行（对照） | 第二轮主臂 | 配对均值差 | 95% 区间 | 结论 |
|---|---|---|---|---|---|
| numeral_recall | 0.891525 | 0.898305 | +0.006780 | [0, +0.020339] | 未检出差异 |
| hallucination_rate | 0.004237 | 0.000000 | −0.004237 | [−0.012712, 0] | 未检出差异 |
| entity_recall | 1.000000 | 1.000000 | 0 | [0, 0] | 未检出差异 |
| profile_gap_per_case | 0.485262 | 0.486128 | +0.000866 | [−0.002907, +0.006040] | 未检出差异 |

- 13 条第一轮有违规的样本里，修掉 2 条数值、1 条幻觉（就是 59 条里唯一有幻觉的那条），新引入 0，无兜底。最终改动 7 条，其中 5 条违规数没变，只是在线风格分略高而被选中，改动很小。
- `hallucination_rate` 降到 0，但只涉及 1 条样本，区间含 0，PLAN 出口条件仍然**未达到**。
- 文体指标基本不动：改 system 消息没有让文风变差，这是事前担心的风险，没有发生。

### 修订轮数分布

| 臂 | 1 轮 | 2 轮 | 3 轮 | 真实解码次数 | 被防护切过的轮 | 最终输出含回显 |
|---|---|---|---|---|---|---|
| agent2-balanced-k2-followup | 46 | 0 | 13 | 20 | 6 | 0 |
| agent2-balanced-k2-restate | 46 | 2 | 11 | 15 | 19 | 0 |
| agent2-balanced-k2-system（主臂） | 46 | 2 | 11 | 13 | 0 | 0 |
| agent2-k0-followup | 41 | 1 | 17 | 26 | 5 | 0 |
| agent2-k0-restate | 41 | 5 | 13 | 19 | 18 | 0 |
| agent2-k0-system | 41 | 6 | 12 | 20 | 0 | 0 |

防护起了作用：六个臂的最终输出都不再含回显。system 格式下一次回显都没有发生，说明回显确实是「反馈放在 user 轮」引起的。

### 事前预期对不对

- followup：预期最终改动接近 0，实际两个来源各 3 条。**预期偏低**。第一轮靠第 3 轮修掉的那 5 条确实没有再出现，但 `balanced-k2` 臂另有 2 条数值在第 3 轮修好了（`k0` 臂 1 条）。
- restate：预期只有正文本来就修好的那几条算数（`balanced-k2` 2 条加 1 条减少，`k0` 4 条），实际分别修掉 3 条和 5 条违规，**大体相符**。

### 两轮合起来怎么读

- **这个环在 59 条上没有可靠地改善保真。** 第一轮主臂的 +0.028 来自一条偶然的路径：第 2 轮整段回显，第 3 轮针对那段回显的反馈反而逼出了一版完整改写。防护打开后同一格式是 +0.017，区间 [0, +0.042]，含 0。
- 探索臂里 `agent2-k0-restate`（+0.031）和 `agent2-k0-system`（+0.040）的 `numeral_recall` 区间不含 0。它们是不给范例的臂，起点更低（0.861），没有做多重比较校正，按规则不替换主臂；照实列出，不据此下结论。
- 修不掉的违规多数是「一块」「一头」「一遍」「两个」。SPEC §1.5.3 已经写明数值规则对「一 + 量词」偏严，这些未必是真的信息丢失。也就是说 13 条里真正可修的事实错误更少，实验的分辨力很低。
- 局限照 §4.9：防护规则是看了第一轮输出之后定的，又在同一批样本上检验；两轮都在同样 59 条上跑；人工抽查 pending。

### 完整汇总（`eval/reports/agent2-eval.md` 原样）

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

## 2026-10-09 EDT T2.5：agent 评估（Phase 2 出口）

数字取自 `eval/reports/agent-eval.*`、四份 `agent-*.json` 和 `agent-run-manifest.json`，不手算。规则是 SPEC §4.8，在写代码和跑 GPU 之前提交（`4ad9826`）。生成在 WSL2（2026-10-08T01:48Z，代码 `e8cb446`，sutong-v2，贪心解码，seed 42），评估在 Windows（`--skip-judge`）。主臂是预注册的 `agent-balanced-k2-followup`，其余三臂是**探索性的**。

### 主臂结论

| 指标 | 检索行（对照） | agent 主臂 | 配对均值差 | 95% 区间 | 结论 |
|---|---|---|---|---|---|
| numeral_recall | 0.891525 | 0.919774 | +0.028249 | [+0.005650, +0.059322] | 改善 |
| hallucination_rate | 0.004237 | 0.004237 | 0 | [0, 0] | 未检出差异 |
| entity_recall | 1.000000 | 1.000000 | 0 | [0, 0] | 未检出差异 |
| profile_gap_per_case | 0.485262 | 0.484792 | −0.000471 | [−0.009576, +0.007578] | 未检出差异 |
| copy_ratio | 0.742007 | 0.750093 | +0.008086 | [+0.001485, +0.017025] | 只报告 |

- **PLAN 出口条件「`hallucination_rate` 明显下降」未达到。** 59 条里只有 1 条有幻觉（「钟爱」），三轮都没修掉，变化幅度为 0。预注册时已经写明下降空间很小。
- `numeral_recall` 的变化幅度是 +0.028（0.892 → 0.920），相对不加任何东西的 sutong-v2 是 +0.059（0.861 → 0.920）。0.919774 没有达到 STATUS §9.5 的 ≥ 0.92。
- 59 条里第一轮有违规的只有 13 条，最终输出不同于对照的 8 条。修掉 4 条数值违规、1 条称谓违规，新引入 0 条，无兜底。
- 这三项保真指标和校验器同源，改善有一部分是构造使然（§4.8）。人工抽查 pending。
- 三种预注册的「假修复」：`cjk_numeral_ratio` 的 z 值 −0.020 [−0.059, 0]，4 条数值修复里有 1 条把阿拉伯数字抄进了输出（反馈里明确要求用中文数字，没有听）；`copy_ratio` 略升且区间不含 0；没有修一处坏一处。

### 修订轮数分布（PLAN T2.5 验收项）

| 臂 | 1 轮 | 2 轮 | 3 轮 | 真实解码次数 | 耗时（秒） |
|---|---|---|---|---|---|
| agent-balanced-k2-followup（主臂） | 46 | 0 | 13 | 26 | 228 |
| agent-balanced-k2-restate | 46 | 9 | 4 | 15 | 197 |
| agent-k0-followup | 41 | 2 | 16 | 30 | 267 |
| agent-k0-restate | 41 | 10 | 8 | 19 | 208 |

主臂没有一条在第 2 轮结束：followup 格式下第 2 轮的 13 条里有 6 条是把修订反馈原样输出（模型把最后一轮 user 当成了要改写的文本），其余的要么几乎没改，要么抄了范例的白话。这些都被挑选规则挡掉了。第 3 轮的反馈针对的是第 2 轮那版，模型这时才写出一版完整的改写，5 条被接受。也就是说 followup 实际只有一次有效的修订机会。

### 坑：restate 的好数字是假的（事后发现）

`agent-balanced-k2-restate` 的 `numeral_recall` 是 0.985876，`agent-k0-restate` 是 0.940113，看起来远好于主臂。核对输出后发现，restate 格式下模型经常在正文后面把修订反馈原样附上，而反馈里用「」引了缺失的数字或称谓。校验器在输出里找到这个片段就判为已修复。（更正，2026-10-09：把抄回来的部分切掉再校验，`balanced-k2` 臂这 10 条里 8 条正文仍有违规、2 条无违规；`k0` 臂 9 条里 5 条仍有违规、4 条无违规。原先写的「正文其实没改」说得过头了，多数没改，少数是真修了。）

| 臂 | 最终输出改动的样本 | 其中含反馈回显 |
|---|---|---|
| agent-balanced-k2-followup（主臂） | 8 | 0 |
| agent-balanced-k2-restate | 11 | 10 |
| agent-k0-followup | 5 | 1 |
| agent-k0-restate | 12 | 9 |

- 这个诊断（`feedback_echo`）是看到结果之后加的，不在 §4.8 的预注册里；判定规则和主臂结论没有因此改动。
- 两个 restate 臂的保真数字**不可引用**。它们的 `profile_gap_per_case` 按预注册规则是「恶化」，字数比上升 4 个点，和「正文后面多了一段指令」一致。
- 预注册的三种假修复没有覆盖这一种。教训：规则校验器只看片段在不在，不看它在哪；把缺失片段原样写进反馈，等于把答案递给了一个会抄的模型。
- 没有做的事：没有改校验器、路由或挑选规则，没有重跑。修这个漏洞是一轮新实验，要先另行预注册（QUESTIONS Q31）。

### 完整汇总（`eval/reports/agent-eval.md` 原样）

主臂是 agent-balanced-k2-followup，对照是 retrieval-balanced-k2（同样 59 条，逐条配对，agent − 对照）。
其余三臂是探索性的，不替换 README 的 agent 行。style_win_rate 本轮不跑。

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

**事后诊断（不在 SPEC 4.8 的预注册里，看到结果之后加的）：反馈回显**

修订反馈里用「」引用了缺失的片段。输出若把反馈原样抄回去，片段就出现在输出里，校验器判为已修复，正文其实没改。下表数的是含反馈引导语或「片段」的输出。

| 臂 | 最终输出改动的样本 | 其中最终输出含反馈回显 | 修订轮含回显 | 第 2 轮含回显 |
|---|---|---|---|---|
| agent-balanced-k2-followup | 8 | 0 | 6 / 26 | 6 / 13 |
| agent-balanced-k2-restate | 11 | 10 | 12 / 17 | 10 / 13 |
| agent-k0-followup | 5 | 1 | 5 / 34 | 5 / 18 |
| agent-k0-restate | 12 | 9 | 13 / 26 | 11 / 18 |

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

## 2026-10-07 EDT T1.6：检索评估收尾（Phase 1 出口）

数字取自分词修正后重算的 `eval/reports/retrieval-sweep.*` 与各 `retrieval-*.json`，不手算。主配置是预注册的 balanced、k = 2；其余 11 个带范例组合是**探索性的**，不替换 README 的 retrieval 行。`style_win_rate` 本轮不跑。

### 主配置对照、13 组全表、三个目标值

（与 `eval/reports/retrieval-sweep.md` 相同。）

主配置是 balanced、k = 2，对照是同一次扫描的 k = 0。
其余组合是探索性的，不替换 README 的 retrieval 行。style_win_rate 本轮不跑。

| 指标 | 均值差（主 − 对照） | 95% 区间 | 结论 |
|---|---|---|---|
| profile_gap_per_case | -0.028729 | [-0.060864, 0.003425] | 未检出差异 |
| hallucination_rate | -0.009887 | [-0.025424, 0.000000] | 未检出差异 |
| entity_recall | 0.000000 | [0.000000, 0.000000] | 未检出差异 |
| numeral_recall | 0.030508 | [-0.003390, 0.070056] | 未检出差异 |
| copy_ratio | 0.032760 | [0.019210, 0.047696] | 只报告 |
| style_distance | 0.007010 | [-0.052414, 0.066518] | 只报告 |
| profile_gap | 0.010471 | — | 只报点估计 |

| 组合 | 角色 | style_distance | profile_gap | profile_gap_per_case | entity_recall | numeral_recall | hallucination_rate | copy_ratio |
|---|---|---|---|---|---|---|---|---|
| retrieval-k0 | 对照 | 0.922513 | 0.203816 | 0.513992 | 1.000000 | 0.861017 | 0.014124 | 0.709247 |
| retrieval-equal-k1 | 探索性 | 0.908714 | 0.205450 | 0.497898 | 1.000000 | 0.875141 | 0.004237 | 0.745687 |
| retrieval-equal-k2 | 探索性 | 0.926546 | 0.204620 | 0.467228 | 1.000000 | 0.892090 | 0.004237 | 0.749246 |
| retrieval-equal-k3 | 探索性 | 0.996126 | 0.189797 | 0.483773 | 1.000000 | 0.883616 | 0.004237 | 0.761473 |
| retrieval-balanced-k1 | 探索性 | 0.919984 | 0.192274 | 0.508809 | 1.000000 | 0.886441 | 0.004237 | 0.731873 |
| retrieval-balanced-k2 | 主配置 | 0.929522 | 0.214287 | 0.485262 | 1.000000 | 0.891525 | 0.004237 | 0.742007 |
| retrieval-balanced-k3 | 探索性 | 0.918607 | 0.216889 | 0.470503 | 1.000000 | 0.863277 | 0.004237 | 0.753109 |
| retrieval-content-k1 | 探索性 | 0.913444 | 0.217614 | 0.493708 | 1.000000 | 0.875141 | 0.004237 | 0.747112 |
| retrieval-content-k2 | 探索性 | 0.922507 | 0.189182 | 0.455869 | 1.000000 | 0.892090 | 0.004237 | 0.753631 |
| retrieval-content-k3 | 探索性 | 0.986370 | 0.187597 | 0.472459 | 1.000000 | 0.889266 | 0.004237 | 0.762296 |
| retrieval-style-k1 | 探索性 | 0.911608 | 0.205993 | 0.498544 | 1.000000 | 0.886441 | 0.004237 | 0.729564 |
| retrieval-style-k2 | 探索性 | 0.933805 | 0.183188 | 0.495522 | 1.000000 | 0.844350 | 0.004237 | 0.739132 |
| retrieval-style-k3 | 探索性 | 0.940808 | 0.178332 | 0.501536 | 1.000000 | 0.889548 | 0.004237 | 0.746687 |

| 目标 | 阈值 | 主配置 | 达到 |
|---|---|---|---|
| numeral_recall | >= 0.92 | 0.891525 | 未达到 |
| hallucination_rate | <= 0.005 | 0.004237 | 达到 |
| profile_gap | < 0.204 | 0.214287 | 未达到 |

### prompt token 分布

取自 `eval/reports/retrieval-sweep-manifest.json`。

| 组合 | mean | std | median | q1 | q3 | min | max |
|---|---|---|---|---|---|---|---|
| retrieval-k0 | 278.5 | 38.1 | 274 | 256 | 286 | 213 | 430 |
| retrieval-equal-k1 | 735.9 | 61.2 | 728 | 693 | 760 | 635 | 924 |
| retrieval-equal-k2 | 1179.9 | 88.5 | 1164 | 1120 | 1222 | 1007 | 1543 |
| retrieval-equal-k3 | 1620.2 | 103.6 | 1611 | 1540 | 1686 | 1452 | 1978 |
| retrieval-balanced-k1 | 722.5 | 66.1 | 713 | 678 | 760 | 623 | 924 |
| retrieval-balanced-k2 | 1170.2 | 81.0 | 1165 | 1100 | 1210 | 1007 | 1395 |
| retrieval-balanced-k3 | 1618.0 | 97.6 | 1600 | 1555 | 1674 | 1451 | 1903 |
| retrieval-content-k1 | 732.6 | 63.2 | 724 | 690 | 760 | 616 | 924 |
| retrieval-content-k2 | 1183.8 | 88.3 | 1174 | 1118 | 1226 | 1060 | 1543 |
| retrieval-content-k3 | 1627.3 | 104.7 | 1632 | 1538 | 1689 | 1454 | 1978 |
| retrieval-style-k1 | 715.3 | 53.4 | 708 | 685 | 742 | 623 | 871 |
| retrieval-style-k2 | 1163.1 | 78.5 | 1156 | 1116 | 1188 | 1012 | 1435 |
| retrieval-style-k3 | 1612.8 | 90.7 | 1600 | 1566 | 1655 | 1422 | 1852 |

### 探索性组合与事后挑选

12 个带范例组合里，`profile_gap_per_case` 最低的是 **content，k = 2**（0.455869）。这是事后从 12 组里挑出来的，**没有**替换 README 的那一行；59 条上挑最好一定偏乐观。12 组的 `hallucination_rate` / `numeral_recall` / `copy_ratio` 等变化方向大体一致，但它们共用同一批 59 条样本和同一个 adapter，不是 12 次独立验证。

### 12 组 hallucination_rate 相同的原因

12 个带范例组合的 `hallucination_rate` 均为 **0.004237**。原因：59 条里只有 `我的帝王生涯_0130` 非零（该条为 0.25），其余 58 条为 0，故均值 = 0.25/59。该条在 12 组输出上虽不完全相同，但规则抽取到的幻觉实体都是同一个表面形式「钟爱」（1 个实体）。对照 k = 0 另有 `妇女生活_0050`、`罂粟之家_0094` 非零，故均值更高（0.014124）。

### 副作用与欠项

主配置相对对照：`exemplar_entity_leak` 两边均为 0；`exemplar_copy_ratio` 配对差区间含 0，未检出相对对照的范例抄写。`copy_ratio` 上升且区间不含 0，数值保真的变化可能部分来自改得更少。

**T1.3 原定「预测向量 vs 直接用输入特征」的端到端生成对照：本次扫描未单独设该对照，未做，不用现有 13 组硬凑结论。** PLAN 要求的 `human_eval.jsonl` 检索召回评估：正文尚未写入，**未做**。

### 分词修正更正（2026-10-07）

`stylometry/features.py` 原先用全局 `jieba.lcut`。`ensure_manual_userdict` 往全局 jieba 加载 67 个人名后，后续文体特征会变。影响很小、结论不变，但评估不能取决于执行顺序。已改为模块私有 `Tokenizer()`、不加载用户词典（与已提交 `style_reference.json` / `lexicon.json` 的干净拟合口径一致；内存重拟合与已提交文件逐项相同）。

三口径对比（主配置对 k = 0 的 `profile_gap_per_case`）：官方曾混用分词时差约 −0.0271；两组都用干净分词时 −0.028729，区间 [−0.060864, +0.003425]，均含 0。审计给出的污染口径下 balanced-k2 曾为 0.486845，干净为 0.485262。

已提交报告中因此变化的数（旧 → 新）：

| 文件 / 指标 | 旧 | 新 |
|---|---|---|
| `lora-eval59.json` `style_distance` | 0.9228076532274074 | 0.9225125975073157 |
| `base-eval59.json` `style_distance` | 1.149013318260184 | 1.149451731933785 |
| 各 `retrieval-*.json` 的 style 相关量 | 见 sweep 重算 | 干净分词下重写 |

`profile_gap` / `profile_gap_per_case` 与三项保真、`copy_ratio` 在 base/lora 上不变。旧条目正文不改写；T0.5 基线表里的 LoRA `style_distance` 0.923 等显示精度仍可读，精确值见本更正与重算报告。

## 2026-10-07 EDT T2.4：TraceEvent 与每轮违规

`TraceEvent` 落地：每节点记 `node / ts / duration_ms / payload`。时钟与单调计时器可注入。payload 只放 id、hash、长度和违规短片段，不含正文。`RoundRecord.violations` 补上该轮完整违规列表。本阶段无新的端到端评估数字。

## 2026-10-07 EDT T2.3：逐条修订 prompt

`numeral_key_surfaces` 与 `extract_quantities` 改为共用 `iter_quantity_hits`，抽取结果不变。`agent/prompts.py` 实现 followup / restate 两种反馈；默认 followup。数字反馈同时要求数值不变与中文数字书写。消息构造函数成为图的默认构造器。

## 2026-10-07 EDT T2.2：保真校验接入，换范例默认关闭

`Violation` 增加 `entity_missing` / `numeral_missing` / `entity_hallucination`，与三项保真指标共用同一份抽取；表面片段留给 T2.3。打分器用 StylePredictor 目标的负 MAE，只破平局。审计表在 `eval/reports/agent-scorer-validation.json`：候选 A 无信息量；候选 B 会偏向照抄白话，故 `re_retrieve_score_threshold` 默认 null。三项指标相对 base/lora 基线报告无变化。

## 2026-10-07 EDT T2.1：Agent 自检重写环图骨架

搭了 `agent/`：`retrieve → generate → verify → score → route`，四个外部能力全部 Protocol 注入，测试只用 Fake。生成上限 `MAX_GENERATIONS = 3`，`RECURSION_LIMIT = 20`（最长合法路径 14 步 + 余量 6）。结束挑选按违规条数 → 分数 → 轮次字典序，不再用 argmax(scores)；SPEC §4.4 已改，理由写在那里。

本轮没有接真实检索、保真、风格距离，也没有修订 prompt。没有新的评估数字。依赖加了 `langgraph`（间接会有 `langchain-core`，代码里不得直接 import）。

## 2026-10-07 EDT T1.5 后半之二：k 扫描脚本与判定规则

判定规则在看到扫描结果之前单独提交（`dfded45`），写在 SPEC §4.7。§4.6 没有改。主配置仍是 balanced、k = 2，对照是同一次扫描的 k = 0。

`scripts/sweep_topk.py` 按检索计划生成 13 组。解码和 `scripts/generate.py` 共用 `greedy_decode`。k = 0 先和 `corpus/generations/lora-eval59.jsonl` 逐条比对，有一条不同就停。`scripts/eval_sweep.py` 用 `--skip-judge` 的路径汇总，不读 `.env`。两个诊断量在 `eval/exemplar_diagnostics.py`。

扫描还没有跑。没有生成文件，没有汇总数字，README 的 retrieval 行仍然空着。待人工在 WSL2 执行 `python -m scripts.sweep_topk --adapter adapters/sutong-v2/adapter`，再在 Windows 上执行 `uv run python -m scripts.eval_sweep`。（2026-10-07 已跑完并写入 T1.6，见该日条目。）

## 2026-10-07 EDT T1.5 后半之一：检索计划与 token 计数器

`QwenPromptTokenizer` 不再自己加载分词器，也不再导入 transformers。计数改成与 `generate_one` 同一条渲染路径：先 `apply_chat_template(..., tokenize=False, add_generation_prompt=True)` 得到字符串，再 `tokenizer(prompt, add_special_tokens=False)` 取 `input_ids` 长度。调用方以后把 `generate.py` 已经加载的那个分词器传进来。

`scripts/build_retrieval_plan.py` 在 CPU 上为 59 条 eval 查询、四种预注册配置各写出融合排名前 3 的范例，报告在 `eval/reports/retrieval-plan.json`。索引池是 `train_documents` 选出的 743 条。depth 20、rrf_k 60、权重都来自 `eval/configs/retrieval.yaml`，与 `FUSION_CONFIGS` 一致。embedding revision 钉为 `5617a9f61b028005a4858fdac845db406aefb181`。四条硬断言在这次运行里都通过：范例不在 `split.eval`、不是查询自己、每条查询每种配置恰好 3 条、content 的 sources 没有 style、style 的 sources 只有 style。

下面是 k = 2 的描述统计。剖面差用了 eval 原文，只描述范例和目标原文的距离，不改 §4.6。

| 配置 | bm25 | dense | style | 只来自 style | 只来自内容两路 | 同作品 | 剖面差均值 |
|---|---|---|---|---|---|---|---|
| equal | 0.983 | 1.000 | 0.331 | 0.000 | 0.669 | 0.992 | 0.775 |
| balanced | 0.559 | 0.559 | 0.873 | 0.322 | 0.127 | 0.737 | 0.661 |
| content | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.992 | 0.828 |
| style | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.373 | 0.626 |

k = 2 的两两 Jaccard（前 2 条 id 集合，对 59 条查询取平均）：

|  | equal | balanced | content | style |
|---|---|---|---|---|
| equal |  | 0.379 | 0.740 | 0.034 |
| balanced |  |  | 0.164 | 0.328 |
| content |  |  |  | 0.006 |

报告里的 `commit` 是生成时的 HEAD `0356e41`。分布、按作品分组和 k = 1、3 都在报告文件里。

## 2026-10-07 EDT T1.5 前半：三路装配与 few-shot prompt

预注册在看到任何生成结果之前单独提交（`cdcd37e`），规则在 SPEC §4.6。本轮没有生成，也没有 k 扫描。主配置 balanced、k = 2 是预注册时按论证选定的，还没有数据，不能把它说成更好。

`retrieval/hybrid.py` 把三路接起来。BM25 和 dense 用查询文本，style 用预测后的 z 向量。权重为 0 的路不查询。三个索引的 id 集合不一致就拒绝构造。`retrieval/prompt.py` 按训练时的 system 文本拼多轮范例，排名第 1 的范例紧挨真正的输入；k = 0 与 `build_messages(vernacular, None)` 相同。prompt token 数加 768 超过 4096 就报错，不截断。

dense 查询向量另写 `scripts/build_query_cache.py`，默认目录 `retrieval/data/query_cache/`，已进 `.gitignore`。只编码 `split == eval` 的白话。非 dry-run 待人工在 WSL2 执行，这次只跑了 `--dry-run`。

## 2026-10-06 EDT T1.4 带权重的 RRF 融合

`retrieval/fusion.py` 把三路 `Hit` 列表融成带出处的排名。默认每路权重为 1、`rrf_k` 为 60，与 SPEC 的等权公式一致。权重可以另给，0 等于这路不参与。`sources` 记下每条文档来自哪几路、在各路排第几。

等权时，只在 style 路排第 1 的文档（1/61）排在两路内容检索都排第 30 的文档（2/90）后面。这是算术事实，权重怎么配留给 T1.5 预注册和 T1.6 实测。这次没有选推荐值，也没有把三路接成检索器。

## 2026-10-06 EDT T1.3 风格索引与预测向量消融

（端到端「预测向量 vs 直接用输入特征」的生成对照原定留到 T1.6；T1.6 扫描未单独设该对照，结论仍止于本条的检索层消融。见 2026-10-07 T1.6。）

预注册规则在看到结果之前单独提交（`8e4b7cc`）。判定只用 743 条训练对的 5 折，eval 59 条只作描述。度量是召回范例与查询原文的逐条剖面差，主分析 k = 3。

**有优势，预测器保留。** k = 3 的五组均值：predicted 0.607，raw 0.714，mean 0.752，oracle 0.454，random 0.998。oracle 最小。`predicted − raw` 的均值是 -0.107，95% 区间 [-0.119, -0.094]；`predicted − mean` 的均值是 -0.145，区间 [-0.156, -0.134]。两个区间都整体小于 0。raw 的均值小于 mean，按输入查比永远查均值有信息量。五折选出的 alpha 都是 10，用全部 743 条再选也是 10。

折外逐维 R² 最高的是 `para_density` 0.883、`dialogue_verb_density` 0.800、`ttr` 0.763。最低的是 `fw_而` 0.134、`fw_之` 0.171、`sent_len_std` 0.197。句长三维也偏低：`sent_len_mean` 0.420、`sent_len_p90` 0.312、`sent_len_std` 0.197。白话能推出段落密度和对白动词，推不准句长起伏和「而」「之」。

按作品分组的 k = 3 配对差取自 `eval/reports/style-predictor-ablation.json` 的 `by_work`（`predicted_minus_raw_mean`、`predicted_minus_mean_mean`）：

| 作品 | predicted − raw | predicted − mean |
|---|---|---|
| 另一种妇女生活 | -0.104 | -0.080 |
| 园艺 | -0.163 | -0.115 |
| 妇女生活 | -0.073 | -0.187 |
| 妻妾成群 | -0.050 | -0.229 |
| 我的帝王生涯 | -0.169 | -0.116 |
| 罂粟之家 | +0.053 | -0.202 |

《罂粟之家》是例外：predicted − raw 为 +0.053，在这部作品上预测版不如直接用输入特征，但仍优于 mean。

eval 59 条不参与判定。k = 3 的描述性均值：predicted 0.638，raw 0.751，mean 0.803，oracle 0.482，random 1.037。方向与训练集一致。

报告里的 `commit` 是生成时的 HEAD，即预注册那次提交。同一 seed 重跑，除时间戳外逐字节相同。没有把 style 路接进融合或 prompt。

## 2026-10-06 EDT profile_gap 口径更正（T0.4 补丁）

`profile_gap` 此前只有手填数字，代码和 `eval/reports/` 里都没有这个字段。文档把它写成「逐条平均 |z| 差」，已发表的 0.450 / 0.494 / 0.204 实际是另一种算法。

两种口径现在都有实现，模型权重和生成结果都没变，这不是改善也不是恶化：

| | profile_gap（集合级） | profile_gap_per_case（逐条） | sent_len_p90 | sent_len_std |
|---|---|---|---|---|
| 白话输入 | 0.450 | 0.623 | -0.61 | -0.58 |
| base | 0.494 | 0.748 | -0.62 | -0.63 |
| LoRA | 0.204 | 0.514 | -0.61 | -0.50 |

集合级先对样本求逐维均值再取绝对值，方向相反的误差会抵消。「缩了一半还多」只对 0.450 → 0.204 成立。逐条是每一条与自己原文的平均 |z| 差，再对样本求均值，0.623 → 0.514。

原先写的分维数字 `sent_len_p90` -0.64 → -0.64、`sent_len_std` -0.70 → -0.62 用现有 lexicon 和 style_reference 复现不出来。上表是报告里 `profile_mean_delta` 的值。

## 2026-10-06 EDT T1.2 dense 语义召回

内存里做余弦，不接 pgvector，也不装 FlagEmbedding。bge-m3 的 dense 向量用 transformers 取最后一层 CLS，再 L2 归一化；索引内部还会再归一化一次。
缓存是 `retrieval/data/embeddings.npy` 和 `embeddings.meta.json`，两个都不进 git。加载时核对 id 顺序、文本 sha256、模型标识、revision、max_length 和维度，任何一项不一致就抛错，不静默重算。
测试全部用 `FakeEmbedder`，没有下载模型，也没有建索引。待人工在 WSL2 执行 `python -m scripts.build_dense_index`。

## 2026-10-06 EDT T1.1 BM25 稀疏召回

`retrieval/bm25.py` 用私有 jieba 分词器对调用方传入的原文做内存索引，查询是白话。
不读 `pairs.jsonl` / `chunks.jsonl`，不写 pickle。k1/b/epsilon 用 `rank_bm25.BM25Okapi` 的默认值，没有检索评估集，不调参。
`Document` 和 `Hit` 放在 `retrieval/types.py`，给后面的 RRF 用。dense / style / fusion 没做。

停用词表 `retrieval/data/stopwords.txt` 共 215 条，是本仓库自行整理的中文虚词、代词、连词、助词、语气词和中英文标点。
不是从哈工大停用词表、百度停用词表、stopwords-iso 或其他公开词表复制的，没有第三方许可证，随仓库按项目代码同一方式分发。
人名、地名、称谓和数词不在表里。

`sample_public.jsonl` 的查询只用于单测冒烟，确认链路能跑通。不把它报成 recall@k 或检索质量。SPEC 4.3 的送分题仍然成立。
SPEC 4.1 写明索引池只能用 `split=train` 的 chunk，eval 原文不得进索引；具体用哪个 train 子集待定。

## 2026-10-05 EDT T0.5 训练与基线（Phase 0 出口）

**训练完成。** `sutong-v2`，Qwen2.5-3B-Instruct + LoRA r=16，743 条训练对，2 epochs，12 分 39 秒，RTX 5060 Laptop (sm_120) / WSL2。可训练参数实测 29,933,568，与 v1 日志一字不差——这从加载的真实模型数出，独立证实 v1 用的是 r=16 而非 SPEC 原记录的 r=32，见 SPEC 1.2 的更正段。最长样本 806 token，`max_seq_length=1536`，零截断。

**loss 曲线。** train 1.36 → 0.78；eval 0.9808（epoch 1）→ 0.9881（epoch 2 回升）。与 v1 的 train 2.726 → 1.936、eval 2.169 → 2.056 → 2.089 **形状一致、绝对值不可比**（掩码方式不同 + 白话侧已全部重新生成）。v2 的 eval 在 epoch 1 即触底，早于 v1 的 1.9；仅记录，不据此改 epoch 数。

**基线数字（59 条 eval，贪心解码，seed 42）。** 下列 `profile_gap` 与分维数字的口径见 2026-10-06「profile_gap 口径更正」，本段正文保留原样。`style_distance` 的第四位小数在 2026-10-07 分词隔离修正后有微调，见同日 T1.6 更正；显示到千分位的表值不变。

| | style_distance | profile_gap | entity_recall | numeral_recall | hallucination_rate | copy_ratio |
|---|---|---|---|---|---|---|
| base | 1.149 | 0.494 | 0.958 | 0.923 | 0.0042 | 0.875 |
| LoRA | 0.923 | **0.204** | **1.000** | 0.861 | 0.0141 | 0.709 |
| 原文 | 1.035 | 0 | — | — | — | — |
| 白话输入 | 0.992 | 0.450 | — | — | — | — |

微调把特征剖面差距从 0.450 压到 0.204，base 反而恶化到 0.494；`entity_recall` 到满分。代价是**事实漂移**：`numeral_recall` 0.923 → 0.861，`hallucination_rate` 0.0042 → 0.0141。这正是 README 列为「已知缺陷 #1」的现象，首次量化。Phase 1 与 Phase 2 的靶子即此两项。

**一个具体的失败。** `sent_len_p90` 与 `sent_len_std` 两维几乎没动（-0.64 → -0.64、-0.70 → -0.62）。模型学会了书面语词、文言虚词与标点密度，**没学会句长节奏**。

**负结果：`style_distance` 的方向标此前写反了。** 它是 z 分数均方根，期望值 1.0 而非 0；原文实测 1.035。LoRA 的 0.923 低于原文 = 向均值回归。白话输入 0.992 几乎正中靶心却与原文剖面差 0.450——标量把半径与方向压成一个数，分不开「没改」与「改好了」。此后降为弱信号，判别改用 `profile_gap`。见 SPEC 3.1 更正段。

**负结果：`style_win_rate` 这一轮不可引用。** 59 条、118 次调用（= 59 × 2，位置互换全部发出）、零解析失败、模型身份闸门确认响应为 `deepseek-v4-pro`——**judge 链路首次对真实 API 验证通过**。但 `tie_rate = 0.49`：近一半样本在 A/B 互换后评委改口。判得一致的 30 条中 25:5 偏向白话，与 `profile_gap` 方向相反。输出非退化（59 条中仅 3 条有轻微重复或缺末尾标点）。最可能原因是 **Q18 的对手选择有混淆**：白话由 LLM 从原文降级改写而来，内容与叙事顺序几乎原样保留，评委虽被要求只判文风，仍大概率在响应内容相似度。对手选择需重做。

**踩的坑。**

1. `torchvision` 从 PyPI 默认源装会是 CPU 版，`import unsloth` 报 `operator torchvision::nms does not exist`。必须从 cu128 源重装成 `0.26.0+cu128`。
2. TRL 0.24.0 改了 API：`SFTConfig(max_seq_length=)` → `max_length=`，`SFTTrainer(tokenizer=)` → `processing_class=`。`unsloth` 必须在 `trl`/`transformers`/`peft` 之前 import，否则补丁不生效。
3. 后台命令结尾加 `| tail -N` 会让退出码变成 `tail` 的——脚本崩了仍报 exit 0。第一次训练失败因此被误判为成功。
4. `.env` 里的 `JUDGE_MODEL` 与 `VERNACULARIZE_MODEL` 仍是废别名 `deepseek-chat`。按既有记录不静默修改用户配置，本次运行在命令行显式覆盖。待所有者自行更新。

**新增。** `scripts/train.py`、`scripts/generate.py`（共用 `build_messages()`，训练与推理的 prompt 必须同源）、`eval/configs/eval59.yaml`、`EvalConfig.split` 过滤（避免把受版权保护的原文再复制一份出来）。

## 2026-10-05 EDT T0.4 评估编排

`eval/judge.py` 做配对比较：每个样本 A/B 互换各问一次，两次不一致记平局。
胜率是 `(胜 + 0.5 × 平) / 样本数`。磁盘缓存键是 prompt 与模型名的 sha256，只存 A/B。
`eval/run_eval.py` 读 `eval/configs/baseline.yaml` 和生成文件。`--skip-judge` 不构造客户端。
本任务没有基线数字：5 条 `sample_public.jsonl` 用 Fake generator 跑通，困惑度和风格胜率在跳过评委时为 null。
词频阈值 200 的地名代价已由所有者量化，结论补进 Q17，不再调整。未开始 T0.5。

**同日验收后两处修正。** 评委默认模型改为 `deepseek-v4-pro`。`deepseek-chat` 已重映射到 Flash，SPEC 3.3 的旧默认和 SPEC 1.5 的禁令冲突。请求名与响应 model 不一致时 `ModelIdentityMismatchError` 对 judge 同样抛错。报告的 `distribution.style_win_rate` 改为胜、负、平、tie_rate 和 n。永远选 A 的评委胜率仍是 0.5，但 tie_rate 为 1，用来区分评委失效和真打平。仍未开始 T0.5。

**同日再拆 identical。** 输出与白话逐字节相同时，互换后的 prompt 撞同一个缓存键，调用次数少一半，还被记成平局。现在这种情况不发请求，单独记 identical。`copy_ratio` 复用白话化的 `text_similarity`，跳过评委也计算。`tie_rate` 只在 identical 少时才解释成位置偏差。`copy_ratio` 的均值随后补进 `aggregate`，排在 `style_win_rate` 和 `ppl` 之间，字段顺序以 `AGGREGATE_FIELDS` 为准。仍未开始 T0.5。

## 2026-10-05 EDT 风格参照与词表口径

词表只用 train 743 对，风格参照用全部 915 条原文。所有者核对过这不是疏漏：
参照是 20 均值加 20 标准差，单条贡献约 0.1%；eval 原文距离均值 1.0007 对 1.0066，
逐条最大差 0.0491，相对变化 0.59%。词表仍必须 train-only，因为它进入特征。记在 SPEC 2.3。
不改拟合代码。

## 2026-10-05 EDT T0.3 保真指标

`eval/metrics.py` 实现 entity_recall、numeral_recall、hallucination_rate，空输入集和空输出按 SPEC 3.2。
数值不另写归一化，直接用白话化闸门的 `extract_quantities`。四条实测 bug 都有单测：万人大军对十万铁骑
recall 小于 1，太医对宫监产生 title violation，垂死的酸气对死尸散发的酸臭之气不崩，三辆对全角３辆
归一化后 recall 为 1。cn2an 覆盖十万、万、三月初九、四斤、二十、一百二十。
专名表从 train 743 条原文抽出 274 个词，eval 59 条未参与，最长 4 字。不装 LAC，实体通道是
jieba 专名加词表，原因在 Q17。`pytest --cov=eval` 覆盖率 100%。未开始 T0.4。

**同日修正实体通道。** 274 词和 jieba 专名共用盲区，主角被整词漏掉，登基又被标成专名。
主通道改为 `corpus/manual_entities.json`（67 条，出现次数均不少于 3），经 `jieba.load_userdict`
强制成词。274 词降为补充。登基、入宫这类动词即使标成专名也不进实体集。四句回归：
颂莲、端白、皇甫夫人、沉草、枫杨树、娴、芝都抽到，登基不在结果里。`eval` 覆盖率仍是 100%。
仍未开始 T0.4。

**同日再滤补充通道。** 当场的 jieba 标记不能单独进实体集，必须已在 274 词里且单独成词，
宫面圣因此被挡下。jieba 通用词频不低于 200 的有 104 个，274 剩 170，东西、明白、阳光、
孙子、白痴都在被去掉的里面。不稳定切分和粘连碎片再去掉 24 个后，补充通道实际保留 146。
阈值记在 Q17。仍未开始 T0.4。

## 2026-10-05 EDT 数值闸门停改（验收后）

802 条 pairs、eval 59/73、备份哈希与 `split.json` 未重切，均已核对。序数单独成类保留。
不再修数值闸门：剩余 113 条就这样。再修最多捞回约 40 条，对训练可忽略，而闸门已连续出过
两个 bug。已知局限记入 SPEC 1.5.3：

1. 可用 eval 是 59/73。缺口 14 条由内容闸门非随机筛掉，系统性排除数字密集段落。
   后续所有基线对比都在这个有偏子集上。
2. 113 条失败里，44 条只缺基数「1」，29 条序数被写成基数。这两类很可能是过严误判，
   不是真实信息丢失。未修复，记为已知偏差。
3. 训练集 743/842，总入选率 87.7%（802/915）。

## 2026-10-05 EDT T0.2 stylometry

按 SPEC 第 2 节实现 20 维特征、书面语/口语词表和 z-score 欧氏距离。词表只用 train
的 743 对，避免 eval 泄漏：书面语 1300 词，口语 1074 词，最长 6 字，只存词和分数。
风格参照用全部 915 条原文，标准差下界没有被打到（最小 0.039）。`sample_public.jsonl`
相对这个参照，5 条原文距离均值 1.355、中位 1.338，对应白话均值 1.538、中位 1.507，
5 对都是原文更近。不要用这 5 条原文自己拟合再比：有的虚词在 5 条里方差为 0，一次出现
就会把距离放大到 10^5。`pytest --cov=stylometry` 覆盖率 98%。未开始 T0.3。

## 2026-10-05 EDT T0.0 数值闸门修正（重跑前）

全量第一次把 155 条挡在 `pairs.jsonl` 外，其中 150 条原因是 `numeral_mismatch`。
闸门比的是数量短语的字符串：`3` 对 `三`、`16` 对 `十六`、两边集合相同都会失败。
这与 SPEC「保值不保形」冲突，按 bug 修正，不调阈值，也不改 prompt。

重跑前写死的规则：两边用 cn2an 归一化后，期望值集合 ⊆ 实际值集合。序数（第N 和
二/三/四太太、二/三/四姨太的排行）不进基数。万/千/百必须并进数字。概数前缀保留，
上万 ≠ 万。语法性「一+通用量词」豁免。「上一次」一类不算数量。初九与 9号 同为基数。
无法归一化的原文数值判失败。只重跑尚未进入 pairs 的失败项，使用新报告，不重切
`split.json`。

**2026-10-05T19:12:07–19:14:21Z 重跑结果**（代码 `f2b5f19`，155 次调用全部为
`deepseek-v4-pro`，模板哈希仍是 `38fd364e…`，temperature 0.2 / top_p 1 /
max_tokens 2048 / thinking disabled / retries 0 / round 2 / seed 42）。启动时
already_done=760、remaining=155。没有接口失败。`split.json` 未改，备份里的
split 哈希与仓库一致。

| 项目 | 修正前 | 修正后 |
|---|---:|---:|
| `pairs.jsonl` | 760 | 802 |
| eval 有 pair / 设计 73 | 54 | 59 |
| 本次仍失败 | — | 113 |

113 条仍不进入 pairs。原因按 id 计：只缺数值 110，只缺保护称谓 2，两项都缺 1。
111 条数值失败里没有无法归一化的脏 token，都能指出缺了哪个值：

- 44 条只缺基数 1（一次、一块、一人这类；「一个」仍豁免）
- 29 条序数被写成基数（第一 → 一）
- 18 条其他基数值对不上
- 8 条只缺 10000（万；其中 1 条白话写成了概数上万）
- 6 条序数整段丢掉
- 4 条序数和基数都缺
- 2 条同时缺 1 或 10000 以及其他值

设计中的 eval 是 73 条。修正后实际可用 59 条，缺口 14 条，全部在这次重跑里失败，
不是因为没跑到。14 条都是数值：只缺基数 1 的 7 条，序数写成基数的 3 条，其他基数
2 条，序数丢掉 1 条，只缺 10000 的 1 条。不重切，历史对比继续用现有 `split.json`。
报告在 `corpus/rebuild_report_full_pro_numeral_fix_20261005.json`。未开始 T0.2。

## 2026-10-05 EDT T0.0 成对替换与类别规则冻结（预览前）

67 项成对替换、类别规则和必须保留写入第一遍 prompt；第二遍改为只补专名与数值。
PINC-4/6 修补下降容忍从历史实验的 0.03 改为 0，只适用于本冻结版本，不改旧分数。
`literary_residue` 不再使用 jieba 成语标记，只检查 67 个已审核示例。未列入示例的书面语
靠类别规则约束，并用 PINC 观察。草案文末待确认备选（与众不同、一举一动、花花斑斑、邋里邋遢、空落落一大片）按原文保留，不在看结果前加强。
获准一次固定 20 条 `deepseek-v4-pro` 预览；全量 907 条未获准。本节在调用前写入，
不在看到结果前改 prompt、词表或门槛。

**2026-10-05T17:16:00–17:16:37Z 预览结果**（commit `c5f342e`，40 个不同响应全部为
`deepseek-v4-pro`，temperature 0.2 / top_p 1 / max_tokens 2048 / thinking disabled，
沿用固定名单与逐条 seed，零重试）。下列为逐句宏均值；样本标准差。历史四列是原报告
复算，与当时公布的 PINC-4/6、sBLEU 均值一致。

| 实验 | PINC-4 ↑ | PINC-6 ↑ | sBLEU ↓ |
|---|---:|---:|---:|
| Flash round2 | 0.511507 | 0.616770 | 61.372738 |
| Pro 对照 | 0.626358 | 0.732838 | 50.919750 |
| 旧两段式第一遍 | 0.583549 | 0.677986 | 53.676443 |
| 旧两段式第二遍 | 0.576793 | 0.669994 | 54.240518 |
| 本次第一遍 | 0.593124 | 0.693721 | 53.180626 |
| 本次第二遍 | 0.589799 | 0.690824 | 53.511691 |

相对 Pro 对照，本次第二遍 PINC-4/6 更低、sBLEU 更高，改写幅度没有改善。
5/20 达到 PINC 双闸，0/20 通过全部确定性闸门；2 条第二遍使 PINC 下降，18 条第二遍
指标不变。完整逐条、分布与 work 分组在本地
`corpus/rebuild_report_two_pass_pairs_20261005.json`。该次预览本身未启动全量。

## 2026-10-05 EDT T0.0 停止优化白话化，改跑全量（调用前）

六轮同模型、同 20 条、同 seed 之后，pro_control 单遍 PINC-4 0.626358 高于成对替换单遍
0.593124 和两段式第二遍 0.589799。不再改白话化 prompt。全量恢复该单遍模板与采样：
`deepseek-v4-pro`，温度 0.2，top_p 1，max_tokens 2048，thinking disabled，retries 0，
round 2，seed 42。PINC-1/2/3/4/6 与 sBLEU 逐条记录但不拦截。放宽 PINC 是下游结果出来之前
的流程决定：预览约 0.63，13 条参照 PINC-4 为 0.784850，这个差距记为已知偏差，
待下游指标后再决定是否重做语料。内容闸门只保留已审核专名/称谓和数值短语；
情节无确定性检查。`chunks.jsonl` 现为 915 条，全部生成。六轮报告不删不改。
本节在全量调用前写入。环境里的 `VERNACULARIZE_MODEL=deepseek-chat` 是 Flash 别名，
全量进程内强制改为 `deepseek-v4-pro`，不改 `.env`。

**2026-10-05T18:28:06–18:41:21Z 全量结果**（code `ea415aa`，915 条全部返回
`deepseek-v4-pro`，模板哈希与 pro_control 相同，零重试，无模型身份错误）。
PINC/sBLEU 是全部 915 条的逐句宏分布，含未进入 pairs 的 155 条。样本标准差：

| 指标 | 均值 | 样本 SD | 中位 | Q1 | Q3 | 最小 | 最大 |
|---|---:|---:|---:|---:|---:|---:|---:|
| PINC-4 | 0.696189 | 0.156800 | 0.724000 | 0.622404 | 0.806495 | 0.000000 | 0.981752 |
| PINC-6 | 0.800388 | 0.153746 | 0.835125 | 0.745688 | 0.905031 | 0.000000 | 1.000000 |
| sBLEU | 44.698226 | 14.542130 | 43.251329 | 34.447385 | 52.561433 | 9.377647 | 100.000000 |

PINC-4 均值 0.696189，与 13 条参照 0.784850 的差距保留为已知偏差。
这不是那 20 条预览的 0.626：全量里《我的帝王生涯》有 379 条，作品构成不同。
760 条进入 `corpus/pairs.jsonl`，155 条因内容闸门排除（数值短语 150、保护词缺失 6，
有 1 条两项都有）。`split.json` 未重切。两份文件已按 SHA256 核对备份到
`D:\Transformers\corpus-backup\`。未开始 T0.2。

## 2026-10-03 EDT / 2026-10-04 UTC T0.0 清单逐项审核草案（等待确认）

**2026-10-05 EDT 人工确认与修订**：替换表改为keep67/drop25/fix4（仿佛转keep；
莫名其妙、小心翼翼、不可思议转drop）；保护表改为keep54/drop53/fix14，序数+称谓
（大少爷、大太太、二/三/四太太、二/三/四姨太）例外keep，以保护排序事实。普通家庭
称谓、长工、管家仍非硬保护。此前的69不是定稿数；后续成对替换草案应为67项，不能为
迎合旧数目补入未经审核词。

用户另确认频次清单只能是类别示例，不得作为穷举禁用词表：其测量显示54,654种四字
连续串中53,265种（97.5%）仅出现一次。下一版prompt必须以类别规则覆盖长尾四字格、
文言虚词、书面单字动词、生僻颜色/质感和书面比拟词；该prompt和67项成对替换尚在
第二道审核草案阶段，未调用模型、未写入现行生成入口。

只执行审核步骤1，新增API调用0次，未修改现行96词资产、prompt、生成入口或历史报告。
完整草案在docs/reviews/T0.0_lexicon_audit.md：替换96项keep69/drop23/fix4；既有
保护候选并集121项keep46/drop61/fix14，另列相同20条语境手工补漏11项。助手标注
不是人工批准的金标准，范围不冒称覆盖全部语料；短词与理由不含可还原原文的句段。

进一步核对：旧口语排除集只约束l分支，i分支仍收入奇怪的是、闭上眼睛等；保护候选
含风格词、普通名词、姓名粘连/截断及漏项。污染机制与既有负结果相容，但无独立消融，
不声称已分离各项因果贡献。家庭称谓、口语常用成语、河东方位等边界在Q14明确待确认。

SPEC1.5.2登记两次人工确认闸门及后续窄修补方案：第二遍仅恢复专名/数值，PINC-4/6
各不得低于第一遍。未实施新prompt/校验，不追溯改写旧0.03规则与结果。确认两表后
才给成对白话目标，再停一次；本轮止于草案，未启动预览、全量或T0.2以后的任务。

## 2026-10-03 T0.0 两段式一次预览结果与恢复审阅

实际运行UTC21:49:09–21:49:43（本地EDT17:49），生成快照86e1941；20条、两遍共40个
不同响应，全部deepseek-v4-pro，temperature0.2/top_p1/max_tokens2048/thinking disabled，
沿用逐条seed、零重试。恢复时发现已完成，新增API调用0次，不再跑第二轮。
下列PINC/sBLEU是固定中文字符口径的逐句宏均值，旧输出补算不替换历史判定。

| 实验 | PINC-4 ↑ | PINC-6 ↑ | sBLEU ↓ | SequenceMatcher（辅助） |
|---|---:|---:|---:|---:|
| Flash round2 | 0.511507 | 0.616770 | 61.372738 | 0.736356 |
| Pro单变量对照 | 0.626358 | 0.732838 | 50.919750 | 0.687311 |
| 两段式第一遍 | 0.583549 | 0.677986 | 53.676443 | 0.694789 |
| 两段式第二遍 | 0.576793 | 0.669994 | 54.240518 | 0.697557 |

相对Pro对照，新设计没有改善整体改写幅度，方差增大。15/20未达PINC双闸，2条修补
导致PINC下降>0.03，15条第二遍全文不变；按原冻结启发式20条均有拒绝理由，不进入全量。
最终PINC-4样本SD0.224854，PINC-6样本SD0.221581，SequenceMatcher样本SD0.170417。
完整20条双遍PINC-1/2/3/4/6、sBLEU、SequenceMatcher、分布及work分组保留本地报告。
实体候选宏recall0.887381，数值宏recall0.940000；有非专名误报和修辞数值误报/漏报，
不能把这些值当真实语义保真率。固定6条助手抽查与真正人工review分开，后者pending/null。

**必须披露的流程违规**：旧冻结前新增词表测试预期错误、失败后编排仍继续提交及调用，
违反全绿要求。恢复时修正测试，永久使用ASCII转义导出，并增加UTF-8/CP936无损测试。
默认临时目录权限失败也留痕，换用新的仓库日志目录运行同一套CPU测试，47项全过；
ruff check、format --check、mypy均通过。没有重写生成commit、原始输出或验收标准。
补修提交只修工程测试/编码并记录结果；不修改prompt或词表、不重跑、不启动907条。

## 2026-10-03 T0.0 模型别名污染与单变量对照

### 2026-10-03T21:19Z 两段式方案执行前预注册

**2026-10-03T21:39Z 执行前追加修订**：尚未调用新方案API，依用户更新将主判据改为
Chen & Dolan (2011) PINC的中文字符适配，输出出现次数为分母，4阶≥0.75、6阶≥0.85。
修补后二者各不得下降超过0.03；旧源侧分母不是PINC的补数。13条复算PINC-1/2/3/4/6为
0.211414/0.562161/0.703106/0.784850/0.877025，与用户值略有差异，不凑数。
新增sBLEU（固定中文逐字、sacrebleu2.6.0、exp平滑、有效阶，0–100逐句宏均值），
SequenceMatcher保留历史口径。源/参考双向评价、文献与无方法创新局限已登记。
T0.2分类器、T0.3 BERTScore和T1.7外部baseline只进PLAN，本轮不实现。
以下21:19的源侧重合方案保留为历史提案，已在新数据生成前被PINC修订替换。

**21:47Z 工程失败留痕**：首次启动在外部调用前因导出词表编码损坏而退出，API调用0次。
同规则重导出并添加完整性测试；损坏资产和零调用报告本地保留。不是看结果后调整方案。

用户取消原四臂消融，改为一次设计变更实验。**主判据从 SequenceMatcher 改为中文去重
N-gram 原样重合率**：6-gram≤0.15、4-gram≤0.25，修补后两项各不得回升>0.03。
13条参照复算0.124365/0.218425；整体相似度区间[0.51,0.61]保留为辅助观察。
删掉等段落数约束，长度兜底改0.6–1.6，数字保值不保形，加入从 raw 抽取的必须替换清单
和具体口语替换例子。两次调用分别负责激进口语化和语义事实修补；温度等采样不变。
这是**看新结果前**的方案修改，不是看到结果后移动门槛。旧 Flash/Pro 全部记录和旧判定
原样保留；新 N-gram 对旧输出的补算注明回顾性。完整方法、假设、停止条件见 SPEC 1.5.1。
Pro 对照均值0.687311、样本标准差0.126368，虽下降但仍超旧区间且最大值增至0.888087，
不能把模型差异当成唯一原因。本次整体设计同时改变多项，不能分辨每项独立贡献。

**发现**：留一法、round1、round2 请求 deepseek-chat，实际全部返回 deepseek-flash
（共 53 次）。以下结果全部标注 **在 deepseek-flash 上得到**，不能引用为 deepseek-chat：

| 实验 | n | 相似度均值 | 样本标准差 |
|---|---:|---:|---:|
| 留一法新 vs 旧 | 13 | 0.687855 | 0.076177 |
| 留一法新 vs 原文 | 13 | 0.750584 | 0.102722 |
| round1 新 vs 原文 | 20 | 0.733518 | 0.096583 |
| round2 新 vs 原文 | 20 | 0.736356 | 0.087840 |

**原因证据**：[官方更新日志](https://api-docs.deepseek.com/updates/) 说明 chat 旧别名指向
Flash 非思考模式。同账户 /models 列出 deepseek-flash、deepseek-v4-pro；两次微型探针分别
返回 chat→flash、pro→pro，支持别名重映射而非账户限制。名称和速度不能证明照抄的因果机制。

**修正**：响应与请求 model 严格不等则抛错并停止整批，禁止生成失败重试掩盖路由错误。
报告保留响应身份、取消与在途项；无有效完整分布时不作区间验收。新增单变量对照入口，
调用前核对 round2 的全部 20 条展开 prompt、范例、逐条 seed 和采样参数。只切换为 Pro，
显式关闭新模型默认思考以保留旧别名的非思考语义。旧执行模式是文档推断，非不可变快照证据。

**范围**：本次只运行一次模型对照并报告后停止，沿用已有门槛，不执行消融或全量。
2025-08 生成模型未知和跨模型参照风险已补入 SPEC / README；版权文本及运行档案仍不提交。

---

## 2026-10-02 项目重启

**事件**：2025-08 的全部产物随 AutoDL 实例释放而永久丢失——907 对语料、v1 adapter、两份生成结果。
AutoDL 文件存储从未开通，本地与 HF Hub 均无备份。

**抢救**：从 2025-08 的运行日志里抠回 30 对 `(白话, 原文)`（14 条完整，6 条附带 v1 模型输出），
存入 `corpus/salvaged_pairs.jsonl`，作为重建的验收依据，不作训练数据。

**同时保住的**：训练超参、loss 轨迹、chunk 规模与命名规则、两个已测出的模型缺陷，
全部从日志复原并写入 `docs/SPEC.md` 第 1.2 节。方法没丢，只丢了产物。

**教训（已写入流程）**：
- 语料与 adapter 必须在产出当天备份到本地 + HF Hub，写进 T0.5 的验收标准
- 数据切分必须固定 seed 并提交 `split.json`，旧切分因未固化而无法还原
- 一次性脚本必须进版本控制，旧 pipeline 全是临时代码，重建时只能照日志反推

**计划调整**：新增 T0.0 语料重建；T0.4 降级为纯链路搭建（不出基线）；
T0.5 合并训练与基线产出；不再复刻 v1 的 3 epochs，直接训 2 epochs。

## 2026-10-02 原始文本已恢复

**来源**：Windows 回收站。三个文件的 `$R` 实体均未被清除，时间戳 2025-08-11 20:25–20:43，
正是当初做这个项目的那一晚。线索来自 `AppData\Roaming\Microsoft\Windows\Recent` 下残留的
`.lnk` 快捷方式——它内嵌了原始路径，指向 Desktop 与 Downloads 下已删除的文件。

**恢复结果**：三个文件覆盖全部六部作品，合计 226,193 中文字，已转 UTF-8 存入 `corpus/raw/`（未入库）。
226,193 ÷ 907 ≈ 249 字/chunk，与从日志反推的 chunk 规模吻合，确认是当初的同一批素材。

**决策**：白话化走**方案 A**（复刻当初风格）。缺陷 #2 不在数据层修，转由检索与 agent 环在推理侧缓解。
连带 `human_eval.jsonl` 不再阻塞 T0.0。

**新发现（已写入 SPEC 1.6）**：
- `妻妾成群.txt` 是四部中篇的合集，切分脚本要按「标题独占一行」分篇
- 必须先匹配 `另一种妇女生活` 再匹配 `妇女生活`，否则标题行被吃掉
- `园艺` 在正文中也作普通词出现，不能用子串搜索定位标题
- `※※※` 是分节符，天然 chunk 边界
- 源文本含繁体残留 `麽`，需规范化为 `么`
- 引号用法在同一部作品内就不一致，`quote_density` 这一维会偏噪，已记为已知局限
