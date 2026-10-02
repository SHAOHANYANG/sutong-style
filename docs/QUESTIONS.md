# QUESTIONS

实现中遇到的歧义、需要人工确认的事项、已知局限。

格式：一条一个小节，写清楚「问题 / 我的假设 / 影响范围」。
不要因为等答案而停工——按假设实现并记在这里。

---

## Q1 既有数据的真实 schema 未经核实 `[阻塞 T0.1，待人工]`

**问题**
`docs/SPEC.md` 第 1.1 节里 `pairs.jsonl` / `eval_results.jsonl` / `base_results.jsonl` 的字段名，是从 2025-08 的运行日志**反推**的，没有打开过文件核实。日志里只能确认：

- 存在「白话」与「原文」两个概念字段（日志打印时的标签是 `[输入-白话]` / `[原文-ground truth]` / `[模型生成]`，但这是打印标签，不等于 JSON 字段名）
- chunk id 形如 `我的帝王生涯_0429`、`妻妾成群_0049`
- 总量 907，切分 train 834 / eval 73
- 产物路径 `/home/shaohan_yang/projects/sutong-style/corpus/`

**现状**
文件在一台远端机器上（SSH 配置指向 AutoDL 租用实例 `connect.west[b|c|e].seetacloud.com`）。尝试 SSH 读取时被本地沙箱的凭据探测策略拦截，未能核实。

**假设**
按 SPEC 1.2 的目标 schema 实现，`scripts/normalize_corpus.py` **必须支持字段名映射配置**（如 `--field-map vernacular=input,original=target`），不要把字段名硬编码。

**影响范围**
T0.1 之后的一切。但影响可控：只要 normalize 脚本有映射层，核实后改一行配置即可。

**需要人工做什么**
把 `pairs.jsonl` 的头两行、以及 `eval_results.jsonl` 的头一行贴过来，然后更新 SPEC 1.1 并移除本条。

---

## Q2 train/eval 切分能否精确还原 `[待确认]`

**问题**
后续所有对比都依赖与 v1 训练时**完全一致**的 73 条 eval 集。如果 `pairs.jsonl` 里没有 `split` 字段，就只能从 `eval_results.jsonl` 的 id 列表反推。

**假设**
`eval_results.jsonl` 的 73 个 id 即 eval 集全集。反推后固化到 `corpus/split.json` 并提交，此后不再变动。

**风险**
如果当时的切分是随机的、且没有固定 seed、且 `eval_results.jsonl` 不完整，则无法精确还原。那种情况下必须重新切分并**重跑 base 与 v1 的生成**，否则基线表不可比。这会让 T0.4 多花一轮 GPU 时间。

---

## Q3 `corpus/human_eval.jsonl` 的 30 条正文待人工填写 `[待人工]`

**状态**
30 条情境种子已预填（`id` / `domain` / `seed` 三个字段齐全），`vernacular` 全为空串。

**约束（重要）**
**这 30 条不得由任何 LLM 代写**，包括 Codex 自己。理由与撰写规范见 `corpus/HUMAN_EVAL_GUIDE.md`：它存在的唯一目的是检验域外泛化，若由 LLM 生成会带上与训练数据同源的分布偏差，分数虚高，测试集作废。

**在填好之前怎么办**
用 `corpus/sample_public.jsonl`（5 条自编样例）跑通链路。涉及 `human_eval` 的评估一律跳过并在报告里标 `null`，不要用空串当输入去跑出一堆 0 分。

**额外要求**
T0.1 需实现 `scripts/validate_human_eval.py`，校验：30 条齐全、每条 100–300 字、实体 ≥ 2、数字 ≥ 1、6 个 domain 分布均匀。前两项之外的两项是硬指标——没有实体和数字，`entity_recall` 与 `numeral_recall` 会因「输入集为空记 1.0」的约定而恒为满分，那两列数字就是假的。

---

## Q4 LICENSE 未定 `[待人工]`

README 末尾留了 TODO。代码本身建议 MIT，但本项目涉及受版权保护的训练语料，adapter 权重的授权条款需要单独考虑。**这是仓库所有者的决定，不要自行选定并提交 LICENSE 文件。**
