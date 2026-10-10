# 项目现状（截至 2026-10-07，Phase 0–1 完成）

这份文档回答三个问题：**做到哪了、试过什么、下一步干什么**。
新接手的实现者或审计者先读这份，再读 `AGENTS.md` / `SPEC.md` / `PLAN.md`。

---

## 1. 这个项目是什么

把现代白话中文改写成苏童文风。Qwen2.5-3B + LoRA 微调，外挂风格范例检索与生成自检重写环。

核心数据构造是**反向伪平行语料**：苏童原文当训练目标，用 LLM 把原文降级改写成白话当训练输入。907 对历史语料已随数据丢失灭失，当前 802 对是 2026-10 重建的。

项目的声明价值**不在方法创新**（伪平行语料、检索增强、自检环都是已有方法），而在**评估严谨度与完整工程实现**。这一点写在 README「已知局限」里，不要在文档里把它拔高成方法创新。

---

## 2. 做到哪了

Phase 0 全部完成。Phase 1（检索）全部完成，出口条件（README 指标表追加 retrieval 行）已满足；主配置相对 k = 0 **未检出差异**（SPEC §4.7）。`human_eval.jsonl` 正文尚未写入，PLAN T1.6 要求的检索召回评估仍欠。Phase 2 四个代码任务与 T2.5 GPU 评估已完成：主臂 `numeral_recall` 0.892 → 0.920（改善，SPEC §4.8），PLAN 出口条件「`hallucination_rate` 明显下降」**未达到**；restate 臂的数字因反馈回显不可引用。第二轮（SPEC §4.9：回显防护 + system 反馈格式）主臂相对检索行**未检出差异**，六个臂的最终输出不再含回显。两轮合起来：这个环在 59 条上没有可靠地改善保真。人工抽查 pending。基座规模对比（SPEC §4.10）：把基座换成 Qwen2.5-14B、其余不变，微调后的判定指标相对 3B **未检出差异**；探针集上两者对训练分布之外的输入都改动很少。

### 提交历史

```
8c3a32c  feat(train): 训出 sutong-v2 基线并填 README 指标表 (T0.5)
9d3ff8e  fix(eval): 把 copy_ratio 均值放进 aggregate (T0.4)
cb7e599  fix(eval): 相同候选记为 identical，并记录 copy_ratio (T0.4)
93d64b3  feat(eval): 编排评估，评委默认 deepseek-v4-pro 并记录胜负平 (T0.4)
fe67dc3  fix(eval): 实体主通道改用人工表，并滤掉补充通道的碎片和高频词 (T0.3)
edf3f5d  feat(eval): 实现实体、数值和称谓保真指标 (T0.3)
c35bfc5  docs(spec): 写明风格参照用全部原文是有意的 (T0.2)
56a48d5  feat(stylometry): 实现 20 维文体特征与风格距离 (T0.2)
```

T0.0（语料重建）的 commit 更早，不在此列。

### 门禁现状

```
ruff check / ruff format --check / mypy   全过（36 源文件）
pytest -m "not gpu"                        88 项通过
eval 覆盖率                                 97%
工作区                                      干净
```

### 语料规模（这组数字经常被搞混，以此为准）

```
907     v1 历史 chunk 数（已灭失，只存在于 2025-08 日志里）
834/73  v1 历史切分（同上）
──────────────────────────────────────────────
915     当前 chunk 数
842/73  当前切分（corpus/split.json，seed 42，永不重切）
802     过内容闸门的 pair
743/59  实际可用：训练 743，评估 59
```

**警告**：`SPEC.md` 的 §1.2 是「从日志反推出的已知参数」，那里的 907 / 834 是数据灭失后仅存的凭证，**一个字都不能改**。不要做全局替换。

---

## 3. Phase 0 的实测结果

59 条 eval 切片，贪心解码，seed 42。

| | style_distance | profile_gap | profile_gap_per_case | entity_recall | numeral_recall | hallucination_rate | copy_ratio | style_win_rate | tie_rate |
|---|---|---|---|---|---|---|---|---|---|
| base（无微调） | 1.149 | 0.494 | 0.748 | 0.958 | 0.923 | 0.0042 | 0.875 | — | — |
| **+ LoRA (sutong-v2)** | 0.923 | **0.204** | **0.514** | **1.000** | 0.861 | 0.0141 | 0.709 | 0.331 | **0.49** |
| *参照：苏童原文* | *1.035* | *0* | *0* | — | — | — | — | — | — |
| *参照：白话输入* | *0.992* | *0.450* | *0.623* | — | — | — | — | — | — |

### 怎么读这张表

**微调是有效的。** 判据是集合级 `profile_gap`：先对全部样本求剖面差的逐维均值，再对 20 维取绝对值求平均。方向相反的误差会抵消。白话 0.450 → LoRA 0.204，这句话里的「缩了一半还多」只在集合级成立；没微调的基座是 0.494。逐条口径 `profile_gap_per_case` 是 0.623 → 0.514，base 是 0.748。

**代价是事实漂移。** `numeral_recall` 0.923 → 0.861，`hallucination_rate` 0.0042 → 0.0141。这正是 README 列为「已知缺陷 #1」的现象，**首次被量化**。Phase 1 和 Phase 2 的靶子就是把这两项拉回来，同时保住文风改善。

**模型没有学会苏童的句长变化。** `lora-eval59.json` 的 `profile_mean_delta`：`sent_len_p90` 白话 -0.61、LoRA -0.61，微调后没有移动；`sent_len_std` 白话 -0.58、LoRA -0.50，移动很小。消融报告的折外 R²：`sent_len_std` 0.197、`sent_len_p90` 0.312，在 20 维里属于偏低的一组（`para_density` 0.883、`dialogue_verb_density` 0.800）。白话输入的这 20 维特征对目标原文的句长变化只有很弱的线性预测力。原先手填的 -0.64 → -0.64、-0.70 → -0.62 对不上这份报告，上面的分维数字取自报告本身。

---

## 4. 两个负结果（重要，不要重新发现一遍）

### 4.1 `style_distance` 的方向标曾经写反了

它定义为 20 维 z 分数向量的欧氏范数除以 √20，即 z 分数的均方根。对一条服从参照分布的文本，每维 E[z²]=1，所以**期望值是 1.0，不是 0**。59 条原文实测 1.035，与理论吻合。

因此：

- 低于 1.0 **不代表更好**，代表向语料均值回归，方差小于真人。LoRA 的 0.923 就是这种情况
- 白话输入实测 0.992，几乎正中靶心，但集合级 `profile_gap` 与原文相差 0.450（逐条是 0.623）。**标量把「离质心多远」和「往哪个方向偏」压成了一个数**，无法区分「完全没改」和「改得完美」

已更正：`style_distance` 降为弱信号。判别看剖面差：集合级 `profile_gap` 没有单条的值，进不了 agent 环；逐条的 `profile_gap_per_case` 可以，但需要 ground truth。`human_eval` 没有原文，两项都算不了，只能退回标量——这是该集合的已知局限。

### 4.2 `style_win_rate` 这一轮不可引用

**好消息**：judge 链路首次对真实 API 验证通过。59 条、118 次调用（= 59 × 2，位置互换全部发出）、零解析失败、模型身份闸门确认响应确为 `deepseek-v4-pro`。

**坏消息**：`tie_rate = 0.49`——近一半样本在 A/B 位置互换后评委改口，这一半是掷骰子。判得一致的 30 条里 25:5 偏向白话输入，**与 `profile_gap` 方向相反**。

排查结论：输出本身不是退化，59 条中仅 3 条有轻微重复或缺末尾标点。最可能的原因是 **Q18 选定的对手有混淆**——那些白话是用 LLM 从原文降级改写来的，内容、意象、叙事顺序几乎原样保留，只替换了词汇层。评委虽被明确要求「只判文风」，仍大概率在响应内容相似度。这是 LLM 评委的典型失效模式。

**待办**：对手选择需要重做。候选方案（都未验证）：

- 换成 base 模型的输出作对手（同内容、不同风格，混淆小）
- 用三元比较而非配对
- 在 prompt 里加入「两段内容相同，只比文体」的显式说明

---

## 5. 已验证的 GPU 环境（不要重装、不要重新探测）

Windows 11 + WSL2 Ubuntu 24.04。环境在 `~/venvs/sutong`（**WSL 本地盘，不在 /mnt/d**，venv 上万个小文件跨文件系统慢十倍）。

```
Python       3.11.17
torch        2.11.0+cu128
torchvision  0.26.0+cu128      ← 必须带 +cu128 后缀
unsloth      2026.9.14
xformers     0.0.35
triton       3.6.0
trl          0.24.0
transformers 5.5.0
GPU          RTX 5060 Laptop，计算能力 (12,0) = sm_120，7.96 GB
```

实测：bf16 矩阵乘通过；Qwen2.5-3B 4bit + LoRA 前向反向通过；训练显存峰值约 3.45 GB，宽裕。

### 踩过的坑（会再踩，写在这里）

1. **`torchvision` 装成 CPU 版**。从 PyPI 默认源装会得到无 CUDA 的构建，`import unsloth` 报 `operator torchvision::nms does not exist`。必须：
   ```bash
   uv pip install --reinstall --no-deps \
     --index-url https://download.pytorch.org/whl/cu128 "torchvision==0.26.0"
   ```
2. **TRL 0.24.0 改了 API**，旧教程全错：
   - `SFTConfig(max_seq_length=)` → `max_length=`
   - `SFTTrainer(tokenizer=)` → `processing_class=`
   - `unsloth` 必须在 `trl` / `transformers` / `peft` **之前** import，否则补丁不生效
3. **后台命令结尾加 `| tail -N` 会吞掉退出码**，脚本崩了仍报 exit 0。第一次训练失败因此被误判为成功。
4. **Windows 上 pytest 的 `tmp_path` 默认路径会 PermissionError**，批量报 ERROR 但不是代码问题。加 `--basetemp=` 指到可写目录。
5. **`.env` 里的 `JUDGE_MODEL` 和 `VERNACULARIZE_MODEL` 仍是废别名 `deepseek-chat`**（2026-04 被官方重定向到 Flash）。按既有记录不静默修改用户配置，运行时在命令行显式覆盖。**待所有者自行更新为 `deepseek-v4-pro`。**

---

## 6. 关键历史决策（不要翻案）

### 6.1 v1 的 LoRA rank 是 16，不是 SPEC 原文写的 32

SPEC 1.2 原记录「r=32，可训练参数 29,933,568」自相矛盾。按 Qwen2.5-3B 的结构（hidden 2048、intermediate 11008、36 层、kv 维 256、七个 target module）实算：

```
r=8    14,966,784
r=16   29,933,568   ← 日志记录的参数量
r=32   59,867,136
```

2026-10-05 本机实测两个 rank 都跑过，与推算完全一致。参数量是 PEFT 的机器输出，`r=32` 是手记错误。已按参数量更正为 r=16，原记录在 SPEC 里划除保留。`scripts/train.py` 启动即断言可训练参数等于 29,933,568。

### 6.2 epochs = 2，不是 3

v1 训了 3 轮但 eval_loss 在 epoch 1.9 触底 2.056、之后回升 2.089，第三轮纯过拟合。v1 已不存在，没有复刻的意义。

v2 实测：train 1.36 → 0.78；eval 0.9808（epoch 1）→ 0.9881（epoch 2 回升）。**与 v1 形状一致、绝对值不可比**——v2 用 `train_on_responses_only`（loss 只算 assistant 段）而 v1 的掩码方式无记录，且白话侧已全部重新生成。

v2 的 eval 在 epoch 1 就触底，比 v1 的 1.9 还早。**仅记录，不据此改 epoch 数**：epochs=2 在见数据前已定，回头按 eval 选 epoch 就是用评估集做模型选择。epoch 1 的检查点保留在 `adapters/sutong-v2/checkpoint-186`。

### 6.3 曾抓到 API 悄悄换模型

请求 `deepseek-chat`，响应标识全部是 `deepseek-flash`，53 次调用无一例外。官方 2026-04-24 把这个旧别名重定向了。**只因为预注册规则要求记录响应模型标识才发现**——否则留一法、round1、round2 三组实验的结论全是废的。

`OpenAICompatibleGenerator` 现在有硬闸：`response.model != self._model` 即抛 `ModelIdentityMismatchError`，所以不会再静默替换。

### 6.4 其他已定案、不再调整的

- **实体抽取**：人工表 `corpus/manual_entities.json`（67 条，每条经语料实测出现 ≥3 次）是主通道，jieba 自动抽的 274 词只作补充。补充通道用 jieba 词频阈值 200 过滤，代价已量化评估（京城 ×53、庐山 ×8、云南 ×5、香港 ×5、苏州 ×4、北平 ×3、上海 ×1 会一起被挡掉），**阈值不再调整**。见 Q17
- **风格参照**用全部 915 条原文（含 eval），刻意为之，影响实测 0.59%。但书面语词表必须只用 train 的 743 对，因为它的命中直接进特征向量。这个不对称是有意的。见 SPEC 2.3
- **风格距离用对角标准化不用马氏距离**：20 维、样本量有限，协方差矩阵求逆数值不稳
- **`corpus/split.json` 永不重切**。重切一次所有历史对比全废

---

## 7. 绝对不能违反的约束

1. **苏童作品受版权保护。** `corpus/raw/` 和 `corpus/*.jsonl` 永不进 git。仓库只放不可还原原文的衍生物：词表、专名表、特征均值方差、指标、hash。白名单例外：`sample_public.jsonl`（5 条自编）、`human_eval.jsonl`、`split.json`、`gazetteer.json`。`eval/reports/*.json` 已核验只含 sha256 与 chunk id，可提交
2. **凭据只在 `.env`**，从环境变量读，不硬编码、不打印、不贴进任何 agent 对话
3. **`corpus/human_eval.jsonl` 的白话必须所有者本人手写**，不能让任何 LLM 代笔（包括你）。理由见下
4. **`corpus/split.json` 永不重切**
5. **人名地名表不能由 agent 编造**，加新条目必须先在语料里验证出现次数

---

## 8. `human_eval` 的两处变更（待实现）

所有者的决定，尚未落地到代码。

### 8.1 数量 30 → 10

`scripts/validate_human_eval.py` 把 `EXPECTED_ROWS = 30`、`EXPECTED_DOMAINS = 6`、每 domain 5 条写死了。改为 10 条；domain 均衡检查改成**记录实际分布但不强制均衡**。`corpus/human_eval.jsonl` 保留前 10 条骨架，`seed` 字段保留。README / SPEC 里的「30 条」改为「10 条」，并在已知局限写明 n=10 是案例性证据，不是统计检验。

### 8.2 删掉「口语化」要求

`corpus/HUMAN_EVAL_GUIDE.md` 的「该有的味道 / 不该有的味道」两节删掉，硬性要求里的「口语化——想象在跟朋友转述一件事」改为「平实的现代中文叙述」。

**理由（这是设计更正不是放宽）**：缺陷 #2 的根因是「训练里的白话是从苏童原文降级生成的，带着原文骨架」。要避开的分布是**「背后有过一篇文学原文」**，不是「书面语」。一段平实的现代书面中文只要不是从文学原文降级来的，就已完全在域外。塞语气词是在测一个不存在的使用场景，且会让失败归因变脏——分不清是没剥掉口语标记还是没上文体。

**保留不动**：100–300 字、≥2 个人名或地名、≥1 个数字、不是 LLM 写的、不要模仿苏童。前两条是 `entity_recall` / `numeral_recall` 唯一的抓手，去掉它们那两列恒为 1.0。

需在指南补一句风险说明：输入偏书面时离苏童更近，模型不作为也能得分，这由 `copy_ratio` + `identical_rate` + `tie_rate` 三个量监控。

---

## 9. Phase 1 — 检索（RAG）（已完成）

出口已满足：README 表格多一行 `retrieval`（balanced，k = 2）。判定与数字见 CHANGELOG T1.6 与 `eval/reports/retrieval-sweep.md`。

### 9.1 为什么不能用普通 RAG

直觉做法是拿输入去检索**语义相似**的段落当范例。这在这里是错的：输入「他每天去河边散步」会召回一堆河边场景，但**需要的不是题材像，是写法像**——同样是日常动作白描的段落，不管写的是河边还是院子。

### 9.2 三路召回 + RRF 融合

```
输入白话
  ├─ BM25 路     关键词召回            T1.1 [CPU]
  ├─ dense 路    bge-m3 向量语义召回    T1.2 [GPU-OPT]
  └─ style 路    20 维文体特征召回      T1.3 [CPU]  ← 设计核心
       └─ RRF 融合  T1.4 → few-shot prompt  T1.5 → 评估  T1.6
```

### 9.3 style 路的关键手法

**不能拿输入白话的特征直接去检索。** 白话的特征是「白话该有的样子」（短句、少虚词），拿它去找会召回语料里最像白话的段落——恰恰是最不该参考的。

正确做法：用 numpy 闭式解做岭回归，不引入 sklearn。**从输入白话的 z 向量预测「它对应的原文应该有什么 z 向量」**，再拿预测向量去 style 索引检索。743 对平行语料正好是这个回归的训练数据。2026-10-06 的预注册消融判定预测向量有优势，这个模块保留。详见 CHANGELOG 和 `eval/reports/style-predictor-ablation.json`。

### 9.4 任务清单与验收

| 任务 | 产出 | 关键验收 |
|---|---|---|
| **T1.1** `[CPU]` | `retrieval/bm25.py`，jieba 分词 + 中文停用词表（自带在 `retrieval/data/stopwords.txt`，**不要运行时下载**） | 索引 `sample_public.jsonl` 后用其中一条白话查询能召回对应 original；停用词生效的单测 |
| **T1.2** `[GPU-OPT]` | `retrieval/dense.py`、`infra/bge_embedder.py`、`tests/fakes.py` 的 `FakeEmbedder`。向量存 pgvector | **所有测试用 `FakeEmbedder`，不下载模型、不连网络、不碰 GPU**。降级路径：bge-m3 在 CPU 上编码全量几分钟可接受，向量缓存到 `retrieval/data/embeddings.npy`（gitignored） |
| **T1.3** `[CPU]` | `retrieval/style_index.py`（numpy 暴力最近邻，**不要上向量库**，915×20 暴力算是微秒级）、`retrieval/style_predictor.py`（Ridge）、`scripts/ablate_style_predictor.py` | 消融脚本输出两组端到端风格分：预测向量 vs 直接用输入特征。**如果预测版没有优势，删掉 `style_predictor.py`**，CHANGELOG 记负结果。这是明确授权的删除 |
| **T1.4** `[CPU]` | `retrieval/fusion.py` | RRF 公式单测（构造已知 rank 列表断言融合顺序）；三路任一为空时不崩 |
| **T1.5** `[CPU]` | `retrieval/prompt.py`、`scripts/sweep_topk.py`（k ∈ {0,1,2,3}） | prompt 构造有单测（断言范例数量、顺序、token 预估）；**必须统计并记录 prompt token 数**——3 条范例轻松超 1500 token，会同时降质降速 |
| **T1.6** `[GPU]` | README 追加 `retrieval` 行 | ✅ 主配置填表；CHANGELOG 含 13 组与 token 分布。`human_eval` 召回评估欠 |

### 9.5 Phase 1 该盯的指标（主配置实测）

| 目标 | 阈值 | 主配置 | 结果 |
|---|---|---|---|
| `numeral_recall` | ≥ 0.92 | 0.891525 | 未达到 |
| `hallucination_rate` | ≤ 0.005 | 0.004237 | 达到 |
| `profile_gap` | < 0.204 | 0.214287 | 未达到 |

相对 k = 0：`profile_gap_per_case` / `numeral_recall` / `hallucination_rate` 均值差方向朝好，但区间均含 0 → 未检出差异。`copy_ratio` 上升且区间不含 0。`exemplar_entity_leak` = 0；未检出相对对照的范例抄写。

---

## 10. 下一步：Phase 2 — Agent 自检重写环

出口：README 多一行 `agent`，且 `hallucination_rate` 明显下降。T2.1–T2.5 已完成，出口条件的后半句未达到（见 CHANGELOG T2.5）。第二轮已堵上反馈回显（SPEC §4.9），主臂未检出差异。遗留：两轮主臂改动样本的人工抽查；数值规则对「一 + 量词」偏严，剩下修不掉的违规多数属于这一类。

```
白话输入 → [检索范例] → [LoRA 生成] → [保真校验 + 风格打分] → [定向修订，最多 3 轮] → 输出 + trace
```

| 任务 | 产出 | 关键验收 |
|---|---|---|
| **T2.1** | `agent/state.py`、`graph.py`、`nodes.py` 骨架，节点先用 Fake | **循环上限有单测**：构造永远违规的 Fake，断言恰好 3 轮后退出且返回非空。`recursion_limit` 与显式 iter 计数两道保险都要有测试 |
| **T2.2** | 节点接入 `eval/fidelity.py` 与 `stylometry/distance.py`，路由阈值放 `agent/config.py` | **断言 agent 模块没有 import `eval/judge.py`**（写成 import 检查测试）。LLM judge 不得进环 |
| **T2.3** | `agent/prompts.py`，逐条具体反馈 | 断言生成的 prompt 含每条的具体词对（如「太医」「宫监」）。**禁止「请更忠实于原意」这类笼统措辞**，写一条断言排除它 |
| **T2.4** | `TraceEvent` 落地，每节点记 `node / ts / duration_ms / payload` | ✅ trace 可 JSON 序列化；事件数与节点执行次数一致；时钟可注入 |
| **T2.5** `[GPU]` | README 追加 `agent` 行 | CHANGELOG 记 `hallucination_rate` 与 `numeral_recall` 变化幅度，以及**修订轮数分布**（多少条一轮过、多少条用到 3 轮） |

**硬禁令**：用 `langgraph` 不用 `langchain`；LLM judge 不得进 agent 循环（太慢，一轮几秒三轮超时），环内只允许确定性指标。这个分层是整套设计的核心取舍。

---

## 11. 已接受的局限（不要重新翻案）

这些都已讨论定案并写进 README「已知局限」，再提一遍只是浪费时间：

- eval 集是 59/73，非随机筛选，数字密集段落被系统性排除。训练 743/842，总入选率 87.7%
- PINC 参照只有 13 条（均值 0.560 / SD 0.085 / 95%CI [0.513, 0.606]），来自 2025-08 人工 review 日志，无法确认是随机抽样还是择优展示。**这是当前最大的方法论软肋**
- 2026-10-03 的留一法 / round1 / round2 都是在 **deepseek-flash** 上得到的，不能当作 `deepseek-chat` 的结果
- 本项目方法层面无新颖性，价值在评估严谨度与完整工程实现
- 语义级漂移抓不到：规则校验能抓实体、数字、称谓的替换，抓不到「垂死的酸气」→「死尸散发的酸臭之气」这类语义扭曲
- 托管 API 只公开模型别名时，日期和采样参数仍不足以锁定不可变底座快照

---

## 12. 常用命令

```bash
# 门禁四件套（Windows 上跑）
uv run ruff check .
uv run ruff format --check .
uv run mypy .
uv run pytest -m "not gpu" -q --cov=eval --cov-report=term-missing
#   注：tmp_path 在 Windows 上会 PermissionError，加 --basetemp=<可写目录>

# 训练与生成（WSL2）
uv run python -m scripts.train --dry-run          # 不碰 GPU，只核对超参
python -m scripts.train --run-id sutong-v2
python -m scripts.generate --run-id base-eval59
python -m scripts.generate --run-id lora-eval59 --adapter adapters/sutong-v2/adapter

# 评估（--skip-judge 不发任何网络请求）
uv run python -m eval.run_eval --config eval/configs/eval59.yaml \
  --generations corpus/generations/lora-eval59.jsonl --run-id lora-eval59 --skip-judge
```

`eval/configs/baseline.yaml` 是 5 条公开样例的冒烟配置，`eval/configs/eval59.yaml` 是真实基线配置。两份都保留，不要互相覆盖。
