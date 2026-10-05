# SPEC — 技术规格

本文件定义**接口、数据结构和算法**。任务顺序见 `docs/PLAN.md`，工作约定见 `AGENTS.md`。

本文件是契约。如果实现中发现规格与现实冲突（尤其是第 1 节的既有数据 schema），**修改本文件并在 commit 里说明**，不要让代码和规格悄悄分叉。

---

## 0. 架构总览

```
                      ┌──────────────────────────────┐
白话输入 ──────────────▶│  retrieve  风格范例检索        │
                      └──────────────┬───────────────┘
                                     ▼
                      ┌──────────────────────────────┐
                      │  generate  LoRA 模型生成       │◀──┐
                      └──────────────┬───────────────┘   │
                                     ▼                   │
                      ┌──────────────────────────────┐   │
                      │  verify    保真校验（确定性）   │   │
                      │  score     风格打分（确定性）   │   │
                      └──────────────┬───────────────┘   │
                                     ▼                   │
                           route ────┴── revise ─────────┘
                             │                    (最多 3 轮)
                             ▼
                           输出 + trace
```

复用关系（重要）：

```
stylometry/ + eval/metrics.py
    ├─▶ eval/run_eval.py   离线评估，产出基线与对比
    ├─▶ agent/nodes.py     在线 reward，驱动 verify / score / route
    └─▶ CI eval gate       拦截指标回归的 PR
```

一份指标代码三处复用，这是本项目的架构主线。所以 `stylometry/` 与 `eval/metrics.py` 必须是无 I/O 的纯函数库。

---

## 1. 既有资产与数据 schema

### 1.1 既有产物已全部灭失

2025 年 8 月的全部产物存放在一台 AutoDL 租用实例上（`/home/shaohan_yang/projects/sutong-style/`）。该实例已被平台释放，系统盘销毁，**数据不可恢复**。AutoDL 的「文件存储」从未开通，本地无任何副本。

灭失清单：

| 文件 | 内容 | 条数 |
|---|---|---|
| `corpus/pairs.jsonl` | `(白话, 原文)` 训练对 | 907 |
| `corpus/eval_results.jsonl` | 微调模型在 eval 集上的生成结果 | 73 |
| `corpus/base_results.jsonl` | 基座模型在同一 eval 集上的结果 | 73 |
| `adapters/qwen2.5-3b-sutong-v1/` | LoRA adapter（r=32） | — |

**结论：语料与模型都要重建。** 重建流程见 `docs/PLAN.md` 的 T0.0。

### 1.2 从日志反推出的已知参数（重建的唯一依据）

以下全部来自 2025-08 的运行日志，是重建时必须对齐的事实：

- **语料范围**：6 部作品 —— `妻妾成群`、`妇女生活`、`另一种妇女生活`、`园艺`、`罂粟之家`、`我的帝王生涯`
- **chunk 规模**：907 个，id 形如 `我的帝王生涯_0429`（零填充 4 位）
- **chunk 长度**：约 200–400 字（从日志里的样本目测）
- **切分**：train 834 / eval 73（约 8%）
- **基座模型**：`Qwen/Qwen2.5-3B-Instruct`，31.2 亿参数
- **LoRA**：r=32，可训练参数 29,933,568（占 0.96%）
- **训练超参**：3 epochs、lr 2e-4 cosine、batch 2 × grad_accum 2、padding-free、gradient offload
- **训练耗时**：43.8 分钟，显存峰值约 2.84 GB
- **loss 轨迹**：train 2.726 → 1.936；**eval 2.169 → 2.056（epoch 1.9 触底）→ 2.089 回升**

最后一条是重建时最有价值的信息：**第 3 个 epoch 是过拟合，重建时直接训 2 epochs**，不要复刻当初的 3 epochs。

### 1.3 残存样本 `corpus/salvaged_pairs.jsonl`

从运行日志里抠回的 30 对 `(白话, 原文)`，六部作品均有覆盖。**它不是训练数据**（量太小），用途是**重建验收**：

```json
{
  "id": "妇女生活_0015",
  "work": "妇女生活",
  "idx": 15,
  "vernacular": "娴在楼下看见那些穿白衣服戴白帽子的人……",
  "original": "（对应的原著段落，受版权保护，此处不写出）",
  "model_output_v1": "娴在楼下的时候看见那些白衣白帽的人……",
  "source_log": "scratch_inference_log.txt",
  "maybe_truncated": false
}
```

- 30 条中 14 条标记为完整（`maybe_truncated: false`）；2026-10-03 复核发现 `妇女生活_0015` 白话/原文中文字数比约 1.95，判定原文侧截断，剔除后只有 **13 条可用参照**。原始文件保留，剔除原因逐条进入重建报告；其余 16 条原已标记截断
- 6 条附带 `model_output_v1`，即当初 v1 模型的实际输出
- 含原著文本，**不提交**（已在 `.gitignore` 覆盖范围内）

用法见 T0.0 的验收标准。

### 1.4 目标 schema

所有下游代码只认这个 schema，由 T0.0 的重建流程直接产出。

`corpus/pairs.jsonl` — 每行：

```json
{
  "id": "妻妾成群_0049",
  "work": "妻妾成群",
  "idx": 49,
  "vernacular": "颂莲没想到飞浦会来。她开了门……",
  "original": "（对应的原著段落，受版权保护，此处不写出）",
  "split": "train"
}
```

`split` 取值 `train` 或 `eval`。旧切分已随数据灭失，**重新切分并用固定 seed**（`seed=42`，比例 eval ≈ 8%，按 `work` 分层以免某部作品全落进 eval），切分结果固化到 `corpus/split.json` 并提交。此后永不变动——所有横向对比都依赖它稳定。

`corpus/generations/<run_id>.jsonl` — 每行：

```json
{
  "id": "妻妾成群_0049",
  "model": "sutong-v1",
  "pipeline": "baseline",
  "output": "……",
  "trace": null,
  "seed": 42
}
```

`pipeline` 取值 `baseline` / `retrieval` / `agent`。取 `agent` 时 `trace` 必填，结构见 4.4。

### 1.5 语料重建：`scripts/vernacularize.py` 规格

这是整个重建的质量瓶颈——训练数据的上限就在这一步。

**做什么**：输入苏童原文 chunk，用 LLM 产出对应的白话版。注意方向是**倒着造数据**：原文是训练**目标**，白话是训练**输入**。

**必须保留**：

- 全部人名、地名、职官称谓（颂莲、陈佐千、枫杨树、燮国、太医……）
- 全部数值（含年份、数量、日期），形式改为口语常用的阿拉伯数字
- 全部情节与对话内容

**必须改变**：

- 书面语 → 口语（「心如死水」→「心里凉透了，啥盼头都没有」）
- 长句 → 短句，拆开定语
- 成语、文言残留 → 大白话
- 不加引号的对话 → 常规对话写法（苏童的无引号对话是**目标侧**特征，不该出现在输入侧）

**禁止**：概括、删掉事实、增补原文没有的情节。当前两段式方案只以原文 **0.6–1.6 倍**中文字数作兜底，不限制段落数；本轮零重试。旧方案的 0.9–1.3、等段落规则保留在历史代码和报告中，不再用于新方案。

**工程要求**（照搬当初 pipeline 已验证的做法）：

- **断点续跑**：启动时扫描已有输出，只处理 remaining。原日志的 `Total chunks: 907, already done: 0, remaining: 907` 就是这个机制
- **并发** + 失败重试，每 20 条打一次进度 `progress: N/907 (ok=, failed=)`
- 凭据走 `LLM_BASE_URL` / `LLM_API_KEY`，模型名走 `VERNACULARIZE_MODEL`（默认 `deepseek-v4-pro`），OpenAI 兼容接口。2025-08 的生成模型未知，不能声称已保持同级；不得使用已重映射到 Flash 的 `deepseek-chat` 代称 Pro
- 普通单条失败不中断全局，最后汇总 failed 列表；**模型身份不一致是致命错误，必须中止整批**

**一个重要的决策点（需要人工拍板，先记进 `docs/QUESTIONS.md`）**：

模型的已知缺陷 #2（域外输入失效）的**根因就在这一步**——LLM 生成的白话仍带着原文的句子骨架和叙事密度，和人类随手写的白话分布不同。旧语料已经灭失，**所以重建时没有"保持兼容"的包袱，可以直接修这个根因**：

- 方案 A（保守）：复刻当初的效果，参照 `salvaged_pairs.jsonl` 的白话风格
- 方案 B（治根）：prompt 里额外要求打散句子边界、加入口语赘余和语气词，让产出的白话更接近真人写法

方案 B 更可能解决缺陷 #2，但需要一把尺子来验证"更接近真人"。那把尺子就是 `corpus/human_eval.jsonl`：用 `stylometry` 算**生成白话**与**人工白话**之间的 `style_distance`，越小说明越接近真实输入分布。

> 这意味着 **`human_eval.jsonl` 至少要先写 10 条**，才能调这个 prompt。原计划把它排在 Phase 1 末尾，现在要提前。见 PLAN 的 T0.0。

**验收**：对 `salvaged_pairs.jsonl` 剔除异常后的 **13 条**执行严格 leave-one-out：生成第 i 条时，候选范例池只能来自其余 12 条，记录全部候选与实际注入的 2–3 条。当前样本的旧白话不能进入它自己的 prompt。报告新白话 vs 旧白话的逐条相似度及分布，供人工检查信息保全（人名、数字、对话一个不少）；不使用该表层指标单独证明事实保真。

13 条参照样本按 `difflib.SequenceMatcher(None, 白话, 原文).ratio()` 实测：均值 **0.560**、标准差 **0.085**（`ddof=1`，复算 0.084897）、95%CI **[0.513, 0.606]**（既定参照值）、最大值 **0.680**。不去空白，不修改标点，不互换参数顺序，保留默认 `autojunk`。复制率校验必须使用同一直接比较口径，所有密度档位的硬上限不得超过 **0.70**，不得为放行个别样本上调全局标准。按原文 chunk 的对白与专名密度分三档：

- 低密度（对白 `< 0.20` 且专名 `< 0.07`）：上限 **0.65**
- 中密度（对白 `>= 0.20` 或专名 `>= 0.07`）：上限 **0.68**
- 高密度（对白 `>= 0.50` 或专名 `>= 0.10`）：上限 **0.70**

对白密度按含引号或说话标记的句段所占中文字比例计算，专名密度按 `jieba.posseg` 的人名、地名、机构名等专名词性所占中文字比例计算。必须从原文计算，不能从模型输出计算。重试后仍超过对应上限的候选不得进入 `pairs.jsonl`，并必须逐条写入 `rebuild_report.json` 的 `copy_similarity_exceptions`（记录最低相似度、对应上限及两项密度），不得静默计为正常通过。

**已拍板：走方案 A**（2026-10-02）。复刻当初的白话风格，参照 `salvaged_pairs.jsonl`。缺陷 #2 本轮不处理，留待后续——理由与后续方案记在 `docs/QUESTIONS.md` 的 Q1。

**预注册预览规则（2026-10-03，替换单边均值门槛）**：20 条预览均值落在 **[0.51, 0.61]**，按本项目既定规则判定为与参照分布兼容。低于 **0.51** 同样停止并报告可能的信息流失，不当作改进；高于 **0.61** 报告过度复制。区间相容是操作性规则，不能据此声称统计学上已证明同分布。密度分档硬上限保持不变。

执行顺序与预先固定的方法：

> 以下单段预览和模型对照规则为历史预注册记录，连同 0.65 / 0.68 / 0.70 闸门保留，
> 仅解释既有报告。当前执行规则以新增 1.5.1 为准；旧结果不删除、不重算覆盖。

1. 先以 13 次单次调用完成留一法，不隐藏重试；报告新 vs 旧和新 vs 原文两组分布。
2. 从当前 chunks 按 `work` 分层、`seed=42` 随机重抽 20 条，排除旧范例 id、原文完全一致项和异常参照 id。采用近似等额分层，六部作品各 3 或 4 条；抽样函数及 seed 入库，同一名单用于两轮。该总体均值不按全语料作品占比加权。
3. 使用已提交的同一方案 A prompt，预览第 1 轮注入 3 条，第 2 轮注入 2 条，按固定 seed 在 work 层间随机轮换；生成 seed 为 `42 + idx + attempt + (round−1)×100000`。`temperature=0.2`、`top_p=1.0`、`max_tokens=2048`，`retries=0`，SDK 隐式重试关闭。最多两轮；若均值低于下界则立即停止并报告，其他情况在两轮后停止等待人工决定。
4. 每轮完整保留 20 条的相似度，即使未通过长度、段落、数字或复制率校验也纳入统计。API 失败记为缺失及原因，不对不足 20 条的结果作区间判定。标准差同时给 `ddof=0` 与 `ddof=1`，四分位用线性插值；报告均值、中位数、四分位、极值以及按 work 的分组结果。
5. 每次运行的 `rebuild_report.json`（分轮文件名为 `rebuild_report_<run>.json`）记录请求模型完整标识、服务返回的模型标识/response id/时间/可用 fingerprint、UTC 调用日期时间、全部采样参数（含 thinking 模式）及逐次实际 seed、prompt git commit 与 SHA256、调用代码 commit、源文件 SHA256、候选池/入选范例、剔除/超限/重试/数字豁免及理由。**响应 model 必须与请求标识严格相等，否则立即抛出 ModelIdentityMismatch，不重试、不接纳输出，并停止尚未发起的调用**；已在途请求无法撤回，仍留痕，整轮标记 aborted_model_mismatch，不作有效实验引用。无固定快照或服务端 seed 生效证明时明确记录未知，不猜测模型版本或确定性。
6. `scripts/prompts/*.txt` 提交完整可复用 prompt 模板；实际展开后的 prompt 包含版权原文和范例，逐次完整存入本地 `corpus/generations/*_attempts.jsonl`，只在报告中记录 hash，不把这些文本提交。所有拒绝输出同样归档，只有通过校验的输出可进入 pairs 文件。全量入口需 `--allow-full`；本轮不执行全量，必须等人工明确放行。

**模型对照预注册（2026-10-03）**：此前留一法、round1、round2 请求 `deepseek-chat`，53 个响应均为 **deepseek-flash**，三组结论全部标注“在 deepseek-flash 上得到”，不可引用为 deepseek-chat 的实验。官方更新日志记录该旧别名改指 Flash 非思考模式，同账户诊断能正常请求并返回 `deepseek-v4-pro`。仅再执行一次 20 条单变量模型对照：以 round2 本地归档为准，原文、完整展开 prompt、候选与实际范例、名单及顺序、逐条 seed、temperature=0.2、top_p=1.0、max_tokens=2048、并发数和零重试完全一致，只改请求模型为 `deepseek-v4-pro`。新请求显式 `thinking.type=disabled`，旧请求的非思考模式来自官方别名文档的推断而非当时显式记录，必须披露。调用前逐条 hash/参数验证，任何差异先报错，不调用 API。沿用 [0.51, 0.61] 及 0.65 / 0.68 / 0.70 原标准，结果与 round2 并排；本次结束即停止，不启动四臂消融或全量。

**已知局限**：参照只有 **13 条**，占历史 907 条的 **1.4%**，来源是 2025-08 的人工 review 日志，无法确认当初随机抽样还是择优展示；这是当前最大的方法论软肋。严格留一法只能防止当前样本进入自己的范例，不能消除日志选择偏差，也不提供一个独立的大样本测试集。SequenceMatcher 是表层重合的代理指标，不直接等价于白话化质量或语义保真；低分可能是有效改写，也可能是信息流失。对白密度和 jieba 专名密度同样是有误差的启发式估计。**2025-08 使用的生成模型未知，本轮既有三组实际使用 Flash，跨模型比较的不确定性比此前记录更大，不能把差异全部归因于 prompt 或模型能力**。托管模型可能只返回可变别名，记录日期和响应元数据仍不能保证底座快照可复现；模型名称及速度本身不能证明其照抄倾向或因果关系。

#### 1.5.1 两段式重设计预注册（2026-10-03T21:19Z，外部生成前声明）

> 本节保留已完成的两段式实验及当时的规则，后续暂停与审核要求见1.5.2。
> 不以新规则覆盖历史结果，现有生成入口不得据本节自动启动下一轮。

**方案变更而非事后改判**：用户取消温度/约束四臂消融，改为一次整体设计实验。
理由是假设保护型约束把段落密度、句长、中文数字形式等风格线索带入白话输入，
而缺少明确的风格破坏指令。本轮同时改设计各项，不能分辨每项的独立效应；
Pro 对照的均值 0.687311、样本标准差 0.126368、极大值 0.888087 保留为历史结果，
Flash round2 的均值 0.736356 等历史判定保持原样。模型差异不是唯一解释，也不能
声称一次小样本已科学排除所有模型效应。

**执行前第二次修订（2026-10-03T21:39Z，尚未调用新方案 API）**：用户要求采用
PINC / sBLEU，替代先前暂定的源端分母重合率。原方案源端去重重合率参照
6-gram=0.124365、4-gram=0.218425 的复算记录保留，但不是本次主判据。
这两种分母不同，不能直接把原指标取反冒称 PINC；历史实验和历史验收结论不覆盖。

**新主判据 PINC（Chen & Dolan, 2011 的中文字符适配）**：分别把原文和白话中的
`U+3400–U+9FFF` 中文字符连起来（会跨越被删除的标点、数字和换行）。对输出的每个
连续 N 字窗口计一次，判断其是否出现在源文本中，重复输出窗口仍分别计数：
`PINC-N = sum(output[i:i+N] not in source_ngrams) / (len(output)-N+1)`。
不是 BLEU 的多重集裁剪，不用源侧分母；输出少于 N 字时记 null/不可评且失败。
总体为逐条非加权均值。原论文是词级且对 1–4 阶取均值，本项目另报该均值及
PINC-1/2/3/4/6；中文字符口径不能与英文词级文献数字直接比较。
13 条复算分别为 **0.211414、0.562161、0.703106、0.784850、0.877025**；用户提供
0.210 / 0.561 / 0.703 / 0.784 / 0.875，与当前数据相差至多0.002025，未完全复现。
输出去重的替代口径分别为0.293445 / 0.591121 / 0.710922 / 0.786699 / 0.877200，
尤其1阶差异很大，故公开固定出现次数口径，不调整算法迎合给定数字。
以下门槛执行前固定，逐条而非仅总体均值：

- 最终白话 **PINC-4 ≥ 0.75、PINC-6 ≥ 0.85**；任何一项不足即失败。
- 第二遍相对第一遍，两项 PINC 的下降各自不得超过 **0.03**（绝对比例差）；这是对
  “修补不得显著增加复制”的方向等价要求，不把两种不同分母度量称为数值等价。
- sBLEU 为输出相对于源原文的 BLEU，0–100，中文字符逐字以空格分开，只保留上述中文。
  固定 sacrebleu 2.6.0，tokenize=none、smooth_method=exp、effective_order=True；逐句统计
  分布和宏均值，不冒称 corpus BLEU。不另设临时 sBLEU 门槛。
- SequenceMatcher 仍按白话 vs 原文、保留空白和默认 autojunk，完整报告；均值区间
  **[0.51, 0.61]** 为双向辅助警报，不再单独决定接纳。旧密度档位分数仍作历史观察，不作为新闸门。
- 最终中文字数比 **[0.6, 1.6]** 兜底；不要求相同段落数，不按长度、句长或段落去拟合原文风格。
- 数值归一化后保留率必须为 1，实体/称谓候选原样保留率必须为 1；实际数量用阿拉伯数字。
  三百→300、三月初九→3月9号、十七州八十县→17个州80个县、万→10000、全角数字统一后比较。
  只按数值集合比较形式不同不算失败；单位、出现次数、角色绑定、近似范围仍由语义核查负责。
  继承 Q6 的语法性“一+通用量词”豁免并逐条记录；人名/称谓中固有的数字字形不得强改。

**固定必须替换清单**：从 raw 自动统计 jieba 的 i 成语及 l 四字候选（含四字格）、
固定书面语候选，以及带预先列明的生僻颜色/质感标志的词；只取频次 ≥2 的 2–4 字词。
分类内按频次降序、词序打破并列，各取前 60 个，再合并去重。短词、频次、分类和源 hash
存 scripts/data/mandatory_replacements.json 并提交；无原文句子、不附上下文。
通用书面语候选及颜色标志由脚本预先声明，抽取/排序过程可复现，不从新输出反推词表。
该清单是“必须替换”而非“只能替换”，不能完整识别全部四字格，这是已知局限。
每条 prompt 注入原文命中的清单及不少于 15 对具体示例；明确“原文中的成语、四字格、
文言词一律不得出现在白话版中”。实体、地名、称谓仍优先保护，不因长度为四字而破坏名字。
执行前发现 l 还包括“干什么”“怎么办”等口语，故 l 只取四字并排除脚本声明的
11 个常见口语四字候选，自动数值掩码与动态成语禁用仅用 i；此校准在生成前完成。
清单词残留和原文中 jieba i 成语原样残留逐条判失败；未识别的书面词需审阅。

**两次调用，硬性分工**：

1. 第一遍激进口语化：只保人名、地名、称谓、数值、完整情节与对话意思，打破句式、
   段落、成语、四字格和文言表达；数字改口语形式。沿用该样本 round2 的两条旧范例。
   输出正文，不输出解释。第一遍测量不通过仍执行第二遍，不隐藏重试。
2. 第二遍只做语义核查与修补：原文和第一遍全文同时输入，不注入旧范例。
   输出 JSON：issues 数组逐项列 category（entity/numeral/event/causality/dialogue/other）、
   lost_or_changed_fact、repair；以及 repaired_text。只补丢失/改变的事实，不改第一遍文体，
   不引回原文词句或成语，不为降低分数删事实。无问题也返回完整 repaired_text。
   JSON 解析失败、API 失败、截断、任何闸门失败都留痕，不额外调用、不筛掉失败输出算均值。
   第二遍自报问题不能作为独立的情节保真证据。

**固定执行与记录**：同一份 20 条名单、原文和范例池，与 Pro 对照源 hash 核对。
两遍均请求 deepseek-v4-pro、非思考、temperature=0.2、top_p=1.0、max_tokens=2048；
各条两遍均沿用 round2 seed `42+idx+1+100000`，并发4，SDK 和业务重试均为0，
只有一次 20 条两段式运行（正常最多40次调用）。身份不一致立即中止尚未发起调用。
所有展开 prompt、第一遍正文、第二遍原始 JSON/正文、问题清单和失败完整留在本地；
报告含日期、响应模型、参数、逐条 seed、prompt/code commit、hash、剔除与豁免原因。
历史 Flash/Pro 档案只读，按新 PINC/sBLEU 口径补算对照列，明确是回顾性补算而非历史预注册指标。
新方案两遍各报告全部20条 PINC-1/2/3/4/6、sBLEU、SequenceMatcher，均值、两种标准差、
中位、四分位、极值、work 分组；实体与数值分母、增删值、豁免、语义修补问题逐条留痕。

**保真抽查**：调用前从固定20条中按 work 分层 seed=42 抽6条，每部作品1条，提交抽样代码。
固定抽查人名/称谓、数值、逐事件、因果、对话、颜色/质感事实；两遍对照，不按通过情况挑样本。
实体指标用 jieba 专名候选加预声明人名/称谓词，数值用 cn2an 归一化；这是 T0.0 生成闸门，
不是尚未实现的 T0.3 正式指标。无候选时 recall=1 同时报告分母0，不声称证明保真。
助手逐事实核对单独记为 assistant_review，不冒称人工；真正 human_review 保持 pending/null，
待仓库所有者 review。全量许可不由自动闸门代替。

**停止与局限**：运行后立即停止，不调 prompt/词表/门槛，不做消融，不运行全量，不动 T0.2+。
只保留中文的 N-gram 会受数字格式、标点拼接和无法避免的实体重合影响，仍是代理指标，
低重合不证明情节完整；13条参照的日志选择偏差和旧模型未知仍存在，六条抽查也不代表总体。
自动词表来自同一 raw（含这些预览作品），是有意的生成词表，不把它包装成独立域外验证。

**双向评价框架**：白话化时源=原文、输出=白话，源侧重合低/PINC高表明不是平凡复制，
但不单独证明内容保真；必须同时核查事实。Phase 1 之后生成时源=白话输入、参考=原文，
源侧重合低用于观察改写，参考侧重合高用于辅助内容核查，两者不能混淆。
文学原文允许大量等价表达：用户对当前 raw 的测量为“泪”105次、上下文去重104种写法
（上下文窗口尚未提供，未独立复核）。因此生成阶段参考答案 n-gram 重合是弱信号，
不得作为主判据。参考重合高也不自动证明风格或语义正确。

#### 1.5.2 两张清单人工确认与收窄修补（2026-10-03 EDT / 2026-10-05 EDT）

**当前状态：第二道成对替换已定稿，仅获准一次 20 条预览**。2026-10-05 EDT 起，
67 项成对替换、类别规则与必须保留写入 `scripts/prompts/two_pass/destroy.txt`，
第二遍收窄规则写入 `repair.txt`。跑完该次预览即停止，等待人工 review，不启动全量。
此前两段式结果无改善，不能靠词性标签充当人工审核。96项替换表与保护候选均有污染；词表排除规则只作用于l分支，i分支
仍收入普通口语。保护表则误含风格词、普通名词并切坏/漏掉真实姓名。这些是已核对的
数据质量问题；未做独立词表消融，不能把它们的独立效应或全部退步归因当成定量结论。

严格分三步，不得合并执行：

1. 本轮只给出逐项keep/drop/fix审核草案，见[两张审核表](reviews/T0.0_lexicon_audit.md)。
   替换表覆盖冻结的全部96项；保护表覆盖此前20条已发送prompt的102个不同候选与
   固定34项的并集121项，另列手工补漏11项。助手草案不是人类批准，不重新自动抽词，
   不应用草案、不修改prompt、不调用生成API；仓库所有者已于2026-10-05 EDT确认并指定修订。
2. 第一道确认之后，才给每个保留替换词提供具体白话目标。fix项未获确认不得混入keep。
   2026-10-05 成对目标已按审核草案定稿，不沿用未经审阅的例子。姓名/地名的边界修正不属于白话目标。
3. 两张表及成对替换目标全部定稿之后，才实施、测试、冻结下一版prompt/校验代码。
   2026-10-05 所有者明确的范围是一次 20 条预览，不是全量许可。

保护清单只收真实人名、地名、职官称谓，删除普通形容词、副词、名词；经确认，家庭称谓
通常不入硬表，但“序数+称谓”是例外，须原样保护以保住排序身份事实。草案将君主/宫廷
称号视为广义职官，将姓氏加称谓的明确个人指代视为人名化称呼。普通家庭称谓及职业不
默认为职官，河东等专名性不足项隔离待确认（Q14）。不在硬表不意味着删除身份、关系、
组织、事件、物品或方位事实。该审核仅覆盖当前20条及既有固定项，不冒称全语料已审核。

**清单定位已修订**：它是书面语类别的审核示例，不是可穷举的禁用词表。用户提供的原始
语料测量为54,654种四字连续串中53,265种（97.5%）只出现一次；频次阈值2会系统性漏掉
长尾表达。因此下一版第一遍prompt须同时列出类别规则：除明确口语例外外，拆解任意
成语/四字格；替换或删除文言虚词；把书面单字动词、生僻颜色/质感、书面比拟词改为
大白话。69这一旧计数在确认修订后变为67（仿佛转keep，三个口语常用成语转drop）；
实际配对草案须以67项为准，不能虚报为69项。2026-10-05 第二道按
`docs/reviews/T0.0_replacement_pairs_draft.md` 原文定稿，文末待确认备选不在预览前改写。
不回溯改写历史 prompt 与历史分数。

**后续第二遍规则（2026-10-05 已写入冻结版本）**：只准恢复第一遍丢失的
真实专名（人名、地名、职官称谓、带序数称谓），及丢失/改错的数值。禁止恢复已替换的形容词、副词、
成语、四字格、文言词、比喻或书面单字动词；禁止以更贴近原文为由改任何其他内容。情节保真仍需独立核查，发现问题
必须记录并失败/交review，不能扩大修补范围。修补后的PINC-4与PINC-6各自不得低于
第一遍（容忍下降为0，不再是旧实验的0.03）；最终双闸仍为0.75/0.85。旧实验的0.03
及其分数/判定原样保留，新限制仅适用于本冻结版本及之后。

**残留门禁同步收窄，且在本次预览调用前登记**：`literary_residue` 只检查上述 67 个已审核示例
是否原样留在白话中。不再把 jieba 成语标记并入门禁；该标记已在 Q14 核对为污染源，不能再充当
人工审核的替代。67 项之外的书面语由 prompt 中的类别规则约束，并用 PINC 观察，不另造自动词表，
也不把“未命中 67 项”解释成允许照抄。

#### 1.5.3 全量白话化预注册（2026-10-05，调用前）

六轮同模型、同 20 条、同 seed 的配对比较里，pro_control 单遍的 PINC-4 均值为
**0.626358**，高于成对替换加类别规则的单遍 **0.593124**，也高于其第二遍 **0.589799**。
此后停止继续优化白话化 prompt。全量使用 pro_control 的已提交模板
`scripts/prompts/vernacularize_a.txt` 及其配套 example/examples/retry 模板，模板哈希必须等于
`38fd364e91c12cf46828e1d5669e80b23e51398536d2364b375b238046afe506`。采样为
`deepseek-v4-pro`、temperature 0.2、top_p 1.0、max_tokens 2048、thinking disabled、
retries 0、round 2、seed 42、并发 4。展开 prompt 先用一条与
`corpus/rebuild_report_model_control_20261003.json` 的档案核对，不一致则不调用。

**放宽 PINC 闸门是看到下游结果之前的流程决定。** PINC 与 sBLEU 逐条记录，不拦截、不重试。
理由是该代理指标没有经过下游训练或风格评估验证。本轮预览 PINC-4 约 0.63，13 条参照复算
PINC-4 为 0.784850。这个差距记为已知偏差，等下游指标出来后再决定是否重做语料。
六轮实验报告保留，不删不改。

进入 `corpus/pairs.jsonl` 的确定性闸门只保留内容保真：

- 已审核且长度不少于 2 的人名、地名、职官/宫廷称号、序数称谓，只要出现在原文中，白话必须原样包含。
  名单在 `scripts/data/content_gate_terms.json`。单字人名「娴」「芝」以及语境专名性不足的「枫杨树」
  只记录、不拦截。jieba 专名标记不进入闸门，原因见 Q14。
- 数值只比值、不比字形。原文和白话各自抽取后，用 `cn2an` 归一化再比较，规则是期望值集合
  ⊆ 实际值集合，白话多出的数量词不算失败。序数（第N，以及二/三/四太太、二/三/四姨太里的排行）
  单独成类，不进基数集合。量级词（万、千、百）必须并进前面的数字，单独的「万」按 10000 计。
  概数前缀保留在键里：将近视同近，差不多视同约，上万与万不是同一个值。语法性「一+通用量词」豁免。
  「上一次」「上一个」里的「上」是「前一个」，不是概数，不进数值集合。初九按日期基数，与 9号 相同。
  原文里抽出来却无法归一化的数值判失败并留痕。2026-10-05 全量第一次用的是短语字符串比对，
  3 与 三、16 与 十六会被误杀；该行为是 bug，本节是修正，不是调阈值。
- 情节没有确定性检查。prompt 仍要求保留情节、因果和对话；漏检是已知局限，不另加第二遍模型。

长度、段落数、复制率、PINC、sBLEU 都记录，都不拦截。内容失败的样本不进入 pairs，逐条写入报告，
不中断其余样本。断点续跑跳过 pairs 里已有的 id，也跳过**本次报告**档案里已经调用过的 id。
修正后的重试换一份新报告，因此只重跑尚未进入 pairs 的失败项，不重跑已通过的样本，也不重切
`corpus/split.json`。
`corpus/chunks.jsonl` 当前 915 条，落在 907±30 内；全量覆盖全部现有 chunk，不再抽成 907。
`corpus/split.json` 已按 seed 42 分层固化，本轮不重切。历史两段式与成对替换代码保留，
默认闸门仍是 historical，只有 `--gate content` 走本节。

**已知局限（2026-10-05，数值闸门停止再改）**

所有者验收 802 条 pairs 后决定不再修数值闸门。再修最多捞回约 40 条，对训练可忽略；
闸门已经连续出过两个 bug，继续改是负收益。下列三条约束此后所有基线：

1. 可用 eval 是 59/73。缺口 14 条是被内容闸门筛掉的，不是随机留出，数字密集的段落被系统性排除。
   之后每一列基线对比都建立在这个有偏子集上。
2. 未入 pairs 的 113 条里，44 条只缺基数「1」，29 条是序数被写成基数。这两类很可能是闸门过严，
   不是真实信息丢失。未修复，记为已知偏差。
3. 训练集 743/842，802 条对 915 条 chunk 的总入选率是 87.7%。

### 1.6 原始文本（已恢复，`corpus/raw/`）

2026-10-02 从 Windows 回收站恢复，三个文件覆盖全部六部作品，均已转为 UTF-8。

> **固定备份位置：`D:\Transformers\corpus-backup\`**（仓库之外，含 `SHA256SUMS` 与 `MANIFEST.md`）。
> `corpus/raw/` 是工作副本，可随时从备份恢复。校验：`sha256sum -c SHA256SUMS`。

| 文件 | 中文字数 | 包含作品 |
|---|---|---|
| `妻妾成群.txt` | 103,040 | **四部合集**：妻妾成群、妇女生活、另一种妇女生活、园艺 |
| `我的帝王生涯.txt` | 93,492 | 我的帝王生涯 |
| `罂粟之家.txt` | 29,661 | 罂粟之家 |

合计 **226,193** 字。除以 907 得每 chunk 约 **249 字**，与 SPEC 1.2 里反推的 chunk 规模吻合——可确认这就是当初的素材，切分目标按此对齐。

**`妻妾成群.txt` 的分篇**（T0.0 的切分脚本必须处理）：

- 四部作品各以**标题独占一行**开头，后接 `第一节` 这类小节标题
- 按标题行出现位置，分界点约在字符偏移 `0 / 34760 / 60563 / 87040`（仅供校验，脚本应按「标题独占一行」判定，不要硬编码偏移）
- ⚠️ **必须先匹配 `另一种妇女生活` 再匹配 `妇女生活`**，否则后者会把前者的标题行吃掉
- ⚠️ `园艺` 二字在正文里也作普通词出现（87127、87760…），**只认独占一行的标题**，不要用子串搜索
- `※※※` 是分节分隔符，**是天然的 chunk 边界**，切分时优先在此断开

**文本清洗（必做）**：

源文件含繁体残留 `麽`（`什麽`、`怎麽`、`那麽`），是文本来源的转换残留，不是作者用字。`scripts/chunk_corpus.py` 应统一规范化为 `么`。同类残留一并处理（如 `裏`→`里`、`牠`→`它`），规则列在脚本里并写单测。

**不要规范化引号。** 源文本里引号用法不一致——`妻妾成群` 部分段落用 `「」`，另一些段落完全不加引号。这种不一致既是来源artifact 也是苏童本人的行文特征，而 stylometry 的第 5 维（`quote_density`）正要测它。

> **已知局限**：由于引号用法在同一部作品内就不一致，`quote_density` 这一维会比其他维度更噪。若 T0.2 的冒烟测试显示该维区分度差，允许在 CHANGELOG 记录后将其权重降低或剔除。

---

## 2. stylometry/ — 文体特征（项目技术核心）

模块 `stylometry/features.py`。纯函数，无 I/O。

### 2.1 特征向量定义（20 维，顺序固定）

分句规则：按 `。！？；` 和换行切分，丢弃长度小于 2 的片段。分词用 `jieba.lcut`。

| # | 名称 | 定义 |
|---|---|---|
| 0 | `sent_len_mean` | 句长（字符数）均值 |
| 1 | `sent_len_std` | 句长标准差 |
| 2 | `sent_len_p90` | 句长 90 分位 |
| 3 | `comma_ratio` | 逗号占全部标点的比例 |
| 4 | `period_ratio` | 句号占全部标点的比例 |
| 5 | `quote_density` | 引号类字符每百字出现次数 |
| 6 | `dialogue_verb_density` | 说 道 问 答 喊 嚷 每百字次数 |
| 7 | `simile_density` | 像 似 仿佛 般 犹如 好比 好像 每百字次数 |
| 8–14 | `fw_的` `fw_了` `fw_着` `fw_地` `fw_而` `fw_其` `fw_之` | 七个虚词各自每百字频率 |
| 15 | `ttr` | type-token ratio（分词后 unique/total） |
| 16 | `avg_word_len` | 平均词长（分词后） |
| 17 | `literary_hit` | 书面语词表命中率，见 2.2 |
| 18 | `cjk_numeral_ratio` | 中文数字字符数 ÷ (中文数字 + 阿拉伯数字) 字符数 |
| 19 | `para_density` | 段落数 ÷ 百字 |

**第 18 维的来历**：基座模型会输出「第４天早晨」「３辆马车」（半角/全角阿拉伯数字），而苏童原文一律写「第四天」「三辆」。这是实测观察到的、廉价且判别力很强的特征，别省掉。

接口：

```python
FEATURE_NAMES: tuple[str, ...]   # 长度 20，顺序即向量顺序

def extract(text: str, lexicon: LiteraryLexicon) -> np.ndarray:
    """返回 shape (20,) 的 float64 向量。空文本返回全零向量。"""
```

### 2.2 书面语词表（从自有数据学，不用外部词表）

`stylometry/lexicon.py`。利用 907 对平行语料自动抽取。对每个词 w：

```
score(w) = log( (count_original(w) + 1) / (count_vernacular(w) + 1) )
```

取 `score > 1.0` 且 `count_original >= 3` 的词构成书面语词表；对称地取 `score < -1.0` 构成口语词表。产物存 `stylometry/data/lexicon.json`，**只存词和分数，不存原文**（规避版权）。

`literary_hit` = 命中书面语词表的词数 ÷ 总词数。

> 这是本项目少有的、别人无法复刻的资产——它来自自有平行语料。README 值得单独讲一节。

### 2.3 风格距离

`stylometry/distance.py`：

```python
class StyleReference:
    """用全部 original 文本拟合出的参考分布。"""
    mean: np.ndarray   # (20,)
    std: np.ndarray    # (20,)，下界截断到 1e-6

    @classmethod
    def fit(cls, texts: list[str], lexicon: LiteraryLexicon) -> "StyleReference": ...

    def distance(self, text: str) -> float:
        """z-score 标准化后的欧氏距离 ÷ sqrt(20)。越小越像苏童。"""
```

拟合结果存 `stylometry/data/style_reference.json`。

风格参照刻意使用全部 915 条原文，包括 eval 里那 59 条。它是描述性统计（20 个均值加 20 个标准差），不是判别模型，单条文档的贡献大约 0.1%。所有者用同一套特征重算过：eval 原文的距离均值，全量参照是 1.0007，改用 train-only 参照是 1.0066，逐条最大差 0.0491，相对变化 0.59%，可忽略。书面语词表则必须只用 train 的 743 对，因为它的命中直接进入特征向量。这个不对称是有意的，不是疏漏。

**为什么不用马氏距离**：20 维、907 样本，协方差矩阵求逆数值不稳。对角标准化（忽略特征间相关性）更稳健，代价可接受。此决定已评估，不要擅自改成马氏距离。

---

## 3. eval/ — 评估体系

`eval/metrics.py` 纯函数无 I/O；`eval/run_eval.py` 负责编排与读写。

### 3.1 指标清单

| 指标 | 类型 | 方向 | 能否进 CI / agent 环 |
|---|---|---|---|
| `style_distance` | 确定性 | 越低越好 | 可以 |
| `entity_recall` | 确定性 | 越高越好 | 可以 |
| `numeral_recall` | 确定性 | 越高越好 | 可以 |
| `hallucination_rate` | 确定性 | 越低越好 | 可以 |
| `style_win_rate` | LLM judge | 越高越好 | **仅离线** |
| `ppl` | 需要 GPU | 参考值 | **仅离线** |

`ppl` 用不带 adapter 的基座算。注意风格化文本的 PPL 天然偏高，**它只用来抓明显崩坏，不作主指标**。

### 3.2 保真指标

`eval/fidelity.py`。

实体抽取用 `LAC`（轻量，够用）。**通用 NER 在民国/古代语境会大面积漏**——颂莲、梅珊、陈佐千、端白、沉草都认不出来。所以必须叠加一张从语料自动抽的专有名词表：

```python
def build_entity_gazetteer(pairs: list[Pair]) -> set[str]:
    """从 original 文本抽高频专有名词（LAC 的 PER/LOC/ORG + 高频未登录词）。
    产物存 corpus/gazetteer.json —— 只有词，不含原文，可提交。"""
```

抽取与比对：

```python
class Facts(BaseModel):
    entities: set[str]
    numerals: set[float]
    titles: set[str]

def extract_facts(text: str, gaz: set[str]) -> Facts: ...
```

- `numerals`：正则抓中文与阿拉伯数字，**用 `cn2an` 归一化后再比对**。必须正确处理 `十万 → 100000`、`万人 → 10000`、`三月初九`、`四斤`、全角 `３辆`。这是日志里实测到的 bug（「万人大军」被改写成「十万铁骑」），单测必须覆盖这几例
- `titles`：称谓职官（太医/宫监/丞相/钦差/长工/管家……）。通用 NER 管不了，用词表 + 同义判定。实测 bug：`太医 → 宫监`

指标定义：

```
entity_recall      = |E_in ∩ E_out| / |E_in|        # E_in 为空时记 1.0
numeral_recall     = |N_in ∩ N_out| / |N_in|        # N_in 为空时记 1.0
hallucination_rate = |E_out \ E_in| / |E_out|       # E_out 为空时记 0.0
```

### 3.3 风格胜率（LLM judge）

`eval/judge.py`。**必须 pairwise，不要绝对打分**——绝对分数在不同批次间漂移严重。

- 输入 `(ground_truth_original, candidate_A, candidate_B)`，问哪个更像苏童
- 每个 case 跑**两次，A/B 位置互换**。两次结论一致才计胜或负；不一致记平局
- `style_win_rate = (胜 + 0.5 × 平) / 总数`
- 平局就是两次位置互换的结论不一致。两个候选逐字节相同时不发请求，记为 `identical`，不混进平局。`distribution.style_win_rate` 记 `{"win": n, "loss": n, "tie": n, "identical": n, "tie_rate": f, "identical_rate": f, "n": n}`。`tie_rate = 平 / n`，`identical_rate = identical / n`。`identical` 在 `style_win_rate` 里仍按 0.5 计入。`tie_rate` 是位置一致性诊断，不是质量指标；只在 `identical_rate` 低时才可解读为位置偏差。`copy_ratio` 是系统输出与白话输入的 SequenceMatcher 比值，复用白话化的 `text_similarity`，`--skip-judge` 时照常计算，写入 `per_case.metrics` 和 `distribution`。`copy_ratio` 高而 `tie_rate` 高，是模型在抄，不是评委失效。`--skip-judge` 时 `distribution.style_win_rate` 仍为 null。README 指标表填写 `style_win_rate` 时必须同时标注 `tie_rate` 和 `copy_ratio`
- 凭据与白话化共用 `LLM_BASE_URL` / `LLM_API_KEY`，模型名单独走 `JUDGE_MODEL`（默认 `deepseek-v4-pro`）。不得使用已重映射到 Flash 的 `deepseek-chat`。两处模型可不同——judge 可以用更强的模型，白话化用更便宜的
- `OpenAICompatibleGenerator` 的 `ModelIdentityMismatchError` 闸门对 judge 同样生效，请求名与响应 `model` 不一致即抛错，所以报告里的 `judge_model_used` 等价于响应模型标识。预注册第 4 条要求记录响应模型标识，judge 侧靠这道硬闸满足，不是漏记
- **凭据只从环境变量或 `.env` 读，禁止硬编码。** `.env` 已在 `.gitignore` 中。任何 `.py` 文件里都不得出现 `sk-` 开头的字符串，T0.1 要加一条测试扫描全仓库断言这一点
- judge 调用**必须带磁盘缓存**，key = `sha256(prompt + model)`，目录 `eval/.judge_cache/`（加入 `.gitignore`）。重跑评估不应该重复付费

### 3.4 报告格式

`eval/reports/<run_id>.json`：

```json
{
  "run_id": "20260205-retrieval-k2",
  "timestamp": "2026-02-05T10:00:00Z",
  "pipeline": "retrieval",
  "model": "sutong-v2",
  "config": {"top_k": 2, "seed": 42},
  "n_cases": 73,
  "aggregate": {
    "style_distance": 0.0,
    "entity_recall": 0.0,
    "numeral_recall": 0.0,
    "hallucination_rate": 0.0,
    "style_win_rate": null,
    "ppl": null
  },
  "per_case": [
    {"id": "妻妾成群_0049", "metrics": {}, "output_hash": "sha256:..."}
  ]
}
```

`per_case` 里**只存 hash 不存全文**（版权）。全文留在 `corpus/generations/`，不提交。

### 3.5 必须产出的基线

T0.4 完成后，下表必须有真实数字填进 `README.md`：

| pipeline | style_distance | entity_recall | numeral_recall | hallucination_rate | style_win_rate |
|---|---|---|---|---|---|
| base（无微调） | | | | | |
| sutong-v1 | | | | | |

**在这张表填完之前，不要开始 Phase 1。**

---

## 4. 检索与 agent

### 4.1 检索（`retrieval/`）

三路召回 + RRF 融合：

| 路 | 实现 | 查询形态 |
|---|---|---|
| sparse | `rank_bm25` + jieba + 中文停用词 | 输入白话文本 |
| dense | `BAAI/bge-m3` → pgvector | 输入白话文本 |
| style | numpy 暴力最近邻（**不用向量库**） | 预测出的 20 维风格向量，见 4.2 |

融合：`score(d) = Σ_i 1/(60 + rank_i(d))`。自己实现，约十行。

索引对象是 `original` 文本——范例要给模型看的是原文，不是白话。

style 路为什么不用向量库：907 条 × 20 维，暴力算是微秒级，上向量库纯属负担。

### 4.2 风格向量预测

`retrieval/style_predictor.py`。用 834 条训练对拟合 `sklearn.linear_model.Ridge`：输入白话的 20 维特征 → 预测对应原文的 20 维特征。检索时用**预测向量**而非输入向量去 style 索引找范例。

理由：输入白话的文体特征本身不像苏童（句子短、口语词多），拿它直接检索会召回语料里最不苏童的那些段落，方向是反的。

> **必须做消融**：对比「用预测向量」与「直接用输入特征向量」的端到端风格分。**如果预测版没有优势，删掉这个模块**，并在 CHANGELOG 记录负结果。不要因为设计精巧就保留无效组件。

### 4.3 检索评估的陷阱（必读）

**不要**用「白话查原文能否召回对应那条」当检索指标。白话是从该原文 LLM 改写出来的，词汇重叠极高，这是送分题，刷出来的 recall@5 是假的。

有效评估只有两种：

1. 用**人工撰写**的白话（`corpus/human_eval.jsonl`，见第 8 节）测召回
2. 看**端到端风格分**有没有改善 ← 这才是真正关心的指标

### 4.4 Agent 图（`agent/`）

LangGraph `StateGraph`：

```python
class AgentState(TypedDict):
    input: str
    exemplars: list[Exemplar]
    candidates: list[str]
    violations: list[Violation]
    scores: list[float]
    iter: int
    trace: list[TraceEvent]
```

节点与边：

```
retrieve → generate → verify → score → route

route ─┬─ "accept"      → END
       ├─ "revise"      → generate     （带 violations 作定向反馈）
       └─ "re_retrieve" → retrieve      （风格分过低时换范例）
```

路由规则：

- 保真有违规 且 `iter < 3` → `revise`
- 保真通过 但 `style_distance` 高于阈值 且 `iter < 3` 且 还没换过范例 → `re_retrieve`
- 否则 → `accept`

**终止规则**：

- `iter >= 3` 强制结束
- 同时设置 LangGraph 的 `recursion_limit`（双保险）
- 结束时返回 `candidates[argmax(scores)]` —— **永不返回失败，永不返回空**

**环内只允许确定性指标。** `verify` 用 3.2 的规则指标，`score` 用 2.3 的风格距离。LLM judge 一轮几秒、三轮直接超时，所以它只能离线用。这个「离线评估与在线 reward 用不同精度指标」的分层是本项目核心设计取舍，不要改。

`trace` 每个节点追加一条 `TraceEvent{node, ts, duration_ms, payload}`，直接喂给前端可视化面板（Phase 4）。

### 4.5 revise 的 prompt 构造

反馈必须**具体**，不要说「请更忠实于原意」。按 violation 类型逐条列出：

```
上一版存在以下问题，请修正后重写，其余部分尽量保留：
- 原文中的「太医」被改成了「宫监」，请改回
- 原文中的数字「万人」被改成了「十万」，请改回
```

---

## 5. 服务层（`api/`）

- FastAPI + pydantic v2
- `POST /v1/transform`：请求体 `{text, style, stream, max_iter, seed}`，SSE 流式。**流式事件要同时推生成 token 和 trace 事件**，前端面板靠这个实时更新
- `GET /v1/styles`：可用 adapter 列表
- `GET /healthz`、`GET /metrics`（Prometheus）
- 生成走 vLLM 的 OpenAI 兼容接口，用 `openai` SDK 调。**不要在 api 进程里 import torch**

vLLM 启动（显存按 8GB 卡估，实测后调）：

```bash
vllm serve Qwen/Qwen2.5-3B-Instruct \
  --enable-lora \
  --lora-modules sutong=/path/to/adapters/qwen2.5-3b-sutong-v2 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85
```

Redis **两层缓存，key 分开**：

- `retr:{sha256(text)}` → 检索结果，TTL 7d（复用率高）
- `gen:{sha256(text+style+params)}` → 最终响应，TTL 1d

分开的理由：同一输入换生成参数时，检索结果仍可复用。

---

## 6. 可观测性与 CI

- `opentelemetry-instrumentation-fastapi` 自动埋 HTTP 层；agent **每个节点手动开 span**
- `prometheus-fastapi-instrumentator` 暴露 `/metrics`
- Grafana 面板四块：p95 延迟、**各节点耗时分解**、**修订轮数分布**、cache hit rate。后两个是本项目特有的，也最能讲出东西
- locust 压测，结果写进 README

CI（GitHub Actions）两个 job：

1. `quality` — ruff check + ruff format --check + mypy + `pytest -m "not gpu"`
2. `eval-gate` — 跑 73 条评估（用 Fake generator，CPU 可跑），与 `main` 分支的 `report.json` 对比，**任一确定性指标劣化超过 2% 则 fail**

---

## 7. 版权与数据合规（不可妥协）

苏童作品受版权保护。

- `corpus/*.jsonl` 全部进 `.gitignore`
- 可提交的衍生物：`lexicon.json`（词 + 分数）、`gazetteer.json`（专有名词表）、`style_reference.json`（均值方差）、`split.json`（id 列表）、`reports/*.json`（指标 + hash）。这些都**不可还原原文**
- 必须提供 `scripts/prepare_corpus.py`，README 写明：「语料因版权不包含在仓库中，请自备文本后运行此脚本」
- adapter 权重传 Hugging Face Hub，model card 写明训练数据来源与用途限制
- README 加一节「数据与版权」

---

## 8. 需要新建的测试数据

| 文件 | 内容 | 用途 | 可提交 |
|---|---|---|---|
| `corpus/human_eval.jsonl` | **30 条人工撰写**的白话段落，100–300 字，题材贴近语料（民国、宅院、宫廷、乡土）但不抄原著 | 测域外泛化（已知缺陷 #2）与检索召回 | 是 |
| `corpus/sample_public.jsonl` | 5 条自编的 `(白话, 伪原文)` 对 | 单测与 quickstart demo | 是 |

`human_eval.jsonl` 没有 ground truth，所以只能用 `style_distance` 和 `hallucination_rate` 评，不能算 recall。这是已知局限，README 写明。

> 这 30 条需要人来写。如果你（agent）无法获得，生成占位文件并在 `docs/QUESTIONS.md` 标明「待人工补充」，先用 `sample_public.jsonl` 跑通链路。

---

## 9. 依赖与技术选型理由

```
核心     python 3.11, uv, pydantic v2, numpy, structlog
中文     jieba, cn2an, LAC
检索     rank_bm25, FlagEmbedding (bge-m3), scikit-learn
agent    langgraph                 ← 注意：不是 langchain
服务     fastapi, sse-starlette, openai, redis, psycopg[binary], pgvector, alembic
观测     opentelemetry-*, prometheus-fastapi-instrumentator
质量     ruff, mypy, pytest, pytest-cov, locust
推理     vllm                      ← 只在 GPU 机器装，放 optional dependency group
```

版本不在本文件里钉死，由 `uv.lock` 负责。

**为什么不用 LangChain**：本项目 style 路的检索查询是一个 20 维 numpy 向量，不是字符串。LangChain 的 `BaseRetriever.invoke(query: str)` 接口假设查询是文本，适配它只能继承基类重写，净收益为负。`EnsembleRetriever` 虽内置 RRF，但它融合的是吃字符串的 retriever，同样用不上。RRF 自己写十行就够。

**为什么用 LangGraph**：类型化 state、条件边、`recursion_limit`、中间状态流式输出（前端 trace 面板需要）、checkpointer 便于调试。四节点规模下它算便利而非必需，但 trace 流式这一项值得。

---

## 10. 相关工作与方法依据

本项目不主张方法创新。伪平行语料构造、风格分类器、词汇改写、PEFT与多阶段转换
均有先例；工程价值是可复现评估、明确失败、完整实现与针对中文文学的实测。
下列文献用于方法定位，不将其他任务/语言的数值或结论直接搬来当本项目证据。

- Chen & Dolan (2011), [Collecting Highly Parallel Data for Paraphrase Evaluation](https://aclanthology.org/P11-1020/)。
  PINC度量候选相对源的词汇新颖性，原定义对1–4阶取均值；参考重合另用于内容评价。
  本项目中文字符适配与原词级指标的区别、重复计数及方向见1.5.1。
- Mukherjee et al. (2025), [Evaluating Text Style Transfer Evaluation: Are There Any Reliable Metrics?](https://www.arxiv.org/pdf/2502.04718v2)。
  讨论风格准确性、内容保留、自然度以及source/reference评价，比较PINC、BERTScore、
  分类器置信度等指标。平凡复制会使源重合虚高，是本项目需防的已知评价风险；
  该文不能证明本项目Flash/Pro模型的具体复制机制，也不是中文作者风格实验。
- [Text Style Transfer with Parameter-efficient LLM Finetuning and Round-trip Translation](https://arxiv.org/html/2602.15013v1) (2026)。
  以round-trip translation降风格造伪平行对，再做LLM参数高效微调，与本管线相近。
  本项目用LLM直接降级，不是该文翻译步骤的复刻，也不把降级造对称为首创。
- [Text Style Transfer: A Review and Experimental Evaluation](https://arxiv.org/pdf/2010.12742)。
  综述含构造伪平行数据等方法及风格、内容、流畅性评价；是领域定位依据。
- [文本风格迁移研究综述（软件学报2022）](https://jos.org.cn/html/2022/12/6544.htm)。
  官方索引摘要支持“保留内容、修改风格”的任务定义；当前网页访问受限，未以未读细节
  支撑具体方法效果，待补全文核验。
- [基于风格化嵌入的中文文本风格迁移（CCL2021）](https://aclanthology.org/2021.ccl-1.26/)。
  中文style embedding先例；支持中文TST已有研究，不证明本项目指标或苏童风格有效。
- 虚词频率支持作者归属的中文实证方向，可补充支撑stylometry第8–14维；
  [The Many Voices of Du Ying: Revisiting the Disputed Writings of Lu Xun and Zhou Zuoren](https://dh-abstracts.library.virginia.edu/works/12053)
  使用虚词特征与logistic regression，不是用户提到的SVM研究。用户提供另一研究的
  “鲁迅1918–1936、57.7万字、虚词频率+SVM、时间漂移”细节尚未找到完整书目，标记待核验，
  不伪造引文。语料限定同期、避免把时期漂移当作者差异仍作为工程约束，非本轮实验结论。

## 11. 目录结构

```
sutong-style/
├── AGENTS.md
├── README.md                   ← 含指标对比表，每阶段更新
├── pyproject.toml
├── docker-compose.yml
├── .github/workflows/ci.yml
├── docs/
│   ├── SPEC.md  PLAN.md  CHANGELOG.md  QUESTIONS.md
├── corpus/                     ← gitignored，白名单除外
│   ├── pairs.jsonl
│   ├── split.json              ✅ 可提交
│   ├── gazetteer.json          ✅
│   ├── human_eval.jsonl        ✅
│   └── sample_public.jsonl     ✅
├── stylometry/                 ← 技术核心，无 I/O
│   ├── features.py  lexicon.py  distance.py
│   └── data/lexicon.json  data/style_reference.json   ✅
├── eval/
│   ├── metrics.py  fidelity.py  judge.py  run_eval.py
│   ├── configs/*.yaml
│   └── reports/*.json          ✅
├── retrieval/
│   └── bm25.py  dense.py  style_index.py  style_predictor.py  fusion.py
├── agent/
│   └── graph.py  nodes.py  state.py  prompts.py
├── api/
│   └── main.py  routes.py  schemas.py  cache.py  telemetry.py
├── infra/                      ← 真实模型与外部服务实现，唯一可 import torch 的地方
│   └── vllm_generator.py  bge_embedder.py
├── web/                        ← Phase 4，Vite + React + TS
├── scripts/
│   └── prepare_corpus.py  normalize_corpus.py  train_v2.py  loadtest.py
└── tests/
    ├── fakes.py                ← 所有 Protocol 的假实现
    └── test_*.py
```
