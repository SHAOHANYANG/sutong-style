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

- 30 条中 14 条完整（`maybe_truncated: false`），其余在日志里被截断
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
- 全部数字（含年份、数量、日期）
- 全部情节与对话内容
- 段落结构（原文分几段，白话也分几段）

**必须改变**：

- 书面语 → 口语（「心如死水」→「心里凉透了，啥盼头都没有」）
- 长句 → 短句，拆开定语
- 成语、文言残留 → 大白话
- 不加引号的对话 → 常规对话写法（苏童的无引号对话是**目标侧**特征，不该出现在输入侧）

**禁止**：概括、缩写、增补原文没有的情节。白话版字数应在原文的 **0.9–1.3 倍**，超出范围的 chunk 要重试并记录。

**工程要求**（照搬当初 pipeline 已验证的做法）：

- **断点续跑**：启动时扫描已有输出，只处理 remaining。原日志的 `Total chunks: 907, already done: 0, remaining: 907` 就是这个机制
- **并发** + 失败重试，每 20 条打一次进度 `progress: N/907 (ok=, failed=)`
- 模型与当初保持同级。凭据走 `LLM_BASE_URL` / `LLM_API_KEY`，模型名走 `VERNACULARIZE_MODEL`（默认 `deepseek-chat`），OpenAI 兼容接口
- 单条失败不中断全局，最后汇总 failed 列表

**一个重要的决策点（需要人工拍板，先记进 `docs/QUESTIONS.md`）**：

模型的已知缺陷 #2（域外输入失效）的**根因就在这一步**——LLM 生成的白话仍带着原文的句子骨架和叙事密度，和人类随手写的白话分布不同。旧语料已经灭失，**所以重建时没有"保持兼容"的包袱，可以直接修这个根因**：

- 方案 A（保守）：复刻当初的效果，参照 `salvaged_pairs.jsonl` 的白话风格
- 方案 B（治根）：prompt 里额外要求打散句子边界、加入口语赘余和语气词，让产出的白话更接近真人写法

方案 B 更可能解决缺陷 #2，但需要一把尺子来验证"更接近真人"。那把尺子就是 `corpus/human_eval.jsonl`：用 `stylometry` 算**生成白话**与**人工白话**之间的 `style_distance`，越小说明越接近真实输入分布。

> 这意味着 **`human_eval.jsonl` 至少要先写 10 条**，才能调这个 prompt。原计划把它排在 Phase 1 末尾，现在要提前。见 PLAN 的 T0.0。

**验收**：对 `salvaged_pairs.jsonl` 里 14 条完整样本的同名 chunk 重跑白话化，人工比对新旧两版。新版在「信息保全」上不得劣于旧版（人名、数字、对话一个不少）。

14 条参照样本按 `difflib.SequenceMatcher(None, 白话, 原文).ratio()` 实测：均值 **0.539**、中位数 **0.558**、最小值 **0.274**、最大值 **0.680**。复制率校验必须使用同一直接比较口径，所有密度档位的硬上限不得超过 **0.70**，不得为放行个别样本上调全局标准。按原文 chunk 的对白与专名密度分三档：

- 低密度（对白 `< 0.20` 且专名 `< 0.07`）：上限 **0.65**
- 中密度（对白 `>= 0.20` 或专名 `>= 0.07`）：上限 **0.68**
- 高密度（对白 `>= 0.50` 或专名 `>= 0.10`）：上限 **0.70**

对白密度按含引号或说话标记的句段所占中文字比例计算，专名密度按 `jieba.posseg` 的人名、地名、机构名等专名词性所占中文字比例计算。必须从原文计算，不能从模型输出计算。重试后仍超过对应上限的候选不得进入 `pairs.jsonl`，并必须逐条写入 `rebuild_report.json` 的 `copy_similarity_exceptions`（记录最低相似度、对应上限及两项密度），不得静默计为正常通过。

**已拍板：走方案 A**（2026-10-02）。复刻当初的白话风格，参照 `salvaged_pairs.jsonl`。缺陷 #2 本轮不处理，留待后续——理由与后续方案记在 `docs/QUESTIONS.md` 的 Q1。

**预览停止条件（2026-10-02 补充）**：20 条预览序列相似度均值必须 **≤ 0.60**、最大值必须 **≤ 0.70**。每次调用从 14 条完整旧样本中可复现地轮换注入 2–3 条真正的 few-shot，排除当前同名样本。此次最多再调整、评测两轮 prompt，之后无论结果如何均停止并报告全部 20 条的分布；达到均值门槛前不得启动全量白话化。

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
- 凭据与白话化共用 `LLM_BASE_URL` / `LLM_API_KEY`，模型名单独走 `JUDGE_MODEL`（默认 `deepseek-chat`）。两处模型可不同——judge 可以用更强的模型，白话化用更便宜的
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

## 10. 目录结构

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
