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

### 1.1 现状（必须先核实）

既有产物在一台 Linux 机器上，路径 `/home/shaohan_yang/projects/sutong-style/`：

| 文件 | 内容 | 条数 |
|---|---|---|
| `corpus/pairs.jsonl` | `(白话, 原文)` 训练对 | 907 |
| `corpus/eval_results.jsonl` | 微调模型在 eval 集上的生成结果 | 73 |
| `corpus/base_results.jsonl` | 基座模型在同一 eval 集上的结果 | 73 |
| `adapters/qwen2.5-3b-sutong-v1/` | LoRA adapter（r=32） | — |

语料覆盖 6 部作品：`妻妾成群`、`妇女生活`、`另一种妇女生活`、`园艺`、`罂粟之家`、`我的帝王生涯`。chunk id 形如 `我的帝王生涯_0429`。

训练切分：train 834 / eval 73。基座模型 `Qwen/Qwen2.5-3B-Instruct`，LoRA r=32，可训练参数 29,933,568（占 0.96%）。

> **以上字段名是从运行日志反推的，未经核实。** T0.1 的第一件事是打开这些文件确认真实 schema，然后更新本节。

### 1.2 目标 schema（规范化后）

所有下游代码只认这个 schema。T0.1 要写 `scripts/normalize_corpus.py` 把实际数据转成它。

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

`split` 取值 `train` 或 `eval`，**必须与 v1 训练时的切分一致**，否则后续所有对比失效。如果原始文件没有这个字段，从 `eval_results.jsonl` 的 id 列表反推 eval 集，并把切分固化到 `corpus/split.json`。

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
- judge 模型从环境变量读：`JUDGE_MODEL`、`JUDGE_BASE_URL`、`JUDGE_API_KEY`，走 OpenAI 兼容接口，默认 `deepseek-chat`
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
