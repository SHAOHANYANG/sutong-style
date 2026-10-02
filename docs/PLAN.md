# PLAN — 任务清单

按顺序做。每个任务一个 commit。标记含义见 `AGENTS.md` 的「GPU 约束」一节：

- `[CPU]` 你能完整做完，必须做完
- `[GPU]` 需要 GPU 执行，你只交付代码 + 可独立运行的脚本 + 运行说明
- `[GPU-OPT]` 需要 GPU 但有降级路径，走降级路径交付

每个任务的「验收」是机械可判定的。**验收不过不要进下一个任务。**

---

## Phase 0 — 语料重建与评估基线（最优先，不可跳过）

目标：先把灭失的语料和模型造回来，再建立唯一裁判。完成后 `README.md` 的指标对比表必须有 base 与微调模型两行真实数字。

> **背景**：2025-08 的全部产物（907 对语料、v1 adapter、两份生成结果）随 AutoDL 实例释放而永久丢失，详见 `docs/SPEC.md` 第 1.1 节。重建成本不高——几十块钱 LLM 调用费加一次 44 分钟训练——但它是一切的前提。

### T0.0 `[CPU]` + `[GPU]` 语料重建

依赖：无（与 T0.1 的骨架搭建可并行，但 T0.2 之后的一切都依赖它）

**这是新增任务，原计划假设语料还在。**

产出：
- `scripts/chunk_corpus.py`：从自备的六部作品文本切 chunk。目标 907 个、每个 200–400 字、id 格式 `{work}_{idx:04d}`（对齐 SPEC 1.2 的已知参数）
- `scripts/vernacularize.py`：按 **SPEC 1.5** 实现。断点续跑、并发、失败重试、每 20 条打进度
- `scripts/split_corpus.py`：**固定 `seed=42`**，按 `work` 分层，eval 约 8%，结果写 `corpus/split.json` 并提交。此后永不变动
- `corpus/pairs.jsonl`（不提交）、`corpus/split.json`（提交）

**前置条件均已就绪，此任务可立即开工**：
- ✅ 六部作品文本已从回收站恢复，在 `corpus/raw/`（已 gitignore）。文件清单与合集分篇规则见 **SPEC 1.6**
- ✅ 白话化方案已拍板：**方案 A**（复刻当初风格，参照 `salvaged_pairs.jsonl`）
- ✅ `human_eval.jsonl` 不再是前置，Phase 1 之前交付即可

切分脚本有四个已知陷阱，**全在 SPEC 1.6，动手前必读**：合集分篇要按「标题独占一行」、`另一种妇女生活` 必须先于 `妇女生活` 匹配、`园艺` 在正文里也是普通词、`※※※` 是天然 chunk 边界。另需做繁体残留清洗（`麽`→`么`），但**不要规范化引号**。

验收：
- 907 ± 30 个 chunk，六部作品都有，无空 chunk
- 分篇正确性有单测：四部中篇的首尾各取一句，断言落在正确的 `work` 下
- 繁体清洗有单测，且断言引号**未**被改动
- `vernacularize.py` 中断后重启只处理 remaining（写一条集成测试，用 Fake LLM）
- 白话版字数落在原文的 0.9–1.3 倍区间，越界的 chunk 被记录在 `corpus/rebuild_report.json`
- **对 `salvaged_pairs.jsonl` 里 14 条完整样本的同名 chunk 重跑，人工比对新旧两版**，新版的人名/数字/对话不得有缺失
- `corpus/split.json` 已提交，`corpus/pairs.jsonl` 未被提交

### T0.1 `[CPU]` 仓库骨架

依赖：无

产出：
- `pyproject.toml`（uv，依赖分组：`core` / `gpu` / `dev`），`ruff` 与 `mypy strict` 配置
- `.gitignore`：挡 `corpus/*.jsonl`（白名单见 SPEC 第 7 节）、`__pycache__`、`unsloth_compiled_cache/`、`*.safetensors`、`eval/.judge_cache/`、`.venv`
- `docs/CHANGELOG.md`、`docs/QUESTIONS.md` 空文件
- `scripts/prepare_corpus.py`：从自备文本切 chunk 产出 `pairs.jsonl` 骨架（切分逻辑与既有 907 chunk 对齐，chunk id 格式 `{work}_{idx:04d}`）
- `scripts/validate_human_eval.py`：校验 `corpus/human_eval.jsonl`。检查 30 条齐全、每条 100–300 字、**实体 ≥ 2**、**数字 ≥ 1**、6 个 domain 分布均匀。后两项是硬指标，缺了会让保真指标恒为满分（见 QUESTIONS 的 Q3）
- `corpus/sample_public.jsonl`：5 条自编样例（**自己写，不要用原著文本**）

`corpus/human_eval.jsonl` 已预填 30 条情境种子，`vernacular` 字段为空、**待人工填写**。不要代写，原因见 `corpus/HUMAN_EVAL_GUIDE.md`。在填好之前，涉及 human_eval 的评估一律跳过并在报告里标 `null`，**不要拿空串当输入跑出一堆 0 分**。

验收：
- `uv sync && uv run ruff check . && uv run mypy . && uv run pytest` 全绿
- `uv run python scripts/validate_human_eval.py corpus/human_eval.jsonl` 能正确报出「30 条待填写」
- `git status` 确认 `corpus/pairs.jsonl` 与 `corpus/salvaged_pairs.jsonl` 都不在待提交列表里

---

### T0.2 `[CPU]` stylometry 特征与风格距离

依赖：T0.1

产出：`stylometry/features.py`、`lexicon.py`、`distance.py`，按 SPEC 第 2 节实现。附 `scripts/build_lexicon.py` 与 `scripts/fit_style_reference.py`。

验收：
- `extract()` 对 20 维向量的每一维都有单测，用手写的小段文本断言期望值
- **第 18 维必须有专门单测**：`"第四天"` → 1.0，`"第４天"` → 0.0，`"第4天"` → 0.0
- 空文本、纯标点、单字这三种退化输入不抛异常
- `StyleReference.distance()` 满足：原文样本的平均距离 **显著低于** 白话样本的平均距离（用 `sample_public.jsonl` 断言）。这是整套特征是否有效的冒烟测试，不过就说明特征设计有问题
- `pytest --cov=stylometry` 覆盖率 ≥ 90%

---

### T0.3 `[CPU]` 保真指标

依赖：T0.2

产出：`eval/fidelity.py`、`eval/metrics.py`，按 SPEC 3.2 实现。附 `scripts/build_gazetteer.py`。

验收 —— **以下实测 bug 必须被抓到，逐条写成单测**：

| 输入 → 输出 | 期望 |
|---|---|
| `万人大军` → `十万铁骑` | `numeral_recall < 1.0` |
| `太医` → `宫监` | `titles` 不匹配，产生 violation |
| `垂死的酸气` → `死尸散发的酸臭之气` | 不要求抓到（语义级，超出规则能力），但**不得崩**，且需在 QUESTIONS.md 记录为已知局限 |
| `三辆马车` → `３辆马车` | `numeral_recall == 1.0`（归一化后相等）|

另外：
- `cn2an` 归一化单测覆盖 `十万` `万` `三月初九` `四斤` `二十` `一百二十`
- 三个指标的边界情况（输入实体集为空、输出为空串）按 SPEC 3.2 的约定返回，有单测
- `pytest --cov=eval` 覆盖率 ≥ 85%

---

### T0.4 `[CPU]` 评估编排

依赖：T0.3

产出：
- `eval/judge.py`（pairwise + 位置互换 + 磁盘缓存）
- `eval/run_eval.py`（CLI：`--config`、`--generations`、`--run-id`、`--skip-judge`）
- `eval/configs/baseline.yaml`

此任务**不产出基线数字**——模型还没训出来。用 `sample_public.jsonl` + Fake generator 把链路跑通即可。

验收：
- `--skip-judge` 模式下不发任何网络请求（`pytest` 断言无出网）
- judge 缓存命中时不重复请求（单测断言调用次数）
- 对 5 条样例能产出格式合法的 report，`aggregate` 六个字段齐全
- 位置互换逻辑有单测：构造一个永远选 A 的 Fake judge，断言胜率收敛到 0.5 而不是 1.0（这是位置偏差的检测手段）

---

### T0.5 `[GPU]` 训练与基线

依赖：T0.0、T0.4

产出：
- `scripts/train.py`：超参照抄 SPEC 1.2 的已知参数（LoRA r=32、lr 2e-4 cosine、batch 2 × grad_accum 2、padding-free、gradient offload），**但 epochs 设为 2**
- `scripts/generate.py`：批量生成，支持 `--adapter`、`--pipeline`、`--seed`，输出符合 SPEC 1.2 的 `generations/*.jsonl`
- README 的「如何在 GPU 机器上复现」一节
- **填好 README 的指标对比表**

为什么是 2 epochs：旧 v1 训了 3 epochs，但 eval_loss 在 epoch 1.9 触底 2.056、之后回升到 2.089——第三个 epoch 纯属过拟合。不要为了"复刻 v1"去训 3 epochs，没有意义，旧 v1 已经不存在了。

**训练完成后立刻备份**：adapter 传 HF Hub 私有库，`pairs.jsonl` 和 `split.json` 下载到本地。这个项目已经因为没备份全丢过一次。

验收：
- 脚本在无 GPU 环境下 `--help` 和 `--dry-run` 正常
- commit message 注明「待人工在 GPU 机器执行」
- 人工执行后：eval_loss 曲线记进 CHANGELOG；README 表格的 `base` 与微调两行填上真实数字
- **adapter 已上传 HF Hub，语料已下载到本地**，CHANGELOG 记录备份位置

> **这是 Phase 0 的出口。表格填完才能进 Phase 1。**

---

## Phase 1 — 检索

目标：给模型「开卷考试」的参考材料。出口是 README 表格多一行 `retrieval`。

### T1.1 `[CPU]` BM25 路

依赖：T0.4

产出：`retrieval/bm25.py`，jieba 分词 + 中文停用词表（停用词表自带在 `retrieval/data/stopwords.txt`，不要运行时下载）。

验收：索引 `sample_public.jsonl` 后，用其中一条的白话查询，能召回对应 original；停用词生效的单测。

### T1.2 `[GPU-OPT]` dense 路

依赖：T1.1

产出：`retrieval/dense.py`、`infra/bge_embedder.py`（`Embedder` Protocol 的真实实现）、`tests/fakes.py` 里的 `FakeEmbedder`。向量存 pgvector。

降级路径：`bge-m3` 在 CPU 上编码 907 条约几分钟，可接受。先用 CPU 跑通并把向量缓存到 `retrieval/data/embeddings.npy`（gitignored）。

验收：所有测试用 `FakeEmbedder`，不下载模型、不连网络、不碰 GPU。

### T1.3 `[CPU]` 风格索引与向量预测（含消融）

依赖：T1.2

产出：`retrieval/style_index.py`（numpy 暴力最近邻）、`retrieval/style_predictor.py`（Ridge）、`scripts/ablate_style_predictor.py`。

验收：
- 消融脚本输出两组端到端风格分：预测向量 vs 直接用输入特征
- 结论写进 `docs/CHANGELOG.md`
- **如果预测版没有优势，删掉 `style_predictor.py`**，CHANGELOG 记录负结果。这是明确授权的删除

### T1.4 `[CPU]` RRF 融合

依赖：T1.3

产出：`retrieval/fusion.py`。

验收：RRF 公式的单测（构造已知 rank 列表，断言融合后顺序）；三路任一为空时不崩。

### T1.5 `[CPU]` few-shot prompt 与 k 扫描

依赖：T1.4

产出：`retrieval/prompt.py`，`scripts/sweep_topk.py`（k ∈ {0,1,2,3}）。

验收：
- prompt 构造有单测（断言范例数量、顺序、token 预估）
- 扫描脚本能用 Fake generator 跑通
- **必须统计并记录 prompt token 数**。3 条范例轻松超 1500 token，会同时降质降速

### T1.6 `[GPU]` retrieval pipeline 评估

依赖：T1.5

产出：README 表格追加 `retrieval` 行；CHANGELOG 记录最优 k 与指标变化。

验收：k 扫描结果以表格形式进 CHANGELOG；检索召回**只用 `human_eval.jsonl` 评**（SPEC 4.3 的陷阱，不要用训练对自测）。

---

## Phase 2 — Agent 自检重写环

目标：治事实漂移。出口是 README 表格多一行 `agent`，且 `hallucination_rate` 明显下降。

### T2.1 `[CPU]` 图骨架

依赖：T1.4（不依赖 T1.6，可与 Phase 1 尾部并行）

产出：`agent/state.py`、`agent/graph.py`、`agent/nodes.py` 骨架，全部节点先用 Fake 实现。

验收：
- 用 `FakeGenerator` 端到端跑通图，`pytest` 可验证
- **循环上限有单测**：构造一个永远违规的 Fake，断言恰好 3 轮后退出且返回非空
- `recursion_limit` 与显式 iter 计数两道保险都有测试

### T2.2 `[CPU]` verify / score / route 接真实指标

依赖：T2.1

产出：节点接入 `eval/fidelity.py` 与 `stylometry/distance.py`。路由阈值放 `agent/config.py`，不要硬编码在逻辑里。

验收：
- **断言 agent 模块没有 import `eval/judge.py`**（写成一条 import 检查测试）。LLM judge 不得进环，见 SPEC 4.4
- 三条路由分支各有单测

### T2.3 `[CPU]` revise prompt

依赖：T2.2

产出：`agent/prompts.py`，按 SPEC 4.5 构造逐条具体反馈。

验收：给定一组 violations，断言生成的 prompt 文本包含每一条的具体词对（如「太医」「宫监」）。禁止出现「请更忠实于原意」这类笼统措辞——写一条断言排除它。

### T2.4 `[CPU]` trace 结构

依赖：T2.3

产出：`TraceEvent` 落地，每节点记录 `node / ts / duration_ms / payload`。

验收：trace 可 JSON 序列化；跑一次图后 trace 事件数与实际节点执行次数一致。

### T2.5 `[GPU]` agent pipeline 评估

依赖：T2.4、T1.6

产出：README 表格追加 `agent` 行；CHANGELOG 记录 `hallucination_rate` 与 `numeral_recall` 的变化幅度，以及**修订轮数分布**（多少条一轮过、多少条用到 3 轮）。

验收：轮数分布以表格进 CHANGELOG。这个数据后面 Grafana 面板要用。

---

## Phase 3 — 服务化

目标：从脚本变成服务。这一阶段指标不变，产出的是工程能力证据和性能数字。

### T3.1 `[CPU]` FastAPI + SSE

依赖：T2.4

产出：`api/main.py`、`routes.py`、`schemas.py`。按 SPEC 第 5 节。

验收：
- `pytest` 用 `httpx.AsyncClient` 测 SSE，断言**既收到 token 事件也收到 trace 事件**
- **断言 `api/` 下没有 import torch**（写成 import 检查测试）
- `/healthz` 在依赖服务不可用时返回 503 而不是崩

### T3.2 `[CPU]` Redis 两层缓存

依赖：T3.1

产出：`api/cache.py`。两层 key 分开，见 SPEC 第 5 节。

验收：用 `fakeredis` 测；断言换生成参数时检索缓存仍命中（这是分两层的全部意义）。

### T3.3 `[CPU]` Postgres + pgvector + alembic

依赖：T3.2

产出：alembic 迁移脚本，建 chunk 元数据表与向量表。

验收：`alembic upgrade head` 与 `downgrade base` 都能跑通。

### T3.4 `[CPU]` docker-compose

依赖：T3.3

产出：`docker-compose.yml`（api / postgres / redis / prometheus / grafana），`.env.example`。vLLM 单独一个 profile，默认不启动。

验收：`docker compose config` 校验通过；README 有一条可复制的启动命令。

### T3.5 `[CPU]` 可观测性

依赖：T3.4

产出：`api/telemetry.py`、Grafana dashboard JSON（提交进仓库）。四块面板见 SPEC 第 6 节。

验收：`/metrics` 返回 Prometheus 格式；agent 每个节点产生一个 span（单测用 in-memory span exporter 断言 span 名与数量）。

### T3.6 `[CPU]` CI

依赖：T3.5

产出：`.github/workflows/ci.yml`，两个 job 见 SPEC 第 6 节。

验收：
- `quality` job 在干净 clone 上能过
- `eval-gate` 用 Fake generator 在 CPU 跑完 73 条
- **故意把某个指标调坏，确认 gate 真的 fail**（这一步必须实际验证，不要假设它能工作）

### T3.7 `[GPU]` vLLM 接入与压测

依赖：T3.6

产出：`infra/vllm_generator.py`、`scripts/loadtest.py`（locust）、README 性能一节。

验收：
- 脚本在无 GPU 时 `--help` 正常
- README 预留性能表格（RPS / p50 / p95 / cache hit rate），人工跑完填入
- 显存提示写进 README：8GB 卡需 `--max-model-len 4096 --gpu-memory-utilization 0.85`，不够就换 AWQ 量化版

---

## Phase 4 — 前端与发布

目标：让人在十秒内看懂这套架构。

### T4.1 `[CPU]` trace 可视化前端

依赖：T3.1

技术栈：Vite + React + TS + Tailwind + shadcn/ui。**不要用 Next.js**（单页，SSR 无收益）。SSE 用 `fetch` + `ReadableStream` 读，**不要用 `EventSource`**（它不支持 POST，而输入是长文本）。

布局：

```
┌─────────────┬──────────────────────────┐
│  白话输入    │  苏童风格输出（流式）      │
├─────────────┴──────────────────────────┤
│  ▸ 检索范例  找到 2 段（风格匹配 0.87/0.81）│
│    └ 《妻妾成群》_0049  对话密集型        │
│  ▸ 生成      第 1 轮 ✓                   │
│  ▸ 保真校验  ⚠ 2 处漂移                  │
│    └ 太医 → 宫监      │ 万人 → 十万      │
│  ▸ 修订      第 2 轮 ✓                   │
│  ▸ 校验      通过  风格 0.84 保真 1.00    │
└────────────────────────────────────────┘
```

**核心要求：不要做成聊天框。** 下方 trace 面板是主角——它让访客不读文档就看懂架构。另加两项：base 模型与微调模型并排对比；第 1 轮与第 2 轮的 diff 高亮。

还要处理后端冷启动状态（HF Spaces 会睡），显示「正在唤醒推理服务…」而不是报错。

验收：
- `npm run build` 产出静态文件
- 后端不可用时页面仍能加载并显示明确状态
- 窄屏（400px）不横向溢出

### T4.2 `[CPU]` README

依赖：T4.1

必须包含，按此顺序：
1. 一句话说明 + 一个 GIF（占位，人工录制后替换）
2. **指标对比表**（base / v1 / v2 / retrieval / agent 五行）
3. 架构图（mermaid）
4. Quickstart（docker compose 一条命令）
5. 「书面语词表是怎么从自有平行语料学出来的」—— 这是别人复刻不了的部分，单独一节
6. 「取舍与坑」—— 为什么不用马氏距离、为什么不用 LangChain、为什么 judge 不进 agent 环、检索评估的陷阱。**区分度最高的一节**
7. 「数据与版权」
8. 性能数字
9. 已知局限（语义级漂移抓不到、human_eval 无 ground truth、域外泛化仍未完全解决）

验收：前三屏之内能看到 GIF 和指标表。

### T4.3 `[CPU]` 部署

依赖：T4.2

产出：前端部署 Vercel/Netlify（静态，永远在线）；后端 HF Spaces + ZeroGPU（免费按需 GPU，闲置休眠）。配置文件与说明进仓库。

验收：README 有可点击的 demo 链接占位；冷启动 UX 已在 T4.1 处理。

### T4.4 `[CPU]` 可选：Claude Skill 包装

依赖：T4.3

一个调 `/v1/transform` 的轻量 skill，README 加一节「也可以在 Claude Code 里用」。**优先级最低，前面全部完成后再做。**

---

## 不要做的事

明确排除，不要「顺手也做了」：

- 用户系统、鉴权、多租户
- 模型训练的 Web UI
- A/B 实验平台
- 除苏童外的其他作者 adapter（多风格是 Phase 5 的事，现在不做）
- DSPy / prompt 自动优化
- 把 LLM judge 做成在线服务
- 数据飞轮（自动找失败样本 → 生成训练对 → 重训）

最后三项都是有价值的后续方向，但不在本轮范围内。想做就先在 `docs/QUESTIONS.md` 里提出来等确认。

---

## 进度自查

每完成一个 Phase，确认四件事（见 `AGENTS.md` 末节）：

1. 跑了完整评估，`eval/reports/*.json` 已提交
2. README 指标表加了一列
3. `docs/CHANGELOG.md` 记录了做了什么、指标变了多少、踩了什么坑
4. **如果指标没改善，照实写。** 不要调参凑数字，不要换指标定义
