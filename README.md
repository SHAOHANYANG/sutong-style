# sutong-style

把现代白话中文改写成苏童文风。Qwen2.5-3B + LoRA 微调，外挂风格范例检索与生成自检重写环。

> 🚧 施工中。实现者请先读 [`AGENTS.md`](AGENTS.md)，再读 [`docs/SPEC.md`](docs/SPEC.md) 和 [`docs/PLAN.md`](docs/PLAN.md)。

<!-- TODO(T4.2): 插入 demo GIF -->

## 效果

<!-- TODO(T0.5/T1.6/T2.5): 每阶段填一行真实数字，不要留占位符 -->

59 条 eval 切片，贪心解码，seed 42。`style_win_rate` 的对手是白话输入（SPEC Q18）。

| pipeline | style_distance → 1.0 | profile_gap ↓ | profile_gap_per_case ↓ | entity_recall ↑ | numeral_recall ↑ | hallucination_rate ↓ | copy_ratio | style_win_rate ↑ | tie_rate |
|---|---|---|---|---|---|---|---|---|---|
| base（Qwen2.5-3B 无微调） | 1.149 | 0.494 | 0.748 | 0.958 | 0.923 | 0.0042 | 0.875 | — | — |
| + LoRA 微调（sutong-v2） | 0.923 | **0.204** | **0.514** | **1.000** | 0.861 | 0.0141 | 0.709 | 0.331 | **0.49** |
| + 风格范例检索（balanced，k = 2） | 0.930 | 0.214 | 0.485 | 1.000 | 0.892 | 0.0042 | 0.742 | — | — |
| + 自检重写环（balanced k = 2，followup） | 0.933 | 0.211 | 0.485 | 1.000 | 0.920 | 0.0042 | 0.750 | — | — |
| + 自检重写环 第二轮（回显防护，system 反馈） | 0.929 | 0.209 | 0.486 | 1.000 | 0.898 | 0.0000 | 0.742 | — | — |
| *参照：苏童原文* | *1.035* | *0* | *0* | — | — | — | — | — | — |
| *参照：白话输入* | *0.992* | *0.450* | *0.623* | — | — | — | — | — | — |

**这张表要对着两行参照读，不要只看升降。**

- `style_distance` 是 20 维 z 分数的均方根，**目标是 1.0 不是 0**。真苏童原文得 1.035，正是理论期望值。LoRA 的 0.923 低于原文，意思是它写得「比苏童更平均苏童」——向语料均值回归，方差比真人小。而白话输入本身就有 0.992，所以这个标量**分不开「完全没改」和「改得完美」**，只能当弱信号。
- `profile_gap` 是集合级：先对全部样本求剖面差 d 的逐维均值，再对 20 维取绝对值求平均。方向相反的误差会抵消。白话 0.450 → LoRA 0.204，这句话里的「缩了一半还多」只在这个口径下成立；base 是 0.494。逐条口径 `profile_gap_per_case` 是每一条与自己原文的平均 \|z\| 差，再对样本求均值：白话 0.623 → LoRA 0.514，base 0.748。`human_eval` 没有 ground truth，两列都算不了。
- `style_win_rate` 这一轮**不可单独引用**：`tie_rate = 0.49`，A/B 位置对调后评委有一半改口。详见下方「已知局限」。

**风格范例检索（主配置 balanced，k = 2）相对同次扫描的 k = 0 对照（SPEC §4.7）。** `profile_gap_per_case`、`numeral_recall`、`hallucination_rate` 的均值差方向都朝「更好」一侧，但 95% 自助区间都含 0，按预注册规则只能写**未检出差异**；`entity_recall` 两边都是 1。`copy_ratio` 上升且区间不含 0：给了范例之后输出离输入更近了，数值保真上的变化可能部分来自模型改得更少，不能单独读成保真改善。两个副作用诊断量：相对对照未检出范例抄写，范例实体泄漏为 0。59 条样本的检验效力有限。本行 `style_win_rate` / `tie_rate` 填「—」（本轮不跑 judge）。完整 13 组与区间见 `eval/reports/retrieval-sweep.md`。

**自检重写环（主臂 balanced k = 2 + followup）相对同样 59 条的检索行（SPEC §4.8）。** `numeral_recall` 0.892 → 0.920，配对均值差 +0.028，95% 自助区间 [+0.006, +0.059] 不含 0，按预注册规则记为**改善**；`entity_recall`、`hallucination_rate` 两边逐条相同；`profile_gap_per_case` 未检出差异；`copy_ratio` +0.008，区间 [+0.001, +0.017]（只报告）。这个改善要打三个折扣：（1）59 条里只有 13 条第一轮有违规，最终输出改动的是 8 条，修掉 5 条违规（4 条数值、1 条称谓），没有新引入；（2）校验器和评估用的是同一套规则，agent 按违规最少挑版本，这三项指标的改善有一部分是构造使然，人工抽查尚未做（pending）；（3）4 条数值修复里有 1 条把阿拉伯数字抄进了输出。PLAN 的出口条件「`hallucination_rate` 明显下降」**未达到**：唯一一条有幻觉的样本三轮都没修掉。`numeral_recall` 0.9198 也差一点没到 0.92 的目标。

**restate 反馈格式的数字不可引用。** 探索臂 `agent-balanced-k2-restate` 的 `numeral_recall` 是 0.986，看起来最好，但它最终改动的 11 条输出里有 10 条是在正文后面把修订反馈原样抄了一遍：反馈里用「」引了缺失的数字，校验器在输出里找到这个数字就判为已修复；把抄回来的部分切掉再校验，这 10 条里有 8 条的正文仍有违规。这是规则校验器的漏洞，事后才发现，不在预注册的三种「假修复」里。主臂的最终输出里没有这种回显，但 followup 格式下第 2 轮有 6 / 13 条是把反馈原样输出，修复全部来自第 3 轮。完整四臂、轮数分布和违规转移见 `eval/reports/agent-eval.md` 与 CHANGELOG T2.5。

**第二轮：堵上回显之后（SPEC §4.9，主臂 balanced k = 2 + system 反馈 + 回显防护）。** 生成之后先切掉抄回来的反馈再校验，并新增一种把反馈放进 system 消息的格式。主臂相对检索行：`numeral_recall` +0.007，区间 [0, +0.020]；`hallucination_rate` −0.0042，区间 [−0.013, 0]（唯一一条有幻觉的样本被修掉了）；`profile_gap_per_case` +0.001。三项的区间都含 0，按预注册规则是**未检出差异**。13 条第一轮有违规的样本里修好 2 条数值、1 条幻觉，没有新引入。六个臂的最终输出里都不再有回显，system 格式下模型一次也没有抄反馈。

两轮合起来的结论：**这个环在 59 条上没有可靠地改善保真。** 第一轮主臂的「改善」来自一条偶然的路径（第 2 轮回显被挡掉之后的第 3 轮）；防护打开后同一格式只剩 +0.017，区间含 0。剩下修不掉的违规大多是「一块」「一遍」「两个」这类「一 + 量词」，SPEC §1.5.3 已经说明数值规则在这里偏严，它们未必是真的信息丢失。样本里真正可修的事实错误太少，这个实验分辨不出小的效果。

**模型没有学会苏童的句长变化。** `eval/reports/lora-eval59.json` 的 `profile_mean_delta`：`sent_len_p90` 白话 -0.61、LoRA -0.61，微调后没有移动；`sent_len_std` 白话 -0.58、LoRA -0.50，移动很小。消融报告的折外 R²：`sent_len_std` 0.197、`sent_len_p90` 0.312，在 20 维里属于偏低的一组（`para_density` 0.883、`dialogue_verb_density` 0.800）。白话输入的这 20 维特征对目标原文的句长变化只有很弱的线性预测力。原先手填的 -0.64 → -0.64、-0.70 → -0.62 对不上这份报告，上面的分维数字取自报告本身。

指标定义见 [SPEC 3.1](docs/SPEC.md#31-指标清单)。填写 `style_win_rate` 时必须同时标注 `tie_rate` 和 `copy_ratio`。`tie_rate` 只在 `identical_rate` 低时才表示位置偏差；`copy_ratio` 高而 `tie_rate` 高是模型在抄，不是评委失效。定义见 [SPEC 3.3](docs/SPEC.md#33-风格胜率llm-judge)。

## 做了什么

这个项目起点是一个能用但有缺陷的微调模型。两个缺陷是实测出来的：

1. **事实漂移** —— 追风格时会改掉内容。实例：「太医」→「宫监」、「万人大军」→「十万铁骑」
2. **域外输入失效** —— 对人类随手写的白话几乎不做风格化。根因是训练数据的「白话」是 LLM 从原文降级生成的，仍带着原文的句子骨架，与真实用户输入存在分布差

针对这两点，在**不改模型权重**的前提下加了三层：

```
白话输入 → [检索风格范例] → [LoRA 生成] → [保真校验 + 风格打分] → [定向修订，最多 3 轮] → 输出 + trace
```

<!-- TODO(T4.2): mermaid 架构图 -->

## 技术要点

<!-- TODO(T4.2): 展开以下各节 -->

### 书面语词表是从自有平行语料学出来的

907 对 `(白话, 原文)` 本身就是一份平行语料。对每个词计算它在原文侧与白话侧的出现比的对数，就能自动分离出书面语词与口语词，不需要任何外部词表。这张表同时服务于文体特征、风格距离和检索。

### 检索的是句法形态，不是题材

语义相似不等于风格可参照：输入「河边散步」去检索，召回一堆河边场景没有用——需要的是**写法模板**（对话密集型？长句白描型？）。所以除主题索引外另建一路 20 维文体特征索引，并用 Ridge 从输入白话的特征预测目标原文该有的特征，拿预测向量去检索。

### 离线评估与在线 reward 用不同精度的指标

LLM judge 准但慢，一轮几秒、三轮超时，所以它只能离线用。agent 环内只允许确定性指标（规则化的保真校验 + 文体距离）。这个分层是整套设计的核心取舍。

## 在 GPU 机器上复现训练

实测环境：Windows 11 + WSL2 Ubuntu 24.04，RTX 5060 Laptop 8GB（Blackwell sm_120）。

```bash
# 1. 环境放 WSL 本地盘，不要放 /mnt/，venv 有上万个小文件，跨文件系统慢十倍
uv venv --python 3.11 ~/venvs/sutong

# 2. torch 必须走 cu128 源，sm_120 需要 CUDA 12.8+
VIRTUAL_ENV=~/venvs/sutong uv pip install torch --index-url https://download.pytorch.org/whl/cu128
VIRTUAL_ENV=~/venvs/sutong uv pip install unsloth trl datasets structlog pydantic

# 3. 坑：torchvision 会从 PyPI 装成 CPU 版，import unsloth 时报
#    "operator torchvision::nms does not exist"。必须从 cu128 源重装
VIRTUAL_ENV=~/venvs/sutong uv pip install --reinstall --no-deps \
  --index-url https://download.pytorch.org/whl/cu128 "torchvision==0.26.0"

# 4. 验证：必须打印 (12, 0)
~/venvs/sutong/bin/python -c "import torch; print(torch.cuda.get_device_capability(0))"
```

验证过的版本组合：`torch 2.11.0+cu128` / `torchvision 0.26.0+cu128` / `unsloth 2026.9.14` / `xformers 0.0.35` / `triton 3.6.0` / `trl 0.24.0`。

**TRL 0.24.0 的 API 与旧教程不同**：`SFTConfig` 用 `max_length` 不是 `max_seq_length`，`SFTTrainer` 用 `processing_class` 不是 `tokenizer`，且 `unsloth` 必须在 `trl` / `transformers` / `peft` 之前 import。

```bash
# 不碰 GPU，只核对超参和数据切分
uv run python -m scripts.train --dry-run

# 训练（12 分 39 秒 / RTX 5060 Laptop）。启动即断言可训练参数 == 29,933,568
python -m scripts.train --run-id sutong-v2

# 生成：不给 --adapter 就是无微调基座
python -m scripts.generate --run-id base-eval59
python -m scripts.generate --run-id lora-eval59 --adapter adapters/sutong-v2/adapter

# 评估：--skip-judge 不发任何网络请求
uv run python -m eval.run_eval --config eval/configs/eval59.yaml \
  --generations corpus/generations/lora-eval59.jsonl --run-id lora-eval59 --skip-judge
```

`scripts/train.py` 与 `scripts/generate.py` 共用同一个 `build_messages()`。**训练与推理的 prompt 必须同源**，差一个 token 则 adapter 失效。

### 建 dense 索引

用已经配好的 `~/venvs/sutong`，不要重装 torch / transformers。第一次运行会把 `BAAI/bge-m3` 下到 Hugging Face 缓存，权重不进仓库。输出是 `retrieval/data/embeddings.npy` 和 `embeddings.meta.json`，两个都在 `.gitignore` 里。

```bash
~/venvs/sutong/bin/python -m scripts.build_dense_index
# 没有 GPU 时的降级路径：
~/venvs/sutong/bin/python -m scripts.build_dense_index --device cpu
```

默认编码 `corpus/pairs.jsonl` 里 `split` 为 train 的原文（当前 743 条，过了内容闸门的配对），max_length 512，batch 8，权重 float32。先加 `--dry-run` 可以只核对条数和路径，不下载模型、不写文件。日志里的 `truncated` 大于 0 表示有 chunk 被截断。缓存和当前语料或模型对不上时，加载会直接报错，不会静默重算。

eval 白话的查询向量单独缓存，不和上面的原文索引写进同一个目录。模型、revision、max_length 必须和原文索引一致，否则加载直接拒绝。

```bash
~/venvs/sutong/bin/python -m scripts.build_query_cache --dry-run
~/venvs/sutong/bin/python -m scripts.build_query_cache
```

输出在 `retrieval/data/query_cache/`，已进 `.gitignore`。同样不要重装 torch / transformers。

### k 扫描

检索计划与 k 扫描已跑完。生成在 WSL2，评估在 Windows（`eval_sweep`，`--skip-judge`）。README 的 retrieval 行填的是主配置 balanced、k = 2。

```bash
~/venvs/sutong/bin/python -m scripts.sweep_topk --dry-run
~/venvs/sutong/bin/python -m scripts.sweep_topk --adapter adapters/sutong-v2/adapter
```

```bash
uv run python -m scripts.eval_sweep
```

扫描先写 k = 0，并和 `corpus/generations/lora-eval59.jsonl` 逐条比对。有一条不同就停，不跑其余 12 组。生成文件在 `corpus/generations/`，不提交。清单在 `eval/reports/retrieval-sweep-manifest.json`。

### agent 评估

四个臂（范例来源 balanced k = 2 / 不给范例，反馈格式 followup / restate），主臂是 `agent-balanced-k2-followup`，判定规则见 [SPEC 4.8](docs/SPEC.md#48-agent-评估的运行臂与判定规则预注册)。README 指标表的 agent 行填的是主臂。

agent 每一轮都要当场做保真检查和文体打分，所以这一步在 WSL2 里还需要 `langgraph`、`cn2an`、`python-dotenv`。先把现有环境的版本锁住再装，保证不改动任何已装的包（torch / transformers 不动）：

```bash
uv pip freeze --python ~/venvs/sutong/bin/python > /tmp/sutong-freeze.txt
uv pip install --python ~/venvs/sutong/bin/python -c /tmp/sutong-freeze.txt \
  "langgraph>=1.2.14" cn2an python-dotenv
```

```bash
~/venvs/sutong/bin/python -m scripts.run_agent_eval --dry-run
~/venvs/sutong/bin/python -m scripts.run_agent_eval --adapter adapters/sutong-v2/adapter
```

```bash
uv run python -m scripts.eval_agent
```

第二轮实验（[SPEC 4.9](docs/SPEC.md#49-agent-第二轮回显防护与-system-反馈格式预注册)：六个臂，打开回显防护，多一种 system 反馈格式）在两条命令后面各加 `--experiment v2`，产物的前缀是 `agent2-`：

```bash
~/venvs/sutong/bin/python -m scripts.run_agent_eval --experiment v2 --adapter adapters/sutong-v2/adapter
```

```bash
uv run python -m scripts.eval_agent --experiment v2
```

`--dry-run` 不加载模型，只核对计划和两个基线文件，并打印每个臂第一轮就有违规的样本数（只有这些样本需要真实生成）。第一轮不重新解码：它的 prompt 和 k 扫描里对应组的逐字相同，输出按 `prompt_sha256` 从 `retrieval-balanced-k2.jsonl` 和 `retrieval-k0.jsonl` 里取，找不到对应的 sha 就在加载模型之前报错退出。修订轮的 prompt 超出 4096 − 768 的预算同样直接报错，不截断。中断后重跑同一条命令会跳过已写完的样本。

生成文件是 `corpus/generations/agent-*.jsonl`，含正文，不提交。清单在 `eval/reports/agent-run-manifest.json`。评估回到 Windows 跑，写出四份 `eval/reports/agent-*.json` 和汇总 `agent-eval.json` / `agent-eval.md`。汇总把指标分成 agent 直接优化的和没有优化的两张表：校验器与评估用的是同一套规则，前一张表的改善有一部分是构造使然，必须连同后一张一起读。

## Quickstart

启动 T3.1 API 骨架（真实 vLLM 装配在 T3.7 接入；当前开发与测试使用注入的 Fake）：

```bash
uv run uvicorn api.main:app --host 0.0.0.0 --port 8000
```

SSE 请求会依次返回 `trace`、带轮次的 `token`，最后返回 `done`：

```bash
curl -N -X POST http://localhost:8000/v1/transform \
  -H "Content-Type: application/json" \
  -d '{"text":"他在河边等了三天。","style":"sutong","stream":true,"max_iter":3,"seed":42}'
```

<!-- TODO(T3.4) -->

```bash
uv sync
docker compose up -d
```

语料因版权不包含在本仓库，见下方「数据与版权」。

T0.0 模型单变量对照（需自备本地 round2 报告及完整档案，先配置 `.env`）：

```bash
uv run python -m scripts.model_control \
  --baseline corpus/rebuild_report_round2_20261003.json \
  --output corpus/preview_pairs_model_control_20261003.jsonl \
  --report corpus/rebuild_report_model_control_20261003.json
```

仅重放原 20 条一次，拒绝改动 prompt / seed / 范例或复用输出路径。结束后等待人工 review，
不自动运行消融或全量。`deepseek-chat` 旧别名不能用于 Pro；现有 `.env` 中旧设置不会被静默修改，
对照入口明确请求 `deepseek-v4-pro`，常规入口需将 `VERNACULARIZE_MODEL` 配为同一标识。

## 性能

当前T0.0一次两段式预览（会调用授权DeepSeek API，需已有本地历史报告，拒绝覆盖）：

```bash
uv run python -m scripts.two_pass_preview
```

调用前要求prompt、词表、代码、SPEC已提交；固定20条和采样参数，最多40次调用，
完整两遍输出和报告仅留本地corpus，不会产生全量pairs。运行后停下来人工review。

<!-- TODO(T3.7): locust 压测后填入 -->

| | RPS | p50 | p95 | cache hit rate |
|---|---|---|---|---|
| | — | — | — | — |

## 取舍与坑

<!-- TODO(T4.2): 这是区分度最高的一节，务必写满 -->

- 为什么风格距离用对角标准化而不是马氏距离
- 为什么只用 `langgraph` 不用 `langchain`
- 为什么 LLM judge 不能进 agent 循环
- 检索评估的陷阱：为什么不能用训练对自测召回
- 风格向量预测的消融结果

## 数据与版权

苏童作品受版权保护，**语料不包含在本仓库中**。仓库只提供：

- 预处理 pipeline（`scripts/prepare_corpus.py`）
- 不可还原原文的衍生物：词表、专有名词表、特征分布的均值方差、评估指标与输出 hash
- 5 条自编示例（`corpus/sample_public.jsonl`）

复现需自备文本。LoRA adapter 权重发布在 Hugging Face Hub，model card 已标注训练数据来源与用途限制。
发布前可运行 `uv run python -m scripts.check_corpus_leak --history` 检查当前跟踪文件及 Git 历史中的原文泄漏。

## 已知局限

- 内容闸门筛过的语料不是随机子集。可用 eval 是 59/73，缺口 14 条，数字密集段落被系统性排除；训练集 743/842，总入选率 87.7%（802/915）。未入选的 113 条里，44 条只缺基数「1」、29 条序数被写成基数，很可能是闸门过严而不是事实丢失，但不再修。此后基线都建立在这个有偏子集上
- 白话化参照仅 13 条，占历史 907 条的 1.4%，来自 2025-08 人工 review 日志，无法确认是随机抽样还是择优展示。这是当前最大的方法论软肋。剔除原文侧疑似截断的 `妇女生活_0015`（长度比约 1.95）并做严格留一法，仍不能消除日志选择偏差
- PINC、sBLEU、SequenceMatcher 都是表层重合代理，不能直接等同质量或信息保真。当前两段式以PINC-4≥0.75、PINC-6≥0.85作为改写闸门；SequenceMatcher均值[0.51,0.61]仅作双向辅助警报。中文字符PINC不能直接与词级文献数值对比。分层预览每部作品3–4条，均值不是按总体作品占比加权的均值
- 本项目方法层面无新颖性，不主张伪平行语料、两段式或检索/自检方法创新；价值在于评估严谨度与完整工程实现，详见SPEC「相关工作与方法依据」。未来生成阶段参考原文n-gram重合只作弱辅助信号，文学表达不唯一
- 两段式事实修补由同一个模型完成，问题清单不是独立保真证据；助手抽查不得冒称人工review。六条固定分层抽查不代表总体，人工review仍待完成
- 托管 API 若只公开模型别名，日期、采样参数和响应标识仍不足以锁定不可变底座快照；报告明确保留这一缺失。jieba 专名与对白密度也是启发式估计，需要人工核对
- 2025-08 白话生成模型未知；2026-10-03 的留一法、round1、round2 均**在 deepseek-flash 上得到**（请求 deepseek-chat 被别名重映射），不能当作 deepseek-chat 的结果。跨模型参照的不确定性比此前更大，不能单凭表层分数推断 prompt 或模型的因果作用
- **`style_distance` 的方向标此前写反了。** 它是 z 分数均方根，一条服从参照分布的文本期望值就是 1.0；59 条原文实测 1.035，与理论吻合。低于 1.0 不代表更好，代表向均值回归。白话输入 0.992 几乎正中靶心，集合级 `profile_gap` 却与原文差 0.450（逐条是 0.623），说明标量半径与方向被混成了一个数。此后 `style_distance` 只作弱信号，判别以剖面差为准；`human_eval` 无 ground truth，只能退回标量
- **2026-10-05 的 `style_win_rate` 0.331 不可单独引用。** 59 条、118 次调用、零解析失败，模型身份闸门确认响应为 `deepseek-v4-pro`，链路本身是通的。但 `tie_rate = 0.49`——近一半样本在 A/B 位置互换后评委改口，这一半是掷骰子。判得一致的 30 条里 25:5 偏向白话，与确定性指标（profile_gap 0.450 → 0.204）方向相反。输出本身不是退化：59 条中仅 3 条有轻微重复或缺末尾标点。最可能的原因是 **Q18 选定的对手有混淆**——白话是用 LLM 从原文降级改写而来，内容、意象与叙事顺序几乎原样保留，评委虽被明确要求只判文风，仍大概率在响应「内容像不像」。对手选择需要重做，不是模型问题
- 语义级漂移抓不到。规则化校验能抓实体、数字、称谓的替换，但抓不到「垂死的酸气」被写成「死尸散发的酸臭之气」这类语义扭曲（人还活着）
- `human_eval.jsonl` 的 30 条人工白话没有 ground truth，只能评风格距离与幻觉率，不能算 recall
- 域外泛化只缓解未解决。根因在训练数据分布，彻底解决需要补一批人类撰写的白话输入重训
- 自检重写环的校验器会被「回显反馈」骗过：输出只要包含缺失的片段就算修复，不管它出现在正文里还是出现在被抄回来的修订指令里。restate 格式下大部分「修复」是这样来的（见上）；第二轮加了回显防护（切掉抄回来的反馈再校验），但防护的规则是照着第一轮观察到的抄法定的，换一批输入可能出现别的抄法
- 自检重写环在 59 条 eval 上没有可靠的保真改善：保真指标与校验规则同源，只有 13 条样本可能受影响，第二轮主臂未检出差异；人工抽查尚未完成
- Phase 1 在 59 条 eval 上未能确认风格范例检索的效果：主配置相对 k = 0 的判定指标区间均含 0（未检出差异）；`human_eval.jsonl` 正文尚未写入，PLAN 要求的检索召回评估仍欠

## License

<!-- TODO(T0.1) -->
