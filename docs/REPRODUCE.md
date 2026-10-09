# 复现指南

本文给出从环境搭建到各阶段评估的完整命令和已知的环境问题。实测环境为 Windows 11 + WSL2 Ubuntu 24.04，RTX 5060 Laptop 8GB。语料因版权不随仓库分发，需自备文本，见 README「数据与版权」。

## 训练、生成与评估

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

## 语料重建的辅助入口

以下两个入口会调用外部 LLM API，需先配置 `.env`，并要求本地已有对应的历史报告。

模型单变量对照（仅重放原 20 条一次，拒绝改动 prompt、seed、范例或复用输出路径）：

```bash
uv run python -m scripts.model_control \n  --baseline corpus/rebuild_report_round2_20261003.json \n  --output corpus/preview_pairs_model_control_20261003.jsonl \n  --report corpus/rebuild_report_model_control_20261003.json
```

两段式预览（固定 20 条和采样参数，最多 40 次调用，输出只留在本地 `corpus/`）：

```bash
uv run python -m scripts.two_pass_preview
```
