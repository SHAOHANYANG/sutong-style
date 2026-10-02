# AGENTS.md — 本仓库的 agent 操作手册

你是本仓库的实现者。开工前**完整读完本文件**，以及 `docs/SPEC.md`（技术规格）和 `docs/PLAN.md`（任务清单）。

## 项目一句话

把现代白话中文改写成苏童文风。已有一个 Qwen2.5-3B + LoRA 的微调模型，它能用但有两个已测出的缺陷：**事实漂移**（人名/数字/称谓被改掉）和**域外输入失效**（对人类随手写的白话几乎不做风格化）。本项目在模型**外面**加三层来解决：评估体系、风格范例检索、生成自检重写环，最后服务化。

**模型权重基本不动。** 唯一允许的训练改动见 `docs/PLAN.md` 的 T0.5。

## 最重要的三条

1. **按 `docs/PLAN.md` 的任务顺序做，一个任务一个 commit。** 不要跳着做，不要把多个任务塞进一个 commit。
2. **Phase 0 必须最先完成。** 它建立评估基线，是后续一切改动的唯一裁判。在 Phase 0 完成前不要写任何检索或 agent 代码。
3. **遇到歧义不要自己拍板也不要停工。** 把问题追加到 `docs/QUESTIONS.md`（没有就创建），按你认为最合理的方式实现并在该文件里记录你的假设，然后继续做不依赖这个答案的部分。

## 环境与命令

Python 3.11，依赖用 `uv` 管理。

```bash
uv sync                      # 安装依赖
uv run ruff check . --fix    # lint
uv run ruff format .         # 格式化
uv run mypy .                # 类型检查
uv run pytest                # 测试
uv run pytest -m "not gpu"   # 只跑不需要 GPU 的测试（CI 用这个）
```

**提交前这四条必须全绿**：`ruff check`、`ruff format --check`、`mypy`、`pytest -m "not gpu"`。

## GPU 约束（重要）

**执行环境**：仓库所有者有一张 **RTX 5060 Laptop（8GB，Blackwell sm_120）**，`[GPU]` 任务由人工在 **WSL2 Ubuntu** 下执行，不是 Windows 原生。你写的运行脚本按 Linux 写，路径不要假设 Windows 盘符；需要装依赖时注意 Blackwell 要 torch ≥ 2.7 + CUDA 12.8。8GB 显存下 vLLM 的参数见 SPEC 第 5 节，不要改大。

**你很可能没有 GPU，也没有远端训练机的访问权限。** 任务在 `docs/PLAN.md` 里标了三种：

- `[CPU]` — 你能完整做完，必须做完
- `[GPU]` — 需要 GPU 才能执行。你的交付物是**代码 + 一个可独立运行的脚本 + README 里的运行说明**，不是运行结果。写完后在 commit message 里注明"待人工在 GPU 机器执行"
- `[GPU-OPT]` — 需要 GPU 但有降级路径，用降级路径交付

为了让 `[CPU]` 的范围尽可能大，**所有模型调用必须走依赖注入**：

```python
# 禁止在业务代码里直接 import torch / transformers / vllm 并调用
# 必须通过 Protocol 注入
class Generator(Protocol):
    def generate(self, prompt: str, **kw: object) -> str: ...

class Embedder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray: ...
```

每个 Protocol 都要提供一个 `Fake*` 实现放在 `tests/fakes.py`，返回确定性的假数据。**所有单测和集成测试都用 Fake 跑，不碰 GPU。** 真实实现放在 `infra/` 下，只在入口处装配。

需要真实 GPU 的测试打 `@pytest.mark.gpu`，CI 不跑。

## 硬性禁止

- ❌ **不要把 `corpus/` 下的语料文件提交进 git。** 苏童作品有版权。`.gitignore` 必须挡掉 `corpus/*.jsonl`。只能提交 `corpus/sample_public.jsonl`（见 SPEC 第 8 节，内容是自己编的，不含任何原著文本）
- ❌ **不要引入 `langchain`。** 只允许 `langgraph`。理由见 `docs/SPEC.md` 第 9 节
- ❌ **不要给 20 维的风格特征向量上向量数据库。** 907 条 × 20 维，numpy 暴力算是微秒级
- ❌ **不要在 agent 循环内部调用 LLM judge。** 环内只允许确定性指标。理由见 SPEC 第 4.3 节
- ❌ **不要重新实现已有的指标。** `stylometry/` 和 `eval/metrics.py` 是单一真相来源，被评估、agent、CI 三处复用
- ❌ **不要自行扩大范围。** 不要加你觉得"顺手也该有"的功能（用户系统、多租户、模型训练 UI、A/B 实验平台……）。PLAN 里没写的不要做
- ❌ **不要把 adapter 权重或任何 >10MB 的二进制提交进 git。** 走 Hugging Face Hub

## 代码约定

- 类型注解全覆盖，`mypy` 配 `strict = true`（第三方库缺 stub 的在 `pyproject.toml` 里单独 ignore，不要用 `# type: ignore` 糊）
- 数据结构用 pydantic v2 `BaseModel`，不用裸 dict 传递跨模块数据
- 纯函数优先。`stylometry/` 和 `eval/metrics.py` 里**不允许有 I/O**，文件读写放在调用方
- 日志用 `structlog`，输出 JSON。不要 `print`
- 所有涉及中文的文件读写显式 `encoding="utf-8"`
- 随机性必须可复现：任何采样、shuffle、模型生成都接受 `seed` 参数

## Commit 规范

Conventional commits，一个任务一条：

```
feat(stylometry): 实现 20 维文体特征抽取 (T0.2)
test(eval): 补充指标模块单测 (T0.3)
docs(spec): 修正 pairs.jsonl schema 以匹配实际数据 (T0.1)
```

末尾不加任何 AI 署名行。

## 每个 Phase 结束时必须做的事

1. 跑一遍完整评估：`uv run python -m eval.run_eval --config eval/configs/<当前阶段>.yaml`
2. 把 `eval/reports/<run_id>.json` 提交进 git（这个文件小，必须进版本控制）
3. 更新 `README.md` 里的**指标对比表**，加一列
4. 在 `docs/CHANGELOG.md` 记录本阶段做了什么、指标变化了多少、遇到了什么坑

**如果某个阶段的指标没有改善，照实写进 CHANGELOG，不要调参数去凑好看的数字，也不要悄悄换掉指标定义。** 负结果也是结果，README 里写清楚更可信。
