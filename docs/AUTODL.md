# 在租用的 GPU 上跑基座规模对比

本文是 SPEC §4.10 的操作步骤：在一台租用的机器上训练 Qwen2.5-14B 的 LoRA 并生成四份输出，再把结果拿回本地评估。以 AutoDL 为例，其他平台的差别只在上传下载的方式。

整个过程约 1–2 小时机时。评估不需要 GPU，在本地做。

## 1. 租机器

- 显存 24 GB 或以上。14B 以 4-bit 加载并训练，峰值在 15 GB 上下。
- 镜像选 PyTorch 2.7 或更高、CUDA 12.8、Python 3.11 或 3.12。RTX 50 系显卡在更旧的 PyTorch 上无法运行。
- 数据盘至少 30 GB，用来放模型缓存。

## 2. 准备环境

登录后在终端里执行。模型缓存放数据盘，系统盘放不下。

```bash
# AutoDL 的学术加速；没有这个文件可以跳过
source /etc/network_turbo

export HF_HOME=/root/autodl-tmp/hf
# 直连 Hugging Face 不通时使用镜像
export HF_ENDPOINT=https://hf-mirror.com

cd /root/autodl-tmp
git clone https://github.com/SHAOHANYANG/sutong-style.git
cd sutong-style
```

安装依赖。训练和生成只需要下面这几个包，不需要装整个项目：

```bash
pip install uv
uv venv --python 3.11 /root/venv
VIRTUAL_ENV=/root/venv uv pip install torch --index-url https://download.pytorch.org/whl/cu128
VIRTUAL_ENV=/root/venv uv pip install unsloth trl datasets structlog pydantic

# torchvision 会被装成 CPU 版，import unsloth 时报错，必须从同一个源重装
VIRTUAL_ENV=/root/venv uv pip install --reinstall --no-deps \
  --index-url https://download.pytorch.org/whl/cu128 torchvision

# 验证：应打印 True 和显卡名称
/root/venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

版本组合与已知问题见 [REPRODUCE.md](REPRODUCE.md)。

## 3. 上传语料

只需要一个文件：本地的 `corpus/pairs.jsonl`，传到机器上的同一位置。

```
/root/autodl-tmp/sutong-style/corpus/pairs.jsonl
```

可以用 JupyterLab 的上传按钮，或者在本地执行（把端口和地址换成实例页面上显示的）：

```bash
scp -P <端口> corpus/pairs.jsonl root@<地址>:/root/autodl-tmp/sutong-style/corpus/pairs.jsonl
```

这个文件含原著文本。只传到自己的私有实例上，跑完后删除，不要把实例保存成公开镜像。

## 4. 运行

先核对要执行的步骤，不加载模型：

```bash
/root/venv/bin/python -m scripts.run_model_size --dry-run
```

正式运行。放到后台，这样关掉网页或断线也不会中断：

```bash
nohup /root/venv/bin/python -m scripts.run_model_size > run.log 2>&1 &
tail -f run.log
```

脚本依次执行五步：训练、基座在 eval 上生成、微调后在 eval 上生成、两者在探针集上生成。每一步是独立的进程，已有产出的步骤会被跳过，所以中途失败后重新执行同一条命令即可续跑。

训练启动时会断言可训练参数为 68,812,800；不相符说明 LoRA 配置或基座与预注册的不一致，脚本会停下。

日志最后一行是 `model_size_complete` 时表示全部完成，仓库根目录下会出现 `model-size-14b-results.tar.gz`。

想同时对比 7B 时，再执行一次并指定基座：

```bash
/root/venv/bin/python -m scripts.run_model_size --base-model Qwen/Qwen2.5-7B-Instruct
```

## 5. 取回结果并评估

把 `model-size-14b-results.tar.gz` 下载到本地仓库的根目录（JupyterLab 里右键下载，或用 `scp`），然后在本地执行：

```bash
tar -xzf model-size-14b-results.tar.gz
uv run python -m scripts.eval_model_size
```

压缩包里是四份生成文件和一份运行清单，不含模型权重。评估写出 `eval/reports/model-size-14b.json` 和 `.md`，以及 14B 两份逐条报告。

3B 在探针集上的两份输出需要在本地生成一次（已有则跳过）：

```bash
python -m scripts.generate --pairs eval/probes/modern_inputs.jsonl --split probe \
  --run-id lora-3b-probe --adapter adapters/sutong-v2/adapter
python -m scripts.generate --pairs eval/probes/modern_inputs.jsonl --split probe \
  --run-id base-3b-probe
```

## 6. 收尾

- 需要保留 14B 的权重时，下载 `adapters/sutong-v2-14b/adapter/` 整个目录（约 280 MB）。
- 删除机器上的 `corpus/pairs.jsonl`。
- 关机或释放实例，避免继续计费。
