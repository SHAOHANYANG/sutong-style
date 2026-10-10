# 在租用的 GPU 上跑基座规模对比

本文是 SPEC §4.10 的操作步骤：在一台租用的机器上训练 Qwen2.5-14B 的 LoRA 并生成四份输出，再把结果拿回本地评估。以 AutoDL 为例，其他平台的差别只在上传下载的方式。下面的命令是 2026-10-10 在一台 RTX 5090 上实际跑通的版本。

整个过程约 1–2 小时机时，其中训练约 8 分钟，四次生成约 19 分钟，其余是装环境和下载。评估不需要 GPU，在本地做。

## 1. 租机器

- 显存 24 GB 或以上。14B 以 4-bit 加载并训练，显存占用约 13 GB。
- 镜像选 PyTorch 2.7 或更高、CUDA 12.8、Python 3.11 或 3.12。RTX 50 系显卡在更旧的 PyTorch 上无法运行。
- 数据盘至少 30 GB，用来放模型缓存。

## 2. 准备环境

登录后在终端里执行。模型缓存放数据盘，系统盘放不下。

```bash
export HF_HOME=/root/autodl-tmp/hf
# 直连 Hugging Face 不通时使用镜像
export HF_ENDPOINT=https://hf-mirror.com
# 经镜像下载时，默认的 Xet 传输会卡在几百 MB 不动，关掉它走普通 HTTP
export HF_HUB_DISABLE_XET=1

cd /root/autodl-tmp
git clone https://github.com/SHAOHANYANG/sutong-style.git
cd sutong-style
```

从国内的机器访问 GitHub 可能失败。此时在本地执行 `git bundle create sutong-main.bundle main`，把这个文件传到 `/root/autodl-tmp/`，再执行 `git clone -b main sutong-main.bundle sutong-style`。

安装依赖。训练和生成只需要下面这几个包，不需要装整个项目。版本与本地环境一致。

```bash
pip install uv
uv venv /root/venv
export VIRTUAL_ENV=/root/venv
MIRROR=https://pypi.tuna.tsinghua.edu.cn/simple
WHEELS=https://mirrors.aliyun.com/pytorch-wheels/cu128/

uv pip install "torch==2.11.0+cu128" "torchvision==0.26.0+cu128" \
  --index-url $MIRROR --find-links $WHEELS

uv pip install --index-url $MIRROR --find-links $WHEELS \
  "torch==2.11.0+cu128" "torchvision==0.26.0+cu128" \
  "unsloth==2026.9.14" "unsloth-zoo==2026.9.9" "trl==0.24.0" "transformers==5.5.0" \
  "peft==0.21.2" "datasets==4.3.0" "bitsandbytes==0.50.2" "accelerate==1.15.0" \
  "xformers==0.0.35" structlog pydantic

# 验证：应打印 True 和显卡名称
/root/venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

踩过的坑：

- 平台自带的加速代理（`/etc/network_turbo`）在下载 PyTorch 的 CUDA 依赖时反复断开，改用上面的国内轮子镜像后正常。
- torch 与 torchvision 必须带 `+cu128` 后缀从轮子镜像安装。从普通 PyPI 源装到的 torchvision 是 CPU 版，import unsloth 时报 `operator torchvision::nms does not exist`。
- 个别 PyPI 镜像缺少新发布包的元数据文件，`uv` 会报 404，换一个镜像即可。

版本组合与其他已知问题见 [REPRODUCE.md](REPRODUCE.md)。

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

先单独把模型下载好（约 11 GB）。下载慢或中断时重复执行这一条即可：

```bash
/root/venv/bin/hf download unsloth/Qwen2.5-14B-Instruct-unsloth-bnb-4bit
```

核对要执行的步骤，不加载模型：

```bash
/root/venv/bin/python -m scripts.run_model_size --dry-run
```

正式运行。放到后台，这样关掉网页或断线也不会中断：

```bash
nohup /root/venv/bin/python -m scripts.run_model_size > run.log 2>&1 < /dev/null &
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
- 删除机器上的 `corpus/pairs.jsonl` 和 `corpus/generations/` 下的生成文件。
- 关机或释放实例，避免继续计费。
