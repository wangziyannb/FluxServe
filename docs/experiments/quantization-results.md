# LLaDA2.0-flash：H100 量化实验结果

2026-09-10 至 2026-09-11，在 4 × H100 80GB HBM3 上完成三轮实验：
HumanEval 固定负载吞吐 50 次、原有 attention 路径的 GSM8K 全量评测 14 次，
以及 FA3 原生 FP8 attention / 显式 FA2 对照的 GSM8K 全量评测 10 次正式测量。
全部运行通过验收；原始结果、校准 scale、命令、逐 rank 指标和 GPU 采样记录均已归档。

## 版本与计时口径

- 前两轮运行代码：`c86f105b5daf6201099c9f9b2918ed6e1ff17a98` 加 `49956ba` 中的同步计时、指标记录与双卡 Graph 支持。
  新一轮使用 `49956bada57ca74a7308acfbf02135eb97a2a80d` 加本次提交的原生 FP8 attention 和显式 FA2/FA3 对照。
  各轮 `manifest.json` 和 `source.patch` 保留测量源码的 SHA256 与补丁；新一轮另有完整运行源码 `source-snapshot.tar.gz`。
- PyTorch `2.8.0+cu128`，FlashInfer `0.6.18`，FlashInfer fork revision
  `5ce4d077c33bbe167386cf5487a4a5bb4bcafbfd`。
- BF16 模型：`inclusionAI/LLaDA2.0-flash`，revision
  `744c3f8c6c8317d2377d6d16d8a3d4be2caef563`。
- FP8 模型：`thnkinbtfly/llada2.0-flash-fp8`，revision
  `85bd9f38034aa46f93135676189acec4d7fc40d3`。
- 全局 batch 8、mini-batch 4，block/page 64，threshold 0.95、low-threshold 0.3；
  FlashInfer paged prefill/cache/KV layout，TP=EP=GPU 数，DP=PP=1。
- Graph 使用默认 decomposed decode。实际计数确认生成期间发生 decode replay，
  本负载的请求 prefill 走 eager；计时期间没有新增 capture 或 invalidation。
- 每批生成前后 CUDA 同步，使用单调时钟，取各 rank 最大生成耗时；耗时归约在计时区间外。
  TPS 为有效 completion tokens / 总生成耗时，包含请求 prefill 与 diffusion decode，
  排除加载、warmup、JIT、Graph capture、结果写盘及准确率计分。
- completion token 数不含 prompt，包含首次 EOS；未生成 EOS 时计非 mask completion tokens。
  保留原有长度分桶/对齐行为，实际长度见逐题结果与 token 记录。
- 完整模型单卡、BF16 模型双卡按权重容量标记不可运行，未作为实测 OOM 或零吞吐。

## GSM8K 两轮合并：权重、KV 与 attention 精度

新一轮于 2026-09-11 17:29 UTC 完成，10 个正式配置及各自 16 题试跑全部通过验收，
正式评测共 13,190 份回答、3,725,522 个 completion tokens。负载沿用前轮完整 test split
1,319 题、4-shot、gen-len 2048 和上面的 batch/解码参数，每配置一遍。
本节仅纳入 UltraChat 校准及无需 KV 校准的组合；前轮 GSM8K train 校准结果单独保留在下一节。

两轮模型 revision、评测输入以及 FP8 模型的 UltraChat scale 相同。
新一轮 BF16 权重配 FP8 KV 使用同一批 512 条 UltraChat 输入另做校准，scale 匹配 BF16 模型，
没有使用 GSM8K train/test 做 KV 校准。原有四条 GSM8K train few-shot 示例仍用于固定评测 prompt。
“全 FP8”表示模型权重量化、FP8 KV 存储及 attention QK/PV 的 FP8 Tensor Core 输入，
不表示归一化、softmax、累加和所有中间输出均为 FP8；attention 输出仍为 BF16。

下表精度顺序为 **权重 / KV / Attention**，每格为 **tokens/s / accuracy**。

| 轮次 | 精度 | GPU 数 | 后端 | eager | Graph |
|---|---|---:|---|---:|---:|
| 旧 | BF16 / BF16 / BF16 | 4 | FA3 / 混合¹ | 139.65 / 96.36% | 223.32 / 96.13% |
| 新 | BF16 / BF16 / BF16 | 4 | FA2 | 116.41 / 95.98% | 230.55 / 95.83% |
| 新 | BF16 / BF16 / BF16 | 4 | FA3 | 121.97 / 96.36% | 246.50 / 96.29% |
| 新 | BF16 / FP8 / FP8 | 4 | FA3 | 107.43 / 95.91% | 249.94 / 96.29% |
| 旧 | FP8 / BF16 / BF16 | 2 | FA3 / 混合¹ | 101.62 / 95.75% | 271.03 / 95.98% |
| 旧 | FP8 / FP8 / BF16 | 2 | FA2 | 97.01 / 96.06% | 145.52 / 96.06% |
| 新 | FP8 / FP8 / FP8 | 2 | FA3 | 94.25 / 96.13% | 297.05 / 96.29% |
| 旧 | FP8 / BF16 / BF16 | 4 | FA3 / 混合¹ | 113.92 / 95.83% | 360.23 / 96.21% |
| 旧 | FP8 / FP8 / BF16 | 4 | FA2 | 93.40 / 96.51% | 169.41 / 95.83% |
| 新 | FP8 / FP8 / FP8 | 4 | FA3 | 85.41 / 95.98% | 358.71 / 96.21% |

¹ 旧版 BF16 KV 的 eager 已使用 Hopper FA3，Graph 为 FA3 prefill + FA2 decode，
因此不能把整个旧矩阵称为 FA2。新版 BF16 FA2/FA3 使用相同分块、页表和 KV 写入适配器。
新一轮 Graph 配置在本负载中实际为 eager prefill + Graph decode；各 rank 都有 decode replay，
计时期间无新增 capture/invalidation。生成结果允许不同，逐题输出与长度差异保留在比较 JSON 中。

- **相对旧版 FP8 KV 路径**：新版全 FP8 Graph 双卡/四卡为 2.04× / 2.12×。
  这同时改变 attention 精度、后端和实现，不能将两倍收益全部归因于单个因素。
- **相对旧版更快的 FP8 权重 / BF16 KV 基线**：双卡 Graph 从 271.03 到 297.05 TPS（+9.6%），
  四卡从 360.23 到 358.71（−0.4%），本次测量基本持平。
- **两卡对四卡**：新版两卡全 FP8 Graph 比新版四卡全 BF16 FA3 快 20.5%，两者均答对 1270/1319。
  旧版两卡 FP8/BF16 也已比旧版四卡全 BF16 快 21.4%，因此这一现象并非本次 FP8 attention 独有。
- **同版本 FA3 / FA2**：全 BF16、四卡 eager +4.8%，Graph +6.9%。
  eager 的 NFE 少 9.0%、整模型前向/秒低 2.4%；Graph 的 NFE 少 3.2%、前向/秒高 4.5%。
  这些是端到端指标，不是独立 attention 算子计时。
- **同为 FA3 的精度比较**：四卡全 FP8 / 全 BF16，Graph +45.5%、eager −30.0%。
  固定 BF16 权重，仅将 KV 与 attention 一起改成 FP8，Graph +1.4%、eager −11.9%。
- **eager 回退**：新版全 FP8 比旧版 FP8/BF16 双卡/四卡分别慢 7.2% / 25.0%。
  新版全 BF16 FA3 eager 也从旧版 139.65 降至 121.97 TPS（−12.7%），两者 tokens、NFE、答对题数相同。
  已观察到端到端回退，具体耗时来源尚未 profiling。

四卡 Graph 对比旧版 FP8/BF16，新版全 FP8 的 tokens 从 368,176 增至 384,971（+4.6%），
NFE 从 63,936 增至 74,848（+17.1%）。整模型前向/秒虽高 11.5%，每次前向的有效 token 产出降低，
最终 TPS 基本持平。两轮代码与路径不同，跨轮结果不属于只改一个变量的消融。
本节 accuracy 为 95.75%–96.51%；每配置一遍，无重复测量方差或 MAD，也未扫描 batch。
这些数据不能证明质量等价，亦不代表各配置的峰值吞吐。

完整记录：[合并报告](../../runs/fa3-gsm8k/COMBINED-RESULTS.md)、
[合并 CSV](../../runs/fa3-gsm8k/combined-matrix.csv)、
[合并 JSON 与精确比值](../../runs/fa3-gsm8k/combined-matrix.json)、
[本轮详细结果](../../runs/fa3-gsm8k/RESULTS.md)、
[本轮验收](../../runs/fa3-gsm8k/completion-audit.json)。
另外保留 [原生 FP8 实现验证及微基准](../../runs/fp8-attention/README.md)，
其中 16 题试验不与本节全量结果混合；SASS 记录确认 FP8 QK/PV Tensor Core 指令。

## 前轮 GSM8K：两种 KV 校准数据

每配置运行完整 test split 1319 题一次，共 18,466 份回答、5,207,076 个 completion tokens。
使用固定 EvalScope revision `acd09b44384d53174768bb1063f675420f76fae9` 的
4-shot 模板（train 前 4 条示例）、答案提取与 numeric accuracy；模型原生 chat template，
生成长度参数 2048，测试题原始顺序。BF16/FP8 tokenizer 的实际输入 token 已逐条核对一致。

两份独立 scale 均以 FP8 模型、BF16 KV、四卡 SDPA eager 校准，batch 8、mini-batch 4、
生成长度参数 2048。每份覆盖全部 32 层，K/V scale 均有限且为正；两卡/四卡共用同一份 scale。

- **UltraChat**：`HuggingFaceH4/ultrachat_200k` 的 `train_sft` 前 512 条，随后 seed 42 shuffle；
  完整对话采用原生 chat template，不添加 generation prompt，右截断至 2048 输入 tokens。
  其中 32 条被截断。数据 revision `8049631c405ae6576f93f445c6b8166f76f5505a`。
- **GSM8K train**：`openai/gsm8k` 的 train 前 512 条，随后 seed 42 shuffle；
  仅提供问题与 zero-shot CoT 指令，不提供 gold answer，模型生成后续内容。
  数据 revision `740312add88f781978c0658806c59bc2815b9866`。
- 两份校准输入与 test 问题均无检测到的逐字问题重合。此检查不排除语义改写或预训练数据重合。
  本轮没有使用上一轮基于 GSM8K test 的 scale。

下表每格为 **accuracy / tokens/s**。每配置只有一次正式测量，因此不报告重复测量的 MAD。

| 模型 / KV | 校准数据 | GPU 数 | eager | Graph |
|---|---|---:|---:|---:|
| BF16 / BF16 | — | 4 | 96.36% / 139.65 | 96.13% / 223.32 |
| FP8 / BF16 | — | 2 | 95.75% / 101.62 | 95.98% / 271.03 |
| FP8 / BF16 | — | 4 | 95.83% / 113.92 | 96.21% / 360.23 |
| FP8 / FP8 | UltraChat | 2 | 96.06% / 97.01 | 96.06% / 145.52 |
| FP8 / FP8 | UltraChat | 4 | 96.51% / 93.40 | 95.83% / 169.41 |
| FP8 / FP8 | GSM8K train | 2 | 96.29% / 94.60 | 96.44% / 151.30 |
| FP8 / FP8 | GSM8K train | 4 | 95.98% / 94.42 | 95.83% / 173.57 |

四卡 Graph 下，FP8/BF16 相对 BF16/BF16 的 TPS 为 1.613×；token 数为 0.981×，
NFE 为 0.930×。双卡 FP8/BF16 Graph 的 TPS 为四卡 BF16/BF16 Graph 的 1.214×，
但这同时改变了模型精度和 TP/EP 规模。eager 中，同四卡 FP8/BF16 的 TPS 为 BF16/BF16 的 0.816×。

前轮 FP8 KV 实现在所有匹配配置中均降低吞吐，Graph 下相对 BF16 KV 慢约 44%–53%。
它保持 BF16 Q/attention 计算，并以 FP8 存储 KV；这一轮尚未测量原生 FP8 attention 计算。
这些端到端数字不能确定反量化、量化写入、mask 或其他算子的耗时占比。

两种校准数据没有一致的准确率优势：GSM8K train 相对 UltraChat 在双卡 eager/Graph
分别多答对 3/5 题，在四卡 eager 少答对 7 题，四卡 Graph 正确总数相同但逐题结果不同。
所有长度耗尽的输出均保留计分，每配置有 9–15 个未生成 EOS 的输出。
每组单次生成、不同 token/NFE 和有限测试集均限制结论，不能据此宣称量化质量等价。

完整数据：[summary.csv](../../runs/gsm8k-kv-calibration/summary.csv)、
[comparisons.csv](../../runs/gsm8k-kv-calibration/comparisons.csv)、
[逐题对照索引](../../runs/gsm8k-kv-calibration/comparisons.json)、
[验收](../../runs/gsm8k-kv-calibration/verification.json)、
[输入/输出完整性](../../runs/gsm8k-kv-calibration/integrity.json)。
CSV 包含 token 数、NFE、每次前向生成 tokens、生成耗时、单卡平均 TPS 和显存峰值。

## HumanEval 固定负载吞吐

HumanEval 原始顺序前 8 题，生成长度参数 512。10 个配置各一次试跑，随后五轮交替顺序的
串行正式测量，共 50 次。下表为 TPS **中位数 / 未缩放 MAD**；本轮没有评估 HumanEval accuracy。

| 模型 / KV | GPU 数 | eager | Graph |
|---|---:|---:|---:|
| BF16 / BF16 | 4 | 181.83 / 1.84 | 297.07 / 0.60 |
| FP8 / BF16 | 2 | 168.33 / 1.27 | 366.60 / 0.31 |
| FP8 / BF16 | 4 | 184.71 / 2.01 | 455.69 / 1.40 |
| FP8 / FP8 | 2 | 126.13 / 0.31 | 271.84 / 0.51 |
| FP8 / FP8 | 4 | 101.42 / 0.73 | 403.75 / 0.33 |

此早期吞吐实验的 KV scale 使用仓库 `data/gsm8k.jsonl` 前 128 条校准，实际来自
**GSM8K test**，不是 train。该 scale 仅保留为这轮 HumanEval 吞吐记录的来源，
没有用于上面的 GSM8K 准确率实验。不能用它做独立 GSM8K test 准确率结论。

eager/Graph 输出差异被允许并单独记录；同配置重复也可能改变 token 数和 NFE。
HumanEval 与 GSM8K 的数据、prompt 和生成长度参数不同，TPS 不应跨数据集直接比较。

完整数据：[summary.csv](../../runs/quant-throughput/summary.csv)、
[50 次原始测量](../../runs/quant-throughput/raw.csv)、
[比值](../../runs/quant-throughput/ratios.csv)、
[输出差异](../../runs/quant-throughput/output-differences.json)、
[验收](../../runs/quant-throughput/verification.json)。

## 归档与复核

实验脚本、汇总、scale、依赖版本和身份清单在 `runs/quant-throughput/`、
`runs/gsm8k-kv-calibration/`、`runs/fa3-gsm8k/`、`runs/fp8-attention/` 中直接跟踪。
每个目录的 `records.tar.gz` 保存完整运行命令、
日志、GPU 采样、逐 rank 指标、生成输出以及准备好的实验输入。
`records-manifest.json` 列出归档及每个成员的 SHA256。原始测量文件未修改；
原命令中的绝对路径保留为历史记录。

模型权重、虚拟环境、编译缓存和可重新下载的 parquet 源文件未打包；
模型/数据的完整 revision、下载脚本及原文件 hash 保留在清单中。
`runs/` 继续忽略未来生成的文件，本次结果作为明确选定的快照提交。

在仓库根目录解包后，可以重新计算 GSM8K 对照与验收，无需启动 GPU 推理：

```bash
tar -xzf runs/quant-throughput/records.tar.gz -C runs/quant-throughput
tar -xzf runs/gsm8k-kv-calibration/records.tar.gz -C runs/gsm8k-kv-calibration
tar -xzf runs/fa3-gsm8k/records.tar.gz -C runs/fa3-gsm8k
tar -xzf runs/fp8-attention/records.tar.gz -C runs/fp8-attention
python runs/gsm8k-kv-calibration/analyze.py
python runs/fa3-gsm8k/write_results_report.py
```

新增归档也可不解包直接复核：`python runs/fa3-gsm8k/verify_records.py`。
该脚本校验逐成员 SHA256、源码快照、scale、正式/试跑结果和两轮合并汇总的数值。

HumanEval 的 `verify_results.py` 还核对原始推理依赖版本，需要匹配该环境。
如需重新生成，应使用独立 checkout/输出副本，配置模型路径并重新生成身份清单与校准文件；
不要覆盖归档的测量记录。脚本中 `.venv` 为推理环境，`.eval-venv` 为固定 EvalScope 的独立计分环境。

前次提交的运行时代码修改：生成计时前后 CUDA 同步、单调时钟与区间外 rank 最大耗时归约；
记录实际权重 dtype 字节数和每 rank 耗时；允许 TP=EP=2 的 FlashInfer paged Graph。
验证包含计时/路由回归、先前实际 CUDA/NCCL 计时检查，以及上面全部正式测量的验收。
本次新增显式启用的 FA3 原生 FP8 attention、设备端动态 Q 量化、block-causal 页前缀适配、
Graph 调度元数据更新及同适配器的 BF16 FA2/FA3 对照。默认路径保持现状。
固定 FlashInfer 0.6.18 的 FA3 plan ABI，不支持同一进程动态切换后端/精度。
83 项回归通过，含独立 PyTorch 数值参考、动态 scale/页表、Graph replay 和模型路由；
记录见 [tests.log](../../runs/fa3-gsm8k/tests.log)。提交前核对 138 个运行源码文件与正式实验哈希一致。
