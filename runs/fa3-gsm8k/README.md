# FA3：UltraChat 校准、GSM8K 全量 accuracy / throughput

本轮与上一轮保持 BF16 checkpoint，不改为 IEEE FP16。每个正式配置完整评测 GSM8K test
1,319 题一次；4-shot、seed 42、global batch 8、mini-batch 4、gen-len 2048、block/page 64、
threshold 0.95、low-threshold 0.3、TP=EP=卡数、DP=PP=1，Graph 使用 decomposed decode。
所有运行串行、各自独立进程，每个配置正式测量前先试跑前 16 题。

| 权重 | KV 存储 | Attention QK/PV 计算 | 后端 | GPU | 模式 |
|---|---|---|---|---|---|
| FP8 | FP8 | FP8 | FA3 | 2、4 | eager、Graph |
| BF16 | BF16 | BF16 | FA3 | 4 | eager、Graph |
| BF16 | FP8 | FP8 | FA3 | 4 | eager、Graph |
| BF16 | BF16 | BF16 | FA2 | 4 | eager、Graph |

共 10 个正式配置。BF16 权重无法容纳在两张 H100 80GB，依据容量排除。
“全 FP8”指量化权重、KV 存储及 attention QK/PV 的 Tensor Core 输入；
不表示归一化、softmax、累加或全部中间输出都使用 8 位。attention 输出仍为 BF16。

FP8 KV 只使用 UltraChat `train_sft` 校准。FP8 模型复用此前已验证的 512 条 scale；
BF16 模型使用同一批 512 条 UltraChat 输入，另外执行 SDPA eager/BF16 KV 校准。
模型路径/权重格式不同，因此不直接共用 FP8 模型的 scale。GSM8K train/test 均不用于
本轮 KV scale 校准；既有四条 GSM8K train few-shot 示例继续用于固定评测 prompt。

当前 FlashInfer FA3 要求 FP8 KV 配 FP8 Q；不支持 BF16 attention + FP8 KV。
因此 BF16 权重 / FP8 KV 一行必须标注 attention FP8，不能解释为仅改动 KV 存储。

`--flashinfer-kernel-backend` 显式固定内核；BF16 FA2/FA3 使用相同的查询分块、页表适配、
KV 写入方式与解码参数。每个 rank 记录真实 wrapper 后端及 Q/KV dtype，并要求 Graph
确实 replay、计时内无新增 capture/invalidation。固定进程配置规避已知的同进程后端混用问题。

`summary.json` / `summary.csv` 随正式配置完成而更新，包含 accuracy、TPS、单卡 TPS、
token 数、NFE、每次前向 token 数、前向次数/秒、显存峰值。`comparisons.json` 自动生成：

- 四卡、同为 FA3：全 FP8 / 全 BF16，表示整体量化收益。
- 四卡、权重/KV/attention 均 BF16：FA3 / FA2，比较相同适配方式下的后端收益。
- 四卡 FA3：BF16 权重 + FP8 KV/attention / 全 BF16，表示 KV 与 attention 联合变化。
- 四卡 FA3：全 FP8 / BF16 权重 + FP8 KV/attention，权重精度改变且 scale 按各模型校准。
- 全 FP8 两卡到四卡加速比及效率；每种组合 Graph / eager。

各比值分别在 eager 和 Graph 内计算，同时给出 token/NFE 比值、accuracy 差值及逐题输出差异。
生成行为随数值精度/内核变化，因此端到端 TPS 比值不能完全归因于计算速度；每配置仅一遍，
不报告重复测量的 MAD，也不宣称质量等价。旧实验中 BF16 eager/Graph 后端选择并不统一，
本轮新增明确固定后端的对照，不将旧行直接当作匹配的 FA2 基线。

文件说明：

- `experiment.py`：准备、试跑、正式运行、验收、评分和汇总；`status.json` 记录队列状态。
- `manifest.json`、`source-snapshot.tar.gz`、`source.patch`：代码/输入/模型身份及哈希。
- `smoke/`、`measurements/`：命令、日志、逐题答案/token IDs、分 batch 计时、各 rank 指标。
- `calibration-bf16-ultrachat/`、`scales-ultrachat-bf16-model.json`：BF16 模型校准及验证。
- `telemetry.jsonl`：每 0.5 秒 GPU 显存、利用率、进程采样；异常退出/外部干扰作废。

正式计时为每个 batch CUDA 同步后的各 rank 最大耗时之和，包含请求 prefill 和 diffusion decode，
排除模型加载、warmup、JIT、Graph capture、写盘和评分。运行前后验证 GPU 独占及清理完成。
运行期间若源码或输入哈希变化则停止，保留失败记录，不自动覆盖或重跑。

本轮已全部完成，见 [最终结果](RESULTS.md) 和 [两轮合并解读](COMBINED-RESULTS.md)。
`records.tar.gz` 保存原始 `smoke/`、`measurements/`、BF16 校准目录与执行日志，
`records-manifest.json` 保存归档及逐成员 SHA256。解包到本目录即可恢复原始记录；
复核完整输入还需解包 `../gsm8k-kv-calibration/records.tar.gz` 到对应目录。
`tests.log` 直接跟踪 83 项回归结果；`completion-audit.json` 为十个正式配置的验收。
无需解包或 GPU，可运行 `python runs/fa3-gsm8k/verify_records.py`，核对两份新增归档的逐文件哈希、
源码快照、scale、十个正式/试跑结果及两轮合并 CSV/JSON 的数值。
