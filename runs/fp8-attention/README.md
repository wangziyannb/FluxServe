# 原生 FP8 attention 实现验证

固定环境：4 × H100 80GB，PyTorch 2.8.0+cu128、FlashInfer 0.6.18 fork 5ce4d077c33bbe167386cf5487a4a5bb4bcafbfd。
模型/数据/scale 使用上一轮已固定的身份；模型为 FP8 LLaDA2.0-flash，KV 均为 FP8，scale 来自 UltraChat train_sft。
完整模型测试为 GSM8K test 前 16 题，4-shot，batch 8/mini-batch 4，生成长度参数 2048，block/page 64，threshold 0.95，seed 42，每配置一次。
原始 token IDs、输出、准确率计分、命令、日志、逐 rank 指标及 0.5 秒 GPU 采样保存在各配置目录；source-hashes.json 固定实际测量源码。

## 完整模型

| GPU | Attention compute | 模式 | TPS | completion tokens | NFE | 正确数 | 显存峰值 GiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 2 | fp8 | eager | 96.89 | 5605 | 858 | 16/16 | 65.18 |
| 2 | fp8 | Graph | 219.56 | 7071 | 1854 | 15/16 | 61.80 |
| 2 | bf16 | Graph | 130.77 | 6638 | 1415 | 16/16 | 61.80 |
| 4 | fp8 | Graph | 394.26 | 5344 | 803 | 15/16 | 36.42 |
| 4 | bf16 | Graph | 113.23 | 6975 | 1949 | 16/16 | 36.42 |

TPS 使用同步后的各 rank 最大生成时间，包含请求 prefill/decode，排除加载、JIT、warmup/capture、写盘和计分。
Graph 配置每个 rank 均发生 decode replay，计时阶段 capture/invalidation 增量均为零。五次运行串行完成，GPU 独占与进程退出检查通过。

| GPU | 新/旧 TPS | 新/旧 token 数 | 新/旧 NFE | 新/旧每秒前向次数 |
|---:|---:|---:|---:|---:|
| 2 | 1.679× | 1.065× | 1.310× | 2.065× |
| 4 | 3.482× | 0.766× | 0.412× | 1.872× |

两种 Graph 的原生 FP8 均少答对一道题（gsm8k/test/12）：双卡输出达到对齐后的长度上限仍未结束，四卡把“开始盈利”回答为第 12 年，gold 为 13。
16 题不能代表全量准确率。尤其四卡的 NFE 大幅下降，端到端 TPS 比值不能全归因于 attention 算子提速。
这些是全量实验之前的实现验证记录；后续完整 GSM8K 结果见 [FA3 结果](../fa3-gsm8k/RESULTS.md)
和 [两轮合并解读](../fa3-gsm8k/COMBINED-RESULTS.md)。路径显式启用，未改变默认 BF16 compute。

## Attention 微基准

相同 batch=4、query length=64、head dim=128、KV dtype/scale 与有效 KV 长度；旧路径为 FA2 BF16 Q / FP8 KV，新路径为 FA3 FP8 Q/K/V。
计入新路径每次调用的 Q 动态量化，排除两边的 plan、KV 页写入与输入准备。CUDA Graph 重放，每轮 100 次，共 5 轮，报告每轮均值的中位数。
这是新旧完整 attention 实现的比较，包含 FA2/FA3 后端差异，不能解释为同一内核仅换 dtype 的收益。

| Q/KV heads | KV length | 旧路径 μs | 新路径 μs | 加速比 |
|---|---:|---:|---:|---:|
| 16/2 | 1024 | 147.62 | 20.65 | 7.15× |
| 16/2 | 2048 | 293.93 | 30.26 | 9.71× |
| 16/2 | 4096 | 570.74 | 50.08 | 11.40× |
| 8/1 | 1024 | 143.35 | 20.66 | 6.94× |
| 8/1 | 2048 | 291.41 | 30.42 | 9.58× |
| 8/1 | 4096 | 558.59 | 50.42 | 11.08× |

编译产物中的 QGMMA.*.F32.E4M3.E4M3 指令见 fp8-instructions.json，确认 FP8 Tensor Core 输入和 FP32 累加。
算子测试以独立 PyTorch block-causal reference 核对；同时覆盖动态 query scale、部分页、位置偏移、GQA、请求换页/换长度、eager 和真实 Graph capture/replay。最终回归输出见 final-tests.log。

最终回归：76 passed（final-tests.log）。模型级不同配置在独立进程运行；混合后端单进程回归曾在旧 BF16 Graph 卡住，已保留 mixed-process-tests-interrupted.log；同一 BF16 用例独立进程通过（isolated-bf16-graph.log）。同进程动态切换后端不在支持范围。

## 归档

`records.tar.gz` 保留五个完整模型试跑目录、测试/微基准日志和 `native-kernel.sass`；
`records-manifest.json` 给出归档及逐文件 SHA256。解包到本目录可恢复原始记录。
`probe.py` 是早期探索脚本，其中旧误差门槛不作为验收依据；正式测试结果以上述回归记录为准。
