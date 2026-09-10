# LLaDA2 FP8 KV 实现与验收记录

日期：2026-09-10。基于 `quantization` 分支 `2972589` 的工作区改动，保留已有未提交的 CUDA Graph 清理修复。

**实现已接入，但完整模型质量和性能验收尚未完成，不能据此宣称完整验收通过或推理加速。**

## 已运行的检查

| 检查 | 环境 | 结果 |
|---|---|---|
| 配置、校准、scale 加载、缓存及相关原有回归 | Python 3.13 / PyTorch 2.13.0+cu130，CUDA 隐藏 | 118 passed，2 skipped |
| 新增 CUDA 数值、SDPA/Flex、FlashInfer paged/ragged、Graph、清理 | H200 / Python 3.12 / PyTorch 2.8.0+cu129 / FlashInfer 0.6.13 | 12 passed |
| 真实两进程 scale MAX 归约 | CPU / Gloo，正常销毁 process group | 两个 rank 导出一致 scale，通过 |
| 独立融合分页写入参考 | 同上；另在主机 PyTorch 2.13 验证 | 非单位 scale、FP8 输入不重复缩放、跨页重写及 dummy/无效 slot 通过 |
| 相同容量缓存数据大小 | 2 层 dense/paged 小缓存 | FP8 数据为 BF16 的 1/2，不含页表、workspace 和临时量 |
| CLI、Python 编译、diff 空白检查 | 当前工作区 | 通过 |
| 完整 32 层模型四卡小样本校准、SDPA / FlashInfer FP8 KV eager | 已有 H200 rootfs；GPU 4–7，TP=4 / EP=4 | 三项均退出码 0；详情见下文 |
| 完整模型 SDPA / FlashInfer FP8 KV 在线服务 | 新构建的只读 Docker runtime；GPU 4–7，TP=4 / EP=4 | 启动预热、健康检查、短生成请求均通过；见 [Docker 部署验证](docker-deployment.md) |
| 完整实验空闲检查 | 8 × H200 NVL | 所有 GPU 均有外部计算进程，完整实验未启动 |

CPU 命令：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=python:flux-kernel/python pytest -q \
  test/runtime/test_kv_quantization.py \
  test/runtime/test_modelopt_fp8.py \
  test/runtime/test_model_runner_cuda_graph_routing.py \
  test/runtime/test_block_diffusion_offline.py \
  test/runtime/test_scheduler_defaults.py \
  test/runtime/test_diffusion_gemma.py \
  test/runtime/test_diffusion_gemma_benchmark.py
```

GPU 命令在已有 H200 runtime rootfs 中加载当前工作区源码，串行执行：

```bash
PYTHONPATH=python:flux-kernel/python pytest -q test/runtime/test_kv_quantization_cuda.py
```

这些小规模功能测试使用共享 GPU，不用于报告吞吐或完整模型峰值。
两层 FlashInfer Graph 测试实际执行 prefill 和 decode replay，覆盖请求切换、长度变化、
预捕获 key 复用、padded/decomposed 两种已有模式，并执行正常清理。
BF16 回归覆盖 SDPA/Flex；不把此前仓库的 FP8 **权重**实验算作 FP8 **KV**验收。

修复测试发现的问题：FP8 输入的 Triton masked load 使用浮点零；Flex 禁用编译器内部自动 Graph，
避免外层 capture 内执行另一个 Graph replay；统一 warmup/replay 的 KV 配置 key，避免重复 capture。

## 指定 GPU 4–7 的完整模型运行检查

按用户后续要求，在已有 `/home/zwang53/fluxserve-h200-rootfs` Docker/chroot 环境中，
对物理 GPU `4,5,6,7` 串行执行小样本 eager 检查。该次允许使用共享 GPU，仅检查能否完成生成和正常退出；
不计入质量、性能、在线服务或 CUDA Graph 验收。

- 环境：Python 3.12、PyTorch 2.8.0+cu129、FlashInfer 0.6.13。
- 完整 checkpoint：`thnkinbtfly/llada2.0-flash-fp8`，revision `85bd9f38034aa46f93135676189acec4d7fc40d3`，32 层。
- 公共参数：`--quantization modelopt_fp8 --tp-size 4 --ep-size 4 --batch-size 1 --mini-batch-size 1 --gen-len 64 --block-length 64 --prefilling-limit 128 --parallel-decoding threshold --threshold 0.95`。
- 校准使用 GSM8K 第一条、SDPA eager、BF16 KV，排除 warmup；随后两个后端都使用同一份 `smoke-scales.json` 和 HumanEval 第一条。
- 短生成长度会经过原有分桶及 EOS 处理，不用本次 token 数或 TPS 比较后端性能。

| 运行 | 有效 KV dtype / 布局 | 结果 |
|---|---|---|
| `calibrate_kv_cache` | BF16 / dense | 4 个 rank 完成生成及 TP/NCCL MAX 归约，导出 32 层共 64 个有限、正的 scale；退出码 0 |
| `bench_offline --attention-backend sdpa --kv-cache-dtype fp8_e4m3` | FP8 E4M3 / dense | 完成生成、写出非空答案；退出码 0 |
| `bench_offline --attention-backend flashinfer --kv-cache-dtype fp8_e4m3` | FP8 E4M3 / paged，公共 FA2 路径 | 完成生成、写出非空答案；退出码 0 |

两次推理的 metrics 均确认 `kv_cache_dtype=fp8_e4m3`、scale 来源为该 JSON，并包含全部 4 个 rank。
未启用 Graph，replay 计数为 0；三个容器均已退出，无本次测试 worker 残留。
这份单样本 scale 仅供运行检查，不替代计划中的 GSM8K 前 128 条校准。

完整命令、启动日志、输出及汇总保存在 `/data/fluxserve-kv-smoke-20260910-rootfs/`：
`run_case.py`、`summary.json`、`smoke-scales.json`，以及各项的
`{calibrate,sdpa,flashinfer}-command.json`、`*-result.json`、`*-launcher.log` 和对应输出子目录。
本机复现入口（依次执行）：

```bash
python /data/fluxserve-kv-smoke-20260910-rootfs/run_case.py calibrate
python /data/fluxserve-kv-smoke-20260910-rootfs/run_case.py sdpa
python /data/fluxserve-kv-smoke-20260910-rootfs/run_case.py flashinfer
```

首次 rootfs 尝试因源码只读挂载阻止 RoPE 算子写入 `objs/` 而失败；改为可写挂载后上述三项通过。
失败日志保留为 `calibrate-readonly-attempt1-*`。在用户指定 rootfs 前启动的原生 host 尝试已停止，不计入结果。

## 尚未验证

| 验收项 | 状态与原因 |
|---|---|
| FP8 checkpoint + GSM8K 前 128 条离线校准 | 未运行；仅完成上面的单样本运行检查，未生成完整验收 scale |
| 同权重 BF16 KV vs FP8 KV 全量 HumanEval 164 | 未运行；pass@1 下降不超过 2 个百分点的门槛尚未验证 |
| SDPA / FlashInfer 单卡 eager、四卡 eager/Graph，各 dtype 5 次 | 未运行；没有 TPS、总峰值显存或 MAD 对比结论 |
| 完整 LLaDA2 模型四卡 TP/EP Graph | 未验证；小模型 Graph 检查不能替代该项 |
| NVFP4 权重 + FP8 KV | 接口保留；需要 Blackwell，H200 不计入通过 |

现有 runtime 的 FlashInfer 0.6.13 公共 FA2 FP8 路径已做小规模验证；
原有 BF16 paged 路径仍要求仓库现有的 native block-extend/offset API。
当前 rootfs 的公共 wrapper 不暴露这些 native 参数，因此该环境不能用于接受 BF16 paged 全模型对比结果。
BF16 原有实现保留；完整 FlashInfer A/B 需使用满足原有 BF16 依赖要求的环境。

空闲检查记录：2026-09-10 05:58 UTC，GPU 0–7 利用率均为 100%，已使用显存约 62–91 GiB/卡，
均存在非本次实验的计算进程。完整验收脚本停止于该检查；随后仅按用户要求另行运行上述四卡小样本检查。

复现校准、服务和完整验收的命令见 [使用指南](../guides/fp8-kv-cache.md)。
