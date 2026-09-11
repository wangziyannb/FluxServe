# LLaDA2 FP8 KV cache

KV cache 量化与权重量化独立。LLaDA2 支持 BF16 或 FP8 E4M3 KV 存储，
可与 BF16、ModelOpt FP8、ModelOpt NVFP4 权重配置组合。
NVFP4 权重组合需要 Blackwell 验证；不支持 NVFP4 KV 或 DiffusionGemma KV 量化。

新环境部署见 [Docker 部署指南](deployment.md)：镜像构建时预编译固定 CUDA 库，
启动时使用实际 KV scale 预热；可用 `serve --warmup-only` 提前准备持久编译缓存。

## 使用

`serve` 和 `bench_offline` 都接受：

```text
--kv-cache-dtype {auto,bf16,fp8_e4m3,fp8}
--kv-cache-scales /path/to/scales.json
--attention-compute-dtype {bf16,fp8}
```

`fp8` 是 `fp8_e4m3` 的别名。`auto` 仅根据 checkpoint 的 KV 声明选择格式，
无 KV 声明时仍为 BF16，即使权重本身是 FP8。显式 `bf16` 或 `fp8_e4m3` 覆盖自动选择。
仅支持静态、每层分别一个 K scale 和一个 V scale 的 FP8 KV 声明。
动态、逐通道、分组、INT8、NVFP4 等声明报错。

启用 FP8 必须提供每层完整、有限、正值的 scale。JSON 优先于 checkpoint scale；
缺失 scale 会提示校准，不使用默认的 1。全零校准观测值对应的 scale 才使用 1。

现有 `thnkinbtfly/llada2.0-flash-fp8` checkpoint 没有 KV scale，需要先校准：

```bash
fluxserve calibrate_kv_cache \
  --model thnkinbtfly/llada2.0-flash-fp8 \
  --quantization modelopt_fp8 \
  --dataset data/gsm8k.jsonl --num-samples 128 \
  --output runs/llada2-kv-scales.json \
  --tp-size 4 --ep-size 4 \
  --batch-size 8 --mini-batch-size 4 \
  --gen-len 512 --block-length 64 --threshold 0.95 \
  --parallel-decoding threshold
```

校准复用离线 benchmark 的数据读取、tokenizer、模型加载和 diffusion 解码。
强制 SDPA eager、BF16 KV，权重仍使用目标格式；启动 warmup 不计入观测。
观测发生在 QK norm/RoPE 后、attention 前，覆盖所有 prefill 调用和 diffusion 迭代。
各 TP rank 对每层 K/V amax 做 MAX 归约，再分别计算 `amax / 448`。

```bash
fluxserve bench_offline \
  --model thnkinbtfly/llada2.0-flash-fp8 \
  --dataset data/humaneval.jsonl \
  --quantization modelopt_fp8 \
  --kv-cache-dtype fp8 --kv-cache-scales runs/llada2-kv-scales.json \
  --attention-backend sdpa \
  --tp-size 4 --ep-size 4 --batch-size 8 --mini-batch-size 4 \
  --gen-len 512 --block-length 64 --threshold 0.95 \
  --parallel-decoding threshold --use-cuda-graph

fluxserve serve \
  --model thnkinbtfly/llada2.0-flash-fp8 \
  --kv-cache-dtype fp8 --kv-cache-scales runs/llada2-kv-scales.json \
  --attention-backend flashinfer --kv-cache-layout paged \
  --flashinfer-prefill-mode paged --flashinfer-cache-mode paged \
  --scheduler-policy paged --tp-size 4 --ep-size 4 \
  --max-model-len 4096 --use-cuda-graph
```

SDPA/Flex 的布局参数由 CLI 自动归一为 dense。默认 FlashInfer FP8 KV 路径使用公共 FA2
paged wrapper；不要求 `flashinfer-dllm` 的同 dtype block-extend 接口。
启动日志输出最终 KV dtype、scale 来源和 attention 路径。

### H100 原生 FP8 attention

默认 `--attention-compute-dtype bf16` 保持上述计算路径。显式使用 `fp8` 可启用
Hopper FA3 的 FP8 QK/PV Tensor Core 计算，输出仍为 BF16，softmax/累加保留高精度。
这与仅将 KV 存为 FP8 是不同的精度配置；需要重新验证模型质量。

在原有 paged FP8 KV 命令上增加：

```text
--attention-compute-dtype fp8
--attention-backend flashinfer
--flashinfer-prefill-mode paged --flashinfer-cache-mode paged --kv-cache-layout paged
--kv-cache-dtype fp8_e4m3 --kv-cache-scales /path/to/scales.json
```

当前范围为 LLaDA2、Hopper（如 H100）、head dimension 64/128，支持 eager 和 paged
prefill/decode CUDA Graph。权重量化独立，KV 沿用已有静态校准 scale；Q 在 QK norm/RoPE
之后按每个 query head 对本次调用的 tokens 动态计算 `amax / 448`（全零 head 使用 1），
通过两阶段 Triton reduction/encode 转为 E4M3。Q scale 全程在设备上计算，Graph replay
也会更新；不需要另做 Q scale 校准，不使用 K scale 代替 Q scale。

每个 diffusion block 被表示为独立的查询段，引用原请求从起点到可见 block 末尾的
KV 页前缀。段内使用非因果 attention，保持块内双向、块间因果的原有语义，支持位置偏移
和部分页。只复制页索引，不复制历史 KV，不构造二次方大小的自定义 mask。

Graph 使用固定查询分段和工作分配，replay 前更新页表、可见长度及 FA3 调度表中的
KV 起点/长度。固定 FlashInfer `0.6.18`（本仓库环境的 fork revision
`5ce4d077c33bbe167386cf5487a4a5bb4bcafbfd`）会在 plan 时复制这些字段，因此仅修改外部
CSR 不够。适配器检查该版本的 SM90 plan ABI；不支持的版本报错，不能无验证地升级。
计算阶段不重新 plan/capture。Graph key 包含 attention compute dtype，防止复用另一精度的图。
计算配置在进程启动时固定；同一进程中动态切换多种 FlashInfer 后端/精度组合不在已验证范围。
回归中这种混用可导致旧 BF16 Graph 卡住，模型级配置对照因此使用独立进程。

不满足硬件、布局、KV dtype 或 mask 契约时明确报错，不静默回退到 BF16。
日志与 benchmark metrics 分别记录 `flashinfer-fa3-fp8`、`attention_compute_dtype`
和 `attention_kernel`。KV 校准命令仍固定为 SDPA/BF16 attention。

### 固定 FA2/FA3 的 BF16 对照

`--flashinfer-kernel-backend {auto,fa2,fa3}` 默认保留原有选择。对 LLaDA2 的 paged
BF16 KV，显式选择 `fa2` 或 `fa3` 会使用同一个查询分块、页表和 KV 写入适配器，
支持 eager 和 CUDA Graph；因此可以在权重、KV、attention 都为 BF16 时比较后端。
FP8 attention 只能选择 `auto` 或 `fa3`。当前 FlashInfer FA3 不支持 BF16 Q / FP8 KV，
该组合明确报错；BF16 权重搭配 FA3 FP8 KV 时也必须显式启用 FP8 attention。

每个 rank 的 benchmark metrics 另存 `observed_block_attention_kernels`，来自实际调用的
wrapper 后端与 Q/KV dtype；Graph key 同时包含后端。各对照使用独立进程，避免后端混用。
比较吞吐时仍需报告生成 token 数与 NFE，不能把生成行为变化全部归因于内核加速。

## Scale 文件及 checkpoint

版本 1 JSON 包含：

- `version=1`、`kv_cache_dtype="fp8_e4m3"`。
- `checkpoint.name` 与 `checkpoint.revision`：Hugging Face 配置的模型身份和 commit hash。
- `weight_format` 与 `model`：权重格式、层数、hidden size、Q/KV heads 和 head dimension。
- `calibration`：数据集路径及 SHA256、样本数/ID、解码长度、阈值、batch 和并行配置。
- `layers`：以层号字符串为键，分别包含 `k_scale`、`v_scale`、`k_amax`、`v_amax`。

加载时严格检查 checkpoint 身份、权重格式和模型维度。请使用与校准时一致的
模型名称或本地路径；同一 checkpoint 换成另一个路径也可能触发身份检查。
不要通过更改模型身份字段把其他模型的 scale 当作已校准结果使用。

checkpoint 接入 `model.layers.N.self_attn.k_proj.k_scale` / `v_proj.v_scale`，
以及 attention 层直接持有的 `k_scale` / `v_scale`；`attention` 和 `self_attn`
两种层名均支持。ModelOpt 的平铺和嵌套 `quantization` 元数据均支持。
线性层的 `weight_scale` 和 `input_scale` 保持独立，不作为 KV scale 使用。

## 存储、计算和 Graph

编码：`cast_fp8(clamp(x / scale, -448, 448))`；解码：`bf16(fp8.float() * scale)`。
持久缓存、prefill 返回值、当前 diffusion block 和完成 block 均采用相同编码。
缓存重排、复制和页复用只移动已编码的数据，避免重复缩放。

- **SDPA/Flex**：持久存储 FP8，仅在计算当前层 attention 时反量化；不同时创建整模型的 BF16 KV 副本。
- **FlashInfer paged**：Triton 融合量化与页写入；Q/输出为 BF16，KV 为 FP8，公共 FA2 wrapper 接收每层 K/V scale。显式 block-causal mask 支持位置偏移和块内双向注意力。
- **FlashInfer ragged 输入**：使用公共 paged FA2 wrapper 的 `page_size=1` 零复制视图。已检查的 FlashInfer 0.6.13 和当前固定的 0.6.18 ragged FA2 实现均未应用 BF16 Q / FP8 KV 路径的 `k_scale`/`v_scale`；此适配避免错误缩放，不创建 BF16 历史缓存。
- **FlashInfer 原生 FP8 compute（显式启用）**：Q/K/V 为 E4M3，FA3 的 QK 和 PV 均使用 FP8 Tensor Core，输出 BF16；路径和 Graph 元数据处理见上节。
- **内存**：相同容量的 FP8 KV 数据字节数恰为 BF16 的一半。页表、mask、workspace、权重、Graph 输出和临时反量化仍有开销；总显存和 TPS 不保证减半或提升。

Graph capture 前完成 scale 固定、wrapper planning、kernel warmup 和固定地址缓冲区分配。
prefill 更新 mask 和页表；FP8 decode 使用固定容量页表及 packed mask，更新现有缓冲区后 replay，
不在 capture/replay 中调用 wrapper planning。Graph key 包含 KV dtype 和各层 scale。
warmup 的预捕获和实际 replay 使用同一 key。

保留已有 Graph 范围：SDPA prefill/decode、Flex 原有 decode 路由、FlashInfer paged 原有 TP/EP 组合。
Flex prefill Graph 及原本不支持的并行组合仍禁用；Flex 编译关闭内部自动 Graph，避免嵌套 capture。
已有通用 Graph 清理改动保留，退出时释放 Graph、缓存和分布式资源。

## 验证与测量

```bash
PYTHONPATH=python:flux-kernel/python pytest -q test/runtime/test_kv_quantization.py
PYTHONPATH=python:flux-kernel/python pytest -q test/runtime/test_kv_quantization_cuda.py
PYTHONPATH=python:flux-kernel/python pytest -q test/runtime/test_native_fp8_attention.py test/runtime/test_native_fp8_attention_cuda.py
```

CUDA 测试使用独立 PyTorch 编码和 attention 参考，默认 `rtol=1e-2, atol=1e-2`。
包含非单位/不同层 scale、跨页写入、部分 prompt、连续重写、请求重排、页复用和 dummy 页，
以及 SDPA/Flex、FlashInfer eager/replay 与清理检查。测试用小张量/两层模型不代表完整模型质量。
原生 FP8 算子测试额外考虑 PV 中 softmax 概率的 FP8 舍入，同时限制相对 L2 误差 `<0.04`
和最大绝对误差 `<0.08`；这只是随机张量测试的门槛，不能替代全量 GSM8K 等质量评估。

每次离线 benchmark 额外生成 `<exp-name>_metrics.json`，记录：

- 每 rank 持久 KV 数据字节数、Graph 输入 KV 字节数。
- PyTorch allocated/reserved 峰值，包含模型加载、warmup 和 capture。
- 实际生成阶段的 TPS、NFE、token 数和 Graph replay 次数（排除 warmup replay）。
- FlashInfer 在计时前的 capture/invalidation 计数，以及计时期间的增量：
  `flashinfer_graph_before_generation` 和 `flashinfer_graph_during_generation`，
  均包含 `prefill`、`decode`、`gemma_decode`、`invalidations`。

LLaDA2 FlashInfer Graph 预热前按 `max(batch_size, mini_batch_size)` 分配持久 KV，
保证 mini-batch 预热和正式 batch 使用相同缓存地址。性能验收除检查每个 rank
确实发生生成 replay 外，还要求 FlashInfer 计时期间没有新增 capture 或缓存失效；
缺少上述计数的旧测量文件也不能通过该检查。

完整验收脚本：

```bash
# 与现有 CI 相同的固定 EvalScope 版本
python -m venv /tmp/evalscope-venv
/tmp/evalscope-venv/bin/python -m pip install \
  'evalscope @ git+https://github.com/modelscope/evalscope.git@acd09b44384d53174768bb1063f675420f76fae9'

python test/benchmark/fluxserve/kv_cache_acceptance.py \
  --stage all --gpus 0,1,2,3 --output-dir runs/kv-cache-acceptance
```

支持 `check`、`calibrate`、`quality`、`performance` 独立阶段。脚本在运行前检查空闲 GPU，
串行执行，保存命令、源文件 hash、日志和显存采样；发现外部 GPU 进程或退出超时则不接受该次运行。
需从可查看 GPU 进程所属进程组的环境运行；容器需保留对应 PID 可见性。
`--python` 指定 FluxServe 运行环境，`--evalscope-python` 指定评测环境。

质量使用全量 HumanEval 164 题、单次生成、temperature 0、生成上限 512；
同一 FP8 权重的 BF16 KV 与 FP8 KV pass@1 差值最多允许 0.02。
性能使用 HumanEval 前 8 条、batch 8、mini-batch 4、生成 512、block 64、threshold 0.95；
在 SDPA 和 FlashInfer 上分别比较单卡 eager、四卡 eager/Graph，两种 KV dtype 各运行 5 次，报告中位数和未缩放 MAD。
设备总显存采用约 0.5 秒间隔的 `nvidia-smi` 采样峰值，同时保留 PyTorch 分配器峰值。

实际结果和未验证范围见 [FP8 KV 验收记录](../experiments/fp8-kv-cache.md)。
