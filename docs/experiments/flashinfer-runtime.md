# FlashInfer runtime 更新

日期：2026-09-10。修复 `06e58f0` 中 Docker 依赖与 BF16 paged attention 的接口不匹配。

## 固定版本与兼容范围

Docker 改为使用 `FLX-OSS/flashinfer-dllm` 的 `dllm/block-decode` 分支，固定
`5ce4d077c33bbe167386cf5487a4a5bb4bcafbfd`（0.6.18，2026-08-26）。原来的
`kennethzhao24/flashinfer-dllm:feature/block-extend` 仍停在
`6055bba6d7defefe69fe246473647165300b557a`（0.6.13），没有公共 paged block-extend 参数。

新版提供公共 paged/ragged wrapper 的 `block_extend`、`block_size`、Q/KV offset 和
Graph 固定 offset buffer。仓库的 BF16 ragged 适配器同步迁移到公共 wrapper，传入
`head_dim_qk`、KV dtype 和 attention scale；相同页表下改变 scale 也会重新 plan。
FP8 KV 保留公共 FA2 和显式 mask 路径。NVFP4 MoE 保留原有 CUTLASS 调用及六项 scale 布局。

CUDA 12.9、Torch 2.8.0+cu129、Triton 3.4.0 保持固定；CUTLASS DSL 更新为 4.7.0，
CUDA Python 固定为 12.9.7。安装 FlashInfer 时用已有包的版本作为 constraints，
防止依赖解析替换 Torch/CUDA 栈。关闭未使用的 FlashInfer NIXL/NCCL EP 原生构建，
FluxServe 原有 TP/EP 通信路径继续使用。

`docker/check_flashinfer.py` 在 dependencies 和最终 runtime 构建阶段检查真实导入、
revision 和接口签名：paged/ragged block-extend、Graph offset buffer、FP8 K/V scale、
分页追加及 NVFP4 MoE。结果写入 `build-info.json` 的 `flashinfer_api_check`，明确标为
`imports_and_signatures_only`；这不是 NVFP4 的 Blackwell 数值验证。

## 验证记录

记录目录：`/data/fluxserve-flashinfer-update-20260910/`。

- `cpu-tests.log`：接口拒绝、ragged 参数/scale 转发、部署和缓存回归，23 passed / 2 skipped。
- `build.log`：最终 Docker 构建；`build-attempt1-pip-check.log` 保留首次检查失败记录。
- `run_gpu_tests.py`：使用构建出的已安装 runtime，单卡执行小张量数值与 Graph 回归。
- `run_smoke.py`：使用同一 runtime，在 GPU 4–7 上串行执行短请求检查。

构建产物为 `/data/fluxserve-deployment-20260910/artifacts/fluxserve-cu129-fi018.docker.tar`，
镜像 tag 为 `fluxserve:cu129-fp8-fi018`，可用 `docker load -i` 加载。归档 SHA256：
`4df56560ca0f99a57f55db4ed166766d3768f8c5f96c10bfd707e86c1d6330ca`。
本次没有覆盖旧版 `fluxserve-cu129-fp8.docker.tar`。

无 GPU 的 Docker build 通过原生库、TVM FFI FP8 bridge、pip runtime 依赖和全部
FlashInfer API 检查，构建架构为 `90a;100a;120`。测试使用同次构建导出的
`flashinfer-0.6.18-runtime`，直接导入镜像安装的包，并核对 attention 适配器文件摘要，
不从宿主机注入 FluxServe 或 flux-kernel 源码。容器 rootfs 只读，使用非 root UID 和
独立可写编译缓存。

`gpu-tests.log`：GPU 7（H200 NVL），21 passed，33.46 秒；共享卡，耗时仅用于定位日志。
覆盖融合 FP8 写入、非单位逐层 scale、跨页/重写/请求切换、BF16/FP8 的
SDPA/Flex 和 FlashInfer 数值与 Graph 回归、native append 及退出清理。
BF16 ragged 仍仅接受 block-aligned prefill；partial prompt 的 paged 路径和随后 decode
均参与检查。数值容差为 `rtol=1e-2, atol=1e-2`。

首次 GPU suite 失败记录保留在 `gpu-tests-initial.log`：新增 BF16 测试缺少 KV offset，
并错误地直接调用不支持非对齐 prefill 的 BF16 ragged 路径，现已修正测试元数据/范围。
此外，修正元数据后，整个混合 backend/KV dtype/Graph 配置 suite 在同一进程执行时，
仍在 BF16 native padded Graph 预热阶段发生非法访存：`gpu-tests-inline.log` 为
10 passed / 1 failed，遇到错误即停止，不能计入通过结果。确切交互原因尚未定位，
不能简单归因为 Graph 模式切换或 KV dtype 切换：`bf16-graph-mode-switch.log` 和
`fp8-eager-then-bf16-padded.log` 两组简化顺序各 2 passed。

服务为每个进程固定一种配置。上述 21 passed 的 suite 将 native padded 配置放入
独立子进程，仍执行真实 capture/replay、请求切换及清理，不掩盖子进程失败。
独立 native padded 检查通过；另用 `bf16-eager-then-padded.log` 检查 BF16 eager
paged/ragged 后进入 native padded Graph 的部署顺序，3 passed。本次没有解决或接受
完整混合配置的单进程运行；它与以下四个独立服务进程的通过结果分开记录。

完整模型检查统一使用 `thnkinbtfly/llada2.0-flash-fp8`，checkpoint revision
`85bd9f38034aa46f93135676189acec4d7fc40d3`，权重均为 `modelopt_fp8`，只切换 KV dtype。
GPU 4–7，TP=4 / EP=4，batch=1，最大上下文 256、生成上限 64、block length 64、
prefill limit 128、paged scheduler 的 device pages=16。Graph 使用默认 `decomposed`
模式，prefill buckets=64/128、decode capture batch=1。输入为 HumanEval 第一条，
检查四 rank 预热、HTTP healthcheck、非空生成及实际 decode replay；不评判答案正确性。
FP8 KV 使用此前单样本 smoke 校准的 `smoke-scales.json`，不是完整质量校准结果。

| FlashInfer 整模型服务 | Health / 生成 HTTP | 生成 token | Prefill / decode replay | Graph fallback |
|---|---|---:|---:|---:|
| FP8 权重 + BF16 KV，eager | 200 / 200 | 48 | 未启用 | 未启用 |
| FP8 权重 + FP8 KV，eager | 200 / 200 | 48 | 未启用 | 未启用 |
| FP8 权重 + BF16 KV，Graph | 200 / 200 | 48 | 1 / 6 | 0 |
| FP8 权重 + FP8 KV，Graph | 200 / 200 | 48 | 1 / 6 | 0 |

四组均有四个 rank 完成预热，镜像 healthcheck 返回 0，生成结果非空。Graph 组启动时
prefill capture=2、decode capture=1，请求完成后的 invalidation=0。
各组 `*-result.json`、请求响应和 `smoke-summary.json` 保存原始结果。

四个服务容器均通过 `docker stop` 结束，返回原有 supervisor 的 SIGTERM 约定值 143；
HTTP shutdown 完成，三个 worker 均收到 shutdown。与旧镜像相同，supervisor 回收时
仍有 semaphore 清理警告，不计作无警告的退出。单卡 Graph 数值用例中的显式 Graph
清理另行通过。本次四卡整模型 Graph 结果只覆盖默认 `decomposed` 模式；native
`padded` 的通过结果来自上述单卡小模型测试。

首次构建中的 `pip check` 被 apt 自带 `devscripts` 的 Debian 版本号阻止。检查现限定为
pip 管理的 runtime 包，不修改 apt 包；该范围的依赖检查通过后才执行 FlashInfer 接口检查。
BuildKit 的 local rootfs exporter 随后停滞并被取消；切换为 Docker archive 和 tar exporter
后均正常结束。`build-attempt2-local-export.log` 和 `rootfs-export.log` 保留对应记录。

完整 HumanEval 质量比较与五次性能实验不属于本次依赖修复。共享 GPU 上的短运行不作为
性能结果；NVFP4 GPU 数值与完整模型执行需要 Blackwell，H200 不计入通过结果。
