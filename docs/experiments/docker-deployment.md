# Docker 部署验证

日期：2026-09-10。基于 `quantization` 分支 `2972589` 的工作区改动。
部署用法见 [Docker deployment](../guides/deployment.md)。

## 构建

`docker/Dockerfile.flux-cu129` 默认目标完整构建通过，安装 FluxServe、
flux-scheduler 和 flux-kernel，并在无 GPU 的构建阶段编译 RMSNorm、RoPE、
activation、MoE 四个原生库。`flux_kernel.precompile --check` 及安装后的包导入检查通过。
此外，构建时编译 TVM FFI 的 CPU/CUDA Torch DLPack 扩展，校验导出 API，并实际将
CPU FP8 E4M3 张量转换为 TVM Tensor；日志记录 `TVM FFI FP8 bridge check: float8_e4m3fn`。

| 项目 | 实际构建配置 |
|---|---|
| 基础环境 | CUDA 12.9.1 / Ubuntu 22.04 / Python 3.12 |
| Torch / Triton | 2.8.0+cu129 / 3.4.0 |
| FlashInfer | 0.6.13，fork revision `6055bba6d7defefe69fe246473647165300b557a` |
| 原生编译架构 | `90a;100a;120` |
| 运行时原生库策略 | `FLUX_KERNEL_REQUIRE_PREBUILT=1`，校验失败直接报错 |

Triton 的 FP8 编解码/分页写入及 FlashInfer 的形状相关 kernel 在目标 GPU
启动预热时编译，编译缓存保存在挂载卷中。这里没有把它们描述为无 GPU 构建阶段
已覆盖所有形状的二进制。

本机 Docker 数据分区剩余约 6.5 GiB，放不下约 22 GiB 的运行环境。
因此使用独立 BuildKit 容器，把构建缓存和导出文件放在 `/data`，未修改全局 Docker 配置。
构建命令：

```bash
docker exec fluxserve-deploy-buildkit buildctl build \
  --frontend dockerfile.v0 \
  --local context=/workspace --local dockerfile=/workspace/docker \
  --opt filename=Dockerfile.flux-cu129 \
  --output type=tar,dest=/artifacts/runtime.tar
```

`/workspace` 是仓库只读挂载，`/artifacts` 对应
`/data/fluxserve-deployment-20260910/artifacts`。`runtime.tar` 是最终运行环境的
文件系统归档，**不是** `docker load` 使用的镜像归档。另已成功导出约 13.6 GiB 的
`artifacts/fluxserve-cu129-fp8.docker.tar`，其标签为 `fluxserve:cu129-fp8`，可在有足够磁盘
的机器执行 `docker load -i fluxserve-cu129-fp8.docker.tar`。已检查归档中的镜像配置，
确认包含 entrypoint、默认参数和 healthcheck。本次没有向默认 Docker daemon
加载或发布镜像。GPU 验证运行这个全新构建导出的 rootfs，未使用宿主机的 Python 包
或可写的仓库源码替代已安装包。

## 自动检查

以下回归命令通过：**94 passed，2 skipped**。

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=python:flux-kernel/python \
  /tmp/fluxserve-pytest-venv/bin/python -m pytest -q \
  test/runtime/test_deployment.py \
  test/runtime/test_engine_executor.py \
  test/runtime/test_distributed_launch.py \
  test/runtime/test_scheduler_defaults.py \
  test/runtime/test_model_runner_cuda_graph_routing.py \
  test/runtime/test_diffusion_gemma_benchmark.py \
  test/runtime/test_kv_quantization.py \
  test/runtime/test_block_diffusion_offline.py
```

部署测试包含预热失败阻止 HTTP 启动、预热前后 RNG/配置/缓存恢复、分页容量限制、
预热与 Graph 准备的调用顺序、只读模型缓存、任意 UID、缓存隔离及入口参数传递。
另覆盖 HTTP 关闭事件完成分布式 worker 清理后才返回，CLI 的兜底清理不重复发送 collective。

额外运行 `test_api_utils.py` 时，原有
`test_startup_banner_uses_color_for_tty` 因期待的 ASCII logo 与当前实现不符而失败。
该测试和 `api_utils.py` 均与 `HEAD` 相同，属于已有问题，未将该次运行计为全部通过。

## GPU 服务运行检查

使用 GPU `4,5,6,7`，TP=4、EP=4，完整 32 层
`thnkinbtfly/llada2.0-flash-fp8` checkpoint，revision
`85bd9f38034aa46f93135676189acec4d7fc40d3`。权重为 ModelOpt FP8，KV 显式设置
`fp8_e4m3`，复用[前一次运行检查](fp8-kv-cache.md)的单样本 `smoke-scales.json`。
这份 scale 不替代正式 GSM8K 前 128 条校准。

测试使用 Docker wrapper/chroot 挂载上述新 rootfs；已安装源码和模型下载缓存只读，
进程以 UID/GID `666388535` 运行，编译缓存和 `/tmp` 单独可写。
该 UID 在镜像内没有 passwd 项。驱动和 GPU 设备由宿主机挂载，Python/CUDA 用户态库
来自本次构建。两种后端串行执行，batch=1、生成上限 64、block=64、prefill limit=128、
max model len=256，未启用 CUDA Graph。

| 检查 | 结果 |
|---|---|
| SDPA / dense FP8 KV 在线服务 | 4 个 rank 完成预热后 HTTP 才可用；镜像 healthcheck 退出码 0；HumanEval 第一条生成返回 HTTP 200 和非空结果 |
| FlashInfer / paged FP8 KV 在线服务 | 公共 FA2 路径；4 个 rank 完成预热后 HTTP 才可用；镜像 healthcheck 退出码 0；HumanEval 第一条生成返回 HTTP 200 和非空结果 |
| 相同缓存的 FlashInfer `--warmup-only` | 4 个 rank 完成预热并退出，退出码 0；3 个 worker 收到 shutdown，无 semaphore 清理警告 |

重启前后检查同一编译缓存中的 **119 个 `.so` / `.cubin` / `.ptx` 文件**：
文件数量、大小、SHA256 和 mtime 全部一致，新增、删除、修改均为 0。
这证明本次相同配置的预热复用了编译产物，不代表所有新形状均免编译。
本次缓存 fingerprint 为 `46468e5449769c14`。

两种后端的 `docker stop` 返回原有 supervisor 约定的 143（SIGTERM），HTTP 关闭事件完成，
3 个 worker 均记录收到 shutdown。supervisor 回收时仍有 PyTorch 的 semaphore 清理警告，
不能把这次 SIGTERM 退出描述为完全无警告的退出。该行为与预热命令的正常退出码分开记录。

## 产物与范围

本机记录目录：`/data/fluxserve-deployment-20260910/`。

- `final-build.log`：最终 Docker 镜像构建和归档导出日志；`runtime-build-with-dlpack.log` 保留 DLPack 编译及 FP8 检查记录。
- `image-archive-info.json`：最终镜像归档的元数据检查；归档旁附 SHA256 文件。
- `artifacts/runtime/opt/fluxserve/build-info.json`：版本、源码摘要、架构和原生库编译标记。
- `cpu-tests.log`：回归结果。
- `dlpack-probe.log`：非 root、只读环境中的 GPU FP8 DLPack 转换通过；没有运行时编译警告。
- `run_smoke.py`：完整部署运行检查脚本；各 case 的命令、日志、HTTP 输出和结果另存 JSON。
- `runtime-cache/`：独立持久编译缓存。
- `cache-before-warmup.json`、`cache-after-warmup.json`、`cache-reuse-result.json`：缓存复用核对。
- `summary.json`：两种服务后端、单次预热退出及缓存复用的汇总。

首次验证发现 COPY 保留源目录权限导致非 root 入口不可读，已通过构建时赋予读取/遍历权限修复。
第二次发现 PyTorch 对没有 passwd 项的 UID 调用 `getpass.getuser()` 失败，入口已补充
默认 `USER`/`LOGNAME`。失败日志保留为 `sdpa-permissions-attempt1-*` 和
`sdpa-uid-attempt2-*`。

服务退出检查还发现 Uvicorn 可能在关闭事件后重新抛出 SIGTERM，使 CLI 的 `finally`
来不及通知其他 rank。已把 executor 清理接入 HTTP 关闭事件，保留 CLI 兜底，并使
分布式关闭幂等，避免重复广播。原有 CUDA Graph 清理改动继续保留。

FlashInfer 的首次只读运行发现 DLPack 扩展默认写入不可写的用户目录，随后回退路径
无法接收 PyTorch 2.8 FP8 张量。已将 CPU/CUDA 扩展纳入无 GPU 镜像构建和校验，
并在入口自动放入持久缓存 `TVM_FFI_CACHE_DIR`。失败记录为 `flashinfer-dlpack-attempt1-*`。

本次只验证部署及短请求运行。共享 GPU 上的启动时间不能用来评估性能；没有运行完整
HumanEval 质量比较、五次性能实验或完整模型 CUDA Graph 验收。
`100a` 和 `120` 架构仅完成交叉编译，未获得对应 GPU 的实机验证。

## Review 后的退出与 Graph 测量修复

2026-09-10，基于 `6d5e183` 修复三处 review 问题：

- 关闭服务时等待 `asyncio.to_thread` 的实际推理线程结束，再清理 worker 和 Graph；
  重复取消等待任务也不能提前释放执行锁。请求释放使用同一执行锁。
- 用所有 rank 的启动结果交换替代仅成功路径的 barrier。rank 0、其他 rank 或多个
  rank 预热失败时，各 rank 协调清理并报告相同错误，CLI 不再追加正常 shutdown 广播。
- FlashInfer Graph 预热前按正式 batch 容量分配 KV，避免 batch 8 / mini-batch 4
  在计时阶段重新分配和 capture；验收拒绝任何 rank 在计时期间新增 capture/invalidation。

CPU 回归结果：**109 passed，2 skipped**（43.55 秒），日志保存在
`/data/fluxserve-review-fixes-20260910/cpu-tests.log`。

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=python:flux-kernel/python \
  /tmp/fluxserve-pytest-venv/bin/python -m pytest -q \
  test/runtime/test_offloaded_shutdown.py \
  test/runtime/test_distributed_startup.py \
  test/runtime/test_benchmark_graph_warmup.py \
  test/runtime/test_deployment.py \
  test/runtime/test_engine_executor.py \
  test/runtime/test_distributed_launch.py \
  test/runtime/test_scheduler_defaults.py \
  test/runtime/test_model_runner_cuda_graph_routing.py \
  test/runtime/test_diffusion_gemma_benchmark.py \
  test/runtime/test_kv_quantization.py \
  test/runtime/test_block_diffusion_offline.py
```

新增测试使用真实线程及双进程 Gloo，覆盖推理成功/异常、重复取消、启动成功及
任意 rank 预热失败、清理 collective 一致性。缓存测试在 CPU 上调用实际 warmup、
generate、分配和失效路径，覆盖 BF16/FP8、batch 8→2→8 的地址复用，并联动指标输出
与验收拒绝逻辑；GPU forward/capture 在该测试中用地址索引记录替代。

这些回归不代表新增的 CUDA/NCCL 或完整模型验证。本轮未重建 Docker 镜像，也未重跑
GPU 服务、质量或性能实验。上文 rootfs 和镜像归档是修复前的产物，部署本轮修复需从
新提交重新构建镜像；此前的 GPU 运行结果不能作为本轮修复的实机验收。

### AnyIO 客户端取消回归

对 `dfceca4` 的后续 review 发现：客户端进入 AnyIO 取消作用域后，`abort()`
等待执行锁时会再次被取消，导致 scheduler 已移除请求，但 `_states`、终止统计和
executor 资源释放未完成。新增测试在修复前复现了残留状态及未执行的 release。

修复后，逻辑终止、状态移除和 abort 输出入队在任何等待之前完成。资源释放交由
引擎持有的独立任务，通过 shield 保护并继续遵守执行锁；服务关闭时等待所有此类
任务结束，释放异常会记录日志。已取消请求的后续推理异常也不会重复计为失败。

本轮相关 CPU 回归：**52 passed**（36.00 秒）。新增 AnyIO 取消作用域与真实推理线程
测试覆盖推理正常返回、推理抛错、同时关闭服务及后台释放抛错。日志：
`/data/fluxserve-cancellation-fix-20260910/cpu-tests.log`。

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=python:flux-kernel/python \
  /tmp/fluxserve-pytest-venv/bin/python -m pytest -q \
  test/runtime/test_request_cancellation.py \
  test/runtime/test_offloaded_shutdown.py \
  test/runtime/test_deployment.py \
  test/runtime/test_engine_executor.py \
  test/runtime/test_distributed_startup.py \
  test/runtime/test_distributed_launch.py \
  test/runtime/test_scheduler_defaults.py
```

本轮同样未重建 Docker 镜像或新增 GPU 实机验收。
