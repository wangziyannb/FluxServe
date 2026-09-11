"""Render final measured results; no inference or GPU work."""
import json
from datetime import datetime
from pathlib import Path

OUT=Path(__file__).resolve().parent
summary=json.loads((OUT/'summary.json').read_text())['groups']
ratios=json.loads((OUT/'comparisons.json').read_text())
audit=json.loads((OUT/'completion-audit.json').read_text())
assert len(summary)==10 and all(r['status']=='passed' for r in summary)
def find(**kw):return next(r for r in summary if all(r[k]==v for k,v in kw.items()))
def label(r):return f"{r['weights'].upper()} / {'FP8' if r['kv']=='fp8_e4m3' else 'BF16'} / {r['attention_compute'].upper()}"
lines=[
'# FA3 / FA2、权重与 KV / attention 精度：最终结果',
'',
'2026-09-11 17:29 UTC 完成。10 个正式配置全部通过验收，每配置 GSM8K test 全量 1,319 题一次，',
'共 13,190 个正式答案、3,725,522 个有效 completion tokens；另有每配置 16 题试跑。正式队列约 7 小时 25 分钟。',
'',
'统一负载：4-shot、seed 42、原始数据顺序、global batch 8、mini-batch 4、gen-len 2048、',
'block/page 64、threshold 0.95、low-threshold 0.3、TP=EP=卡数、DP=PP=1。',
'只使用 UltraChat 512 条校准。FP8 与 BF16 权重分别使用匹配本模型的 KV scale；没有使用 GSM8K 做 KV 校准。',
'表中的 attention 精度指 QK/PV Tensor Core 输入，FP8 路径的输出仍为 BF16，softmax/累加保留更高精度。',
'',
'## 吞吐与 accuracy',
'',
'| 权重 / KV / Attention | 后端 | 卡数 | eager TPS | eager accuracy | Graph TPS | Graph accuracy |',
'|---|---|---:|---:|---:|---:|---:|',
]
for r in summary:
 if r['graph']:continue
 g=find(weights=r['weights'],kv=r['kv'],attention_compute=r['attention_compute'],kernel_backend=r['kernel_backend'],gpus=r['gpus'],graph=True)
 lines.append(f"| {label(r)} | {r['kernel_backend'].upper()} | {r['gpus']} | {r['tps']:.2f} | {100*r['accuracy']:.2f}% ({r['correct']}/1319) | {g['tps']:.2f} | {100*g['accuracy']:.2f}% ({g['correct']}/1319) |")
lines += ['',
'## 回答实验问题',
'',
'- **同为 FA3、四卡，全 FP8 / 全 BF16**：eager 0.700×（−30.0%），Graph 1.455×（+45.5%）。本轮优势主要在 Graph 模式。',
'- **权重/KV/attention 均 BF16、四卡，FA3 / FA2**：eager 1.048×（+4.8%），Graph 1.069×（+6.9%）。',
'  两者使用相同分块、页表和 KV 写入适配器；仍有数值与生成行为差异，不能把 TPS 比值直接当作算子加速比。',
'- **BF16 权重下，FA3 FP8 KV+attention / BF16 KV+attention**：eager 0.881×（−11.9%），Graph 1.014×（+1.4%）。',
'  这同时改变 KV 存储和 attention 计算精度，不是只量化 KV 的实验。',
'- **FA3 FP8 KV+attention 下，FP8 / BF16 权重**：eager 0.795×，Graph 1.435×。两种权重使用各自匹配的 UltraChat scale。',
'- **全 FP8 两卡到四卡**：eager 0.906×，Graph 1.208×；按加速比除以 2 的扩展效率分别为 45.3%、60.4%。',
]
two=find(weights='fp8',gpus=2,graph=True);bf=find(weights='bf16',kv='bf16',kernel_backend='fa3',graph=True)
lines += [f"- **两卡全 FP8 Graph / 四卡全 BF16 FA3 Graph**：{two['tps']/bf['tps']:.3f}×（+{100*(two['tps']/bf['tps']-1):.1f}%），两者 accuracy 均为 1270/1319；单卡平均吞吐比值为 {2*two['tps']/bf['tps']:.3f}×。",'',
'## 生成行为与时间',
'',
'所有 TPS 都是有效 completion tokens / 各 batch 的最大 rank 同步生成耗时之和。',
'耗时包含请求 prefill 和 diffusion decode，排除加载、warmup、JIT、Graph capture、写盘和评分。',
'显存为所有 rank 中最大的设备使用峰值，包含加载/warmup/capture，不仅是 KV cache。',
'',
'| 权重 / KV / Attention | 后端 | 卡数 | 模式 | Tokens | NFE | Tokens/NFE | 生成秒数 | 前向/秒 | 单卡 TPS | 显存峰值 GiB/卡 |',
'|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|']
for r in summary:
 lines.append(f"| {label(r)} | {r['kernel_backend'].upper()} | {r['gpus']} | {'Graph' if r['graph'] else 'eager'} | {r['generated_tokens']:,} | {r['nfe']:,} | {r['tokens_per_forward']:.3f} | {r['generation_seconds']:.2f} | {r['forwards_per_second']:.2f} | {r['per_gpu_tps']:.2f} | {r['peak_device_gib']:.2f} |")
lines += ['',
'FA3 / FA2 的 BF16 eager 对照中，FA3 的 NFE 少 9.0%，前向次数/秒反而低 2.4%；',
'Graph 对照中，FA3 的 NFE 少 3.2%，前向次数/秒高 4.5%。这些都是整模型前向指标，不能替代 attention 算子计时。',
'四卡 FA3 全 FP8 / 全 BF16 的 Graph 对照中，tokens 多 4.6%、NFE 多 16.6%、前向次数/秒高 62.1%，最终 TPS 高 45.5%。',
'',
'## Graph 与数值差异',
'',
'在本轮 batch 8 / mini-batch 4 的完整负载中，五个 Graph 配置的 prefill 均为 eager（每 rank 330 次），',
'prefill Graph replay 为 0；decode 则在各 rank 实际 replay。Graph 名称表示启用 Graph 的配置，不代表全部阶段均被捕获。',
'所有配置的计时区间内新增 capture/invalidation 均为 0。Graph decode 支持 batch 1、2、4，3 条请求拆为 2+1。',
'',
'| 权重 / KV / Attention | 后端 | 卡数 | Graph/eager TPS | 不同 token 序列题数 | 不同 completion 长度题数 |',
'|---|---|---:|---:|---:|---:|']
for r in summary:
 if not r['graph']:continue
 comparison=next(x for x in ratios if x['comparison']=='graph-versus-eager' and x['numerator']==r['name'])
 lines.append(f"| {label(r)} | {r['kernel_backend'].upper()} | {r['gpus']} | {comparison['tps_ratio']:.3f}× | {comparison['different_token_sequences']} | {comparison['different_completion_lengths']} |")
lines += ['',
'本轮每配置只测一遍，没有多次运行的方差或 MAD；batch 未扫描，不能视为各配置峰值吞吐。',
'Accuracy 在 95.83%–96.36% 之间，仅是本次完整 GSM8K 样本的观察，不能据此宣称质量等价。',
'允许 eager/Graph 和各精度输出不同；逐题答案、token IDs、EOS 情况、评分和分 batch NFE/计时均保留。',
'旧版本全 BF16 eager 使用 Hopper block-extend FA3，旧 Graph 为 eager FA3 prefill + FA2 decode；',
'本轮 FA2 对照为显式固定后端并使用与本轮 FA3 相同的适配器，因此不能混淆新旧基线。',
'',
'## 校准与验收',
'',
'FP8 模型 UltraChat scale SHA256：`c14d91c079e88fd57ffe0296264a9a4ed802f2694dacff40d7b0bc898c2f2dfb`。',
'BF16 模型 UltraChat scale SHA256：`30930c4445821d435cbc850d21587021c8235f25204335a6d3d2e6e5d31bc419`。',
'两份 scale 均覆盖 32 层，有限且为正，并校验模型身份与校准输入哈希。运行源码/输入哈希在整个队列中保持不变。',
'实际 wrapper 的后端/Q/KV dtype 在各 rank 中记录并验收；独立进程串行运行，0.5 秒采样设备状态并验证独占与清理。',
'运行均成功退出，没有失败的试跑/正式配置；最终四张 GPU 均无实验进程、显存约 1 MiB。83 项相关测试通过。',
'',
'- [配置与执行说明](README.md)',
'- [汇总 CSV](summary.csv)、[汇总 JSON](summary.json)、[完整比较 JSON](comparisons.json)',
'- [完成验收](completion-audit.json)、[代码/输入身份](manifest.json)、[环境](environment.json)',
'- `measurements/<配置>/`：逐题答案与评分、token IDs、各 rank 显存/Graph/实际后端、命令/日志/遥测。',
'- `source-snapshot.tar.gz`：运行时源码快照（包括当时未提交的源码）。',
]
(OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
print('Wrote',OUT/'RESULTS.md')
print('Graph 2GPU FP8 / 4GPU BF16 FA3:',two['tps']/bf['tps'])
