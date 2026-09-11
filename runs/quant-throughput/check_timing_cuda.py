import json
import os
from pathlib import Path
import tempfile
import torch
import torch.multiprocessing as mp
from fluxserve.bench_offline import _start_generation_timing, _finish_generation_timing


def worker(rank, rendezvous, output):
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group("nccl", init_method=rendezvous, rank=rank, world_size=2)
    try:
        # Prime NCCL before timing, just as benchmark warmup does.
        value = torch.ones(1, device=f"cuda:{rank}")
        torch.distributed.all_reduce(value)
        start_event, stop_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start = _start_generation_timing(f"cuda:{rank}")
        start_event.record()
        torch.cuda._sleep(100_000_000 * (rank + 1))
        stop_event.record()
        local, maximum = _finish_generation_timing(start, f"cuda:{rank}")
        gpu_seconds = start_event.elapsed_time(stop_event) / 1000
        assert maximum >= local >= gpu_seconds * 0.98
        times = [None, None]
        torch.distributed.all_gather_object(times, {"rank": rank, "local_seconds": local,
            "maximum_seconds": maximum, "cuda_event_seconds": gpu_seconds})
        assert times[0]["maximum_seconds"] == times[1]["maximum_seconds"]
        assert maximum == max(row["local_seconds"] for row in times)
        if rank == 0:
            Path(output).write_text(json.dumps({"status": "passed", "ranks": times}, indent=2) + "\n")
    finally:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="flux-timing-") as folder:
        mp.spawn(worker, args=(f"file://{folder}/rdzv", str(Path(__file__).parent / "timing-cuda.json")), nprocs=2)
