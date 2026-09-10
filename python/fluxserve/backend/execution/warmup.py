"""Representative eager warmup before the HTTP startup event completes."""

import logging
import math
import time

import torch


logger = logging.getLogger(__name__)


def warmup_shapes(runner):
    block = int(runner.block_length)
    limit = int(runner.max_length)
    if limit < block:
        raise ValueError("max_model_len must fit at least one diffusion block")
    max_batch = min(
        int(runner.runner_config.mini_batch_size),
        int(runner.server_args.max_num_seqs),
    )
    if max_batch < 1:
        raise ValueError("Startup warmup requires a positive batch size")
    if getattr(runner, "requires_prompt_lengths", False):
        prompt = 1
    else:
        # Include complete prompt blocks and, when capacity permits, a partial
        # prompt block. Even the smallest configuration still exercises decode.
        prompt = min(max(block, int(runner.prefilling_limit)), limit - block)
        prompt = max(0, prompt // block * block)
        if prompt + block + 1 <= limit:
            prompt += 1
    return [(batch, prompt) for batch in sorted({1, max_batch})]


def prepare_scheduler_warmup_cache(runner, batch, prompt, gen_length):
    """Use the final pool geometry so FP8 scatter compiles for serving capacity."""
    pages = int(runner.server_args.scheduler_num_device_pages)
    page_size = int(runner.runner_config.page_size)
    # Page zero is reserved by the scheduler. Limit the synthetic batch/context
    # to what the configured pool can accommodate, just as admission would.
    if pages < 2:
        raise ValueError("Paged startup warmup requires at least two device pages")
    prefix_pages = min(prompt // page_size, pages - 2)
    prompt = min(prompt, prefix_pages * page_size)
    per_request = math.ceil((prompt + gen_length) / page_size)
    batch = min(batch, (pages - 1) // per_request)
    if batch < 1:
        raise ValueError("Paged startup warmup cannot fit one diffusion block")
    runner.ensure_paged_kv_cache(num_device_pages=pages)
    cache = runner.past_key_values
    cache.set_page_tables(
        list(range(batch)),
        [list(range(1 + row * per_request, 1 + (row + 1) * per_request)) for row in range(batch)],
    )
    return cache, batch, prompt


@torch.no_grad()
def warmup_online_runner(runner):
    shapes = warmup_shapes(runner)
    config = runner.runner_config
    saved = {name: getattr(runner, name) for name in (
        "early_stop", "enable_cuda_graph", "num_forwards", "max_length",
    ) if hasattr(runner, name)}
    old_gen_length = config.gen_length
    old_cache = getattr(runner, "past_key_values", None)
    fi_graph = getattr(runner, "flashinfer_graph_runner", None)
    has_fi_graph = hasattr(runner, "flashinfer_graph_runner")
    device = torch.device(runner.device)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    eos_ids = getattr(runner.decoder, "eos_ids", (runner.decoder.eos_id,))
    token = next(i for i in range(32) if i != runner.decoder.mask_id and i not in eos_ids)
    started = time.perf_counter()
    logger.info(
        "Startup eager warmup: backend=%s kv_dtype=%s scales=%s shapes=%s; HTTP is not ready",
        config.attention_backend, getattr(runner, "kv_cache_dtype", "bf16"),
        getattr(getattr(runner, "kv_quantization", None), "source", "none"), shapes,
    )
    try:
        runner.early_stop = False
        runner.enable_cuda_graph = False
        if has_fi_graph:
            runner.flashinfer_graph_runner = None
        with torch.random.fork_rng(devices=devices):
            for batch, prompt in shapes:
                config.gen_length = min(int(runner.block_length), int(runner.max_length) - prompt)
                paged_cache = None
                if (
                    getattr(runner.server_args, "scheduler_policy", "default") == "paged"
                    and hasattr(runner, "ensure_paged_kv_cache")
                ):
                    paged_cache, batch, prompt = prepare_scheduler_warmup_cache(
                        runner, batch, prompt, config.gen_length,
                    )
                logger.info(
                    "Startup warmup batch=%d prompt_tokens=%d gen_tokens=%d scheduler_pool=%s",
                    batch, prompt, config.gen_length, paged_cache is not None,
                )
                inputs = torch.full((batch, prompt), token, dtype=torch.long, device=device)
                if getattr(runner, "requires_prompt_lengths", False):
                    runner.generate(inputs, prompt_lengths=[prompt] * batch)
                elif paged_cache is None:
                    runner.generate(inputs)
                else:
                    runner.generate(inputs, kv_cache=paged_cache)
                    # No synthetic pages or request data enter the live scheduler.
                    paged_cache.data.zero_()
                    paged_cache.page_table.zero_()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
    finally:
        config.gen_length = old_gen_length
        for name, value in saved.items():
            setattr(runner, name, value)
        runner.past_key_values = old_cache
        if has_fi_graph:
            runner.flashinfer_graph_runner = fi_graph
    logger.info("Startup eager warmup complete in %.2fs", time.perf_counter() - started)
    return {"warmup_seconds": time.perf_counter() - started}
