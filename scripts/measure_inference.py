import enum
import math
import logging
from typing import List, Optional

import numpy as np
from PIL import Image
import torch
import clip

from rich.console import Console
from rich.table import Table
import typer

app = typer.Typer()


@app.callback()
def base(verbose: bool = False):
    """
    CLIP Performance
    Get stats about the inference time / throughput for CLIP models
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s (%(threadName)s): %(name)s - %(levelname)s - %(message)s",
    )
    global logger
    logger = logging.getLogger()


class _CLIPModel(str, enum.Enum):
    pass


AVAILABLE_MODELS = clip.available_models()
CLIPModel = _CLIPModel("CLIPModel", {x: x for x in AVAILABLE_MODELS})

ITERATIONS = 100
BATCH_SIZES = [2**x for x in range(6)]


def get_timings_for_batch_size(model, preprocess, device, batch_size):
    def _preprocess():
        imarray = np.random.rand(336, 336, 3) * 255
        pil_image = Image.fromarray(imarray.astype("uint8")).convert("RGB")
        preprocessed = preprocess(pil_image).to(device)
        return preprocessed

    model_input = torch.stack([_preprocess() for _ in range(batch_size)], dim=0)
    # See https://pytorch.org/docs/stable/generated/torch.cuda.Event.html
    starter, ender = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )

    # WARMUP
    logger.info("Warming up")
    for _ in range(5):
        with torch.no_grad():
            _ = model.encode_image(model_input)
    logger.info("Warm-up done.")
    logger.info(f"Memory: {torch.cuda.memory_allocated()}")

    timings = np.zeros((ITERATIONS, 1))
    for rep in range(ITERATIONS):
        with torch.no_grad():
            starter.record()
            _ = model.encode_image(model_input)
            ender.record()

            torch.cuda.synchronize()

            timings[rep] = starter.elapsed_time(ender)
    del model_input
    return timings


def get_stats_table():
    table = Table(title=f"CLIP stats")
    table.add_column("Model", justify="right")
    table.add_column("Batch Size", justify="right")
    table.add_column("Avg. Inference time(ms)", justify="right")
    table.add_column("Images / sec", justify="right")
    return table


def calculate(timings, batch_size):
    total_inference_time = np.sum(timings)
    mean_inference = total_inference_time / ITERATIONS
    throughput = 1e3 * batch_size * ITERATIONS / total_inference_time

    return mean_inference, throughput


def add_stats(table, model_name, batch_size, timings):

    mean, throughput = calculate(timings, batch_size)
    table.add_row(
        model_name if batch_size == BATCH_SIZES[0] else "",
        f"{batch_size}",
        f"{mean:.3f}",
        f"{throughput:.3f}",
        end_section=batch_size == BATCH_SIZES[-1],
    )

    return table


def get_stats_for_model(model_name, device):
    logger.info(f"Loading model '{model_name}'")
    model, preprocess = clip.load(model_name, device)
    logger.info("Done")

    _get_timings_for_batch_size = lambda batch_size: get_timings_for_batch_size(
        model, preprocess, device, batch_size
    )
    BATCH_STATS = {}
    for batch_size in BATCH_SIZES:
        logger.info(f"Recording stats for batch_size: {batch_size}")
        timings = _get_timings_for_batch_size(batch_size)
        BATCH_STATS[batch_size] = timings

    return BATCH_STATS


@app.command()
def run(
    min_batch_size: int = typer.Option(
        1, min=1, help="min batch size to run the experiment with"
    ),
    max_batch_size: int = typer.Option(
        32, min=1, help="max batch size to run the experiment with"
    ),
    iterations: int = typer.Option(
        100, min=1, help="Number of iterations to run the models for"
    ),
    models: Optional[List[CLIPModel]] = typer.Argument(
        None,
        show_default=False,
        help="Models to run the experiment for (uses all models if not provided)",
    ),
):
    if not models:
        models = list(CLIPModel)

    global BATCH_SIZES, ITERATIONS
    ITERATIONS = iterations
    if min_batch_size > max_batch_size:
        raise typer.BadParameter(
            f"min batch size ({min_batch_size}) must be less than max batch size ({max_batch_size})"
        )
    elif min_batch_size == max_batch_size:
        BATCH_SIZES = (min_batch_size,)
    else:
        _second = math.ceil(math.log(1 + min_batch_size, 2))
        _last = math.ceil(math.log(max_batch_size, 2))

        BATCH_SIZES = (
            (min_batch_size,)
            + tuple(2**x for x in range(_second, _last))
            + (max_batch_size,)
        )
    logger.info(
        f"PARAMETERS: \nMODELS: {[x.value for x in models]}\nBATCH SIZES: {BATCH_SIZES}\nITERATIONS: {ITERATIONS}"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    stats_table = get_stats_table()
    try:
        for model in models:
            model_stats = get_stats_for_model(model.value, device)

            for batch_size in model_stats:
                add_stats(stats_table, model.value, batch_size, model_stats[batch_size])

    finally:
        Console().print(stats_table)


if __name__ == "__main__":

    app()
