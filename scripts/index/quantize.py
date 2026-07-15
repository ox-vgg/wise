import logging
import math
from pathlib import Path

import numpy as np

import faiss
import faiss.contrib.ivf_tools
import faiss.contrib.inspect_tools

logger = logging.getLogger(__name__)

DEFAULT_MIN_AGREEMENT = 0.95
DEFAULT_SELF_QUERIES = 1_000
AGREEMENT_K = 10

# Vectors per add_preassigned call while copying lists
_COPY_BATCH = 65_536

# IVF-SQ8 scanners exist only for these; anything else aborts at search time.
SUPPORTED_METRICS = (faiss.METRIC_L2, faiss.METRIC_INNER_PRODUCT)


class ConversionError(Exception):
    """Catch all error for conversion of IVF index"""

def load_queries(path: Path, d: int) -> np.ndarray:
    """External query matrix: non-empty float32 .npy of shape (n, d) —
    anything else is refused."""
    try:
        q = np.load(path)
    except Exception as e:
        raise ConversionError(f"cannot load queries from {path}: {e}") from e
    if not isinstance(q, np.ndarray):
        raise ConversionError(f"queries must be a float32 array, got {type(q).__name__}: {path}")
    if q.dtype != np.float32:
        raise ConversionError(f"queries must be float32, got dtype {q.dtype}: {path}")
    if q.ndim != 2 or q.shape[1] != d or q.shape[0] == 0:
        raise ConversionError(
            f"queries shape {q.shape} does not match (n > 0, d={d}): {path}"
        )
    if not np.isfinite(q).all():
        # NaN rows return only -1 fillers and would be silently excluded,
        # gating agreement on whatever finite remainder is left
        raise ConversionError(f"queries contain non-finite values (NaN/Inf): {path}")
    return np.ascontiguousarray(q)

def load_ivf_index(path: Path) -> faiss.IndexIVFFlat:
    """Open memory-mapped read-only; only a top-level, non-empty, L2/IP
    IndexIVFFlat is accepted."""
    
    # load the index in read only, memory mapped mode
    try:
        index = faiss.read_index(
            str(path), faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY
        )
    except RuntimeError as e:
        raise ConversionError(f"cannot read source index {path}: {e}") from e
    
    # support only IVF Index for now
    if type(index) is not faiss.IndexIVFFlat:
        raise ConversionError(
            f"source must be a top-level IndexIVFFlat, got {type(index).__name__}"
        )
    
    if index.metric_type not in SUPPORTED_METRICS:
        raise ConversionError(
            f"source metric_type={index.metric_type} is not supported: "
            f"only {SUPPORTED_METRICS} are supported"
        )
    
    # cannot convert empty index
    if index.ntotal == 0:
        raise ConversionError(f"source index is empty (ntotal == 0): {path}")
    
    # nprobe must at least be 1 for an IVF index
    if index.nprobe < 1:
        # inherited by the twin, where it would make every search raise
        raise ConversionError(f"source has invalid nprobe={index.nprobe}: {path}")
    
    return index

def get_assigned_vectors(invlists, list_no: int, d: int) -> tuple[np.ndarray, np.ndarray]:
    """One inverted list of an IVFFlat as (ids, float32 vectors)."""
    ids, codes = faiss.contrib.inspect_tools.get_invlist(invlists, list_no)
    return ids, codes.view(np.float32).reshape(-1, d)

def sample_vectors(src, n: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform sample without replacement of vectors reconstructed from src's
    inverted lists via reconstruct_from_offset — position-keyed, so no direct
    map is needed (the source may be memory-mapped), codec-general, and only
    the sampled rows are ever materialized, never a whole list."""
    offsets = np.concatenate(
        [[0], np.cumsum(faiss.contrib.ivf_tools.get_invlist_sizes(src.invlists))]
    )
    ntotal = int(offsets[-1])
    if n >= ntotal:
        positions = np.arange(ntotal)
    else:
        positions = np.sort(rng.choice(ntotal, size=n, replace=False))
    
    list_nos = np.searchsorted(offsets, positions, side="right") - 1
    local_offsets = positions - offsets[list_nos]
    
    out = np.empty((len(positions), src.d), dtype=np.float32)
    for row, list_no, offset in zip(out, list_nos, local_offsets):
        src.reconstruct_from_offset(int(list_no), int(offset), faiss.swig_ptr(row))
    return out

def compute_running_minmax(index: faiss.IndexIVFFlat, residual: bool = True) -> np.ndarray:
    """Per-dimension [vmin; vmax] over every stored vector, optionally subtracting the coarse centroid for residuals.

    Raises ConversionError if any dimension has non-finite values (NaN/Inf).
    """
    d = index.d
    vmin = np.full(d, np.inf, dtype=np.float32)
    vmax = np.full(d, -np.inf, dtype=np.float32)

    centroids = index.quantizer.reconstruct_n(0, index.nlist) if residual else None
    sizes = faiss.contrib.ivf_tools.get_invlist_sizes(index.invlists)
    
    nonempty = np.flatnonzero(sizes)
    if nonempty.size == 0:
        # without this, the ±inf accumulators would masquerade as NaN input
        raise ConversionError("source index has no stored vectors (ntotal == 0)")
    
    for list_no in nonempty:
        _, vecs = get_assigned_vectors(index.invlists, int(list_no), d)
        if centroids is not None:
            # get_invlist hands back a private copy, so subtract in place
            np.subtract(vecs, centroids[list_no], out=vecs)
        np.minimum(vmin, vecs.min(axis=0), out=vmin)
        np.maximum(vmax, vecs.max(axis=0), out=vmax)
    
    # NaN/Inf anywhere in the source propagates into the accumulator, so one
    # O(d) check detects a poisoned source (garbage ranges) in full.
    bad = np.flatnonzero(~(np.isfinite(vmin) & np.isfinite(vmax)))
    if bad.size:
        raise ConversionError(
            "source contains non-finite values (NaN/Inf); SQ ranges would be "
            f"meaningless in dimension(s) {bad[:16].tolist()}"
            + (f" + {bad.size - 16} more" if bad.size > 16 else "")
        )
    
    return np.stack([vmin, vmax])

def copy_invlists_and_vectors(src: faiss.IndexIVFFlat, dst: faiss.IndexIVF, list_nos: list[int]) -> None:
    """Copy a batch of inverted lists verbatim, preserving ids and assignments."""

    d = src.d
    ids, vecs, assignments = [], [], []
    for list_no in list_nos:
        i, v = get_assigned_vectors(src.invlists, list_no, d)
        ids.append(i)
        vecs.append(v)
        assignments.append(np.full(len(i), list_no, dtype=np.int64))

    all_vecs = np.concatenate(vecs)
    all_assignments = np.concatenate(assignments)
    all_ids = np.concatenate(ids)
    faiss.contrib.ivf_tools.add_preassigned(dst, all_vecs, all_assignments, all_ids)

def convert_ivf_index(
    src: faiss.IndexIVFFlat,
    train_vectors: np.ndarray | None = None,
    residual: bool = True
) -> faiss.IndexIVFScalarQuantizer:
    """SQ8 twin of an IVFFlat: cloned centroids, SQ ranges fit exactly (the
    default) or to train_vectors, lists copied with precomputed assignments,
    inherited metric/nprobe/direct-map. The cloned coarse quantizer is already
    trained, so only the scalar-quantizer ranges are fit — no clustering.

    train_vectors=None fits ranges by exact_ranges and injects them by
    training the SQ on the 2-row [vmin; vmax] matrix — bit-identical to
    native training on the same vectors, without materializing them (native
    train() silently subsamples above 100k vectors, so only the exact path
    actually sees every vector). Passing train_vectors keeps faiss-native
    training"""
    d, nlist = src.d, src.nlist
    if train_vectors is not None:
        if (
            not isinstance(train_vectors, np.ndarray)
            or train_vectors.dtype != np.float32
            or train_vectors.ndim != 2
            or train_vectors.shape[1] != d
            or len(train_vectors) == 0
        ):
            raise ConversionError(
                f"train_vectors must be a non-empty float32 matrix of shape (n, {d})"
            )
        if not np.isfinite(train_vectors).all():
            # faiss-native training would silently skip NaN and fit garbage
            raise ConversionError(
                "train_vectors contain non-finite values (NaN/Inf)"
            )

    # copy the centroids
    quantizer = faiss.clone_index(src.quantizer)
    dst = faiss.IndexIVFScalarQuantizer(
        quantizer, d, nlist, faiss.ScalarQuantizer.QT_8bit, src.metric_type
    )
    dst.by_residual = residual  # before range fit: ranges are fit to residuals

    # if train vectors are provided, use them; otherwise compute the exact min/max over all stored vectors
    if train_vectors is None:
        dst.sq.train(compute_running_minmax(src, residual))
        dst.is_trained = True
    else:
        # faiss returns 0 for "no cap" on index types without a limit
        cap = dst.train_encoder_num_vectors()
        if 0 < cap < len(train_vectors):
            logger.warning(
                "faiss subsamples native SQ training to %d of the %d provided "
                "vectors; the default exact fit sees every stored vector",
                cap,
                len(train_vectors),
            )
        dst.train(train_vectors)  # ty: ignore[missing-argument]
    
    sizes = faiss.contrib.ivf_tools.get_invlist_sizes(src.invlists)
    
    batch: list[int] = []
    batched = 0
    for list_no in np.flatnonzero(sizes):
        batch.append(int(list_no))
        batched += int(sizes[list_no])
        if batched >= _COPY_BATCH:
            copy_invlists_and_vectors(src, dst, batch)
            batch, batched = [], 0
    if batch:
        copy_invlists_and_vectors(src, dst, batch)
    
    if src.direct_map.type != faiss.DirectMap.NoMap:
        dst.set_direct_map_type(src.direct_map.type)
    dst.nprobe = src.nprobe
    
    return dst

def agreement_at_k(
    candidate, reference, queries: np.ndarray, k: int = AGREEMENT_K
) -> tuple[float, int]:
    """Agreement@K: mean per-query |top-K(candidate) ∩ top-K(reference)| / K — 
    consistency is measured against the reference, not accuracy.

    Ids are compared as sets (faiss permits duplicate ids; they collapse).
    When either side returns fewer than K results (-1 fillers: tiny index,
    sparse probed lists) the row is scored against the larger returned set,
    so a bit-perfect twin scores 1.0 while extra junk against a short
    reference is still penalized. Rows where both sides return nothing carry
    no signal and are excluded. 
    
    Returns (agreement, rows scored)."""

    _, cand = candidate.search(queries, k)
    _, ref = reference.search(queries, k)
    scores = []
    for c, r in zip(cand, ref):
        c_ids = np.unique(c[c != -1])  # only the -1 filler is "missing";
        r_ids = np.unique(r[r != -1])  # genuine negative ids are kept
        denominator = max(len(c_ids), len(r_ids))
        if denominator == 0:
            continue
        scores.append(np.intersect1d(c_ids, r_ids).size / denominator)
    if not scores:
        raise ConversionError("neither index returned results for any query")
    return float(np.mean(scores)), len(scores)

def human_readable_bytes(n: int) -> str:
    """Convert a byte count to a human-readable string with binary prefixes."""
    if n < 0:
        raise ValueError(f"byte count must be non-negative, got {n}")
    
    out_n = n
    for unit in ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]:
        if out_n < 1024:
            return f"{out_n:.2f} {unit}"
        out_n /= 1024

    return f"{out_n:.2f} EiB"

def convert_index(
    src_path: Path,
    dst_path: Path,
    train_sample: int | None = None,
    residual: bool = True,
    queries: Path | None = None,
    skip_check: bool = False,
    seed: int = 0,
):
    """Convert src → dst on disk, then measure Agreement@10 against the source
    at its own nprobe (unless skip_check). SQ ranges are fit exactly by
    default; train_sample opts back into the seeded sampled fit. The artifact is
    written even when agreement is low — callers judge report.agreement
    against their threshold (DEFAULT_MIN_AGREEMENT) and decide the exit code."""

    src_path, dst_path = Path(src_path), Path(dst_path)
    if dst_path.exists():
        raise ConversionError(f"output path already exists, refusing: {dst_path}")
    
    if not dst_path.parent.is_dir():
        raise ConversionError(f"output directory does not exist: {dst_path.parent}")
    
    if train_sample is not None and train_sample <= 0:
        raise ConversionError(f"train_sample must be positive, got {train_sample}")
    
    if seed < 0:
        raise ConversionError(f"seed must be non-negative, got {seed}")
    
    if queries is not None and skip_check:
        raise ConversionError(
            "queries were provided but skip_check is set — refusing to silently "
            "skip validation against an explicitly requested query set"
        )

    logger.info("Loading source IVF index from %s", src_path)
    src = load_ivf_index(src_path)

    # Validate external queries before the (expensive) conversion.
    q = None
    if queries is not None:
        if not queries.is_file():
            raise ConversionError(f"queries path is not a file: {queries}")
        logger.info("Loading queries from %s", queries)
        q = load_queries(queries, src.d)

    train_vectors = None
    range_fit, n_train = "exact", int(src.ntotal)
    if train_sample is not None:
        logger.info("Sampling %d vectors for SQ range fitting", train_sample)
        rng = np.random.default_rng(seed)
        train_vectors = sample_vectors(src, train_sample, rng)
        range_fit, n_train = "sampled", len(train_vectors)

    logger.info(
        "Converting IVF index to SQ8 with residual=%s (using %s vectors)",
        residual,
        "all" if range_fit == "exact" else f"{n_train} sampled",
    )
    dst = convert_ivf_index(src, train_vectors, residual=residual)

    # Measure agreement on the in-memory twin BEFORE writing, so every
    # ConversionError this function can raise leaves nothing on disk; a low
    # agreement value never blocks the write — callers judge it afterwards.
    agreement = None
    n_queries = 0
    if not skip_check:
        if q is None:
            # +1: self-queries decorrelated from the SQ training sample
            logger.info("Sampling %d self-queries", DEFAULT_SELF_QUERIES)
            rng_q = np.random.default_rng(seed + 1)
            q = sample_vectors(src, DEFAULT_SELF_QUERIES, rng_q)

        logger.info(
            "Checking agreement@%d against original index on %d queries",
            AGREEMENT_K,
            len(q),
        )
        agreement, n_queries = agreement_at_k(dst, src, q)

    # Write to a sibling temp file and rename: a crash mid-write shouldn't
    # leave a truncated artifact at dst_path.
    tmp_path = dst_path.with_name(dst_path.name + ".partial")
    logger.info("Writing converted SQ8 index to %s", dst_path)
    try:
        faiss.write_index(dst, str(tmp_path))
        tmp_path.replace(dst_path)
    except RuntimeError as e:
        tmp_path.unlink(missing_ok=True)  # e.g. truncated file after disk-full
        raise ConversionError(f"cannot write output index {dst_path}: {e}") from e

    src_size = src_path.stat().st_size
    dst_size = dst_path.stat().st_size
    size_factor = src_size / dst_size if dst_size > 0 else float("inf")
    logger.info(
        "Wrote %s: %s (vs %s, %.2fx), nprobe=%d, range_fit=%s, n_train=%d, n_queries=%d, agreement@%d=%.4f.",
        dst_path,
        human_readable_bytes(dst_size),
        human_readable_bytes(src_size),
        size_factor,
        src.nprobe,
        range_fit,
        n_train,
        n_queries,
        AGREEMENT_K,
        agreement if agreement is not None else float("nan"),
    )


def main():
    import tempfile
    import typer
    import faiss.contrib.datasets
    import faiss.contrib.evaluation

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s (%(threadName)s): %(name)s - %(levelname)s - %(message)s",
    )
    app = typer.Typer()

    def get_nlist(nb: int) -> int:
        return 10 * int(math.floor(math.sqrt(nb)))

    def get_dummy_dataset(
        d: int = 128, nb: int = 1_000_000, nq: int = 1_000, seed: int = 42
    ):

        nlist = get_nlist(nb)
        nt = min(nlist * 50, nb)
        logger.info(
            "Creating synthetic dataset: d=%d, nt=%d, nb=%d, nq=%d", d, nt, nb, nq
        )
        ds = faiss.contrib.datasets.SyntheticDataset(
            d, nt, nb, nq, metric="IP", seed=seed
        )
        return ds

    def build_index(ds):

        nlist = get_nlist(ds.nb)
        quantizer = faiss.IndexFlatIP(ds.d)
        index = faiss.IndexIVFFlat(quantizer, ds.d, nlist, faiss.METRIC_INNER_PRODUCT)

        xt = ds.get_train()
        logger.info("Training IVF index - %d clusters on %d vectors", nlist, len(xt))
        index.train(xt)

        xb = ds.get_database()
        logger.info("Adding %d vectors to IVF index", len(xb))
        index.add(xb)

        return index

    def eval(index, gt, queries, top_k=10):
        # check against groundtruth recall
        _, Iref = index.search(queries, top_k)
        recall = faiss.contrib.evaluation.knn_intersection_measure(Iref, gt)
        return recall

    def _write_index(index, path: Path):
        logger.info("Writing IVF index to %s", path)
        faiss.write_index(index, str(path))
        logger.info(
            "IVF index written to %s: %s",
            path,
            human_readable_bytes(path.stat().st_size),
        )

    @app.command()
    def self_test(
        d: int = typer.Option(64, help="Dimensionality of the dummy vectors"),
        nb: int = typer.Option(
            1_000_000, help="Number of vectors to add to the dummy index"
        ),
        nq: int = typer.Option(1_000, help="Number of queries to test agreement"),
        nprobe: int = typer.Option(32, help="Default nprobe for the dummy index"),
        seed: int = typer.Option(42, help="Random seed for reproducibility"),
    ):
        # create a fake index and convert it to SQ8, then check agreement

        ds = get_dummy_dataset(d, nb, nq, seed=seed)
        index = build_index(ds)
        index.nprobe = nprobe

        with tempfile.TemporaryDirectory() as tmpdir_path:
            index_path = Path(tmpdir_path) / "dummy_ivf.index"
            output_path = Path(tmpdir_path) / "dummy_sq8.index"
            queries_path = Path(tmpdir_path) / "dummy_queries.npy"

            queries = ds.get_queries()
            with open(queries_path, "wb") as f:
                np.save(f, queries)

            _write_index(index, index_path)

            logger.info("Converting IVF index to SQ8 with residual=%s", True)
            convert_index(index_path, output_path, queries=queries_path, seed=seed)

            index = load_ivf_index(index_path)
            converted_index = faiss.read_index(
                str(output_path), faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY
            )

            # check against groundtruth recall
            logger.info(
                "Comparing %d-recall@%d on %d queries", AGREEMENT_K, AGREEMENT_K, nq
            )
            gt = ds.get_groundtruth(AGREEMENT_K)
            recall = eval(index, gt, queries, top_k=AGREEMENT_K)
            recall_converted = eval(converted_index, gt, queries, top_k=AGREEMENT_K)

            logger.info(
                "%d-recall@%d src=%.4f, converted=%.4f",
                AGREEMENT_K,
                AGREEMENT_K,
                recall,
                recall_converted,
            )

    @app.command()
    def write_dummy_index(
        path: Path = typer.Argument(..., help="Path to write a dummy IVF index"),
        d: int = typer.Option(128, help="Dimensionality of the dummy vectors"),
        nb: int = typer.Option(
            1_000_000, help="Number of vectors to add to the dummy index"
        ),
        nprobe: int = typer.Option(32, help="Default nprobe for the dummy index"),
    ):
        """Write a small dummy IVF index for testing."""
        nq = 1_000
        ds = get_dummy_dataset(d, nb, nq=nq)
        index = build_index(ds)
        index.nprobe = nprobe
        _write_index(index, path)

    @app.command()
    def convert(
        src_path: Path = typer.Argument(..., help="Path to the source IVF index"),
        dst_path: Path = typer.Argument(..., help="Path to write the converted SQ8 index"),
        train_sample: int | None = typer.Option(
            None,
            help="Number of vectors to sample for SQ range fitting; default is exact fit",
        ),
        residual: bool = typer.Option(
            True,
            help="Whether to convert to residual SQ8 (default) or non-residual",
        ),
        queries: Path | None = typer.Option(
            None,
            help="Path to a .npy file of queries for agreement checking; if not provided, self-queries are used",
        ),
        skip_check: bool = typer.Option(
            False,
            help="Skip agreement check; useful for large indices where queries are not available",
        ),
        seed: int = typer.Option(
            0,
            help="Random seed for sampling; default is 0",
        ),
    ):
        """Convert an IVF index to an SQ8 index and optionally check agreement."""
        convert_index(src_path, dst_path, train_sample, residual, queries, skip_check, seed)

    app()
    
if __name__ == '__main__':
    main()