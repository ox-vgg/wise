import itertools


def argsort(seq):
    return sorted(range(len(seq)), key=seq.__getitem__)


def batched(iterable, n: int):
    it = iter(iterable)
    while True:
        batch = list(itertools.islice(it, n))
        if not batch:
            return
        yield batch
