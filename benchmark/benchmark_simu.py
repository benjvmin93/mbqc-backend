import timeit
from copy import deepcopy
import time


class BenchmarkSimu:
    def bench_functions(
        self,
        functions,
        it=1000,
        label1: str = None,
        label2: str = None,
        args1=[],
        args2=[],
    ):
        labels = (functions[0].__name__, functions[1].__name__)
        if label1 is not None and label2 is not None:
            labels = (label1, label2)

        t1 = timeit.timeit(
            "f1(*args1)", number=it, globals={"f1": functions[0], "args1": args1}
        )

        t2 = timeit.timeit(
            "f2(*args2)", number=it, globals={"f2": functions[1], "args2": args2}
        )

        return {labels[0]: t1, labels[1]: t2}

    def bench_class_functions(
        self, obj1, obj2, functions, it=1000, args1=[], args2=[], labels=(None, None)
    ):
        if labels == (None, None):
            labels = (functions[0].__name__, functions[1].__name__)
        t1 = 0.0
        for _ in range(it):
            loop_t1 = time.perf_counter()
            functions[0](deepcopy(obj1), *args1)
            loop_t2 = time.perf_counter()
            t1 += loop_t2 - loop_t1
        t2 = 0.0
        for _ in range(it):
            loop_t1 = time.perf_counter()
            functions[1](deepcopy(obj2), *args2)
            loop_t2 = time.perf_counter()
            t2 += loop_t2 - loop_t1
        t1 /= it
        t2 /= it
        return {f"{labels[0]}": t1, f"{labels[1]}": t2}
