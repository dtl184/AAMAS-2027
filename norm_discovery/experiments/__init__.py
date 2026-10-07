# Limit BLAS/OpenMP threads per process: experiments parallelise over processes, and multi-threaded BLAS in every
# worker oversubscribes the CPU badly.  Must run before numpy is imported (python -m experiments.<x> imports this
# package first).
import os as _os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")
