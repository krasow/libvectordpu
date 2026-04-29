# upmem-vector

please run to commit stuff

```bash
bash format.sh 
```

to run the test suite, enable the upmem env
```
make test
```

## Current limitation

The public API is intentionally kept stable, but the current JIT lowering does
not yet compactly inline every JIT-produced intermediate into indirect local
updates.

In practice, this means patterns like kmeans assignment followed by
`dpu_local_vector::apply(...)` can still fall back to materializing the
intermediate label vector and/or compiling a larger update kernel than fits in
IRAM. That is a backend lowering limit, not a user-facing API requirement.
