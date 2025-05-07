# CGNPU

## `compiler`
Implements proof-of-concept compiler for C++ to our own bytecode format that emphasizes matrix multiplication primitives. The IR generation and ANE implementation are stubbed/hardcoded.

### Usage
- Within the `compiler` directory, execute `./scripts/run_example.sh larger_matrix_input`. 
- Set the `GPU_ONLY` environmental variable to test without heterogeneous compute.