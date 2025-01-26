# WebGPU Tensor

1. A Tensor (ndarray) API with common operations accelerated by WebGPU
2. A low-level PyCuda like API to easily transfer data from CPU to GPU, execute WebGPU code, and transfer back from GPU to CPU.

**Roadmap**

- [x] Copy the basic API structure of PyCuda
- [ ] Have kernel running with multiple workgroups
- [x] Memcpy and MemAlloc for other data types (uint and so on).
- [ ] Tensor constructor (shape, strides, ...)
- [ ] Tensor Reshape Ops (transpose, reshape, expand, ...)
- [ ] Tensor Binary Ops (add, sub, mult, div, ...)
- [ ] Tensor Unary Ops (ReLU, Sigmoid, ...)
- [ ] Example of ML model implemented with Tensor

## Usage

TODO!


## Dev

```bash
pnpm install
pnpm dev
```

the `dev.js` is imported via the website `index.html`, but the actual package is in `src/`. The dev mode is just for when working on the package.
