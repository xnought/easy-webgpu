# ezwebgpu

WebGPU API to move data from CPU to GPU and execute kernels more easily for compute applications.

**Roadmap**

- [x] Copy the basic API structure of PyCuda
- [ ] Have kernel running with multiple workgroups
- [ ] Memcpy and MemAlloc for other data types (uint and so on).
- [ ] Multiple kernels executing one after another 
- [ ] Example of Matrix multiply followed by ReLU
- [ ] Benchmarks

## Dev

```bash
pnpm install
pnpm dev
```

the `dev.js` is imported via the website `index.html`, but the actual package is in `src/`. The dev mode is just for when working on the package.