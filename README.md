# EasyWebGPU 

Easily transfer data from CPU to GPU, execute WebGPU code, and transfer back from GPU to CPU. Our API is way simpler than regular WebGPU compute pipelines!

The API _roughly_ follows high-level functions from [PyCuda](https://homepages.math.uic.edu/~jan/mcs572f16/mcs572notes/lec29.html). See the [Usage](#usage) section below for examples on how to use the API.

**Roadmap**

- [x] Copy the basic API structure of PyCuda
- [ ] Have kernel running with multiple workgroups
- [ ] Memcpy and MemAlloc for other data types (uint and so on).
- [ ] Multiple kernels executing one after another 
- [ ] Example of Matrix multiply followed by ReLU
- [ ] Benchmarks


## Usage

TODO!


## Dev

```bash
pnpm install
pnpm dev
```

the `dev.js` is imported via the website `index.html`, but the actual package is in `src/`. The dev mode is just for when working on the package.
