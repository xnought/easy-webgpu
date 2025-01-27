# Compute WebGPU 

A PyCuda like API to easily transfer data from CPU to GPU, execute WebGPU code, and transfer back from GPU to CPU.

All code in [`webgpu-compute/index.js`](webgpu-compute/index.js) with no dependencies other than being run in a WebGPU compatible browser (ie chrome).

## Usage

### Initialize GPU

```js
import { GPU } from "webgpu-compute";
const gpu = await GPU.init(); 
gpu.printDeviceInfo(); // > {architecture : "rdna-1" description : "" device : "" vendor : "amd"}
```

### Allocate GPU Buffers

```js
import { GPU } from "webgpu-compute";
const gpu = await GPU.init(); 

const gpuBuffer = gpu.memAlloc(16); // allocated 16 bytes on the GPU
```

### Copy CPU data into GPU

```js
import { GPU } from "webgpu-compute";
const gpu = await GPU.init(); 

const cpuBuffer = new Float32Array([1,2,3,4]);
const gpuBuffer = gpu.memAlloc(cpuBuffer.byteLength); 
gpu.memcpyHostToDevice(gpuBuffer, cpuBuffer);

await gpu.printGPUBuffer(gpuLength); // > [1,2,3,4]
```

### Copy GPU data back to CPU 

(Uses variables in last example)

```js
import { GPU } from "webgpu-compute";
const gpu = await GPU.init(); 

// Send GPU data back to CPU
const otherCpuBuffer = new Float32Array(cpuBuffer.length);
await gpu.memcpyDeviceToHost(otherCpuBuffer, gpuBuffer);
console.log(otherCpuBuffer); // > [1,2,3,4]
```

### Execute WebGPU Kernels 

This example squares the data `[1,2,3,4]` in place with the final result being `[1,4,9,16]`.

```js
const gpu = await GPU.init();

// cpu data
const cpuData = new Float32Array([1, 2, 3, 4]); // data
const cpuLength = new Uint32Array([cpuData.length]); // length of data

// move cpu data to gpu
const gpuData = gpu.memAlloc(cpuData.byteLength);
const gpuLength = gpu.memAlloc(cpuLength.byteLength);
gpu.memcpyHostToDevice(gpuData, cpuData);
gpu.memcpyHostToDevice(gpuLength, cpuLength);

// initialize webgpu kernel to square all elements in data
const module = gpu.SourceModule(`
	@group(0) @binding(0) var<storage, read_write> data: array<f32>;
	@group(0) @binding(1) var<storage, read> length: u32;

	@compute @workgroup_size(256)
	fn square(@builtin(global_invocation_id) gid : vec3u) {
		if(gid.x < length) {
			data[gid.x] = data[gid.x]*data[gid.x];
		}
	}
`);

// execute kernel
const square = module.getFunction("square");
const workgroups = [1];
square(workgroups, gpuData, gpuLength);

// bring result back to cpu
await gpu.memcpyDeviceToHost(cpuData, gpuData);
console.log(cpuData); // > [1, 4, 9, 16]
```


## Dev

```bash
pnpm install
pnpm dev
```

the `dev.js` is imported via the website `index.html`, but the actual package is in `src/`. The dev mode is just for when working on the package.
