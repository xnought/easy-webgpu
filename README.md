# webgpu-compute library 

A PyCuda like API to easily transfer data from CPU to GPU, execute WebGPU code, and transfer back from GPU to CPU.


## Install

No dependencies and the code is simple enough for you to edit if you need to.

All code in [`webgpu-compute.js`](webgpu-compute.js).

You can manually download the file or just do

```bash
wget https://raw.githubusercontent.com/xnought/webgpu-compute/refs/heads/main/webgpu-compute.js 
```

in your JS project.

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

const gpuBuffer = await gpu.memAlloc(16); // allocated 16 bytes on the GPU
```

### Free GPU Buffers

```js
import { GPU } from "webgpu-compute";
const gpu = await GPU.init(); 

const gpuBuffer = await gpu.memAlloc(16);
gpu.free(gpuBuffer); // freed on GPU!
```

### Copy CPU data into GPU

```js
import { GPU } from "webgpu-compute";
const gpu = await GPU.init(); 

const cpuBuffer = new Float32Array([1,2,3,4]);
const gpuBuffer = await gpu.memAlloc(cpuBuffer.byteLength); 
await gpu.memcpyHostToDevice(gpuBuffer, cpuBuffer);

await gpu.printGPUBuffer(gpuBuffer); // > [1,2,3,4]
```

### Copy GPU data back to CPU 

(Uses variables in [last section](#copy-cpu-data-into-gpu))

```js
import { GPU } from "webgpu-compute";
const gpu = await GPU.init(); 

// Send GPU data back to CPU
const otherCpuBuffer = new Float32Array(cpuBuffer.length);
await gpu.memcpyDeviceToHost(otherCpuBuffer, gpuBuffer);
console.log(otherCpuBuffer); // > [1,2,3,4]
```

### Execute WebGPU Kernels 

This [example](./example/index.js) squares a million 3s `[3,3,3, ..., 3]` in place with the final result being a million 9s.

```js
example();

async function example() {
	const gpu = await GPU.init();

	// cpu data
	const cpuData = new Float32Array(1e6).fill(3); // data
	const cpuLength = new Uint32Array([cpuData.length]); // length of data

	// move cpu data to gpu
	console.time("ALLOCATION");
	const gpuData = await gpu.memAlloc(cpuData.byteLength);
	const gpuLength = await gpu.memAlloc(cpuLength.byteLength);
	console.timeEnd("ALLOCATION");

	console.time("HOST TO DEVICE");
	await gpu.memcpyHostToDevice(gpuData, cpuData);
	await gpu.memcpyHostToDevice(gpuLength, cpuLength);
	console.timeEnd("HOST TO DEVICE");

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
	console.time("SQUARE");
	await square(workgroups, gpuData, gpuLength);
	console.timeEnd("SQUARE");

	// bring result back to cpu
	console.time("PRINT");
	await gpu.memcpyDeviceToHost(cpuData, gpuData);
	console.log(cpuData); 
	console.timeEnd("PRINT");
}
```

`Console Out ->`

```js
ALLOCATION: 0.197021484375 ms
index.js:46 HOST TO DEVICE: 950.698974609375 ms
index.js:66 SQUARE: 5.678955078125 ms
index.js:71 Float32Array(1000000) [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, …]
index.js:72 PRINT: 12.450927734375 ms
```


## Dev

```bash
cd example
pnpm install
pnpm dev
```
