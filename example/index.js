import { GPU } from "../webgpu-compute";

async function example() {
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
}

example();
