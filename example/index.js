import { GPU, assert } from "../webgpu-compute";

main();

async function main() {
	await test();
	await example();
	// await memoryStressTest();
}

async function memoryStressTest() {
	const gpu = await GPU.init();

	console.log("Memory Stress Test Starting");

	const bytes = 268435456;
	let maxIters = 10000;
	let i = 0;
	let buffers = [];
	while (i < maxIters) {
		const b = gpu.memAlloc(bytes);
		buffers.push(b);
		gpu.free(b);
		i++;
	}

	console.log("Memory Stress Test Ended");
}

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

async function testMemAllocAndCopy() {
	const gpu = await GPU.init();
	const c = new Float32Array([1.0, 2.0, 3.0]);
	const result = new Float32Array(c.length);

	const cGPU = gpu.memAlloc(c.byteLength);
	gpu.memcpyHostToDevice(cGPU, c);

	await gpu.memcpyDeviceToHost(result, cGPU);
	let same = true;
	for (let i = 0; i < c.length; i++) {
		if (c[i] !== result[i]) {
			same = false;
			break;
		}
	}
	assert(
		same,
		"[testMemAllocAndCopy] Data not copied to or from GPU correctly."
	);
	console.log("[testMemAllocAndCopy] PASSED");

	gpu.free(cGPU);
}

async function testSingleWorkgroup() {
	const gpu = await GPU.init();

	const length = 128;
	const cpuA = new Float32Array(length).fill(0).map((_) => Math.random());
	const cpuB = new Float32Array(length).fill(0).map((_) => Math.random());
	const cpuC = new Float32Array([0]);

	const gpuA = gpu.memAlloc(cpuA.byteLength);
	const gpuB = gpu.memAlloc(cpuB.byteLength);
	const gpuC = gpu.memAlloc(cpuC.byteLength);

	gpu.memcpyHostToDevice(gpuA, cpuA);
	gpu.memcpyHostToDevice(gpuB, cpuB);
	gpu.memcpyHostToDevice(gpuC, cpuC);

	const THREADS_PER_BLOCK = 256;
	const mod = gpu.SourceModule(/*wgsl*/ `
      @group(0) @binding(0) var<storage, read> a: array<f32>;
      @group(0) @binding(1) var<storage, read> b: array<f32>;
      @group(0) @binding(2) var<storage, read_write> c: f32;

      var<workgroup> partialSums: array<f32, ${THREADS_PER_BLOCK}>;

      @compute @workgroup_size(${THREADS_PER_BLOCK})
      fn myDot(@builtin(global_invocation_id) gid : vec3u, @builtin(local_invocation_id) lid : vec3u) {
        if(gid.x < ${length}) {
          partialSums[lid.x] += a[gid.x]*b[gid.x];
        }

        workgroupBarrier();
        if(lid.x == 0) {
          var summed: f32 = 0.0;
          for(var i: u32 = 0; i < ${THREADS_PER_BLOCK}; i++) {
            summed += partialSums[i];
          }
          c += summed;
        }
      }
    `);
	const dot = mod.getFunction("myDot", true);
	dot(
		[1],
		{ binding: 0, resource: { buffer: gpuA } },
		{ binding: 1, resource: { buffer: gpuB } },
		{ binding: 2, resource: { buffer: gpuC } }
	);

	// copy back the result and compare
	await gpu.memcpyDeviceToHost(cpuC, gpuC);
	let actual = cpuA.reduce((acc, _, i) => {
		return acc + cpuA[i] * cpuB[i];
	}, 0);

	assert(
		actual.toFixed(0) === cpuC[0].toFixed(0),
		"[testSingleWorkgroup] incorrect dot product"
	);
	console.log("[testSingleWorkgroup] PASSED");

	gpu.free(gpuA);
	gpu.free(gpuB);
	gpu.free(gpuC);
}

async function testOtherTypes() {
	const gpu = await GPU.init();

	const length = 256;
	const cpuA = new Uint32Array(length).fill(0).map((_, i) => i);
	const cpuB = new Uint32Array(length).fill(1);
	const cpuC = new Uint32Array([0]);
	const cpuN = new Uint32Array([length]);

	const gpuA = gpu.memAlloc(cpuA.byteLength);
	const gpuB = gpu.memAlloc(cpuB.byteLength);
	const gpuC = gpu.memAlloc(cpuC.byteLength);
	const gpuN = gpu.memAlloc(cpuN.byteLength);

	gpu.memcpyHostToDevice(gpuA, cpuA);
	gpu.memcpyHostToDevice(gpuB, cpuB);
	gpu.memcpyHostToDevice(gpuC, cpuC);
	gpu.memcpyHostToDevice(gpuN, cpuN);

	const THREADS_PER_BLOCK = 256;
	const mod = gpu.SourceModule(/*wgsl*/ `
      @group(0) @binding(0) var<storage, read> a: array<u32>;
      @group(0) @binding(1) var<storage, read> b: array<u32>;
      @group(0) @binding(2) var<storage, read_write> c: u32;
      @group(0) @binding(3) var<storage, read> n: u32;

	  var<workgroup> partialSums: array<u32, ${THREADS_PER_BLOCK}>;

      @compute @workgroup_size(${THREADS_PER_BLOCK})
      fn myDot(@builtin(global_invocation_id) gid : vec3u, @builtin(local_invocation_id) lid : vec3u) {

        if(gid.x < n) {
          partialSums[lid.x] = a[gid.x]*b[gid.x];
        }
        workgroupBarrier();

        if(lid.x == 0) {
          var summed: u32 = 0;
          for(var i: u32 = 0; i < ${THREADS_PER_BLOCK}; i++) {
            summed += partialSums[i];
          }
		  c += summed;
        }
      }
    `);
	const dot = mod.getFunction("myDot");
	dot([1], gpuA, gpuB, gpuC, gpuN);

	// copy back the result and compare
	await gpu.memcpyDeviceToHost(cpuC, gpuC);
	let actual = cpuA.reduce((acc, _, i) => {
		return acc + cpuA[i] * cpuB[i];
	}, 0);

	assert(
		actual.toFixed(0) === cpuC[0].toFixed(0),
		"[testOtherTypes] incorrect dot product"
	);
	console.log("[testMultipleWorkgroups] PASSED");

	gpu.free(gpuA);
	gpu.free(gpuB);
	gpu.free(gpuC);
}

async function test() {
	await testMemAllocAndCopy();
	await testSingleWorkgroup();
	await testOtherTypes();
}
