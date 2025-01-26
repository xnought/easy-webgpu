/**
 * EZWEBGPU Library
 * AIM TO MIMIC PyCUDA https://homepages.math.uic.edu/~jan/mcs572f16/mcs572notes/lec29.html
 * Tooks tons of code from https://developer.chrome.com/docs/capabilities/web-apis/gpu-compute
 */

function assert(truth, msg = "ASSERT FAILED") {
	if (!truth) throw new Error(msg);
}

class GPU {
	constructor(device) {
		this.device = device;
	}
	static async init() {
		const adapter = await navigator.gpu.requestAdapter();
		assert(adapter, "adapter exists");
		const device = await adapter.requestDevice();
		assert(device, "device exists");
		return new GPU(device);
	}
	memAlloc(
		bytes,
		usage = GPUBufferUsage.STORAGE |
			GPUBufferUsage.COPY_DST |
			GPUBufferUsage.COPY_SRC
	) {
		assert(bytes > 0);
		const buffer = this.device.createBuffer({
			size: bytes,
			usage,
		});
		return buffer;
	}
	memcpyHostToDevice(gpuBuffer, cpuBuffer) {
		this.device.queue.writeBuffer(gpuBuffer, 0, cpuBuffer, 0);
	}
	async memcpyDeviceToHost(hostBuffer, deviceBuffer) {
		hostBuffer.set(
			await this.mapGPUToCPU(deviceBuffer, hostBuffer.constructor)
		);
	}
	free(buffer) {
		buffer.destroy();
	}
	async printGPUBuffer(buffer, label = "", TypedArray = Float32Array) {
		const d = await this.mapGPUToCPU(buffer, TypedArray);
		console.log(label, Array.from(d));
	}
	printDeviceInfo() {
		console.table(this.device.adapterInfo);
	}

	// this function may or may not leak. idk
	async mapGPUToCPU(gpuSrcBuffer, TypedArray = Float32Array) {
		const tempDstBuffer = this.memAlloc(
			gpuSrcBuffer.size,
			GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
		);
		const copyEncoder = this.device.createCommandEncoder();
		copyEncoder.copyBufferToBuffer(
			gpuSrcBuffer,
			0,
			tempDstBuffer,
			0,
			gpuSrcBuffer.size
		);
		this.device.queue.submit([copyEncoder.finish()]);
		await tempDstBuffer.mapAsync(GPUMapMode.READ);
		const result = new TypedArray(tempDstBuffer.getMappedRange());
		return result;
	}

	SourceModule(kernel) {
		return new SourceModule(this, kernel);
	}
}

class SourceModule {
	constructor(gpu, kernel) {
		this.gpu = gpu;
		this.device = gpu.device;
		this.kernel = kernel;
	}
	getFunctionExplicitBindings(name) {
		const mod = this.device.createShaderModule({ code: this.kernel });
		const computePipeline = this.device.createComputePipeline({
			layout: "auto",
			compute: {
				module: mod,
				entryPoint: name,
			},
		});
		const bindGroupLayout = computePipeline.getBindGroupLayout(0);
		return (workgroups, ...bindings) => {
			assert(workgroups !== undefined);

			const bindGroup = this.device.createBindGroup({
				layout: bindGroupLayout,
				entries: bindings,
			});
			const commandEncoder = this.device.createCommandEncoder();
			const passEncoder = commandEncoder.beginComputePass();
			passEncoder.setPipeline(computePipeline);
			passEncoder.setBindGroup(0, bindGroup);
			passEncoder.dispatchWorkgroups(...workgroups);
			passEncoder.end();

			this.device.queue.submit([commandEncoder.finish()]);
		};
	}
	getFunctionOnlyBuffers(name) {
		const gpuFunc = this.getFunctionExplicitBindings(name);
		return (workgroups, ...buffers) => {
			const inferredBindingsFromBuffers = buffers.map(
				(buffer, binding) => ({
					binding,
					resource: { buffer },
				})
			);
			gpuFunc(workgroups, ...inferredBindingsFromBuffers);
		};
	}
	/**
	 * Given the entryName of the kernel program (ie main for 'fn main') return a callable gpu function
	 * which takes the workgroups and the buffers as arguments.
	 *
	 * If explicitBindings is set to true, then must specify binding number for each buffer,
	 * otherwise just provide the list of buffers and binding number inferred by position
	 *
	 * @param {string} name
	 * @param {boolean?} explicitBindings
	 * @returns {(workgroups: number[], ...bindings: {binding: number, resource: {buffer: GPUBuffer}}[] | GPUBuffer[]) => void}
	 */
	getFunction(name, explicitBindings = false) {
		return explicitBindings
			? this.getFunctionExplicitBindings(name)
			: this.getFunctionOnlyBuffers(name);
	}
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

export async function test() {
	await testMemAllocAndCopy();
	await testSingleWorkgroup();
	await testOtherTypes();
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
	const mod = gpu.SourceModule(`
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
	const mod = gpu.SourceModule(`
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

export async function dev() {
	await test();
}
