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
		hostBuffer.set(await this.mapGPUToCPU(deviceBuffer));
	}
	free(buffer) {
		buffer.destroy();
	}
	async printGPUBuffer(buffer, label = "") {
		const d = await this.mapGPUToCPU(buffer);
		console.log(label, Array.from(d));
	}
	printDeviceInfo() {
		console.table(this.device.adapterInfo);
	}

	// this function may or may not leak. idk
	async mapGPUToCPU(gpuSrcBuffer) {
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
		const result = new Float32Array(tempDstBuffer.getMappedRange());
		return result;
	}

	/**
	 * @param {{code: string, buffers: {buffer: GPUBuffer, type: "storage" | "read-only-storage" | undefined}[], dispatchWorkgroups: number[], entryPoint?: string}
	 */
	execKernel({ code, buffers, dispatchWorkgroups, entryPoint = "main" }) {
		const mod = this.device.createShaderModule({ code });
		const computePipeline = this.device.createComputePipeline({
			layout: "auto",
			compute: {
				module: mod,
				entryPoint,
			},
		});
		const bindGroupLayout = computePipeline.getBindGroupLayout(0);
		const bindGroup = this.device.createBindGroup({
			layout: bindGroupLayout,
			entries: parseEntries(buffers),
		});
		const commandEncoder = this.device.createCommandEncoder();
		const passEncoder = commandEncoder.beginComputePass();
		passEncoder.setPipeline(computePipeline);
		passEncoder.setBindGroup(0, bindGroup);
		passEncoder.dispatchWorkgroups(...dispatchWorkgroups);
		passEncoder.end();
		this.device.queue.submit([commandEncoder.finish()]);
	}

	SourceModule(kernel) {
		return new SourceModule(this, kernel);
	}
}

function parseLayout(buffers) {
	return buffers.map((d, i) => {
		return {
			binding: i,
			visibility: GPUShaderStage.COMPUTE,
			buffer: { type: d.type ?? "read-only-storage" },
		};
	});
}

function parseEntries(buffers) {
	return buffers.map((d, i) => {
		return { binding: i, resource: { buffer: d.buffer } };
	});
}

class SourceModule {
	constructor(gpu, kernel) {
		this.gpu = gpu;
		this.device = gpu.device;
		this.kernel = kernel;
	}
	getFunction(name) {
		const mod = this.device.createShaderModule({ code: this.kernel });
		const computePipeline = this.device.createComputePipeline({
			layout: "auto",
			compute: {
				module: mod,
				entryPoint: name,
			},
		});
		const bindGroupLayout = computePipeline.getBindGroupLayout(0);
		return ({ inputs, workgroups }) => {
			assert(inputs !== undefined);
			assert(workgroups !== undefined);

			const bindGroup = this.device.createBindGroup({
				layout: bindGroupLayout,
				entries: inputs,
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
	await testSourceModule();
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
	const code = `
      @group(0) @binding(0) var<storage, read> a: array<f32>;
      @group(0) @binding(1) var<storage, read> b: array<f32>;
      @group(0) @binding(2) var<storage, read_write> c: f32;

      var<workgroup> partialSums: array<f32, ${THREADS_PER_BLOCK}>;

      @compute @workgroup_size(${THREADS_PER_BLOCK})
      fn main(@builtin(global_invocation_id) gid : vec3u, @builtin(local_invocation_id) lid : vec3u) {
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
    `;
	gpu.execKernel({
		code,
		buffers: [
			{ buffer: gpuA, type: "read-only-storage" },
			{ buffer: gpuB, type: "read-only-storage" },
			{ buffer: gpuC, type: "storage" },
		],
		dispatchWorkgroups: [1],
	});

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
async function testSourceModule() {
	const gpu = await GPU.init();

	const length = 256;
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
	const dot = mod.getFunction("myDot");
	dot({
		inputs: [
			{ binding: 0, resource: { buffer: gpuA } },
			{ binding: 1, resource: { buffer: gpuB } },
			{ binding: 2, resource: { buffer: gpuC } },
		],
		workgroups: [1],
	});

	// copy back the result and compare
	await gpu.memcpyDeviceToHost(cpuC, gpuC);
	let actual = cpuA.reduce((acc, _, i) => {
		return acc + cpuA[i] * cpuB[i];
	}, 0);

	assert(
		actual.toFixed(0) === cpuC[0].toFixed(0),
		"[testSourceModule] incorrect dot product"
	);
	console.log("[testSourceModule] PASSED");

	gpu.free(gpuA);
	gpu.free(gpuB);
	gpu.free(gpuC);
}

export async function dev() {
	await testSourceModule();
}
