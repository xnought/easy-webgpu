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
		const bindGroupLayout = this.device.createBindGroupLayout({
			entries: parseLayout(buffers),
		});
		const bindGroup = this.device.createBindGroup({
			layout: bindGroupLayout,
			entries: parseEntries(buffers),
		});
		const computePipeline = this.device.createComputePipeline({
			layout: this.device.createPipelineLayout({
				bindGroupLayouts: [bindGroupLayout],
			}),
			compute: {
				module: mod,
				entryPoint,
			},
		});
		const commandEncoder = this.device.createCommandEncoder();
		const passEncoder = commandEncoder.beginComputePass();
		passEncoder.setPipeline(computePipeline);
		passEncoder.setBindGroup(0, bindGroup);
		passEncoder.dispatchWorkgroups(...dispatchWorkgroups);
		passEncoder.end();
		this.device.queue.submit([commandEncoder.finish()]);
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

	gpu.free(cGPU);
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

export async function test() {
	await testMemAllocAndCopy();
}

export async function dev() {
	const gpu = await GPU.init();
	gpu.printDeviceInfo();

	const cpuA = new Float32Array([1, 2, 3, 4, 5]);
	const cpuB = new Float32Array([1, 2, 3, 4, 5]);
	const cpuC = new Float32Array([0]);

	const gpuA = gpu.memAlloc(cpuA.byteLength);
	const gpuB = gpu.memAlloc(cpuB.byteLength);
	const gpuC = gpu.memAlloc(cpuC.byteLength);

	gpu.memcpyHostToDevice(gpuA, cpuA);
	gpu.memcpyHostToDevice(gpuB, cpuB);
	gpu.memcpyHostToDevice(gpuC, cpuC);

	gpu.printGPUBuffer(gpuA, "A");
	gpu.printGPUBuffer(gpuB, "B");
	gpu.printGPUBuffer(gpuC, "C");

	// dot
	const THREADS_PER_BLOCK = 5;
	const code = `
      @group(0) @binding(0) var<storage, read> a: array<f32>;
      @group(0) @binding(1) var<storage, read> b: array<f32>;
      @group(0) @binding(2) var<storage, read_write> c: f32;

      var<workgroup> partialSums: array<f32, ${THREADS_PER_BLOCK}>;

      @compute @workgroup_size(${THREADS_PER_BLOCK})
      fn main(@builtin(global_invocation_id) gid : vec3u, @builtin(local_invocation_id) lid : vec3u) {
        partialSums[lid.x] = a[gid.x]*b[gid.x];
        workgroupBarrier();

        if(lid.x == 0) {
          var summed: f32 = 0.0;
          for(var i: u32 = 0; i < ${THREADS_PER_BLOCK}; i++) {
            summed += partialSums[i];
          }
          c = summed;
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
	console.log(
		"Result gpu",
		cpuC[0],
		"vs actual dot",
		cpuA.reduce((acc, _, i) => {
			return acc + cpuA[i] * cpuB[i];
		}, 0)
	);

	gpu.free(gpuA);
	gpu.free(gpuB);
	gpu.free(gpuC);
}
