/**
 * EZWEBGPU Library
 * AIM TO MIMIC PyCUDA https://homepages.math.uic.edu/~jan/mcs572f16/mcs572notes/lec29.html
 * Tooks tons of code from https://developer.chrome.com/docs/capabilities/web-apis/gpu-compute
 */

function assert(truth, msg = "ASSERT FAILED") {
	if (!truth) throw new Error(msg);
}

function mapBufferToGPU(gpuBuffer, cpuBuffer) {
	new Float32Array(gpuBuffer.getMappedRange()).set(cpuBuffer);
	gpuBuffer.unmap();
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
	mem_alloc(
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
	memcpy_htod(gpuBuffer, cpuBuffer) {
		this.device.queue.writeBuffer(gpuBuffer, 0, cpuBuffer, 0);
	}
	async memcpy_dtoh(hostBuffer, deviceBuffer) {
		hostBuffer.set(await this.mapGPUToCPU(deviceBuffer));
	}
	free(buffer) {
		buffer.destroy();
	}
	async printGPUBuffer(buffer) {
		const d = await this.mapGPUToCPU(buffer);
		console.log("GPU", buffer);
		console.log("CPU", d);
	}
	printDeviceInfo() {
		console.table(this.device.adapterInfo);
	}

	// this function may or may not leak. idk
	async mapGPUToCPU(gpuSrcBuffer) {
		const tempDstBuffer = this.mem_alloc(
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

	async execCopyCommand(dst, src) {}
}

class SourceModule {
	constructor(gpu, kernel) {
		this.gpu = gpu;
		this.kernel = kernel;
	}
	get_function() {}
}

async function test_mem_alloc_cpy() {
	const gpu = await GPU.init();
	const c = new Float32Array([1.0, 2.0, 3.0]);
	const result = new Float32Array(c.length);

	const cGPU = gpu.mem_alloc(c.byteLength);
	gpu.memcpy_htod(cGPU, c);

	await gpu.memcpy_dtoh(result, cGPU);
	let same = true;
	for (let i = 0; i < c.length; i++) {
		if (c[i] !== result[i]) {
			same = false;
			break;
		}
	}
	assert(
		same,
		"[mem_alloc and memcpy] Data not copied to or from GPU correctly."
	);

	gpu.free(cGPU);
}

export async function test() {
	await test_mem_alloc_cpy();
}

export async function dev() {
	console.log("here!");
}
