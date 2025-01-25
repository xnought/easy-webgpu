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
	async memCopyDeviceToHost(hostBuffer, deviceBuffer) {
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
}

class SourceModule {
	constructor(gpu, kernel) {
		this.gpu = gpu;
		this.kernel = kernel;
	}
	get_function() {}
}

async function testMemAllocAndCopy() {
	const gpu = await GPU.init();
	const c = new Float32Array([1.0, 2.0, 3.0]);
	const result = new Float32Array(c.length);

	const cGPU = gpu.memAlloc(c.byteLength);
	gpu.memcpyHostToDevice(cGPU, c);

	await gpu.memCopyDeviceToHost(result, cGPU);
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

export async function test() {
	await testMemAllocAndCopy();
}

export async function dev() {
	console.log("here!");
}
