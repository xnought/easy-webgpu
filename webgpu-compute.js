/**
 * AIM TO MIMIC PyCUDA https://homepages.math.uic.edu/~jan/mcs572f16/mcs572notes/lec29.html
 * Tooks tons of code from https://developer.chrome.com/docs/capabilities/web-apis/gpu-compute
 *
 * SEE DOCUMENTATION AT https://github.com/xnought/webgpu-compute
 */

export function assert(truth, msg = "ASSERT FAILED") {
	if (!truth) throw new Error(msg);
}

export class GPU {
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
	async deviceSynchronize() {
		await this.device.queue.onSubmittedWorkDone();
	}

	/**
	 * @param {number} bytes
	 * @param {GPUBufferUsage} usage
	 * @returns {GPUBuffer}
	 */
	memAlloc(bytes, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC) {
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
		hostBuffer.set(await this.mapGPUToCPU(deviceBuffer, hostBuffer.constructor));
	}
	free(buffer) {
		buffer.destroy();
	}
	async printGPUBuffer(buffer, TypedArray = Float32Array) {
		const d = await this.mapGPUToCPU(buffer, TypedArray);
		console.log(Array.from(d), d.constructor.name);
	}
	printDeviceInfo() {
		console.table(this.device.adapterInfo);
	}

	// this function may or may not leak. idk
	async mapGPUToCPU(gpuSrcBuffer, TypedArray = Float32Array) {
		const tempDstBuffer = await this.memAlloc(gpuSrcBuffer.size, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
		const copyEncoder = this.device.createCommandEncoder();
		copyEncoder.copyBufferToBuffer(gpuSrcBuffer, 0, tempDstBuffer, 0, gpuSrcBuffer.size);
		this.device.queue.submit([copyEncoder.finish()]);
		await tempDstBuffer.mapAsync(GPUMapMode.READ);

		const result = new TypedArray(tempDstBuffer.getMappedRange());
		return result;
	}

	SourceModule(kernel) {
		return new SourceModule(this, kernel);
	}
}

export class SourceModule {
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
			const inferredBindingsFromBuffers = buffers.map((buffer, binding) => ({
				binding,
				resource: { buffer },
			}));
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
		return explicitBindings ? this.getFunctionExplicitBindings(name) : this.getFunctionOnlyBuffers(name);
	}
}
