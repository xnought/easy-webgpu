function assert(truth, msg = "ASSERT FAILED") {
	if (!truth) throw new Error(msg);
}

// AIM TO MIMIC PyCUDA https://homepages.math.uic.edu/~jan/mcs572f16/mcs572notes/lec29.html
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
	mem_alloc(bytes) {
		assert(bytes > 0);
		const buffer = this.device.createBuffer({
			size: bytes,
			usage: GPUBufferUsage.STORAGE,
		});
		return buffer;
	}
	memcpy_dtoh() {}
	memcpy_htod() {}
	free(buffer) {
		buffer.destroy();
	}
}

class SourceModule {
	constructor(gpu, kernel) {
		this.gpu = gpu;
		this.kernel = kernel;
	}
	get_function() {}
}

export async function dev() {
	const a = new Float32Array([1, 2, 3]);
	const b = new Float32Array([3, 2, 1]);
	const c = new Float32Array([0]);

	const gpu = await GPU.init();
	const aGPU = gpu.mem_alloc(a.byteLength);
	gpu.free(aGPU);
	// const bGPU = gpu.mem_alloc(b.byteLength);
	// const cGPU = gpu.mem_alloc(c.byteLength);

	// gpu.memcpy_htod(aGPU, a);
	// gpu.memcpy_htod(bGPU, b);
	// gpu.memcpy_htod(cGPU, c);

	// // Then perform computation
	// const mod = new SourceModule(gpu, `some kernel function here`);
	// const func = mod.get_function("dot");
	// func(aGPU, bGPU, cGPU);

	// grab the result from cGPU and put into c
	// gpu.memcpy_dtoh(c, cGPU);
}
