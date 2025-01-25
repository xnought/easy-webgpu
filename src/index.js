// AIM TO MIMIC PyCUDA https://homepages.math.uic.edu/~jan/mcs572f16/mcs572notes/lec29.html
class GPU {
	mem_alloc() {}
	memcpy_dtoh() {}
	memcpy_htod() {}
}

class SourceModule {
	constructor(gpu, kernel) {
		this.gpu = gpu;
		this.kernel = kernel;
	}
	get_function() {}
}

export function dev() {
	const a = new Float32Array([1, 2, 3]);
	const b = new Float32Array([3, 2, 1]);
	const c = new Float32Array([0]);

	const gpu = new GPU();
	const aGPU = gpu.mem_alloc(a.byteLength);
	const bGPU = gpu.mem_alloc(b.byteLength);
	const cGPU = gpu.mem_alloc(c.byteLength);

	gpu.memcpy_htod(aGPU, a);
	gpu.memcpy_htod(bGPU, b);
	gpu.memcpy_htod(cGPU, c);

	// Then perform computation
	// const mod = new SourceModule(gpu, `some kernel function here`);
	// const func = mod.get_function("dot");
	// func(aGPU, bGPU, cGPU);

	// grab the result from cGPU and put into c
	gpu.memcpy_dtoh(c, cGPU);
}
