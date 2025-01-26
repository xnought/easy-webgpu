import { GPU } from "./gpu";

export class Tensor {
	/**
	 * @param {GPU} gpu
	 * @param {ArrayBufferLike} typedArrayData
	 */
	constructor(gpu, typedArrayData) {
		this.gpu = gpu;
		this.data = gpu.memAlloc(typedArrayData.byteLength);
		gpu.memcpyHostToDevice(this.data, typedArrayData);
		this.dtype = typedArrayData.constructor;
	}

	static fill(gpu, length, fillValue, dtype) {
		return new Tensor(gpu, new dtype(length).fill(fillValue));
	}
	static zeros(gpu, length, dtype) {
		return Tensor.fill(gpu, length, 0, dtype);
	}
	static ones(gpu, length, dtype) {
		return Tensor.fill(gpu, length, 1, dtype);
	}
	async print() {
		console.log(`Tensor(this.dtype=${this.dtype.name}, `);
		await this.gpu.printGPUBuffer(this.data, "this.data=", this.dtype);
		console.log(")");
	}
	free() {
		this.gpu.free(this.data);
		this.data = undefined;
	}
}

export async function dev() {
	const gpu = await GPU.init();
	// const a = new Tensor(gpu, new Float32Array([1, 2, 3, 4]));
	const a = Tensor.ones(gpu, 16, Float32Array);
	await a.print();
	a.free();
}
