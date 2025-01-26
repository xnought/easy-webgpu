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
	print() {
		console.log(`Tensor(dtype=${this.dtype.name})`);
		this.gpu.printGPUBuffer(this.data, "this.data", this.dtype);
	}
	free() {
		this.gpu.free(this.data);
		this.data = undefined;
	}
}

export async function dev() {
	const gpu = await GPU.init();
	const a = new Tensor(gpu, new Float32Array([1, 2, 3]));
	a.print();
	a.free();
}
