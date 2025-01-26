import { GPU } from "./gpu";

function toNewGPUBuffer(gpu, typedArrayData) {
	const gpuBuffer = gpu.memAlloc(typedArrayData.byteLength);
	gpu.memcpyHostToDevice(gpuBuffer, typedArrayData);
	return gpuBuffer;
}

function totalLength(shape) {
	let prod = 1;
	for (const s of shape) {
		prod *= s;
	}
	return prod;
}

const SHAPE_TYPE = Uint32Array;
export class Tensor {
	/**
	 * @param {GPU} gpu
	 * @param {ArrayBufferLike} typedArrayData
	 */
	constructor(gpu, typedArrayData, shape) {
		this.gpu = gpu;
		this.data = toNewGPUBuffer(gpu, typedArrayData);
		this.dtype = typedArrayData.constructor;
		this.shape = toNewGPUBuffer(gpu, SHAPE_TYPE.from(shape));
	}

	static fill(gpu, shape, fillValue, dtype) {
		return new Tensor(
			gpu,
			new dtype(totalLength(shape)).fill(fillValue),
			shape
		);
	}
	static zeros(gpu, shape, dtype) {
		return Tensor.fill(gpu, shape, 0, dtype);
	}
	static ones(gpu, shape, dtype) {
		return Tensor.fill(gpu, shape, 1, dtype);
	}
	async print() {
		console.log(`Tensor(this.dtype=${this.dtype.name}`);
		await this.gpu.printGPUBuffer(this.data, "this.data=", this.dtype);
		await this.gpu.printGPUBuffer(this.shape, "this.shape=", SHAPE_TYPE);
		console.log(")");
	}
	free() {
		this.gpu.free(this.data);
		this.data = undefined;
		this.gpu.free(this.shape);
		this.shape = undefined;
	}
}

export async function dev() {
	const gpu = await GPU.init();
	// const a = new Tensor(gpu, new Float32Array([1, 2, 3, 4]), [4, 1]);
	const a = Tensor.ones(gpu, [4, 1], Float32Array);
	await a.print();
	a.free();
}
