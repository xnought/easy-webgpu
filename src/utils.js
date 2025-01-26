export function assert(truth, msg = "ASSERT FAILED") {
	if (!truth) throw new Error(msg);
}
