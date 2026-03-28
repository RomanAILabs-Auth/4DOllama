/* Pass 9: CPU link stub when MIR marks GPU-candidate `par` inside `spacetime:`.
 * Real CUDA/PTX lowering is a future pipeline stage; this keeps the link line valid.
 */
void roma4d_cuda_offload_stub(void) {}
