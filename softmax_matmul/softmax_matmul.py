import torch
import triton
import triton.language as tl
import torch.nn.functional as F


def softmax_mult(x, V, dim=-1):
    return F.softmax(x, dim=dim) @ V

@triton.jit
def fused_softmax_kernel(
    x_ptr, v_ptr, out_ptr,
    stride_xbatch, stride_xrow, stride_xcol,
    stride_vbatch, stride_vrow, stride_vcol,
    stride_obatch, stride_orow, stride_ocol,
    d1: tl.constexpr, d2: tl.constexpr, d3: tl.constexpr,
    BLOCK_1: tl.constexpr, BLOCK_2: tl.constexpr,
):
    # Each program:
    # - reads a strip of shape (BLOCK_1, d2) from x
    # - reads a strip of shape (d2, BLOCK_2) from v
    # - write a block of shape (BLOCK_1, BLOCK_2) to o
    pid_batch = tl.program_id(0)
    pid_row = tl.program_id(1)
    pid_col = tl.program_id(2)

    x_block = tl.make_block_ptr(
        x_ptr + pid_batch * stride_xbatch,        # take the right batch
        shape=(d1, d2),                           # shape of parent tensor
        strides=(stride_xrow, stride_xcol),       # strides of parent tensor
        offsets=(pid_row * BLOCK_1, 0),           # add offset to start at the right row
        block_shape=(BLOCK_1, d2),                # desired shape of output block
        order=(1, 0),                             # prioritize dimension 1 over dimension 0 for memory access (optimization)
    )
    v_block = tl.make_block_ptr(
        v_ptr + pid_batch * stride_vbatch, 
        shape=(d2, d3),                   
        strides=(stride_vrow, stride_vcol),
        offsets=(0, pid_col * BLOCK_2),   
        block_shape=(d2, BLOCK_2),       
        order=(0, 1),                   
    )
    out_block = tl.make_block_ptr(
        out_ptr + pid_batch * stride_obatch, 
        shape=(d1, d3),                   
        strides=(stride_orow, stride_ocol),
        offsets=(pid_row * BLOCK_1, pid_col * BLOCK_2),   
        block_shape=(BLOCK_1, BLOCK_2),       
        order=(1, 0),                   
    )

    # for x, we cannot auto-pad with -inf so be pad manually 
    # this is not a pb for v since we need to pad with zeros
    x_rows = tl.load(x_block, boundary_check=(0,1), padding_option="")     # (BLOCK_1, d2)
    x_rows = tl.where(tl.arange(0, BLOCK_1) < d1, x_rows, float('-inf'))   
    v_cols = tl.load(v_block, boundary_check=(0,1), padding_option="zero") # (d2, BLOCK_2)

    max_x = x_rows.max(axis=1)  # (BLOCK_1,)
    exps = tl.exp(x_rows - max_x[:, None]) # (B1, d2)
    sums = exps.sum(axis=1) # (B1,)
    softmax = exps / sums[:, None]  # (B1, d2)

    tl.store(out_block, tl.dot(softmax, v_cols), boundary_check=(0,1))


def fused_softmax(x, v, BLOCK_1=16, BLOCK_2=16):
    """
    Compute softmax using Triton kernel with online algorithm.

    Args:
        x: Input tensor of shape (batch_size, d1, d2)
        v: Input tensor of shape (batch_size, d2, d3)

    Output: tensor of shape (batch_size, d1, d3) equal to softmax(x) @ v
    """
    *bs, d1, d2 = x.shape
    *bs_, d2_, d3 = v.shape
    assert bs == bs_ and d2 == d2_
    x = x.reshape(-1, d1, d2)
    v = v.reshape(-1, d2, d3)

    n_batch = x.shape[0]
    out = torch.empty_like(x)

    # Calculate grid dimensions
    grid = (n_batch, triton.cdiv(d1, BLOCK_1), triton.cdiv(d3, BLOCK_2))

    # Launch kernel
    softmax_matmult_kernel[grid](
        x, v, out,
        *x.stride(), *v.stride(), *out.stride(),
        d1, d2, d3,
        BLOCK_1=BLOCK_1, BLOCK_2=BLOCK_2,
    )

    return out.reshape(bs + [d1, d3])
