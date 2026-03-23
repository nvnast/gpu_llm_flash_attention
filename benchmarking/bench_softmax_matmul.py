import torch
from itertools import product
from softmax_matmul.softmax_matmul import fused_softmax, softmax_mult
import pandas as pd
import numpy as np

VERBOSE = True

def time_loop(fn, iters, warmup=5):
    # Warmup phase
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    
    torch.cuda.memory.reset_peak_memory_stats()
    # Timing phase
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()  # Wait for previous work to finish
        start.record()            # Record start event
        fn()                      # Execute the function
        end.record()              # Record end event
        torch.cuda.synchronize()  # Wait for fn() to finish
        
        times.append(start.elapsed_time(end))
    return times, torch.cuda.memory.max_memory_allocated() / 1000000

if __name__ == "__main__":
    device = 'cuda'
    batch_sizes = [4, 16, 64, 256]
    d1s = [256, 1024, 4096]
    d2s = [256, 1024, 4096]
    d3s = [256, 1024, 4096]
    block_sizes = [32, 64, 128]
    
    df = pd.DataFrame(columns=['batch_size', 'd1', 'd2', 'd3', 'triton', 'BLOCK', 'forward_ms_mean', 'forward_ms_std', 'forward_peak_MiB'])
    for i, (batch_size, d1, d2, d3, block_size) in enumerate(product(batch_sizes, d1s, d2s, d3s, block_sizes)):
        x = torch.randn(batch_size, d1, d2)
        v = torch.randn(batch_size, d2, d3)

        if VERBOSE:
            print(f"batch_size={batch_size}, (d1,d2,d3)=({d1},{d2},{d3}), block_size={block_size}")

        # Triton
        times, mem = time_loop(
            lambda: fused_softmax(x, v, BLOCK_1=block_size, BLOCK_2=block_size),
            10
        )
        df._append([batch_size, d1, d2, d3, True, block_size, np.mean(times).item(), np.std(times).item(), mem])
        if VERBOSE:
            print(f"   with triton: {np.mean(times).item()}ms, {mem}MiB max usage")

        # PyTorch  TODO
        times, mem = time_loop(
            lambda: softmax_mult(x, v),
            10
        )
        df._append([batch_size, d1, d2, d3, False, np.nan, np.mean(times).item(), np.std(times).item(), mem])
        if VERBOSE:
            print(f"   with pytorch: {np.mean(times).item()}ms, {mem}MiB max usage\n")
            
    out_path = "outputs/softmax_matmul_benchmark.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")
