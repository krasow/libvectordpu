#include <alloc.h>
#include <barrier.h>
#include <common.h>
#include <config.h>
#include <defs.h>
#include <mram.h>
#include <perfcounter.h>
#include <stdint.h>
#include <stdio.h>

__host DPU_LAUNCH_ARGS args;

BARRIER_INIT(my_barrier, NR_TASKLETS);
__dma_aligned uint64_t reduction_scratchpad[NR_TASKLETS];

// Shared WRAM workspace for tasklets to avoid stack overflow.
// Max size: input (1) + operands (3) + stack (4) = 8 blocks.
#define TASKLET_WORKSPACE_SIZE (8 * BLOCK_SIZE * MINIMUM_WRITE_SIZE)
__dma_aligned uint8_t dpu_workspace[NR_TASKLETS][TASKLET_WORKSPACE_SIZE];

#include "./kernels.h"

int main(void) {
  if (args.kernel < KERNEL_COUNT) {
    return kernels[args.kernel]();
  } else {
    return -1;
  }
}
