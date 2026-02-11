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

#include "./kernels.h"

int main(void) {
  if (args.kernel < KERNEL_COUNT) {
    return kernels[args.kernel]();
  } else {
    // invalid kernel ID
    return -1;
  }
}
