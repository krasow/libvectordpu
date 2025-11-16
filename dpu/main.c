#include <alloc.h>
#include <barrier.h>
#include <common.h>
#include <defs.h>
#include <mram.h>
#include <stdint.h>

__host DPU_LAUNCH_ARGS args;

BARRIER_INIT(my_barrier, NR_TASKLETS);

#include "binary.inl"
#include "reduce.inl"
#include "unary.inl"

#define UNARY_KERNELS(TYPE) \
    unary_##TYPE##_negate,  \
    unary_##TYPE##_abs

#define BINARY_KERNELS(TYPE) \
    binary_##TYPE##_add,     \
    binary_##TYPE##_subtract

#define REDUCTION_KERNELS(TYPE) \
    reduction_##TYPE##_sum,     \
    reduction_##TYPE##_product, \
    reduction_##TYPE##_max,     \
    reduction_##TYPE##_min

#define ALL_UNARY_KERNELS \
    UNARY_KERNELS(float), \
    UNARY_KERNELS(int), \
    UNARY_KERNELS(double)

#define ALL_BINARY_KERNELS \
    BINARY_KERNELS(float), \
    BINARY_KERNELS(int), \
    BINARY_KERNELS(double)

#define ALL_REDUCTION_KERNELS \
    REDUCTION_KERNELS(float), \
    REDUCTION_KERNELS(int), \
    REDUCTION_KERNELS(double)

int (*kernels[KERNEL_COUNT])(void) = {
    ALL_UNARY_KERNELS,
    ALL_BINARY_KERNELS,
    ALL_REDUCTION_KERNELS
};

int main(void) {
  // args.kernel indicates which kernel to run
  if (args.kernel < KERNEL_COUNT) {
    return kernels[args.kernel]();
  } else {
    // invalid kernel ID
    return -1;
  }
}