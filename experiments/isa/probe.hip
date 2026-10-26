#include <hip/hip_runtime.h>
typedef short v2s __attribute__((ext_vector_type(2)));
typedef _Float16 v2h __attribute__((ext_vector_type(2)));
__global__ void k(int* o, const int* a, const int* b, const v2h* ha, const v2h* hb, float* f){
  int i=threadIdx.x;
  o[i]   = __builtin_amdgcn_sdot4(a[i], b[i], o[i], false);
  o[i+1] = __builtin_amdgcn_udot4(a[i], b[i], o[i+1], false);
  o[i+2] = __builtin_amdgcn_sdot8(a[i], b[i], o[i+2], false);
  o[i+3] = __builtin_amdgcn_udot8(a[i], b[i], o[i+3], false);
  f[i]   = __builtin_amdgcn_fdot2(ha[i], hb[i], f[i], false);
}
