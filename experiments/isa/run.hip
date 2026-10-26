#include <hip/hip_runtime.h>
#include <cstdio>
typedef _Float16 v2h __attribute__((ext_vector_type(2)));
__global__ void k(float* f, int* o) {
  v2h a = {(_Float16)100.0f, (_Float16)-200.0f};
  v2h b = {(_Float16)3.0f, (_Float16)4.0f};
  f[0] = __builtin_amdgcn_fdot2(a, b, 0.0f, false);   // expect 300 - 800 = -500
  f[1] = __builtin_amdgcn_fdot2(a, b, 0.0f, true);    // clamp modifier
  int x8 = 0x7F80FF01;  // int8: 1, -1, -128, 127
  int y8 = 0x01010101;
  o[0] = __builtin_amdgcn_sdot4(x8, y8, 0, false);    // 1-1-128+127 = -1
  int x4 = 0x8F7A1E01;  // nibbles low->high: 1,0,14->-2,1,10->-6,7,15->-1,8->-8
  int y4 = 0x11111111;
  o[1] = __builtin_amdgcn_sdot8(x4, y4, 0, false);    // sum of signed nibbles
  o[2] = __builtin_amdgcn_udot8(x4, y4, 0, false);    // sum of unsigned nibbles
}
int main() {
  float* f; int* o; hipMallocManaged(&f, 8); hipMallocManaged(&o, 16);
  k<<<1, 1>>>(f, o); hipDeviceSynchronize();
  printf("fdot2 noclamp=%g clamp=%g | sdot4=%d sdot8=%d udot8=%d\n", f[0], f[1], o[0], o[1], o[2]);
}
