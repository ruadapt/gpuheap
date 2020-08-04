#ifndef UTIL_CUH
#define UTIL_CUH

#define DECLARE_SMEM(Type, Name, Count) Type * Name = (Type *)(smem + curr_smem_offset); curr_smem_offset += sizeof(Type) * (Count); curr_smem_offset = (curr_smem_offset + 3) & ~0x03

#define SINGLE_THREADED if(threadIdx.x == 0)

#define ALIGN_SMEM_8 curr_smem_offset = (curr_smem_offset + 7) & ~0x07

#endif