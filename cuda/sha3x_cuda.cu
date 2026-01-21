#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>

#define CUDA_OK(x) do { if ((x) != cudaSuccess) return 1; } while(0)

extern "C" {

struct __align__(8) sha3x_batch_result_t {
    uint32_t found;
    uint32_t _pad0;
    uint64_t nonce;
    uint8_t  hash[32];
    uint64_t hashes_done;
    float    elapsed_ms;
    uint32_t _pad1;
};

// Host-mapped result that GPU writes on success
struct __align__(8) host_mapped_result_t {
    volatile uint32_t found;
    uint32_t _pad0;
    uint64_t nonce;
    uint8_t  hash[32];
};

static int g_blocks  = 0;
static int g_threads = 0;

static host_mapped_result_t* g_hres = nullptr;
static host_mapped_result_t* g_dres = nullptr;

static cudaEvent_t g_ev_start = nullptr;
static cudaEvent_t g_ev_stop  = nullptr;

// Job constants
__device__ __constant__ uint64_t c_mining_hash[4]; // 32 bytes as 4 LE u64 lanes
__device__ __constant__ uint64_t c_target_u64;     // compare against leading_u64_be
__device__ __constant__ uint32_t c_xn_enabled;
__device__ __constant__ uint32_t c_xn;

__device__ unsigned int       d_found = 0;
__device__ unsigned long long d_nonce = 0;

__device__ __forceinline__ uint64_t rotl64(uint64_t x, unsigned int n) {
    return (x << n) | (x >> (64u - n));
}

__device__ __forceinline__ uint64_t bswap64(uint64_t x) {
    unsigned int lo = (unsigned int)(x);
    unsigned int hi = (unsigned int)(x >> 32);
    lo = __byte_perm(lo, 0, 0x0123);
    hi = __byte_perm(hi, 0, 0x0123);
    return ((uint64_t)lo << 32) | (uint64_t)hi;
}

// Coherent global load (bypass L1) so other SMs see the winning atomic quickly.
__device__ __forceinline__ unsigned int load_found_cg() {
    unsigned int v;
    const unsigned int* p = (const unsigned int*)&d_found;
    asm volatile ("ld.global.cg.u32 %0, [%1];" : "=r"(v) : "l"(p));
    return v;
}

#define KECCAKF_ROUND(RC) do {                                                          \
  uint64_t c0 = a0 ^ a5 ^ a10 ^ a15 ^ a20;                                              \
  uint64_t c1 = a1 ^ a6 ^ a11 ^ a16 ^ a21;                                              \
  uint64_t c2 = a2 ^ a7 ^ a12 ^ a17 ^ a22;                                              \
  uint64_t c3 = a3 ^ a8 ^ a13 ^ a18 ^ a23;                                              \
  uint64_t c4 = a4 ^ a9 ^ a14 ^ a19 ^ a24;                                              \
  uint64_t d0 = c4 ^ rotl64(c1, 1);                                                     \
  uint64_t d1 = c0 ^ rotl64(c2, 1);                                                     \
  uint64_t d2 = c1 ^ rotl64(c3, 1);                                                     \
  uint64_t d3 = c2 ^ rotl64(c4, 1);                                                     \
  uint64_t d4 = c3 ^ rotl64(c0, 1);                                                     \
  a0  ^= d0; a5  ^= d0; a10 ^= d0; a15 ^= d0; a20 ^= d0;                                \
  a1  ^= d1; a6  ^= d1; a11 ^= d1; a16 ^= d1; a21 ^= d1;                                \
  a2  ^= d2; a7  ^= d2; a12 ^= d2; a17 ^= d2; a22 ^= d2;                                \
  a3  ^= d3; a8  ^= d3; a13 ^= d3; a18 ^= d3; a23 ^= d3;                                \
  a4  ^= d4; a9  ^= d4; a14 ^= d4; a19 ^= d4; a24 ^= d4;                                \
                                                                                        \
  uint64_t t = a1;                                                                      \
  a1  = rotl64(a6,  44);                                                                \
  a6  = rotl64(a9,  20);                                                                \
  a9  = rotl64(a22, 61);                                                                \
  a22 = rotl64(a14, 39);                                                                \
  a14 = rotl64(a20, 18);                                                                \
  a20 = rotl64(a2,  62);                                                                \
  a2  = rotl64(a12, 43);                                                                \
  a12 = rotl64(a13, 25);                                                                \
  a13 = rotl64(a19, 8);                                                                 \
  a19 = rotl64(a23, 56);                                                                \
  a23 = rotl64(a15, 41);                                                                \
  a15 = rotl64(a4,  27);                                                                \
  a4  = rotl64(a24, 14);                                                                \
  a24 = rotl64(a21, 2);                                                                 \
  a21 = rotl64(a8,  55);                                                                \
  a8  = rotl64(a16, 45);                                                                \
  a16 = rotl64(a5,  36);                                                                \
  a5  = rotl64(a3,  28);                                                                \
  a3  = rotl64(a18, 21);                                                                \
  a18 = rotl64(a17, 15);                                                                \
  a17 = rotl64(a11, 10);                                                                \
  a11 = rotl64(a7,  6);                                                                 \
  a7  = rotl64(a10, 3);                                                                 \
  a10 = rotl64(t,   1);                                                                 \
                                                                                        \
  uint64_t b0, b1, b2, b3, b4;                                                          \
  b0 = a0;  b1 = a1;  b2 = a2;  b3 = a3;  b4 = a4;                                      \
  a0 = b0 ^ ((~b1) & b2);                                                               \
  a1 = b1 ^ ((~b2) & b3);                                                               \
  a2 = b2 ^ ((~b3) & b4);                                                               \
  a3 = b3 ^ ((~b4) & b0);                                                               \
  a4 = b4 ^ ((~b0) & b1);                                                               \
  b0 = a5;  b1 = a6;  b2 = a7;  b3 = a8;  b4 = a9;                                      \
  a5 = b0 ^ ((~b1) & b2);                                                               \
  a6 = b1 ^ ((~b2) & b3);                                                               \
  a7 = b2 ^ ((~b3) & b4);                                                               \
  a8 = b3 ^ ((~b4) & b0);                                                               \
  a9 = b4 ^ ((~b0) & b1);                                                               \
  b0 = a10; b1 = a11; b2 = a12; b3 = a13; b4 = a14;                                     \
  a10 = b0 ^ ((~b1) & b2);                                                              \
  a11 = b1 ^ ((~b2) & b3);                                                              \
  a12 = b2 ^ ((~b3) & b4);                                                              \
  a13 = b3 ^ ((~b4) & b0);                                                              \
  a14 = b4 ^ ((~b0) & b1);                                                              \
  b0 = a15; b1 = a16; b2 = a17; b3 = a18; b4 = a19;                                     \
  a15 = b0 ^ ((~b1) & b2);                                                              \
  a16 = b1 ^ ((~b2) & b3);                                                              \
  a17 = b2 ^ ((~b3) & b4);                                                              \
  a18 = b3 ^ ((~b4) & b0);                                                              \
  a19 = b4 ^ ((~b0) & b1);                                                              \
  b0 = a20; b1 = a21; b2 = a22; b3 = a23; b4 = a24;                                     \
  a20 = b0 ^ ((~b1) & b2);                                                              \
  a21 = b1 ^ ((~b2) & b3);                                                              \
  a22 = b2 ^ ((~b3) & b4);                                                              \
  a23 = b3 ^ ((~b4) & b0);                                                              \
  a24 = b4 ^ ((~b0) & b1);                                                              \
                                                                                        \
  a0 ^= (RC);                                                                           \
} while (0)

#define KECCAKF_1600_PERMUTE_24() do {                                                  \
  KECCAKF_ROUND(0x0000000000000001ULL);                                                 \
  KECCAKF_ROUND(0x0000000000008082ULL);                                                 \
  KECCAKF_ROUND(0x800000000000808AULL);                                                 \
  KECCAKF_ROUND(0x8000000080008000ULL);                                                 \
  KECCAKF_ROUND(0x000000000000808BULL);                                                 \
  KECCAKF_ROUND(0x0000000080000001ULL);                                                 \
  KECCAKF_ROUND(0x8000000080008081ULL);                                                 \
  KECCAKF_ROUND(0x8000000000008009ULL);                                                 \
  KECCAKF_ROUND(0x000000000000008AULL);                                                 \
  KECCAKF_ROUND(0x0000000000000088ULL);                                                 \
  KECCAKF_ROUND(0x0000000080008009ULL);                                                 \
  KECCAKF_ROUND(0x000000008000000AULL);                                                 \
  KECCAKF_ROUND(0x000000008000808BULL);                                                 \
  KECCAKF_ROUND(0x800000000000008BULL);                                                 \
  KECCAKF_ROUND(0x8000000000008089ULL);                                                 \
  KECCAKF_ROUND(0x8000000000008003ULL);                                                 \
  KECCAKF_ROUND(0x8000000000008002ULL);                                                 \
  KECCAKF_ROUND(0x8000000000000080ULL);                                                 \
  KECCAKF_ROUND(0x000000000000800AULL);                                                 \
  KECCAKF_ROUND(0x800000008000000AULL);                                                 \
  KECCAKF_ROUND(0x8000000080008081ULL);                                                 \
  KECCAKF_ROUND(0x8000000000008080ULL);                                                 \
  KECCAKF_ROUND(0x0000000080000001ULL);                                                 \
  KECCAKF_ROUND(0x8000000080008008ULL);                                                 \
} while (0)

// SHA3-256 for 41-byte message:
// nonce(8 LE) || mining_hash(32) || 0x01
// Padding: byte41 ^= 0x06, byte135 ^= 0x80
__device__ __forceinline__ void sha3_256_41(uint64_t mh0, uint64_t mh1, uint64_t mh2, uint64_t mh3, uint64_t nonce, uint64_t out[4]) {
    uint64_t a0  = nonce;
    uint64_t a1  = mh0;
    uint64_t a2  = mh1;
    uint64_t a3  = mh2;
    uint64_t a4  = mh3;

    uint64_t a5  = 0x0000000000000601ULL; // byte40=0x01, byte41=0x06
    uint64_t a6  = 0, a7  = 0, a8  = 0, a9  = 0;
    uint64_t a10 = 0, a11 = 0, a12 = 0, a13 = 0, a14 = 0, a15 = 0;
    uint64_t a16 = 0x8000000000000000ULL;
    uint64_t a17 = 0, a18 = 0, a19 = 0, a20 = 0, a21 = 0, a22 = 0, a23 = 0, a24 = 0;

    KECCAKF_1600_PERMUTE_24();

    out[0] = a0; out[1] = a1; out[2] = a2; out[3] = a3;
}

// SHA3-256 for 32-byte message:
// Padding: byte32 ^= 0x06, byte135 ^= 0x80
__device__ __forceinline__ void sha3_256_32(const uint64_t in[4], uint64_t out[4]) {
    uint64_t a0  = in[0];
    uint64_t a1  = in[1];
    uint64_t a2  = in[2];
    uint64_t a3  = in[3];
    uint64_t a4  = 0x0000000000000006ULL;
    uint64_t a5  = 0, a6  = 0, a7  = 0, a8  = 0, a9  = 0;
    uint64_t a10 = 0, a11 = 0, a12 = 0, a13 = 0, a14 = 0, a15 = 0;
    uint64_t a16 = 0x8000000000000000ULL;
    uint64_t a17 = 0, a18 = 0, a19 = 0, a20 = 0, a21 = 0, a22 = 0, a23 = 0, a24 = 0;

    KECCAKF_1600_PERMUTE_24();

    out[0] = a0; out[1] = a1; out[2] = a2; out[3] = a3;
}

__device__ __forceinline__ void sha3x(uint64_t mh0, uint64_t mh1, uint64_t mh2, uint64_t mh3, uint64_t nonce, uint64_t out[4]) {
    uint64_t t1[4], t2[4];
    sha3_256_41(mh0, mh1, mh2, mh3, nonce, t1);
    sha3_256_32(t1, t2);
    sha3_256_32(t2, out);
}

__global__ __launch_bounds__(256) void sha3x_kernel(
    uint64_t start,
    uint32_t iters_per_thread,
    host_mapped_result_t* __restrict__ out
) {
    const uint64_t tid  = (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;
    const uint64_t step = (uint64_t)gridDim.x  * (uint64_t)blockDim.x;

    const uint64_t mh0 = c_mining_hash[0];
    const uint64_t mh1 = c_mining_hash[1];
    const uint64_t mh2 = c_mining_hash[2];
    const uint64_t mh3 = c_mining_hash[3];

    const uint32_t xn_en = c_xn_enabled;
    const uint64_t xn    = (uint64_t)(c_xn & 0xFFFFu);

    uint64_t ctr   = start + tid; // xn mode counter
    uint64_t nonce = start + tid; // plain nonce

#pragma unroll 1
    for (uint32_t i = 0; i < iters_per_thread; ++i) {
        if ((i & 0xFFu) == 0u && load_found_cg()) return;

        const uint64_t n = xn_en ? ((ctr << 16) | xn) : nonce;

        uint64_t h[4];
        sha3x(mh0, mh1, mh2, mh3, n, h);

        // leading 8 bytes of hash as BE
        const uint64_t lead_be = bswap64(h[0]);

        if (lead_be <= c_target_u64) {
            if (atomicCAS(&d_found, 0u, 1u) == 0u) {
                d_nonce = (unsigned long long)n;

                out->nonce = n;
                ((uint64_t*)out->hash)[0] = h[0];
                ((uint64_t*)out->hash)[1] = h[1];
                ((uint64_t*)out->hash)[2] = h[2];
                ((uint64_t*)out->hash)[3] = h[3];

                __threadfence_system();
                out->found = 1u;
            }
            return;
        }

        if (xn_en) ctr += step;
        else       nonce += step;
    }
}

int sha3x_cuda_init(int device, int requested_blocks, int requested_threads, int* out_blocks, int* out_threads) {
    CUDA_OK(cudaSetDevice(device));

    cudaDeviceProp prop;
    CUDA_OK(cudaGetDeviceProperties(&prop, device));

    int threads = (requested_threads > 0) ? requested_threads : 256;
    int blocks  = (requested_blocks  > 0) ? requested_blocks  : (prop.multiProcessorCount * 4);

    if (blocks < 1) blocks = 1;
    if (blocks > 65535) blocks = 65535;

    g_blocks  = blocks;
    g_threads = threads;

    if (out_blocks)  *out_blocks  = g_blocks;
    if (out_threads) *out_threads = g_threads;

    if (!g_hres) {
        CUDA_OK(cudaSetDeviceFlags(cudaDeviceMapHost));
        CUDA_OK(cudaHostAlloc((void**)&g_hres, sizeof(host_mapped_result_t), cudaHostAllocMapped));
        memset((void*)g_hres, 0, sizeof(host_mapped_result_t));
        CUDA_OK(cudaHostGetDevicePointer((void**)&g_dres, (void*)g_hres, 0));
    }

    if (!g_ev_start) CUDA_OK(cudaEventCreate(&g_ev_start));
    if (!g_ev_stop)  CUDA_OK(cudaEventCreate(&g_ev_stop));

    {
        uint32_t z32 = 0;
        uint64_t t = 0xFFFFFFFFFFFFFFFFULL;
        CUDA_OK(cudaMemcpyToSymbol(c_target_u64, &t, sizeof(uint64_t)));
        CUDA_OK(cudaMemcpyToSymbol(c_xn_enabled, &z32, sizeof(uint32_t)));
        CUDA_OK(cudaMemcpyToSymbol(c_xn, &z32, sizeof(uint32_t)));
        CUDA_OK(cudaMemcpyToSymbol(d_found, &z32, sizeof(uint32_t)));
        uint64_t z64 = 0;
        CUDA_OK(cudaMemcpyToSymbol(d_nonce, &z64, sizeof(uint64_t)));
    }

    return 0;
}

int sha3x_cuda_set_job(const uint8_t* mining_hash32, uint64_t target_u64, uint16_t xn, int xn_enabled) {
    uint64_t mh[4];
    // Changing endianness
    // This is potential UB if not 8-byte aligned, but it always is
    mh[0] = *reinterpret_cast<const uint64_t*>(mining_hash32 + 0);
    mh[1] = *reinterpret_cast<const uint64_t*>(mining_hash32 + 8);
    mh[2] = *reinterpret_cast<const uint64_t*>(mining_hash32 + 16);
    mh[3] = *reinterpret_cast<const uint64_t*>(mining_hash32 + 24);

    uint32_t xn_en = (xn_enabled != 0) ? 1u : 0u;
    uint32_t xn32  = (uint32_t)xn;

    // Job changes are rare - keep it simple/safe.
    CUDA_OK(cudaDeviceSynchronize());

    CUDA_OK(cudaMemcpyToSymbol(c_mining_hash, mh, sizeof(mh)));
    CUDA_OK(cudaMemcpyToSymbol(c_target_u64, &target_u64, sizeof(uint64_t)));
    CUDA_OK(cudaMemcpyToSymbol(c_xn_enabled, &xn_en, sizeof(uint32_t)));
    CUDA_OK(cudaMemcpyToSymbol(c_xn, &xn32, sizeof(uint32_t)));
    return 0;
}

int sha3x_cuda_run_batch(uint64_t start, uint32_t iters_per_thread, sha3x_batch_result_t* out_host) {
    if (!g_hres || !g_dres || !g_ev_start || !g_ev_stop) return 2;

    // Reset per-batch flags (host + device)
    g_hres->found = 0u;
    g_hres->nonce = 0ull;
    // Hash left as-is; only read when found==1.

    {
        uint32_t z32 = 0;
        CUDA_OK(cudaMemcpyToSymbol(d_found, &z32, sizeof(uint32_t)));
        uint64_t z64 = 0;
        CUDA_OK(cudaMemcpyToSymbol(d_nonce, &z64, sizeof(uint64_t)));
    }

    CUDA_OK(cudaEventRecord(g_ev_start, 0));
    sha3x_kernel<<<g_blocks, g_threads>>>(start, iters_per_thread, g_dres);
    CUDA_OK(cudaEventRecord(g_ev_stop, 0));
    CUDA_OK(cudaEventSynchronize(g_ev_stop));

    float ms = 0.0f;
    CUDA_OK(cudaEventElapsedTime(&ms, g_ev_start, g_ev_stop));

    sha3x_batch_result_t r;
    r.found = g_hres->found;
    r._pad0 = 0;
    r.nonce = g_hres->nonce;
    memcpy(r.hash, (const void*)g_hres->hash, 32);
    r.hashes_done = (uint64_t)g_blocks * (uint64_t)g_threads * (uint64_t)iters_per_thread;
    r.elapsed_ms = ms;
    r._pad1 = 0;

    *out_host = r;
    return 0;
}

void sha3x_cuda_shutdown() {
    if (g_ev_start) { cudaEventDestroy(g_ev_start); g_ev_start = nullptr; }
    if (g_ev_stop)  { cudaEventDestroy(g_ev_stop);  g_ev_stop  = nullptr; }
    if (g_hres)     { cudaFreeHost((void*)g_hres);  g_hres = nullptr; g_dres = nullptr; }
}

}
