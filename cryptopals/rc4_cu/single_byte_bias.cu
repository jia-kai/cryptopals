#include "./rc4.cuh"
#include "./rng.cuh"
#include "./cuda_helper.cuh"

constexpr uint32_t
    KEY_LEN = 16,       //!< key length in bytes
    STREAM_LEN = 512,   //!< length of key streams to be generated
    OUTPUT_TRANSACTION_SIZE = 16;   //!< number of bytes to be written once
static_assert(KEY_LEN % 8 == 0, "KEY_LEN must be multiples of 8");
static_assert(STREAM_LEN % OUTPUT_TRANSACTION_SIZE == 0,
        "invalid STREAM_LEN and OUTPUT_TRANSACTION_SIZE config");


struct ThreadWorkspace {
    uint8_t rc4_state[256];
    uint8_t key[KEY_LEN];
    uint32_t _pad;  //!< to reduce bank conflict
};

struct OutputTransaction {
    uint8_t data[OUTPUT_TRANSACTION_SIZE];
};

/*!
 * \brief generate RC4 key streams and store it in output global memory
 * \param seed0 seed for the xorshift RNG
 * \param seed1 seed for the xorshift RNG
 * \param nr_sample number of keys to be sampled per thread
 * \param output keystream in global memory, layout: [grid_size, nr_sample,
 *      STREAM_LEN / OUTPUT_TRANSACTION_SIZE, block_size,
 *      OUTPUT_TRANSACTION_SIZE]
 */
__global__ void rc4_gen_keystream(
        uint64_t seed0, uint64_t seed1,
        uint32_t nr_sample, uint8_t *output) {
    extern __shared__ ThreadWorkspace thread_workspace_storage[];
    uint64_t xorshift128_state[2];
    xorshift128_state[0] = threadIdx.x ^ seed0;
    xorshift128_state[1] = blockIdx.x ^ seed1;
    output += blockIdx.x * nr_sample * STREAM_LEN * blockDim.x;

    OutputTransaction *out_tx = reinterpret_cast<OutputTransaction*>(output);
    out_tx += threadIdx.x;

    ThreadWorkspace *workspace = thread_workspace_storage + threadIdx.x;

    for (uint32_t sample_idx = 0; sample_idx < nr_sample; ++ sample_idx) {
#pragma unroll
        for (uint32_t i = 0; i < KEY_LEN / 8; ++ i) {
            uint64_t cur = xorshift128plus(xorshift128_state);
            // use uint32_t to avoid align address
            uint32_t *dst = reinterpret_cast<uint32_t*>(workspace->key);
            dst[i * 2 + 0] = cur;
            dst[i * 2 + 1] = cur >> 32;
        }
        rc4_key_sched<KEY_LEN>(workspace->rc4_state, workspace->key);
        uint8_t rc4_i = 0, rc4_j = 0;

        for (uint32_t i = 0; i < STREAM_LEN; i += OUTPUT_TRANSACTION_SIZE) {
            OutputTransaction cur;
#pragma unroll
            for (uint32_t j = 0; j < OUTPUT_TRANSACTION_SIZE; ++ j) {
                cur.data[j] = rc4_next(workspace->rc4_state, rc4_i, rc4_j);
            }

            *out_tx = cur;
            out_tx += blockDim.x;
        }
    }
}

//! rc4 key stream implemented on CPU, for testing purpose
class RC4StreamCPU {
    uint8_t m_state[256], m_i = 0, m_j = 0;

    public:
        RC4StreamCPU(const uint8_t *key, size_t key_len) {
            for (int i = 0; i < 256; ++ i) {
                m_state[i] = i;
            }
            int j = 0;
            for (int i=0; i < 256; ++i) {
                j = (j + m_state[i] + key[i % key_len]) % 256;
                std::swap(m_state[i], m_state[j]);
            }
        }

        uint8_t next() {
            uint8_t i = m_i, j = m_j;
            uint8_t *s = m_state;
            ++ i;
            j += s[i];
            std::swap(s[i], s[j]);
            auto ret = s[(s[i] + s[j]) % 256];
            m_i = i; m_j = j;
            return ret;
        }
};

//! testcase for rc4_gen_keystream
void test_rc4_gen_keystream() {
    constexpr int
        GRID_SIZE = 4, BLOCK_SIZE = 32, NR_SAMPLE = 3,
        OUTPUT_SIZE = GRID_SIZE * BLOCK_SIZE * NR_SAMPLE * STREAM_LEN;
    constexpr uint64_t SEED0 = 123, SEED1 = 456;
    auto output_gpu = cuda_new_arr<uint8_t>(OUTPUT_SIZE);
    rc4_gen_keystream <<< GRID_SIZE, BLOCK_SIZE,
                      sizeof(ThreadWorkspace) * BLOCK_SIZE >>>(
                              SEED0, SEED1, NR_SAMPLE, output_gpu.get());
    std::unique_ptr<uint8_t[]> output_cpu(new uint8_t[OUTPUT_SIZE]);
    CUDA_CHECK(cudaMemcpy(output_cpu.get(), output_gpu.get(), OUTPUT_SIZE,
                cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());
    for (int block = 0; block < GRID_SIZE; ++ block) {
        for (int thread = 0; thread < BLOCK_SIZE; ++ thread) {
            uint64_t xorshift128_state[2] = {thread ^ SEED0, block ^ SEED1};
            for (int sample_idx = 0; sample_idx < NR_SAMPLE; ++ sample_idx) {
                uint8_t key[KEY_LEN];
                for (uint32_t i = 0; i < KEY_LEN / 8; ++ i) {
                    reinterpret_cast<uint64_t*>(key)[i] = xorshift128plus(
                            xorshift128_state);
                }
                RC4StreamCPU rc4{key, KEY_LEN};

                for (uint32_t i = 0; i < STREAM_LEN; ++ i) {
                    uint32_t x = i / OUTPUT_TRANSACTION_SIZE,
                             y = i % OUTPUT_TRANSACTION_SIZE;
                    uint8_t get = output_cpu[
                        block * (NR_SAMPLE * STREAM_LEN * BLOCK_SIZE) +
                        sample_idx * (STREAM_LEN * BLOCK_SIZE) +
                        x * (BLOCK_SIZE * OUTPUT_TRANSACTION_SIZE) +
                        thread * OUTPUT_TRANSACTION_SIZE +
                        y
                    ];
                    uint8_t expect = rc4.next();
                    if (get != expect) {
                        fprintf(stderr, "rc4 check failed: at "
                                "block=%d thread=%d sample=%d stream=%d: "
                                "expect=%d get=%d\n",
                                block, thread, sample_idx, i,
                                expect, get);
                        abort();
                    }
                }
            }
        }
    }
    printf("test_rc4_gen_keystream() passed\n");
}

int main() {
    test_rc4_gen_keystream();
}

// vim: ft=cuda syntax=cuda.doxygen foldmethod=marker foldmarker=f{{{,f}}}
