#include "./rc4.cuh"
#include "./rng.cuh"
#include "./cuda_helper.cuh"

#include <atomic>
#include <mutex>
#include <cassert>

constexpr uint32_t
    KEY_LEN = 16,       //!< key length in bytes
    STREAM_LEN = 512,   //!< length of key streams to be generated
    KEY_STREAM_TRX_SIZE = 16,   //!< number of bytes to be written once
    BYTE_COUNT_STREAM_UNROLL = 16;  //!< stream unroll in get_byte_counts_kern
static_assert(KEY_LEN % 8 == 0, "KEY_LEN must be multiples of 8");
static_assert(STREAM_LEN % KEY_STREAM_TRX_SIZE == 0,
        "invalid STREAM_LEN and KEY_STREAM_TRX_SIZE config");

struct KeyStreamThreadWorkspace {
    uint8_t rc4_state[256];
    uint8_t key[KEY_LEN];
    uint32_t _pad;  //!< to reduce bank conflict
};

struct __align__(KEY_STREAM_TRX_SIZE) KeyStreamTrx {
    uint32_t _data[KEY_STREAM_TRX_SIZE / 4];
};
static_assert(sizeof(KeyStreamTrx) == KEY_STREAM_TRX_SIZE, "bad KeyStreamTrx");

union KeyStreamTrxUnion {
    KeyStreamTrx trx;
    uint32_t u32[KEY_STREAM_TRX_SIZE / 4];
    uint8_t u8[KEY_STREAM_TRX_SIZE];
};
static_assert(sizeof(KeyStreamTrxUnion) == sizeof(KeyStreamTrx) &&
        KEY_STREAM_TRX_SIZE % 4 == 0, "bad KeyStreamTrxUnion");

/*!
 * \brief generate RC4 key streams and store it in output global memory
 * \param seed0 seed for the xorshift RNG
 * \param seed1 seed for the xorshift RNG
 * \param nr_sample number of keys to be sampled per thread
 * \param output keystream in global memory, layout: [grid_size, nr_sample,
 *      STREAM_LEN / KEY_STREAM_TRX_SIZE, block_size, KEY_STREAM_TRX_SIZE]
 *
 * required shared memory: sizeof(KeyStreamThreadWorkspace) * blockDim.x
 */
__global__ void rc4_gen_keystream_kern(
        uint64_t seed0, uint64_t seed1,
        uint32_t nr_sample, uint8_t *output) {
    extern __shared__ KeyStreamThreadWorkspace thread_workspace_storage[];
    uint64_t xorshift128_state[2];
    xorshift128_state[0] = threadIdx.x ^ seed0;
    xorshift128_state[1] = blockIdx.x ^ seed1;
    output += blockIdx.x * nr_sample * STREAM_LEN * blockDim.x;

    KeyStreamTrx * __restrict__ out_tx =
        reinterpret_cast<KeyStreamTrx*>(output);
    out_tx += threadIdx.x;

    auto workspace = thread_workspace_storage + threadIdx.x;

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

        for (uint32_t i = 0; i < STREAM_LEN; i += KEY_STREAM_TRX_SIZE) {
            KeyStreamTrxUnion cur;
#pragma unroll
            for (uint32_t j = 0; j < KEY_STREAM_TRX_SIZE; ++ j) {
                cur.u8[j] = rc4_next(workspace->rc4_state, rc4_i, rc4_j);
            }

            *out_tx = cur.trx;
            out_tx += blockDim.x;
        }
    }
}

void rc4_gen_keystream(
        int grid_size, int block_size,
        uint64_t seed0, uint64_t seed1,
        uint32_t nr_sample, uint8_t *output) {
    assert(reinterpret_cast<uintptr_t>(output) % KEY_STREAM_TRX_SIZE == 0);
    rc4_gen_keystream_kern <<< grid_size, block_size,
        sizeof(KeyStreamThreadWorkspace) * block_size >>> (
                seed0, seed1, nr_sample, output);
}

/*!
 * \brief get count of occurrences of bytes at each stream location
 * \param size_x size of input
 * \param size_y size of input
 * \param input input byte streams, layout:
 *      [size_x, STREAM_LEN / KEY_STREAM_TRX_SIZE, size_y, KEY_STREAM_TRX_SIZE]
 * \param[in,out] output byte occurrence counts that would be increased inplace
 *      layout: [gridDim.x, STREAM_LEN, 256]
 *
 * requirements:
 *      1. shared memory: blockDim.x * (blockDim.x + 4) + blockDim.x * 256
 *      2. (blockDim.x * gridDim.x * BYTE_COUNT_STREAM_UNROLL) divides
 *              (size_x * size_y)
 *      3. KEY_STREAM_TRX_SIZE divides blockDim.x
 */
__global__ void get_byte_counts_kern(
        uint32_t size_x, uint32_t size_y,
        const KeyStreamTrx * __restrict__ input,
        uint32_t * __restrict__ output) {
    constexpr uint32_t
        STREAM_UNROLL = BYTE_COUNT_STREAM_UNROLL,
        NR_TRX_PER_STREAM = STREAM_LEN / KEY_STREAM_TRX_SIZE;

    extern __shared__ uint8_t shared_mem_storage[];
    const uint32_t
        nr_stream = size_x * size_y,
        //! stride for matrices in shared memory, to avoid bank conflict
        stream_cache_stride = blockDim.x + 4,
        stream_cache_size = blockDim.x * stream_cache_stride,
        byte_cnt_size = blockDim.x * 256,
        stream_idx_begin =
            blockIdx.x * blockDim.x * STREAM_UNROLL + threadIdx.x,
        stream_idx_strd = blockDim.x * gridDim.x * STREAM_UNROLL,
        nr_trx_per_block = blockDim.x / KEY_STREAM_TRX_SIZE;

    output += blockIdx.x * STREAM_LEN * 256;

    uint8_t *stream_cache = shared_mem_storage, //!< [stream, position]
            *local_stream_cache =
                stream_cache + threadIdx.x * stream_cache_stride,
            //! [position, 256]
            *byte_cnt = shared_mem_storage + stream_cache_size,
            *byte_cnt_local = byte_cnt + threadIdx.x * 256;

    for (uint32_t trx_pos = 0; trx_pos < NR_TRX_PER_STREAM;
            trx_pos += nr_trx_per_block) {
        // process bytes [byte_pos:byte_pos+blockDim.x] from all streams

        uint32_t
            nr_trx_cur = min(nr_trx_per_block, NR_TRX_PER_STREAM - trx_pos);

        for (uint32_t stream_idx0 = stream_idx_begin; stream_idx0 < nr_stream;
                stream_idx0 += stream_idx_strd) {

            cuda_bzero_shared4_async(byte_cnt, byte_cnt_size);

            // read from multiple streams and incr byte_cnt
            for (uint32_t stream_unroll = 0; stream_unroll < STREAM_UNROLL;
                    ++ stream_unroll) {
                uint32_t stream_idx = stream_idx0 + stream_unroll * blockDim.x,
                         stream_x = stream_idx / size_y,
                         stream_y = stream_idx - stream_x * size_y;
                uint32_t inp_base_offset =
                    (stream_x * NR_TRX_PER_STREAM + trx_pos) * size_y +
                    stream_y;

                // load inputs into stream_cache
                auto dst = reinterpret_cast<uint32_t*>(local_stream_cache);
                for (uint32_t i = 0; i < nr_trx_cur; ++ i) {
                    uint32_t off = inp_base_offset + i * size_y;

                    // prefetch the next line
                    cuda_prefetch_l1(&input[off + size_y]);

                    KeyStreamTrxUnion inp;
                    inp.trx = input[off];
#pragma unroll
                    for (uint32_t j = 0; j < KEY_STREAM_TRX_SIZE / 4; ++ j) {
                        *(dst ++) = inp.u32[j];
                    }
                }
                __syncthreads();

                // read stream_cache and incr byte_cnt
                if (threadIdx.x < nr_trx_cur * KEY_STREAM_TRX_SIZE) {
                    auto col = stream_cache + threadIdx.x;
                    for (uint32_t i = 0; i < blockDim.x; ++ i) {
                        ++ byte_cnt_local[col[i * stream_cache_stride]];
                    }
                }
            }

            // add byte_cnt to output
            __syncthreads();
            uint32_t nr_out = nr_trx_cur * KEY_STREAM_TRX_SIZE * 256,
                     out_off = trx_pos * KEY_STREAM_TRX_SIZE * 256;
            for (uint32_t i = threadIdx.x; i < nr_out; i += blockDim.x) {
                output[out_off + i] += byte_cnt[i];
            }
            __syncthreads();
        }
    }
}

void get_byte_counts(
        int grid_size, int block_size,
        uint32_t size_x, uint32_t size_y,
        const uint8_t* __restrict__ input,
        uint32_t * __restrict__ output) {
    assert(size_x * size_y %
            (grid_size * block_size * BYTE_COUNT_STREAM_UNROLL) == 0);
    assert(block_size % KEY_STREAM_TRX_SIZE == 0);
    assert(reinterpret_cast<uintptr_t>(input) % KEY_STREAM_TRX_SIZE == 0);
    get_byte_counts_kern <<<
        grid_size, block_size,
        block_size * (block_size + 4) + block_size * 256 >>>
            (size_x, size_y,
             reinterpret_cast<const KeyStreamTrx*>(input), output);
}

//! sum-reduce [nr_row, nr_col] matrix into [1, nr_col] and store in data
__global__ void reduce_rows_inplace_kern(
        uint32_t nr_row, uint32_t nr_col, uint32_t *data) {
    for (uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
            c < nr_col; c += gridDim.x * blockDim.x) {
        uint32_t sum = 0;
        for (uint32_t r = 0; r < nr_row; ++ r) {
            sum += data[r * nr_col + c];
        }
        data[c] = sum;
    }
}

void reduce_rows_inplace(uint32_t nr_row, uint32_t nr_col, uint32_t *data) {
    static std::atomic_bool launch_config_init{false};
    static std::mutex launch_config_mtx;
    static int grid_size, block_size;
    if (!launch_config_init) {
        std::lock_guard<std::mutex> lg{launch_config_mtx};
        if (!launch_config_init) {
            CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(
                    &grid_size, &block_size, reduce_rows_inplace_kern));
            grid_size = std::max<int>(grid_size * 4, nr_col / block_size);
            printf("reduce_rows_inplace: grid=%d block=%d\n",
                    grid_size, block_size);
            launch_config_init = true;
        }
    }

    reduce_rows_inplace_kern <<< grid_size, block_size >>> (
            nr_row, nr_col, data);
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
    rc4_gen_keystream(GRID_SIZE, BLOCK_SIZE, SEED0, SEED1,
            NR_SAMPLE, output_gpu.get());
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
                    uint32_t x = i / KEY_STREAM_TRX_SIZE,
                             y = i % KEY_STREAM_TRX_SIZE;
                    uint8_t get = output_cpu[
                        block * (NR_SAMPLE * STREAM_LEN * BLOCK_SIZE) +
                        sample_idx * (STREAM_LEN * BLOCK_SIZE) +
                        x * (BLOCK_SIZE * KEY_STREAM_TRX_SIZE) +
                        thread * KEY_STREAM_TRX_SIZE +
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
    fprintf(stderr, "test_rc4_gen_keystream() passed\n");
}

void test_get_byte_counts() {
    constexpr int
        GRID_SIZE = 3, BLOCK_SIZE = 32, SIZE_X = 96, SIZE_Y = 48,
        INPUT_SIZE = SIZE_X * SIZE_Y * STREAM_LEN,
        OUTPUT_SIZE = GRID_SIZE * STREAM_LEN * 256,
        NR_STREAM_PER_BLOCK = BLOCK_SIZE * BYTE_COUNT_STREAM_UNROLL;
    std::unique_ptr<uint8_t[]> input_cpu{new uint8_t[INPUT_SIZE]};
    std::unique_ptr<uint32_t[]> output_cpu{new uint32_t[OUTPUT_SIZE]};
    auto input_gpu = cuda_new_arr<uint8_t>(INPUT_SIZE);
    auto output_gpu = cuda_new_arr<uint32_t>(OUTPUT_SIZE);
    CUDA_CHECK(cudaMemset(output_gpu.get(), 0, sizeof(uint32_t) * OUTPUT_SIZE));
    {
        uint64_t state[2] = {123, 456};
        uint64_t *dst = reinterpret_cast<uint64_t*>(input_cpu.get());
        for (int i = 0; i < INPUT_SIZE / 8; ++ i) {
            dst[i] = xorshift128plus(state);
        }
    }

    CUDA_CHECK(cudaMemcpy(input_gpu.get(), input_cpu.get(), INPUT_SIZE,
                cudaMemcpyHostToDevice));
    get_byte_counts(GRID_SIZE, BLOCK_SIZE, SIZE_X, SIZE_Y,
            input_gpu.get(), output_gpu.get());
    CUDA_CHECK(cudaMemcpy(output_cpu.get(), output_gpu.get(),
                OUTPUT_SIZE * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());
    static int stat[STREAM_LEN][256], stat_tot[STREAM_LEN][256];
    memset(stat_tot, 0, sizeof(stat_tot));

    for (int block = 0; block < GRID_SIZE; ++ block) {
        memset(stat, 0, sizeof(stat));
        for (int stream_base = block * NR_STREAM_PER_BLOCK;
                stream_base < SIZE_X * SIZE_Y;
                stream_base += GRID_SIZE * NR_STREAM_PER_BLOCK) {
            for (int stream_idx = stream_base;
                    stream_idx < stream_base + NR_STREAM_PER_BLOCK;
                    ++ stream_idx) {
                assert(stream_idx < SIZE_X * SIZE_Y);
                int stream_y = stream_idx % SIZE_Y,
                    stream_x = stream_idx / SIZE_Y,
                    off = stream_x * STREAM_LEN * SIZE_Y +
                        stream_y * KEY_STREAM_TRX_SIZE;
                for (uint32_t i = 0;
                        i < STREAM_LEN / KEY_STREAM_TRX_SIZE; ++ i) {
                    for (uint32_t j = 0; j < KEY_STREAM_TRX_SIZE; ++ j) {
                        int tot_off = off +
                            i * SIZE_Y * KEY_STREAM_TRX_SIZE +
                            j;
                        assert(tot_off < INPUT_SIZE);
                        int val = input_cpu[tot_off];
                        ++ stat[i * KEY_STREAM_TRX_SIZE + j][val];
                    }
                }
            }
        }
        for (int i = 0; i < STREAM_LEN; ++ i) {
            for (int j = 0; j < 256; ++ j ){
                int expect = stat[i][j],
                    offset = block * STREAM_LEN * 256 + i * 256 + j,
                    get = output_cpu[offset];
                if (expect != get) {
                    fprintf(stderr, "stat check failed at "
                            "block=%d pos=%d char=%d offset=%d: "
                            "expect=%d get=%d\n",
                            block, i, j, offset, expect, get);
                    abort();
                }
                stat_tot[i][j] += expect;
            }
        }
    }
    fprintf(stderr, "test_get_byte_counts() passed\n");

    reduce_rows_inplace(GRID_SIZE, STREAM_LEN * 256, output_gpu.get());
    CUDA_CHECK(cudaMemcpy(output_cpu.get(), output_gpu.get(),
                STREAM_LEN * 256 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int i = 0; i < STREAM_LEN; ++ i) {
        for (int j = 0; j < 256; ++ j) {
            int expect = stat_tot[i][j],
                get = output_cpu[i * 256 + j];
            if (expect != get) {
                fprintf(stderr, "reduce check failed at "
                        "%d,%d: expect=%d get=%d\n",
                        i,j, expect, get);
                abort();
            }
        }
    }
}

//! testcase for get_byte_counts

int main() {
    test_get_byte_counts();
    test_rc4_gen_keystream();
}

// vim: ft=cuda syntax=cuda.doxygen foldmethod=marker foldmarker=f{{{,f}}}
