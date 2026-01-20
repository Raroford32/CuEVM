// CuEVM: CUDA Ethereum Virtual Machine implementation
// GPU Mutation Engine Implementation for NVIDIA B300
// SPDX-License-Identifier: MIT

#include <CuEVM/fuzzing/mutation.cuh>
#include <cuda_runtime.h>
#include <cstring>

namespace CuEVM {
namespace fuzzing {

// ============================================================================
// Interesting Values Definitions (declared in mutation.cuh)
// ============================================================================

// 8-bit interesting values
__constant__ int8_t INTERESTING_8_VALUES[NUM_INTERESTING_8] = {
    -128, -1, 0, 1, 16, 32, 64, 100, 127
};

// 16-bit interesting values
__constant__ int16_t INTERESTING_16_VALUES[NUM_INTERESTING_16] = {
    -32768, -129, -128, -1, 0, 1, 127, 128, 255, 256,
    512, 1000, 1024, 4096, 32767
};

// 32-bit interesting values
__constant__ int32_t INTERESTING_32_VALUES[NUM_INTERESTING_32] = {
    -2147483648, -100663046, -32769, -32768, -129, -128, -1,
    0, 1, 127, 128, 255, 256, 512, 1000, 1024, 4096, 32767,
    32768, 65535, 65536, 100663045, 2147483647
};

// 64-bit interesting values (for Solidity uint256 boundaries)
__constant__ int64_t INTERESTING_64_VALUES[NUM_INTERESTING_64] = {
    0LL,
    1LL,
    -1LL,
    255LL,
    256LL,
    65535LL,
    65536LL,
    0x7FFFFFFFLL,
    0x80000000LL,
    0xFFFFFFFFLL,
    0x100000000LL,
    0x7FFFFFFFFFFFFFFFLL,
    (int64_t)0x8000000000000000ULL,
    -1LL  // 0xFFFFFFFFFFFFFFFF
};

// ============================================================================
// EVM Interesting Values (256-bit)
// ============================================================================

// Pre-defined interesting 256-bit values for Solidity
__device__ __constant__ uint32_t EVM_INTERESTING_256[][8] = {
    {0, 0, 0, 0, 0, 0, 0, 0},                                        // 0
    {1, 0, 0, 0, 0, 0, 0, 0},                                        // 1
    {0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF,                 // MAX_UINT256
     0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF},
    {0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF,                 // MAX_UINT256 - 1
     0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF},
    {0, 0, 0, 0, 0, 0, 0, 0x80000000},                               // MIN_INT256
    {0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF,                 // MAX_INT256
     0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x7FFFFFFF},
    {0, 0, 0, 0, 0, 0, 1, 0},                                        // 2^64
    {0, 0, 0, 0, 0, 0, 0, 1},                                        // 2^224
    {0, 0, 0, 0, 1, 0, 0, 0},                                        // 2^128
    {0xFFFFFFFF, 0, 0, 0, 0, 0, 0, 0},                               // 2^32 - 1
    {0, 1, 0, 0, 0, 0, 0, 0},                                        // 2^32
    {0xFFFFFFFF, 0xFFFFFFFF, 0, 0, 0, 0, 0, 0},                      // 2^64 - 1
    {0, 0, 1, 0, 0, 0, 0, 0},                                        // 2^64
    // Common Ether values
    {0x4A817C80, 0xDE0B6B3, 0, 0, 0, 0, 0, 0},                      // 1 ETH in wei (10^18)
    {0x2D79883D, 0x8AC72304, 0x89E8, 0, 0, 0, 0, 0},                // 10000 ETH
    // Common addresses
    {0xDEADBEEF, 0xCAFEBABE, 0x12345678, 0x9ABCDEF0, 0, 0, 0, 0},
};
constexpr uint32_t NUM_EVM_INTERESTING = 16;

// Common function selectors
__device__ __constant__ uint8_t COMMON_SELECTORS[][4] = {
    {0xa9, 0x05, 0x9c, 0xbb},  // transfer(address,uint256)
    {0x23, 0xb8, 0x72, 0xdd},  // transferFrom(address,address,uint256)
    {0x09, 0x5e, 0xa7, 0xb3},  // approve(address,uint256)
    {0x70, 0xa0, 0x82, 0x31},  // balanceOf(address)
    {0xdd, 0x62, 0xed, 0x3e},  // allowance(address,address)
    {0x40, 0xc1, 0x0f, 0x19},  // mint(address,uint256)
    {0x42, 0x96, 0x6c, 0x68},  // burn(uint256)
    {0x79, 0xcc, 0x67, 0x90},  // burnFrom(address,uint256)
    {0x18, 0x16, 0x0d, 0xdd},  // totalSupply()
    {0x06, 0xfd, 0xde, 0x03},  // name()
    {0x95, 0xd8, 0x9b, 0x41},  // symbol()
    {0x31, 0x3c, 0xe5, 0x67},  // decimals()
    {0xb6, 0xb5, 0x5f, 0x25},  // deposit()
    {0x2e, 0x1a, 0x7d, 0x4d},  // withdraw(uint256)
    {0x3c, 0xcf, 0xd6, 0x0b},  // stake(uint256)
    {0x2e, 0x17, 0xde, 0x78},  // unstake(uint256)
};
constexpr uint32_t NUM_COMMON_SELECTORS = 16;

// ============================================================================
// Mutation Dictionary Implementation
// ============================================================================

__host__ __device__ void mutation_dictionary_t::init() {
    num_entries = 0;
    next_insert_idx = 0;
    num_addresses = 0;
    num_selectors = 0;
    num_values = 0;
}

__host__ __device__ bool mutation_dictionary_t::add_entry(const uint8_t* data, uint8_t length,
                                                          DictionaryEntryType type, uint32_t pc) {
    if (length > 64) length = 64;

    // Check for duplicates (simple linear search - could optimize with hashing)
    for (uint32_t i = 0; i < num_entries; i++) {
        if (entries[i].length == length && entries[i].entry_type == (uint8_t)type) {
            bool match = true;
            for (uint8_t j = 0; j < length && match; j++) {
                if (entries[i].data[j] != data[j]) match = false;
            }
            if (match) {
                entries[i].hit_count++;
                return false;  // Already exists
            }
        }
    }

    // Add new entry
    uint32_t idx;
    if (num_entries < MAX_DICTIONARY_SIZE) {
        idx = num_entries++;
    } else {
        // Replace oldest entry (FIFO)
        idx = next_insert_idx;
        next_insert_idx = (next_insert_idx + 1) % MAX_DICTIONARY_SIZE;
    }

    for (uint8_t i = 0; i < length; i++) {
        entries[idx].data[i] = data[i];
    }
    entries[idx].length = length;
    entries[idx].entry_type = (uint8_t)type;
    entries[idx].hit_count = 1;
    entries[idx].source_pc = pc;

    // Update type-specific index
    switch (type) {
        case DictionaryEntryType::ADDRESS:
            if (num_addresses < 256) {
                address_indices[num_addresses++] = idx;
            }
            break;
        case DictionaryEntryType::FUNCTION_SELECTOR:
            if (num_selectors < 256) {
                selector_indices[num_selectors++] = idx;
            }
            break;
        case DictionaryEntryType::UINT256_VALUE:
        case DictionaryEntryType::BYTES32_VALUE:
            if (num_values < 256) {
                value_indices[num_values++] = idx;
            }
            break;
        default:
            break;
    }

    return true;
}

__host__ __device__ const dictionary_entry_t* mutation_dictionary_t::get_random(curandState* rng,
                                                                                 DictionaryEntryType type) {
    if (num_entries == 0) return nullptr;

#ifdef __CUDA_ARCH__
    uint32_t rand_val = curand(rng);
#else
    uint32_t rand_val = rand();
#endif

    if (type == (DictionaryEntryType)255) {
        // Any type
        return &entries[rand_val % num_entries];
    }

    // Type-specific lookup
    switch (type) {
        case DictionaryEntryType::ADDRESS:
            if (num_addresses > 0) {
                return &entries[address_indices[rand_val % num_addresses]];
            }
            break;
        case DictionaryEntryType::FUNCTION_SELECTOR:
            if (num_selectors > 0) {
                return &entries[selector_indices[rand_val % num_selectors]];
            }
            break;
        case DictionaryEntryType::UINT256_VALUE:
        case DictionaryEntryType::BYTES32_VALUE:
            if (num_values > 0) {
                return &entries[value_indices[rand_val % num_values]];
            }
            break;
        default:
            break;
    }

    return &entries[rand_val % num_entries];
}

__host__ __device__ void mutation_dictionary_t::update_hit_count(uint32_t idx) {
    if (idx < num_entries) {
        entries[idx].hit_count++;
    }
}

// ============================================================================
// Mutation Input Implementation
// ============================================================================

__host__ __device__ void mutation_input_t::init(uint32_t max_size) {
    capacity = max_size;
    length = 0;
    num_params = 0;
    for (int i = 0; i < 4; i++) selector[i] = 0;
    for (int i = 0; i < 32; i++) {
        param_offsets[i] = 0;
        param_types[i] = 0;
    }
}

__host__ __device__ void mutation_input_t::copy_from(const mutation_input_t& other) {
    if (capacity < other.length) return;

    length = other.length;
    for (uint32_t i = 0; i < length; i++) {
        data[i] = other.data[i];
    }
    for (int i = 0; i < 4; i++) selector[i] = other.selector[i];
    num_params = other.num_params;
    for (uint32_t i = 0; i < num_params && i < 32; i++) {
        param_offsets[i] = other.param_offsets[i];
        param_types[i] = other.param_types[i];
    }
    // Copy 256-bit values
    for (int i = 0; i < 8; i++) {
        value._limbs[i] = other.value._limbs[i];
        gas_limit._limbs[i] = other.gas_limit._limbs[i];
        sender._limbs[i] = other.sender._limbs[i];
        receiver._limbs[i] = other.receiver._limbs[i];
        block_number._limbs[i] = other.block_number._limbs[i];
        timestamp._limbs[i] = other.timestamp._limbs[i];
        basefee._limbs[i] = other.basefee._limbs[i];
        prevrandao._limbs[i] = other.prevrandao._limbs[i];
    }
}

__host__ __device__ void mutation_input_t::parse_abi() {
    if (length < 4) return;

    // Extract selector
    for (int i = 0; i < 4; i++) {
        selector[i] = data[i];
    }

    // Parse parameters (32-byte chunks)
    num_params = 0;
    for (uint32_t offset = 4; offset + 32 <= length && num_params < 32; offset += 32) {
        param_offsets[num_params] = offset;
        // Simple type detection based on leading zeros
        uint32_t leading_zeros = 0;
        for (uint32_t i = 0; i < 32 && data[offset + i] == 0; i++) {
            leading_zeros++;
        }
        if (leading_zeros >= 12) {
            param_types[num_params] = (uint8_t)abi::ABIType::ADDRESS;  // Likely address
        } else if (leading_zeros >= 24) {
            param_types[num_params] = (uint8_t)abi::ABIType::UINT64;
        } else {
            param_types[num_params] = (uint8_t)abi::ABIType::UINT256;
        }
        num_params++;
    }
}

__host__ __device__ void mutation_input_t::reserialize_abi() {
    // Ensure selector is at the start
    for (int i = 0; i < 4; i++) {
        data[i] = selector[i];
    }
    // Parameters should already be in place
}

// ============================================================================
// GPU RNG State Implementation
// ============================================================================

__host__ void gpu_rng_state_t::init(uint32_t num_threads, uint64_t seed) {
    num_states = num_threads;
    cudaMalloc(&states, num_threads * sizeof(curandState));

    // Initialize RNG states on GPU
    uint32_t block_size = 256;
    uint32_t num_blocks = (num_threads + block_size - 1) / block_size;
    kernel_init_rng<<<num_blocks, block_size>>>(states, num_threads, seed);
    cudaDeviceSynchronize();
}

__host__ void gpu_rng_state_t::free() {
    if (states) {
        cudaFree(states);
        states = nullptr;
    }
}

// ============================================================================
// GPU Mutation Engine Implementation
// ============================================================================

__host__ GPUMutationEngine::GPUMutationEngine(uint32_t num_instances, uint64_t seed) {
    rng_state_.init(num_instances, seed);

    cudaMallocManaged(&dictionary_, sizeof(mutation_dictionary_t));
    dictionary_->init();

    // Default mutation weights
    for (int i = 0; i < 64; i++) mutation_weights_[i] = 10;
    mutation_weights_[(int)MutationType::FLIP_BIT_1] = WEIGHT_BIT_FLIP;
    mutation_weights_[(int)MutationType::FLIP_BYTE_1] = WEIGHT_BYTE_FLIP;
    mutation_weights_[(int)MutationType::ARITH_INC_8] = WEIGHT_ARITH_INC;
    mutation_weights_[(int)MutationType::ARITH_DEC_8] = WEIGHT_ARITH_DEC;
    mutation_weights_[(int)MutationType::INTERESTING_8] = WEIGHT_INTERESTING;
    mutation_weights_[(int)MutationType::DICT_INSERT] = WEIGHT_DICTIONARY;
    mutation_weights_[(int)MutationType::HAVOC_SINGLE] = WEIGHT_HAVOC;
    mutation_weights_[(int)MutationType::SPLICE] = WEIGHT_SPLICE;

    max_mutations_ = 16;
    abi_aware_ = true;
}

__host__ GPUMutationEngine::~GPUMutationEngine() {
    rng_state_.free();
    if (dictionary_) {
        cudaFree(dictionary_);
    }
}

__device__ MutationType GPUMutationEngine::select_mutation_type(curandState* rng) {
    uint32_t total_weight = 0;
    for (int i = 0; i < (int)MutationType::NUM_MUTATION_TYPES; i++) {
        total_weight += mutation_weights_[i];
    }

    uint32_t rand_val = curand(rng) % total_weight;
    uint32_t cumulative = 0;

    for (int i = 0; i < (int)MutationType::NUM_MUTATION_TYPES; i++) {
        cumulative += mutation_weights_[i];
        if (rand_val < cumulative) {
            return (MutationType)i;
        }
    }

    return MutationType::FLIP_BIT_1;
}

__device__ uint32_t GPUMutationEngine::select_offset(uint32_t length, curandState* rng) {
    if (length == 0) return 0;
    return curand(rng) % length;
}

__device__ mutation_result_t GPUMutationEngine::mutate(mutation_input_t* input, curandState* rng) {
    MutationType type = select_mutation_type(rng);
    return mutate_typed(input, type, rng);
}

__device__ mutation_result_t GPUMutationEngine::mutate_typed(mutation_input_t* input, MutationType type, curandState* rng) {
    mutation_result_t result;
    result.type = type;
    result.success = false;
    result.size_delta = 0;

    if (input->length == 0) return result;

    result.offset = select_offset(input->length, rng);

    switch (type) {
        case MutationType::FLIP_BIT_1:
            flip_bit(input->data, input->length, result.offset, 1);
            result.success = true;
            break;

        case MutationType::FLIP_BIT_2:
            flip_bit(input->data, input->length, result.offset, 2);
            result.success = true;
            break;

        case MutationType::FLIP_BIT_4:
            flip_bit(input->data, input->length, result.offset, 4);
            result.success = true;
            break;

        case MutationType::FLIP_BYTE_1:
            flip_byte(input->data, input->length, result.offset, 1);
            result.success = true;
            break;

        case MutationType::FLIP_BYTE_2:
            flip_byte(input->data, input->length, result.offset, 2);
            result.success = true;
            break;

        case MutationType::FLIP_BYTE_4:
            flip_byte(input->data, input->length, result.offset, 4);
            result.success = true;
            break;

        case MutationType::ARITH_INC_8:
            arith_mutation(input->data, input->length, result.offset, 1, true, (curand(rng) % ARITH_MAX_DELTA) + 1);
            result.success = true;
            break;

        case MutationType::ARITH_DEC_8:
            arith_mutation(input->data, input->length, result.offset, 1, false, (curand(rng) % ARITH_MAX_DELTA) + 1);
            result.success = true;
            break;

        case MutationType::ARITH_INC_16:
            arith_mutation(input->data, input->length, result.offset, 2, true, (curand(rng) % ARITH_MAX_DELTA) + 1);
            result.success = true;
            break;

        case MutationType::ARITH_DEC_16:
            arith_mutation(input->data, input->length, result.offset, 2, false, (curand(rng) % ARITH_MAX_DELTA) + 1);
            result.success = true;
            break;

        case MutationType::ARITH_INC_32:
            arith_mutation(input->data, input->length, result.offset, 4, true, (curand(rng) % ARITH_MAX_DELTA) + 1);
            result.success = true;
            break;

        case MutationType::ARITH_DEC_32:
            arith_mutation(input->data, input->length, result.offset, 4, false, (curand(rng) % ARITH_MAX_DELTA) + 1);
            result.success = true;
            break;

        case MutationType::INTERESTING_8:
        case MutationType::INTERESTING_16:
        case MutationType::INTERESTING_32:
        case MutationType::INTERESTING_64:
            interesting_mutation(input->data, input->length, result.offset,
                                (type == MutationType::INTERESTING_8) ? 1 :
                                (type == MutationType::INTERESTING_16) ? 2 :
                                (type == MutationType::INTERESTING_32) ? 4 : 8, rng);
            result.success = true;
            break;

        case MutationType::INTERESTING_256:
            if (result.offset + 32 <= input->length) {
                uint32_t idx = curand(rng) % NUM_EVM_INTERESTING;
                for (int i = 0; i < 8; i++) {
                    uint32_t val = EVM_INTERESTING_256[idx][i];
                    input->data[result.offset + i*4] = val & 0xFF;
                    input->data[result.offset + i*4 + 1] = (val >> 8) & 0xFF;
                    input->data[result.offset + i*4 + 2] = (val >> 16) & 0xFF;
                    input->data[result.offset + i*4 + 3] = (val >> 24) & 0xFF;
                }
                result.success = true;
            }
            break;

        case MutationType::DICT_INSERT:
        case MutationType::DICT_OVERWRITE:
            apply_dictionary(input, rng);
            result.success = true;
            break;

        case MutationType::HAVOC_SINGLE:
            havoc(input, rng, 1);
            result.success = true;
            break;

        case MutationType::HAVOC_MULTI:
            havoc(input, rng, 2 + (curand(rng) % 6));
            result.success = true;
            break;

        case MutationType::EVM_ADDRESS:
            mutate_address(input, result.offset, rng);
            result.success = true;
            break;

        case MutationType::EVM_UINT256:
            mutate_uint256(input, result.offset, rng);
            result.success = true;
            break;

        case MutationType::EVM_SELECTOR:
            mutate_selector(input, rng);
            result.success = true;
            break;

        case MutationType::EVM_CALLDATA:
            mutate_calldata(input, rng);
            result.success = true;
            break;

        case MutationType::DELETE_BYTES:
            if (input->length > 8) {
                uint32_t count = 1 + (curand(rng) % 4);
                if (result.offset + count <= input->length) {
                    delete_bytes(input, result.offset, count);
                    result.size_delta = -(int32_t)count;
                    result.success = true;
                }
            }
            break;

        case MutationType::CLONE_BYTE:
            if (input->length > 1 && input->length < input->capacity - 4) {
                uint32_t src = curand(rng) % input->length;
                uint32_t count = 1 + (curand(rng) % 4);
                if (input->length + count <= input->capacity) {
                    clone_bytes(input, src, result.offset, count);
                    result.size_delta = count;
                    result.success = true;
                }
            }
            break;

        case MutationType::SWAP_BYTES:
            if (input->length > 4) {
                uint32_t offset2 = curand(rng) % input->length;
                uint32_t count = 1 + (curand(rng) % 4);
                if (result.offset + count <= input->length && offset2 + count <= input->length) {
                    swap_bytes(input->data, result.offset, offset2, count);
                    result.success = true;
                }
            }
            break;

        case MutationType::SHUFFLE_BYTES:
            if (input->length > 4) {
                uint32_t count = 4 + (curand(rng) % 12);
                if (result.offset + count <= input->length) {
                    shuffle_bytes(input->data, result.offset, count, rng);
                    result.success = true;
                }
            }
            break;

        case MutationType::BOUNDARY_LOW:
            // Set to boundary value (0 or 1)
            if (result.offset + 32 <= input->length) {
                for (uint32_t i = 0; i < 31; i++) {
                    input->data[result.offset + i] = 0;
                }
                input->data[result.offset + 31] = curand(rng) % 2;
                result.success = true;
            }
            break;

        case MutationType::BOUNDARY_HIGH:
            // Set to max boundary
            if (result.offset + 32 <= input->length) {
                for (uint32_t i = 0; i < 32; i++) {
                    input->data[result.offset + i] = 0xFF;
                }
                result.success = true;
            }
            break;

        case MutationType::BOUNDARY_POWER2:
            // Set to power of 2
            if (result.offset + 32 <= input->length) {
                for (uint32_t i = 0; i < 32; i++) {
                    input->data[result.offset + i] = 0;
                }
                uint32_t bit_pos = curand(rng) % 256;
                uint32_t byte_pos = bit_pos / 8;
                uint32_t bit_in_byte = bit_pos % 8;
                input->data[result.offset + 31 - byte_pos] = 1 << bit_in_byte;
                result.success = true;
            }
            break;

        default:
            break;
    }

    return result;
}

__device__ void GPUMutationEngine::flip_bit(uint8_t* data, uint32_t length, uint32_t offset, uint8_t width) {
    if (offset >= length) return;
    for (uint8_t i = 0; i < width && offset < length; i++) {
        uint8_t bit = i % 8;
        data[offset] ^= (1 << bit);
        if ((i + 1) % 8 == 0) offset++;
    }
}

__device__ void GPUMutationEngine::flip_byte(uint8_t* data, uint32_t length, uint32_t offset, uint8_t width) {
    for (uint8_t i = 0; i < width && offset + i < length; i++) {
        data[offset + i] ^= 0xFF;
    }
}

__device__ void GPUMutationEngine::arith_mutation(uint8_t* data, uint32_t length, uint32_t offset,
                                                   uint8_t width, bool increment, int32_t delta) {
    if (offset + width > length) return;

    switch (width) {
        case 1: {
            if (increment) {
                data[offset] += delta;
            } else {
                data[offset] -= delta;
            }
            break;
        }
        case 2: {
            uint16_t val = data[offset] | (data[offset + 1] << 8);
            if (increment) val += delta;
            else val -= delta;
            data[offset] = val & 0xFF;
            data[offset + 1] = (val >> 8) & 0xFF;
            break;
        }
        case 4: {
            uint32_t val = data[offset] | (data[offset + 1] << 8) |
                          (data[offset + 2] << 16) | (data[offset + 3] << 24);
            if (increment) val += delta;
            else val -= delta;
            data[offset] = val & 0xFF;
            data[offset + 1] = (val >> 8) & 0xFF;
            data[offset + 2] = (val >> 16) & 0xFF;
            data[offset + 3] = (val >> 24) & 0xFF;
            break;
        }
        default:
            break;
    }
}

__device__ void GPUMutationEngine::interesting_mutation(uint8_t* data, uint32_t length, uint32_t offset,
                                                        uint8_t width, curandState* rng) {
    if (offset + width > length) return;

    switch (width) {
        case 1: {
            uint32_t idx = curand(rng) % NUM_INTERESTING_8;
            data[offset] = (uint8_t)INTERESTING_8_VALUES[idx];
            break;
        }
        case 2: {
            uint32_t idx = curand(rng) % NUM_INTERESTING_16;
            int16_t val = INTERESTING_16_VALUES[idx];
            data[offset] = val & 0xFF;
            data[offset + 1] = (val >> 8) & 0xFF;
            break;
        }
        case 4: {
            uint32_t idx = curand(rng) % NUM_INTERESTING_32;
            int32_t val = INTERESTING_32_VALUES[idx];
            data[offset] = val & 0xFF;
            data[offset + 1] = (val >> 8) & 0xFF;
            data[offset + 2] = (val >> 16) & 0xFF;
            data[offset + 3] = (val >> 24) & 0xFF;
            break;
        }
        case 8: {
            uint32_t idx = curand(rng) % NUM_INTERESTING_64;
            int64_t val = INTERESTING_64_VALUES[idx];
            for (int i = 0; i < 8; i++) {
                data[offset + i] = (val >> (i * 8)) & 0xFF;
            }
            break;
        }
        default:
            break;
    }
}

__device__ void GPUMutationEngine::clone_bytes(mutation_input_t* input, uint32_t src_offset,
                                               uint32_t dst_offset, uint32_t count) {
    if (input->length + count > input->capacity) return;

    // Shift data to make room
    for (int32_t i = input->length - 1; i >= (int32_t)dst_offset; i--) {
        input->data[i + count] = input->data[i];
    }

    // Copy bytes
    for (uint32_t i = 0; i < count; i++) {
        input->data[dst_offset + i] = input->data[src_offset + i + (src_offset >= dst_offset ? count : 0)];
    }

    input->length += count;
}

__device__ void GPUMutationEngine::delete_bytes(mutation_input_t* input, uint32_t offset, uint32_t count) {
    if (offset + count > input->length) return;

    for (uint32_t i = offset; i + count < input->length; i++) {
        input->data[i] = input->data[i + count];
    }

    input->length -= count;
}

__device__ void GPUMutationEngine::insert_bytes(mutation_input_t* input, uint32_t offset,
                                                const uint8_t* data, uint32_t count) {
    if (input->length + count > input->capacity) return;

    // Shift existing data
    for (int32_t i = input->length - 1; i >= (int32_t)offset; i--) {
        input->data[i + count] = input->data[i];
    }

    // Insert new data
    for (uint32_t i = 0; i < count; i++) {
        input->data[offset + i] = data[i];
    }

    input->length += count;
}

__device__ void GPUMutationEngine::overwrite_bytes(mutation_input_t* input, uint32_t offset,
                                                   const uint8_t* data, uint32_t count) {
    for (uint32_t i = 0; i < count && offset + i < input->length; i++) {
        input->data[offset + i] = data[i];
    }
}

__device__ void GPUMutationEngine::swap_bytes(uint8_t* data, uint32_t offset1, uint32_t offset2, uint32_t count) {
    for (uint32_t i = 0; i < count; i++) {
        uint8_t tmp = data[offset1 + i];
        data[offset1 + i] = data[offset2 + i];
        data[offset2 + i] = tmp;
    }
}

__device__ void GPUMutationEngine::shuffle_bytes(uint8_t* data, uint32_t offset, uint32_t count, curandState* rng) {
    for (uint32_t i = count - 1; i > 0; i--) {
        uint32_t j = curand(rng) % (i + 1);
        uint8_t tmp = data[offset + i];
        data[offset + i] = data[offset + j];
        data[offset + j] = tmp;
    }
}

__device__ void GPUMutationEngine::havoc(mutation_input_t* input, curandState* rng, uint32_t num_mutations) {
    for (uint32_t i = 0; i < num_mutations; i++) {
        // Exclude complex mutations from havoc to avoid exponential growth
        MutationType type = (MutationType)(curand(rng) % 20);
        mutate_typed(input, type, rng);
    }
}

__device__ void GPUMutationEngine::splice(mutation_input_t* dst, const mutation_input_t* src1,
                                          const mutation_input_t* src2, curandState* rng) {
    if (src1->length == 0 || src2->length == 0) return;

    uint32_t split1 = curand(rng) % src1->length;
    uint32_t split2 = curand(rng) % src2->length;

    // Take first part from src1, second part from src2
    uint32_t new_len = split1 + (src2->length - split2);
    if (new_len > dst->capacity) new_len = dst->capacity;

    for (uint32_t i = 0; i < split1 && i < new_len; i++) {
        dst->data[i] = src1->data[i];
    }
    for (uint32_t i = 0; i + split1 < new_len; i++) {
        dst->data[split1 + i] = src2->data[split2 + i];
    }

    dst->length = new_len;
}

__device__ void GPUMutationEngine::crossover(mutation_input_t* dst, const mutation_input_t* src1,
                                             const mutation_input_t* src2, curandState* rng) {
    if (src1->length == 0 || src2->length == 0) return;

    // Two-point crossover
    uint32_t min_len = (src1->length < src2->length) ? src1->length : src2->length;
    uint32_t pt1 = curand(rng) % min_len;
    uint32_t pt2 = pt1 + (curand(rng) % (min_len - pt1));

    dst->length = min_len;

    for (uint32_t i = 0; i < min_len; i++) {
        if (i < pt1 || i >= pt2) {
            dst->data[i] = src1->data[i];
        } else {
            dst->data[i] = src2->data[i];
        }
    }
}

__device__ void GPUMutationEngine::mutate_address(mutation_input_t* input, uint32_t offset, curandState* rng) {
    if (offset + 32 > input->length) return;

    // Address is 20 bytes, right-padded in 32-byte slot
    // Zero out first 12 bytes
    for (int i = 0; i < 12; i++) {
        input->data[offset + i] = 0;
    }

    // Generate random address or use dictionary
    if (dictionary_->num_addresses > 0 && (curand(rng) % 4) < 3) {
        const dictionary_entry_t* entry = dictionary_->get_random(rng, DictionaryEntryType::ADDRESS);
        if (entry && entry->length >= 20) {
            for (int i = 0; i < 20; i++) {
                input->data[offset + 12 + i] = entry->data[i];
            }
            return;
        }
    }

    // Random address
    for (int i = 0; i < 20; i++) {
        input->data[offset + 12 + i] = curand(rng) & 0xFF;
    }
}

__device__ void GPUMutationEngine::mutate_uint256(mutation_input_t* input, uint32_t offset, curandState* rng) {
    if (offset + 32 > input->length) return;

    uint32_t strategy = curand(rng) % 10;

    switch (strategy) {
        case 0:  // Zero
            for (int i = 0; i < 32; i++) input->data[offset + i] = 0;
            break;
        case 1:  // One
            for (int i = 0; i < 31; i++) input->data[offset + i] = 0;
            input->data[offset + 31] = 1;
            break;
        case 2:  // Max
            for (int i = 0; i < 32; i++) input->data[offset + i] = 0xFF;
            break;
        case 3:  // Power of 2
        {
            for (int i = 0; i < 32; i++) input->data[offset + i] = 0;
            uint32_t bit = curand(rng) % 256;
            input->data[offset + 31 - bit / 8] = 1 << (bit % 8);
            break;
        }
        case 4:  // EVM interesting value
        {
            uint32_t idx = curand(rng) % NUM_EVM_INTERESTING;
            for (int i = 0; i < 8; i++) {
                uint32_t val = EVM_INTERESTING_256[idx][i];
                input->data[offset + i*4] = val & 0xFF;
                input->data[offset + i*4 + 1] = (val >> 8) & 0xFF;
                input->data[offset + i*4 + 2] = (val >> 16) & 0xFF;
                input->data[offset + i*4 + 3] = (val >> 24) & 0xFF;
            }
            break;
        }
        case 5:  // Dictionary value
            if (dictionary_->num_values > 0) {
                const dictionary_entry_t* entry = dictionary_->get_random(rng, DictionaryEntryType::UINT256_VALUE);
                if (entry && entry->length >= 32) {
                    for (int i = 0; i < 32; i++) {
                        input->data[offset + i] = entry->data[i];
                    }
                }
            }
            break;
        default:  // Random
            for (int i = 0; i < 32; i++) {
                input->data[offset + i] = curand(rng) & 0xFF;
            }
            break;
    }
}

__device__ void GPUMutationEngine::mutate_selector(mutation_input_t* input, curandState* rng) {
    if (input->length < 4) return;

    uint32_t strategy = curand(rng) % 4;

    switch (strategy) {
        case 0:  // Common selector
        {
            uint32_t idx = curand(rng) % NUM_COMMON_SELECTORS;
            for (int i = 0; i < 4; i++) {
                input->data[i] = COMMON_SELECTORS[idx][i];
                input->selector[i] = COMMON_SELECTORS[idx][i];
            }
            break;
        }
        case 1:  // Dictionary selector
            if (dictionary_->num_selectors > 0) {
                const dictionary_entry_t* entry = dictionary_->get_random(rng, DictionaryEntryType::FUNCTION_SELECTOR);
                if (entry && entry->length >= 4) {
                    for (int i = 0; i < 4; i++) {
                        input->data[i] = entry->data[i];
                        input->selector[i] = entry->data[i];
                    }
                }
            }
            break;
        default:  // Random selector
            for (int i = 0; i < 4; i++) {
                input->data[i] = curand(rng) & 0xFF;
                input->selector[i] = input->data[i];
            }
            break;
    }
}

__device__ void GPUMutationEngine::mutate_calldata(mutation_input_t* input, curandState* rng) {
    if (!abi_aware_ || input->num_params == 0) {
        // Random mutation if not ABI-aware
        mutate(input, rng);
        return;
    }

    // Pick a random parameter to mutate
    uint32_t param_idx = curand(rng) % input->num_params;
    uint32_t offset = input->param_offsets[param_idx];
    abi::ABIType type = (abi::ABIType)input->param_types[param_idx];

    abi::mutate_by_type(input->data, offset, type, rng);
}

__device__ void GPUMutationEngine::mutate_value(mutation_input_t* input, curandState* rng) {
    uint32_t strategy = curand(rng) % 6;

    switch (strategy) {
        case 0:  // Zero
            for (int i = 0; i < 8; i++) input->value._limbs[i] = 0;
            break;
        case 1:  // Small value
        {
            for (int i = 1; i < 8; i++) input->value._limbs[i] = 0;
            input->value._limbs[0] = curand(rng) % 1000;
            break;
        }
        case 2:  // 1 ETH equivalent
        {
            for (int i = 2; i < 8; i++) input->value._limbs[i] = 0;
            input->value._limbs[0] = 0x4A817C80;  // 10^18 low bits
            input->value._limbs[1] = 0xDE0B6B3;   // 10^18 high bits
            break;
        }
        case 3:  // Max available (simulated)
            for (int i = 0; i < 8; i++) input->value._limbs[i] = 0xFFFFFFFF;
            break;
        default:  // Random
        {
            for (int i = 0; i < 8; i++) {
                input->value._limbs[i] = curand(rng);
            }
            break;
        }
    }
}

__device__ void GPUMutationEngine::mutate_gas(mutation_input_t* input, curandState* rng) {
    uint32_t strategy = curand(rng) % 4;

    // Clear high bits
    for (int i = 2; i < 8; i++) input->gas_limit._limbs[i] = 0;

    switch (strategy) {
        case 0:  // Minimum gas
            input->gas_limit._limbs[0] = 21000;
            input->gas_limit._limbs[1] = 0;
            break;
        case 1:  // Standard gas limit
            input->gas_limit._limbs[0] = 3000000;
            input->gas_limit._limbs[1] = 0;
            break;
        case 2:  // High gas
            input->gas_limit._limbs[0] = 30000000;
            input->gas_limit._limbs[1] = 0;
            break;
        default:  // Random
            input->gas_limit._limbs[0] = curand(rng) % 50000000;
            input->gas_limit._limbs[1] = 0;
            break;
    }
}

__device__ void GPUMutationEngine::mutate_sender(mutation_input_t* input, curandState* rng) {
    // Zero high bytes (address is 20 bytes)
    for (int i = 5; i < 8; i++) input->sender._limbs[i] = 0;
    input->sender._limbs[4] &= 0xFFFF;  // Only low 4 bytes of limb 4

    if (dictionary_->num_addresses > 0 && (curand(rng) % 3) < 2) {
        const dictionary_entry_t* entry = dictionary_->get_random(rng, DictionaryEntryType::ADDRESS);
        if (entry && entry->length >= 20) {
            // Copy address to sender
            for (int i = 0; i < 5; i++) {
                input->sender._limbs[i] =
                    entry->data[i*4] | (entry->data[i*4+1] << 8) |
                    (entry->data[i*4+2] << 16) | (entry->data[i*4+3] << 24);
            }
            return;
        }
    }

    // Generate random sender
    for (int i = 0; i < 5; i++) {
        input->sender._limbs[i] = curand(rng);
    }
}

__device__ void GPUMutationEngine::mutate_block_context(mutation_input_t* input, curandState* rng) {
    uint32_t field = curand(rng) % 4;

    switch (field) {
        case 0:  // Block number
            input->block_number._limbs[0] = 15000000 + (curand(rng) % 5000000);
            for (int i = 1; i < 8; i++) input->block_number._limbs[i] = 0;
            break;
        case 1:  // Timestamp
            // Current-ish timestamp
            input->timestamp._limbs[0] = 1700000000 + (curand(rng) % 100000000);
            for (int i = 1; i < 8; i++) input->timestamp._limbs[i] = 0;
            break;
        case 2:  // Basefee
            input->basefee._limbs[0] = curand(rng) % 1000000000000;  // Up to 1000 Gwei
            for (int i = 1; i < 8; i++) input->basefee._limbs[i] = 0;
            break;
        case 3:  // Prevrandao
            for (int i = 0; i < 8; i++) {
                input->prevrandao._limbs[i] = curand(rng);
            }
            break;
    }
}

__host__ __device__ void GPUMutationEngine::add_to_dictionary(const uint8_t* data, uint8_t length,
                                                               DictionaryEntryType type, uint32_t pc) {
    dictionary_->add_entry(data, length, type, pc);
}

__device__ void GPUMutationEngine::apply_dictionary(mutation_input_t* input, curandState* rng) {
    const dictionary_entry_t* entry = dictionary_->get_random(rng);
    if (!entry) return;

    uint32_t offset = select_offset(input->length, rng);

    // Overwrite or insert based on type
    if (curand(rng) % 2 == 0) {
        // Overwrite
        overwrite_bytes(input, offset, entry->data, entry->length);
    } else {
        // Insert if space available
        if (input->length + entry->length <= input->capacity) {
            insert_bytes(input, offset, entry->data, entry->length);
        }
    }
}

__device__ void GPUMutationEngine::gradient_mutate(mutation_input_t* input, uint32_t target_offset,
                                                   bool increase, curandState* rng) {
    if (target_offset + 32 > input->length) return;

    // Gradient-guided mutation: try to move value toward target
    uint32_t delta = 1 + (curand(rng) % 16);

    if (increase) {
        // Try to increase value
        uint64_t val = 0;
        for (int i = 0; i < 8; i++) {
            val |= ((uint64_t)input->data[target_offset + 24 + i]) << (i * 8);
        }
        val += delta;
        for (int i = 0; i < 8; i++) {
            input->data[target_offset + 24 + i] = (val >> (i * 8)) & 0xFF;
        }
    } else {
        // Try to decrease value
        uint64_t val = 0;
        for (int i = 0; i < 8; i++) {
            val |= ((uint64_t)input->data[target_offset + 24 + i]) << (i * 8);
        }
        if (val >= delta) val -= delta;
        for (int i = 0; i < 8; i++) {
            input->data[target_offset + 24 + i] = (val >> (i * 8)) & 0xFF;
        }
    }
}

__host__ void GPUMutationEngine::set_mutation_weights(const uint8_t* weights) {
    memcpy(mutation_weights_, weights, 64);
}

__host__ void GPUMutationEngine::set_max_mutations(uint32_t max) {
    max_mutations_ = max;
}

__host__ void GPUMutationEngine::enable_abi_aware(bool enable) {
    abi_aware_ = enable;
}

__host__ void GPUMutationEngine::mutate_batch(mutation_input_t* inputs, uint32_t num_inputs,
                                              uint32_t mutations_per_input, cudaStream_t stream) {
    mutation_result_t* results;
    cudaMalloc(&results, num_inputs * mutations_per_input * sizeof(mutation_result_t));

    uint32_t block_size = 256;
    uint32_t num_blocks = (num_inputs + block_size - 1) / block_size;

    kernel_mutate_batch<<<num_blocks, block_size, 0, stream>>>(
        this, inputs, num_inputs, mutations_per_input, rng_state_.states, results
    );

    cudaFree(results);
}

// ============================================================================
// Sequence Mutator Implementation
// ============================================================================

__host__ __device__ void sequence_t::init(uint32_t max_txs) {
    capacity = max_txs;
    num_transactions = 0;
    seed = 0;
}

__host__ __device__ void sequence_t::add_transaction(const transaction_t& tx) {
    if (num_transactions < capacity) {
        transactions[num_transactions] = tx;
        transactions[num_transactions].tx_index = num_transactions;
        num_transactions++;
    }
}

__host__ __device__ void sequence_t::remove_transaction(uint32_t index) {
    if (index >= num_transactions) return;
    for (uint32_t i = index; i < num_transactions - 1; i++) {
        transactions[i] = transactions[i + 1];
        transactions[i].tx_index = i;
    }
    num_transactions--;
}

__host__ __device__ void sequence_t::reorder(uint32_t from, uint32_t to) {
    if (from >= num_transactions || to >= num_transactions || from == to) return;
    transaction_t tmp = transactions[from];
    if (from < to) {
        for (uint32_t i = from; i < to; i++) {
            transactions[i] = transactions[i + 1];
            transactions[i].tx_index = i;
        }
    } else {
        for (uint32_t i = from; i > to; i--) {
            transactions[i] = transactions[i - 1];
            transactions[i].tx_index = i;
        }
    }
    transactions[to] = tmp;
    transactions[to].tx_index = to;
}

__host__ __device__ void sequence_t::copy_from(const sequence_t& other) {
    num_transactions = (other.num_transactions < capacity) ? other.num_transactions : capacity;
    seed = other.seed;
    for (uint32_t i = 0; i < num_transactions; i++) {
        transactions[i] = other.transactions[i];
    }
}

__host__ SequenceMutator::SequenceMutator(GPUMutationEngine* engine) : engine_(engine) {}

__device__ void SequenceMutator::mutate_sequence(sequence_t* seq, curandState* rng) {
    if (seq->num_transactions == 0) return;

    uint32_t operation = curand(rng) % 8;

    switch (operation) {
        case 0:  // Mutate random transaction
            mutate_transaction(seq, curand(rng) % seq->num_transactions, rng);
            break;
        case 1:  // Swap two transactions
            if (seq->num_transactions > 1) {
                swap_transactions(seq, curand(rng) % seq->num_transactions,
                                  curand(rng) % seq->num_transactions);
            }
            break;
        case 2:  // Duplicate transaction
            if (seq->num_transactions < seq->capacity) {
                duplicate_transaction(seq, curand(rng) % seq->num_transactions);
            }
            break;
        case 3:  // Delete transaction
            if (seq->num_transactions > 1) {
                delete_transaction(seq, curand(rng) % seq->num_transactions);
            }
            break;
        case 4:  // Reorder
            if (seq->num_transactions > 1) {
                seq->reorder(curand(rng) % seq->num_transactions,
                            curand(rng) % seq->num_transactions);
            }
            break;
        case 5:  // Mutate sender pattern
            mutate_sender_pattern(seq, rng);
            break;
        case 6:  // Mutate value flow
            mutate_value_flow(seq, rng);
            break;
        default:  // Mutate all transactions
            for (uint32_t i = 0; i < seq->num_transactions; i++) {
                mutate_transaction(seq, i, rng);
            }
            break;
    }
}

__device__ void SequenceMutator::insert_transaction(sequence_t* seq, uint32_t index, curandState* rng) {
    if (seq->num_transactions >= seq->capacity) return;

    // Shift transactions
    for (uint32_t i = seq->num_transactions; i > index; i--) {
        seq->transactions[i] = seq->transactions[i - 1];
        seq->transactions[i].tx_index = i;
    }

    // Create new transaction (copy from adjacent and mutate)
    if (index > 0) {
        seq->transactions[index] = seq->transactions[index - 1];
    }
    seq->transactions[index].tx_index = index;
    seq->num_transactions++;

    engine_->mutate(&seq->transactions[index].input, rng);
}

__device__ void SequenceMutator::delete_transaction(sequence_t* seq, uint32_t index) {
    seq->remove_transaction(index);
}

__device__ void SequenceMutator::duplicate_transaction(sequence_t* seq, uint32_t index) {
    if (seq->num_transactions >= seq->capacity || index >= seq->num_transactions) return;

    seq->transactions[seq->num_transactions] = seq->transactions[index];
    seq->transactions[seq->num_transactions].tx_index = seq->num_transactions;
    seq->num_transactions++;
}

__device__ void SequenceMutator::swap_transactions(sequence_t* seq, uint32_t idx1, uint32_t idx2) {
    if (idx1 >= seq->num_transactions || idx2 >= seq->num_transactions) return;

    transaction_t tmp = seq->transactions[idx1];
    seq->transactions[idx1] = seq->transactions[idx2];
    seq->transactions[idx2] = tmp;

    seq->transactions[idx1].tx_index = idx1;
    seq->transactions[idx2].tx_index = idx2;
}

__device__ void SequenceMutator::splice_sequences(sequence_t* dst, const sequence_t* src1,
                                                  const sequence_t* src2, curandState* rng) {
    if (src1->num_transactions == 0 || src2->num_transactions == 0) return;

    uint32_t split1 = curand(rng) % src1->num_transactions;
    uint32_t split2 = curand(rng) % src2->num_transactions;

    dst->num_transactions = 0;

    // Copy first part from src1
    for (uint32_t i = 0; i < split1 && dst->num_transactions < dst->capacity; i++) {
        dst->transactions[dst->num_transactions] = src1->transactions[i];
        dst->transactions[dst->num_transactions].tx_index = dst->num_transactions;
        dst->num_transactions++;
    }

    // Copy second part from src2
    for (uint32_t i = split2; i < src2->num_transactions && dst->num_transactions < dst->capacity; i++) {
        dst->transactions[dst->num_transactions] = src2->transactions[i];
        dst->transactions[dst->num_transactions].tx_index = dst->num_transactions;
        dst->num_transactions++;
    }
}

__device__ void SequenceMutator::mutate_transaction(sequence_t* seq, uint32_t tx_index, curandState* rng) {
    if (tx_index >= seq->num_transactions) return;

    engine_->mutate(&seq->transactions[tx_index].input, rng);
}

__device__ void SequenceMutator::mutate_sender_pattern(sequence_t* seq, curandState* rng) {
    // Apply same sender mutation across all transactions
    evm_word_t new_sender;
    for (int i = 0; i < 5; i++) new_sender._limbs[i] = curand(rng);
    for (int i = 5; i < 8; i++) new_sender._limbs[i] = 0;

    for (uint32_t i = 0; i < seq->num_transactions; i++) {
        for (int j = 0; j < 8; j++) {
            seq->transactions[i].input.sender._limbs[j] = new_sender._limbs[j];
        }
    }
}

__device__ void SequenceMutator::mutate_value_flow(sequence_t* seq, curandState* rng) {
    // Create ascending/descending value pattern
    bool ascending = curand(rng) % 2;
    uint64_t base_value = curand(rng) % 1000000;
    uint64_t delta = curand(rng) % 10000;

    for (uint32_t i = 0; i < seq->num_transactions; i++) {
        uint64_t value = ascending ? (base_value + i * delta) : (base_value - i * delta);
        seq->transactions[i].input.value._limbs[0] = value & 0xFFFFFFFF;
        seq->transactions[i].input.value._limbs[1] = (value >> 32) & 0xFFFFFFFF;
        for (int j = 2; j < 8; j++) {
            seq->transactions[i].input.value._limbs[j] = 0;
        }
    }
}

// ============================================================================
// ABI Helper Implementations
// ============================================================================

namespace abi {

__device__ ABIType detect_param_type(const uint8_t* data, uint32_t offset, uint32_t length) {
    if (offset + 32 > length) return ABIType::UINT256;

    // Count leading zeros
    uint32_t leading_zeros = 0;
    for (uint32_t i = 0; i < 32 && data[offset + i] == 0; i++) {
        leading_zeros++;
    }

    if (leading_zeros >= 12 && leading_zeros < 32) {
        return ABIType::ADDRESS;  // 20-byte address
    } else if (leading_zeros >= 24) {
        return ABIType::UINT64;
    } else if (leading_zeros >= 28) {
        return ABIType::UINT32;
    } else if (leading_zeros >= 30) {
        return ABIType::UINT16;
    } else if (leading_zeros >= 31) {
        return ABIType::UINT8;
    }

    return ABIType::UINT256;
}

__device__ uint32_t get_type_size(ABIType type) {
    switch (type) {
        case ABIType::UINT8:
        case ABIType::INT8:
        case ABIType::BOOL:
        case ABIType::BYTES1:
            return 1;
        case ABIType::UINT16:
        case ABIType::INT16:
        case ABIType::BYTES2:
            return 2;
        case ABIType::UINT32:
        case ABIType::INT32:
        case ABIType::BYTES4:
        case ABIType::FUNCTION:
            return 4;
        case ABIType::UINT64:
        case ABIType::INT64:
        case ABIType::BYTES8:
            return 8;
        case ABIType::UINT128:
        case ABIType::INT128:
        case ABIType::BYTES16:
            return 16;
        case ABIType::ADDRESS:
            return 20;
        case ABIType::UINT256:
        case ABIType::INT256:
        case ABIType::BYTES32:
        default:
            return 32;
    }
}

__device__ void mutate_by_type(uint8_t* data, uint32_t offset, ABIType type, curandState* rng) {
    uint32_t strategy = curand(rng) % 4;

    switch (type) {
        case ABIType::ADDRESS:
            // Zero prefix, then 20 random bytes
            for (int i = 0; i < 12; i++) data[offset + i] = 0;
            for (int i = 12; i < 32; i++) data[offset + i] = curand(rng) & 0xFF;
            break;

        case ABIType::BOOL:
            for (int i = 0; i < 31; i++) data[offset + i] = 0;
            data[offset + 31] = curand(rng) % 2;
            break;

        case ABIType::UINT8:
        case ABIType::INT8:
            for (int i = 0; i < 31; i++) data[offset + i] = 0;
            if (strategy == 0) data[offset + 31] = 0;
            else if (strategy == 1) data[offset + 31] = 0xFF;
            else data[offset + 31] = curand(rng) & 0xFF;
            break;

        case ABIType::UINT256:
        case ABIType::INT256:
        case ABIType::BYTES32:
        default:
            if (strategy == 0) {
                // Zero
                for (int i = 0; i < 32; i++) data[offset + i] = 0;
            } else if (strategy == 1) {
                // Max
                for (int i = 0; i < 32; i++) data[offset + i] = 0xFF;
            } else {
                // Random
                for (int i = 0; i < 32; i++) data[offset + i] = curand(rng) & 0xFF;
            }
            break;
    }
}

__device__ void generate_by_type(uint8_t* data, uint32_t offset, ABIType type, curandState* rng) {
    mutate_by_type(data, offset, type, rng);  // Same logic for generation
}

__device__ bool lookup_selector(const uint8_t* selector, ABIType* param_types, uint32_t* num_params) {
    // This would normally require a full selector database
    // For now, return false (unknown selector)
    return false;
}

}  // namespace abi

// ============================================================================
// CUDA Kernel Implementations
// ============================================================================

__global__ void kernel_init_rng(curandState* states, uint32_t num_states, uint64_t seed) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_states) return;

    curand_init(seed, idx, 0, &states[idx]);
}

__global__ void kernel_mutate_batch(
    GPUMutationEngine* engine,
    mutation_input_t* inputs,
    uint32_t num_inputs,
    uint32_t mutations_per_input,
    curandState* rng_states,
    mutation_result_t* results
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_inputs) return;

    curandState* rng = &rng_states[idx];

    for (uint32_t m = 0; m < mutations_per_input; m++) {
        mutation_result_t result = engine->mutate(&inputs[idx], rng);
        if (results) {
            results[idx * mutations_per_input + m] = result;
        }
    }
}

__global__ void kernel_havoc_batch(
    GPUMutationEngine* engine,
    mutation_input_t* inputs,
    uint32_t num_inputs,
    uint32_t havoc_iterations,
    curandState* rng_states
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_inputs) return;

    curandState* rng = &rng_states[idx];
    engine->havoc(&inputs[idx], rng, havoc_iterations);
}

__global__ void kernel_splice_batch(
    GPUMutationEngine* engine,
    mutation_input_t* dst,
    const mutation_input_t* src1,
    const mutation_input_t* src2,
    uint32_t num_pairs,
    curandState* rng_states
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pairs) return;

    curandState* rng = &rng_states[idx];
    engine->splice(&dst[idx], &src1[idx], &src2[idx], rng);
}

__global__ void kernel_mutate_sequences(
    SequenceMutator* mutator,
    sequence_t* sequences,
    uint32_t num_sequences,
    curandState* rng_states
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_sequences) return;

    curandState* rng = &rng_states[idx];
    mutator->mutate_sequence(&sequences[idx], rng);
}

// ============================================================================
// Host Helper Functions
// ============================================================================

__host__ void allocate_mutation_inputs(mutation_input_t** inputs, uint32_t num_inputs, uint32_t max_size) {
    cudaMallocManaged(inputs, num_inputs * sizeof(mutation_input_t));

    for (uint32_t i = 0; i < num_inputs; i++) {
        cudaMallocManaged(&(*inputs)[i].data, max_size);
        (*inputs)[i].init(max_size);
    }
}

__host__ void free_mutation_inputs(mutation_input_t* inputs, uint32_t num_inputs) {
    for (uint32_t i = 0; i < num_inputs; i++) {
        if (inputs[i].data) {
            cudaFree(inputs[i].data);
        }
    }
    cudaFree(inputs);
}

__host__ void allocate_sequences(sequence_t** sequences, uint32_t num_sequences, uint32_t max_txs) {
    cudaMallocManaged(sequences, num_sequences * sizeof(sequence_t));

    for (uint32_t i = 0; i < num_sequences; i++) {
        cudaMallocManaged(&(*sequences)[i].transactions, max_txs * sizeof(transaction_t));
        (*sequences)[i].init(max_txs);
    }
}

__host__ void free_sequences(sequence_t* sequences, uint32_t num_sequences) {
    for (uint32_t i = 0; i < num_sequences; i++) {
        if (sequences[i].transactions) {
            cudaFree(sequences[i].transactions);
        }
    }
    cudaFree(sequences);
}

}  // namespace fuzzing
}  // namespace CuEVM
