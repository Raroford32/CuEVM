// CuEVM: CUDA Ethereum Virtual Machine implementation
// GPU Mutation Engine for NVIDIA B300 Smart Contract Fuzzing
// SPDX-License-Identifier: MIT

#ifndef _CUEVM_FUZZING_MUTATION_H_
#define _CUEVM_FUZZING_MUTATION_H_

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdint>
#include <CuEVM/utils/arith.cuh>

namespace CuEVM {
namespace fuzzing {

// ============================================================================
// Configuration Constants for B300 Optimization
// ============================================================================

constexpr uint32_t MAX_MUTATION_SIZE = 4096;           // Max bytes to mutate
constexpr uint32_t MAX_DICTIONARY_SIZE = 1024;         // Dictionary entries
constexpr uint32_t MAX_INTERESTING_VALUES = 256;       // Interesting value pool
constexpr uint32_t MUTATION_STACK_SIZE = 16;           // Havoc mutation stack
constexpr uint32_t MAX_SPLICE_LENGTH = 512;            // Max splice size
constexpr uint32_t ARITH_MAX_DELTA = 35;               // Max arithmetic delta

// Mutation type weights (0-255 for probability weighting)
constexpr uint8_t WEIGHT_BIT_FLIP = 20;
constexpr uint8_t WEIGHT_BYTE_FLIP = 20;
constexpr uint8_t WEIGHT_ARITH_INC = 15;
constexpr uint8_t WEIGHT_ARITH_DEC = 15;
constexpr uint8_t WEIGHT_INTERESTING = 25;
constexpr uint8_t WEIGHT_DICTIONARY = 30;
constexpr uint8_t WEIGHT_HAVOC = 40;
constexpr uint8_t WEIGHT_SPLICE = 15;
constexpr uint8_t WEIGHT_COPY = 10;
constexpr uint8_t WEIGHT_INSERT = 10;
constexpr uint8_t WEIGHT_DELETE = 10;
constexpr uint8_t WEIGHT_OVERWRITE = 15;
constexpr uint8_t WEIGHT_CROSSOVER = 20;

// ============================================================================
// Mutation Types
// ============================================================================

enum class MutationType : uint8_t {
    // Bit-level mutations
    FLIP_BIT_1 = 0,
    FLIP_BIT_2 = 1,
    FLIP_BIT_4 = 2,

    // Byte-level mutations
    FLIP_BYTE_1 = 3,
    FLIP_BYTE_2 = 4,
    FLIP_BYTE_4 = 5,

    // Arithmetic mutations
    ARITH_INC_8 = 6,
    ARITH_DEC_8 = 7,
    ARITH_INC_16 = 8,
    ARITH_DEC_16 = 9,
    ARITH_INC_32 = 10,
    ARITH_DEC_32 = 11,
    ARITH_INC_64 = 12,
    ARITH_DEC_64 = 13,

    // Interesting value replacements
    INTERESTING_8 = 14,
    INTERESTING_16 = 15,
    INTERESTING_32 = 16,
    INTERESTING_64 = 17,
    INTERESTING_256 = 18,

    // Dictionary-based
    DICT_INSERT = 19,
    DICT_OVERWRITE = 20,

    // Structural mutations
    CLONE_BYTE = 21,
    DELETE_BYTES = 22,
    INSERT_BYTES = 23,
    OVERWRITE_BYTES = 24,
    SWAP_BYTES = 25,
    SHUFFLE_BYTES = 26,

    // Havoc (random multi-mutation)
    HAVOC_SINGLE = 27,
    HAVOC_MULTI = 28,

    // Cross-input mutations
    SPLICE = 29,
    CROSSOVER = 30,

    // EVM-specific mutations
    EVM_ADDRESS = 31,
    EVM_UINT256 = 32,
    EVM_BYTES32 = 33,
    EVM_SELECTOR = 34,
    EVM_CALLDATA = 35,

    // Boundary mutations
    BOUNDARY_LOW = 36,
    BOUNDARY_HIGH = 37,
    BOUNDARY_POWER2 = 38,

    // Gradient-guided
    GRADIENT_INC = 39,
    GRADIENT_DEC = 40,

    NUM_MUTATION_TYPES = 41
};

// ============================================================================
// Interesting Values for Smart Contracts
// ============================================================================

// 8-bit interesting values
__constant__ int8_t INTERESTING_8_VALUES[] = {
    -128, -1, 0, 1, 16, 32, 64, 100, 127
};
constexpr uint32_t NUM_INTERESTING_8 = 9;

// 16-bit interesting values
__constant__ int16_t INTERESTING_16_VALUES[] = {
    -32768, -129, -128, -1, 0, 1, 127, 128, 255, 256,
    512, 1000, 1024, 4096, 32767
};
constexpr uint32_t NUM_INTERESTING_16 = 15;

// 32-bit interesting values
__constant__ int32_t INTERESTING_32_VALUES[] = {
    -2147483648, -100663046, -32769, -32768, -129, -128, -1,
    0, 1, 127, 128, 255, 256, 512, 1000, 1024, 4096, 32767,
    32768, 65535, 65536, 100663045, 2147483647
};
constexpr uint32_t NUM_INTERESTING_32 = 23;

// 64-bit interesting values (for Solidity uint256 boundaries)
__constant__ int64_t INTERESTING_64_VALUES[] = {
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
constexpr uint32_t NUM_INTERESTING_64 = 14;

// EVM-specific interesting values
struct evm_interesting_t {
    evm_word_t value;
    const char* description;
};

// ============================================================================
// Dictionary Entry for Smart Contract Fuzzing
// ============================================================================

struct dictionary_entry_t {
    uint8_t data[64];           // Entry data (max 64 bytes)
    uint8_t length;             // Actual length
    uint8_t entry_type;         // Type: address, selector, value, etc.
    uint16_t hit_count;         // How often this produced new coverage
    uint32_t source_pc;         // Where this value was observed
};

enum class DictionaryEntryType : uint8_t {
    ADDRESS = 0,
    FUNCTION_SELECTOR = 1,
    UINT256_VALUE = 2,
    BYTES32_VALUE = 3,
    STRING_VALUE = 4,
    ARRAY_LENGTH = 5,
    STORAGE_SLOT = 6,
    BLOCK_VALUE = 7,
    COMPARISON_OPERAND = 8,
    MAGIC_CONSTANT = 9
};

// ============================================================================
// Mutation Dictionary
// ============================================================================

struct mutation_dictionary_t {
    dictionary_entry_t entries[MAX_DICTIONARY_SIZE];
    uint32_t num_entries;
    uint32_t next_insert_idx;

    // Type-specific indices for efficient lookup
    uint16_t address_indices[256];
    uint16_t selector_indices[256];
    uint16_t value_indices[256];
    uint16_t num_addresses;
    uint16_t num_selectors;
    uint16_t num_values;

    __host__ __device__ void init();
    __host__ __device__ bool add_entry(const uint8_t* data, uint8_t length, DictionaryEntryType type, uint32_t pc);
    __host__ __device__ const dictionary_entry_t* get_random(curandState* rng, DictionaryEntryType type = (DictionaryEntryType)255);
    __host__ __device__ void update_hit_count(uint32_t idx);
};

// ============================================================================
// Input Representation for Mutation
// ============================================================================

struct mutation_input_t {
    uint8_t* data;              // Raw input bytes
    uint32_t length;            // Current length
    uint32_t capacity;          // Max allocated size

    // EVM-specific parsed structure
    uint8_t selector[4];        // Function selector
    uint32_t num_params;        // Number of ABI parameters
    uint32_t param_offsets[32]; // Offset of each parameter
    uint8_t param_types[32];    // Type of each parameter

    // Transaction context
    evm_word_t value;           // msg.value
    evm_word_t gas_limit;       // Gas limit
    evm_word_t sender;          // msg.sender
    evm_word_t receiver;        // Target address

    // Block context
    evm_word_t block_number;
    evm_word_t timestamp;
    evm_word_t basefee;
    evm_word_t prevrandao;

    __host__ __device__ void init(uint32_t max_size);
    __host__ __device__ void copy_from(const mutation_input_t& other);
    __host__ __device__ void parse_abi();
    __host__ __device__ void reserialize_abi();
};

// ============================================================================
// Mutation Result
// ============================================================================

struct mutation_result_t {
    MutationType type;
    uint32_t offset;
    uint32_t length;
    int32_t size_delta;         // Change in input size
    bool success;
    uint32_t mutation_id;       // For tracking/replay
};

// ============================================================================
// GPU Random Number Generator State
// ============================================================================

struct gpu_rng_state_t {
    curandState* states;        // Per-thread RNG states
    uint32_t num_states;

    __host__ void init(uint32_t num_threads, uint64_t seed);
    __host__ void free();
};

// ============================================================================
// GPU Mutation Engine
// ============================================================================

class GPUMutationEngine {
public:
    __host__ GPUMutationEngine(uint32_t num_instances, uint64_t seed = 0);
    __host__ ~GPUMutationEngine();

    // Single mutation operations
    __device__ mutation_result_t mutate(mutation_input_t* input, curandState* rng);
    __device__ mutation_result_t mutate_typed(mutation_input_t* input, MutationType type, curandState* rng);

    // Batch mutations
    __host__ void mutate_batch(mutation_input_t* inputs, uint32_t num_inputs,
                               uint32_t mutations_per_input, cudaStream_t stream = 0);

    // Havoc mutation (multiple random mutations)
    __device__ void havoc(mutation_input_t* input, curandState* rng, uint32_t num_mutations);

    // Splice two inputs
    __device__ void splice(mutation_input_t* dst, const mutation_input_t* src1,
                           const mutation_input_t* src2, curandState* rng);

    // Crossover two inputs
    __device__ void crossover(mutation_input_t* dst, const mutation_input_t* src1,
                              const mutation_input_t* src2, curandState* rng);

    // EVM-specific mutations
    __device__ void mutate_address(mutation_input_t* input, uint32_t offset, curandState* rng);
    __device__ void mutate_uint256(mutation_input_t* input, uint32_t offset, curandState* rng);
    __device__ void mutate_selector(mutation_input_t* input, curandState* rng);
    __device__ void mutate_calldata(mutation_input_t* input, curandState* rng);
    __device__ void mutate_value(mutation_input_t* input, curandState* rng);
    __device__ void mutate_gas(mutation_input_t* input, curandState* rng);
    __device__ void mutate_sender(mutation_input_t* input, curandState* rng);
    __device__ void mutate_block_context(mutation_input_t* input, curandState* rng);

    // Dictionary operations
    __host__ __device__ void add_to_dictionary(const uint8_t* data, uint8_t length,
                                                DictionaryEntryType type, uint32_t pc);
    __device__ void apply_dictionary(mutation_input_t* input, curandState* rng);

    // Gradient-guided mutation
    __device__ void gradient_mutate(mutation_input_t* input, uint32_t target_offset,
                                    bool increase, curandState* rng);

    // Configuration
    __host__ void set_mutation_weights(const uint8_t* weights);
    __host__ void set_max_mutations(uint32_t max);
    __host__ void enable_abi_aware(bool enable);

    // Get dictionary
    __host__ __device__ mutation_dictionary_t* get_dictionary() { return dictionary_; }

private:
    gpu_rng_state_t rng_state_;
    mutation_dictionary_t* dictionary_;
    uint8_t mutation_weights_[64];
    uint32_t max_mutations_;
    bool abi_aware_;

    // Internal mutation implementations
    __device__ void flip_bit(uint8_t* data, uint32_t length, uint32_t offset, uint8_t width);
    __device__ void flip_byte(uint8_t* data, uint32_t length, uint32_t offset, uint8_t width);
    __device__ void arith_mutation(uint8_t* data, uint32_t length, uint32_t offset, uint8_t width, bool increment, int32_t delta);
    __device__ void interesting_mutation(uint8_t* data, uint32_t length, uint32_t offset, uint8_t width, curandState* rng);
    __device__ void clone_bytes(mutation_input_t* input, uint32_t src_offset, uint32_t dst_offset, uint32_t count);
    __device__ void delete_bytes(mutation_input_t* input, uint32_t offset, uint32_t count);
    __device__ void insert_bytes(mutation_input_t* input, uint32_t offset, const uint8_t* data, uint32_t count);
    __device__ void overwrite_bytes(mutation_input_t* input, uint32_t offset, const uint8_t* data, uint32_t count);
    __device__ void swap_bytes(uint8_t* data, uint32_t offset1, uint32_t offset2, uint32_t count);
    __device__ void shuffle_bytes(uint8_t* data, uint32_t offset, uint32_t count, curandState* rng);

    __device__ MutationType select_mutation_type(curandState* rng);
    __device__ uint32_t select_offset(uint32_t length, curandState* rng);
};

// ============================================================================
// Sequence Mutation (for multi-transaction fuzzing)
// ============================================================================

struct transaction_t {
    mutation_input_t input;
    uint32_t sequence_id;
    uint32_t tx_index;
    bool is_deploy;             // CREATE/CREATE2
};

struct sequence_t {
    transaction_t* transactions;
    uint32_t num_transactions;
    uint32_t capacity;
    uint64_t seed;              // For deterministic replay

    __host__ __device__ void init(uint32_t max_txs);
    __host__ __device__ void add_transaction(const transaction_t& tx);
    __host__ __device__ void remove_transaction(uint32_t index);
    __host__ __device__ void reorder(uint32_t from, uint32_t to);
    __host__ __device__ void copy_from(const sequence_t& other);
};

class SequenceMutator {
public:
    __host__ SequenceMutator(GPUMutationEngine* engine);

    // Sequence-level mutations
    __device__ void mutate_sequence(sequence_t* seq, curandState* rng);
    __device__ void insert_transaction(sequence_t* seq, uint32_t index, curandState* rng);
    __device__ void delete_transaction(sequence_t* seq, uint32_t index);
    __device__ void duplicate_transaction(sequence_t* seq, uint32_t index);
    __device__ void swap_transactions(sequence_t* seq, uint32_t idx1, uint32_t idx2);
    __device__ void splice_sequences(sequence_t* dst, const sequence_t* src1, const sequence_t* src2, curandState* rng);

    // Mutate individual transaction in sequence
    __device__ void mutate_transaction(sequence_t* seq, uint32_t tx_index, curandState* rng);

    // Mutate sender pattern across sequence
    __device__ void mutate_sender_pattern(sequence_t* seq, curandState* rng);

    // Mutate value flow across sequence
    __device__ void mutate_value_flow(sequence_t* seq, curandState* rng);

private:
    GPUMutationEngine* engine_;
};

// ============================================================================
// ABI-Aware Mutation Helpers
// ============================================================================

namespace abi {

// ABI type codes
enum class ABIType : uint8_t {
    UINT8 = 0, UINT16 = 1, UINT32 = 2, UINT64 = 3, UINT128 = 4, UINT256 = 5,
    INT8 = 6, INT16 = 7, INT32 = 8, INT64 = 9, INT128 = 10, INT256 = 11,
    ADDRESS = 12,
    BOOL = 13,
    BYTES1 = 14, BYTES2 = 15, BYTES4 = 16, BYTES8 = 17, BYTES16 = 18, BYTES32 = 19,
    BYTES_DYN = 20,
    STRING = 21,
    ARRAY_FIXED = 22,
    ARRAY_DYN = 23,
    TUPLE = 24,
    FUNCTION = 25
};

__device__ ABIType detect_param_type(const uint8_t* data, uint32_t offset, uint32_t length);
__device__ uint32_t get_type_size(ABIType type);
__device__ void mutate_by_type(uint8_t* data, uint32_t offset, ABIType type, curandState* rng);
__device__ void generate_by_type(uint8_t* data, uint32_t offset, ABIType type, curandState* rng);

// Parse function selector to get expected parameter types
__device__ bool lookup_selector(const uint8_t* selector, ABIType* param_types, uint32_t* num_params);

}  // namespace abi

// ============================================================================
// CUDA Kernels
// ============================================================================

// Kernel to initialize RNG states
__global__ void kernel_init_rng(curandState* states, uint32_t num_states, uint64_t seed);

// Kernel to mutate a batch of inputs
__global__ void kernel_mutate_batch(
    GPUMutationEngine* engine,
    mutation_input_t* inputs,
    uint32_t num_inputs,
    uint32_t mutations_per_input,
    curandState* rng_states,
    mutation_result_t* results
);

// Kernel to perform havoc mutation
__global__ void kernel_havoc_batch(
    GPUMutationEngine* engine,
    mutation_input_t* inputs,
    uint32_t num_inputs,
    uint32_t havoc_iterations,
    curandState* rng_states
);

// Kernel to splice inputs pairwise
__global__ void kernel_splice_batch(
    GPUMutationEngine* engine,
    mutation_input_t* dst,
    const mutation_input_t* src1,
    const mutation_input_t* src2,
    uint32_t num_pairs,
    curandState* rng_states
);

// Kernel to mutate sequences
__global__ void kernel_mutate_sequences(
    SequenceMutator* mutator,
    sequence_t* sequences,
    uint32_t num_sequences,
    curandState* rng_states
);

// ============================================================================
// Host Helper Functions
// ============================================================================

__host__ void allocate_mutation_inputs(mutation_input_t** inputs, uint32_t num_inputs, uint32_t max_size);
__host__ void free_mutation_inputs(mutation_input_t* inputs, uint32_t num_inputs);
__host__ void copy_inputs_to_device(mutation_input_t* d_inputs, const mutation_input_t* h_inputs, uint32_t num_inputs);
__host__ void copy_inputs_to_host(mutation_input_t* h_inputs, const mutation_input_t* d_inputs, uint32_t num_inputs);

__host__ void allocate_sequences(sequence_t** sequences, uint32_t num_sequences, uint32_t max_txs);
__host__ void free_sequences(sequence_t* sequences, uint32_t num_sequences);

}  // namespace fuzzing
}  // namespace CuEVM

#endif  // _CUEVM_FUZZING_MUTATION_H_
