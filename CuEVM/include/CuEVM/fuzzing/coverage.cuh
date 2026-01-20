// CuEVM: CUDA Ethereum Virtual Machine implementation
// GPU Coverage Instrumentation for NVIDIA B300 Smart Contract Fuzzing
// SPDX-License-Identifier: MIT

#ifndef _CUEVM_FUZZING_COVERAGE_H_
#define _CUEVM_FUZZING_COVERAGE_H_

#include <cuda_runtime.h>
#include <cstdint>
#include <CuEVM/utils/arith.cuh>

namespace CuEVM {
namespace fuzzing {

// Coverage map sizes optimized for B300 (SM 103)
constexpr uint32_t COVERAGE_MAP_SIZE = 65536;           // 64KB coverage bitmap
constexpr uint32_t BRANCH_COVERAGE_SIZE = 32768;        // 32K branch coverage entries
constexpr uint32_t OPCODE_COVERAGE_SIZE = 256;          // All EVM opcodes
constexpr uint32_t STORAGE_COVERAGE_SIZE = 16384;       // Storage slot coverage
constexpr uint32_t CALL_COVERAGE_SIZE = 4096;           // Call target coverage
constexpr uint32_t PC_COVERAGE_SIZE = 65536;            // Program counter coverage
constexpr uint32_t EDGE_COVERAGE_SIZE = 131072;         // Edge coverage (pc_from -> pc_to)

// Coverage hit counter types
using coverage_counter_t = uint8_t;       // Saturating counter
using coverage_bitmap_t = uint32_t;       // Bitmap word

// Branch distance quantization for gradient guidance
constexpr uint32_t DISTANCE_BUCKETS = 16;
constexpr uint64_t DISTANCE_THRESHOLDS[DISTANCE_BUCKETS] = {
    0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096, 16384, 65536, UINT64_MAX
};

/**
 * Edge coverage entry tracking source->destination transitions
 */
struct edge_coverage_entry_t {
    uint32_t pc_from;
    uint32_t pc_to;
    uint32_t hit_count;
    uint32_t contract_id;
};

/**
 * Branch coverage entry with distance tracking for gradient-guided fuzzing
 */
struct branch_coverage_entry_t {
    uint32_t pc;
    uint32_t true_target;
    uint32_t false_target;
    uint8_t taken_true;
    uint8_t taken_false;
    uint8_t distance_bucket;  // Quantized distance for JUMPI condition
    uint64_t min_distance;    // Minimum observed distance to flip branch
};

/**
 * Storage coverage entry for tracking SLOAD/SSTORE patterns
 */
struct storage_coverage_entry_t {
    uint32_t pc;
    uint32_t slot_hash;       // Hash of storage slot
    uint8_t is_read;
    uint8_t is_write;
    uint8_t is_warm;
    uint8_t value_changed;
};

/**
 * Call coverage entry for tracking inter-contract calls
 */
struct call_coverage_entry_t {
    uint32_t pc;
    uint32_t caller_contract_id;
    uint32_t callee_address_hash;
    uint8_t opcode;           // CALL, CALLCODE, DELEGATECALL, STATICCALL
    uint8_t success;
    uint8_t is_precompile;
    uint8_t value_transferred;
};

/**
 * Opcode execution statistics
 */
struct opcode_stats_t {
    uint64_t execution_count;
    uint64_t gas_used_total;
    uint32_t max_stack_depth;
    uint32_t error_count;
};

/**
 * Per-contract coverage data
 */
struct contract_coverage_t {
    uint32_t contract_id;
    uint32_t code_size;
    uint32_t unique_pcs_hit;
    uint32_t unique_branches_hit;
    uint32_t unique_edges_hit;
    float pc_coverage_percent;
    float branch_coverage_percent;
    float edge_coverage_percent;
};

/**
 * GPU Coverage Map - Main coverage tracking structure
 * Designed for efficient parallel updates on B300
 */
struct gpu_coverage_map_t {
    // Primary coverage bitmaps (atomically updated)
    coverage_counter_t* pc_bitmap;              // [PC_COVERAGE_SIZE]
    coverage_counter_t* edge_bitmap;            // [EDGE_COVERAGE_SIZE]
    coverage_counter_t* opcode_counters;        // [OPCODE_COVERAGE_SIZE]

    // Detailed coverage tracking
    branch_coverage_entry_t* branch_entries;    // [BRANCH_COVERAGE_SIZE]
    storage_coverage_entry_t* storage_entries;  // [STORAGE_COVERAGE_SIZE]
    call_coverage_entry_t* call_entries;        // [CALL_COVERAGE_SIZE]

    // Statistics
    opcode_stats_t* opcode_stats;               // [OPCODE_COVERAGE_SIZE]
    contract_coverage_t* contract_coverage;     // Per-contract stats

    // Counters
    uint32_t num_branch_entries;
    uint32_t num_storage_entries;
    uint32_t num_call_entries;
    uint32_t num_contracts;

    // Global statistics
    uint64_t total_instructions_executed;
    uint64_t total_branches_executed;
    uint64_t total_storage_ops;
    uint64_t total_calls;
    uint64_t total_gas_used;

    // Coverage metrics
    uint32_t unique_pcs;
    uint32_t unique_edges;
    uint32_t unique_branches;
    float overall_coverage;

    // Bitmap for quick "new coverage" detection
    coverage_bitmap_t* virgin_bits;             // [COVERAGE_MAP_SIZE / 32]

    __host__ __device__ void init();
    __host__ __device__ void reset();
    __host__ __device__ void merge(const gpu_coverage_map_t& other);
};

/**
 * Per-instance coverage state (thread-local during execution)
 */
struct instance_coverage_t {
    // Hash-based compact representation for GPU efficiency
    uint32_t edge_hashes[256];                  // Recent edge hashes
    uint32_t edge_hash_idx;

    uint32_t branch_hashes[64];                 // Recent branch decisions
    uint32_t branch_hash_idx;

    uint32_t storage_hashes[64];                // Recent storage accesses
    uint32_t storage_hash_idx;

    // Quick stats for this instance
    uint32_t pcs_hit;
    uint32_t edges_hit;
    uint32_t branches_taken;
    uint32_t storage_ops;
    uint32_t calls_made;

    // Last PC for edge tracking
    uint32_t last_pc;
    uint32_t last_opcode;

    __host__ __device__ void init();
    __host__ __device__ void record_pc(uint32_t pc);
    __host__ __device__ void record_edge(uint32_t from_pc, uint32_t to_pc);
    __host__ __device__ void record_branch(uint32_t pc, bool taken, uint64_t distance);
    __host__ __device__ void record_storage(uint32_t pc, uint32_t slot_hash, bool is_write);
    __host__ __device__ void record_call(uint32_t pc, uint32_t target_hash, uint8_t opcode, bool success);
};

/**
 * Coverage instrumentation hooks for EVM execution
 */
class CoverageInstrumentation {
public:
    __host__ __device__ CoverageInstrumentation(gpu_coverage_map_t* global_map, instance_coverage_t* instance);

    // Pre-execution hooks
    __host__ __device__ void on_instruction_start(uint32_t pc, uint8_t opcode);

    // Post-execution hooks
    __host__ __device__ void on_instruction_end(uint32_t pc, uint8_t opcode, uint32_t error_code);

    // Branch coverage
    __host__ __device__ void on_jump(uint32_t from_pc, uint32_t to_pc);
    __host__ __device__ void on_jumpi(uint32_t pc, uint32_t target, bool taken,
                                       const evm_word_t& condition);

    // Storage coverage
    __host__ __device__ void on_sload(uint32_t pc, const evm_word_t& slot, bool warm);
    __host__ __device__ void on_sstore(uint32_t pc, const evm_word_t& slot,
                                        const evm_word_t& old_value, const evm_word_t& new_value);

    // Call coverage
    __host__ __device__ void on_call(uint32_t pc, uint8_t opcode, const evm_word_t& target,
                                      const evm_word_t& value, bool success);

    // Memory coverage
    __host__ __device__ void on_memory_access(uint32_t pc, uint32_t offset, uint32_t size, bool is_write);

    // Comparison coverage (for gradient-guided fuzzing)
    __host__ __device__ void on_comparison(uint32_t pc, uint8_t opcode,
                                            const evm_word_t& a, const evm_word_t& b,
                                            const evm_word_t& result);

    // Return/revert coverage
    __host__ __device__ void on_return(uint32_t pc, bool success, uint32_t return_size);

    // Merge instance coverage to global
    __host__ __device__ void finalize();

private:
    gpu_coverage_map_t* global_map_;
    instance_coverage_t* instance_;

    __host__ __device__ uint32_t hash_edge(uint32_t from, uint32_t to);
    __host__ __device__ uint32_t hash_slot(const evm_word_t& slot);
    __host__ __device__ uint8_t quantize_distance(uint64_t distance);
    __host__ __device__ uint64_t compute_branch_distance(const evm_word_t& condition);
};

/**
 * Coverage map allocator for B300
 */
class CoverageMapAllocator {
public:
    __host__ static gpu_coverage_map_t* allocate_global(uint32_t num_contracts = 1);
    __host__ static instance_coverage_t* allocate_instances(uint32_t num_instances);
    __host__ static void free_global(gpu_coverage_map_t* map);
    __host__ static void free_instances(instance_coverage_t* instances);

    // Pinned memory for efficient host-device transfer
    __host__ static gpu_coverage_map_t* allocate_pinned();
    __host__ static void copy_to_host(gpu_coverage_map_t* host_map, const gpu_coverage_map_t* device_map);
};

/**
 * Coverage serialization for corpus management
 */
struct coverage_snapshot_t {
    uint8_t* pc_bitmap_data;
    uint32_t pc_bitmap_size;
    uint8_t* edge_bitmap_data;
    uint32_t edge_bitmap_size;
    uint32_t unique_pcs;
    uint32_t unique_edges;
    uint32_t unique_branches;
    float coverage_score;
    uint64_t timestamp;

    __host__ void serialize(void* buffer, size_t* size);
    __host__ static coverage_snapshot_t deserialize(const void* buffer, size_t size);
    __host__ bool has_new_coverage(const coverage_snapshot_t& baseline);
    __host__ float novelty_score(const coverage_snapshot_t& baseline);
};

/**
 * AFL-style coverage bitmap operations
 */
namespace bitmap_ops {
    __host__ __device__ uint32_t hash_pc(uint32_t pc, uint32_t prev_pc);
    __host__ __device__ void increment_counter(coverage_counter_t* bitmap, uint32_t index);
    __host__ __device__ bool check_virgin(coverage_bitmap_t* virgin, uint32_t index);
    __host__ __device__ void mark_virgin(coverage_bitmap_t* virgin, uint32_t index);
    __host__ uint32_t count_bits(const coverage_counter_t* bitmap, uint32_t size);
    __host__ uint32_t count_nonzero(const coverage_counter_t* bitmap, uint32_t size);
    __host__ void merge_bitmaps(coverage_counter_t* dst, const coverage_counter_t* src, uint32_t size);
    __host__ bool has_new_bits(const coverage_counter_t* current, const coverage_counter_t* virgin, uint32_t size);
}

// CUDA kernel for batch coverage merging
__global__ void kernel_merge_coverage(
    gpu_coverage_map_t* global_map,
    instance_coverage_t* instances,
    uint32_t num_instances
);

// CUDA kernel for computing coverage statistics
__global__ void kernel_compute_coverage_stats(
    gpu_coverage_map_t* map,
    uint32_t* unique_pcs,
    uint32_t* unique_edges,
    float* coverage_score
);

// CUDA kernel for virgin bits detection
__global__ void kernel_detect_new_coverage(
    gpu_coverage_map_t* current,
    gpu_coverage_map_t* baseline,
    uint32_t* new_coverage_flags,
    uint32_t num_instances
);

}  // namespace fuzzing
}  // namespace CuEVM

#endif  // _CUEVM_FUZZING_COVERAGE_H_
