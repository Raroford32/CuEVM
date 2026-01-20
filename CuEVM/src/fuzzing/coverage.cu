// CuEVM: CUDA Ethereum Virtual Machine implementation
// GPU Coverage Instrumentation Implementation for NVIDIA B300
// SPDX-License-Identifier: MIT

#include <CuEVM/fuzzing/coverage.cuh>
#include <cuda_runtime.h>
#include <cstring>

namespace CuEVM {
namespace fuzzing {

// ============================================================================
// GPU Coverage Map Implementation
// ============================================================================

__host__ __device__ void gpu_coverage_map_t::init() {
    num_branch_entries = 0;
    num_storage_entries = 0;
    num_call_entries = 0;
    num_contracts = 0;
    total_instructions_executed = 0;
    total_branches_executed = 0;
    total_storage_ops = 0;
    total_calls = 0;
    total_gas_used = 0;
    unique_pcs = 0;
    unique_edges = 0;
    unique_branches = 0;
    overall_coverage = 0.0f;
}

__host__ __device__ void gpu_coverage_map_t::reset() {
    if (pc_bitmap) {
        for (uint32_t i = 0; i < PC_COVERAGE_SIZE; i++) {
            pc_bitmap[i] = 0;
        }
    }
    if (edge_bitmap) {
        for (uint32_t i = 0; i < EDGE_COVERAGE_SIZE; i++) {
            edge_bitmap[i] = 0;
        }
    }
    if (opcode_counters) {
        for (uint32_t i = 0; i < OPCODE_COVERAGE_SIZE; i++) {
            opcode_counters[i] = 0;
        }
    }
    init();
}

__host__ __device__ void gpu_coverage_map_t::merge(const gpu_coverage_map_t& other) {
    // Merge bitmap counters using saturating addition
    for (uint32_t i = 0; i < PC_COVERAGE_SIZE; i++) {
        uint16_t sum = (uint16_t)pc_bitmap[i] + (uint16_t)other.pc_bitmap[i];
        pc_bitmap[i] = (sum > 255) ? 255 : (coverage_counter_t)sum;
    }

    for (uint32_t i = 0; i < EDGE_COVERAGE_SIZE; i++) {
        uint16_t sum = (uint16_t)edge_bitmap[i] + (uint16_t)other.edge_bitmap[i];
        edge_bitmap[i] = (sum > 255) ? 255 : (coverage_counter_t)sum;
    }

    for (uint32_t i = 0; i < OPCODE_COVERAGE_SIZE; i++) {
        uint16_t sum = (uint16_t)opcode_counters[i] + (uint16_t)other.opcode_counters[i];
        opcode_counters[i] = (sum > 255) ? 255 : (coverage_counter_t)sum;
    }

    // Merge statistics
    total_instructions_executed += other.total_instructions_executed;
    total_branches_executed += other.total_branches_executed;
    total_storage_ops += other.total_storage_ops;
    total_calls += other.total_calls;
    total_gas_used += other.total_gas_used;
}

// ============================================================================
// Instance Coverage Implementation
// ============================================================================

__host__ __device__ void instance_coverage_t::init() {
    edge_hash_idx = 0;
    branch_hash_idx = 0;
    storage_hash_idx = 0;
    pcs_hit = 0;
    edges_hit = 0;
    branches_taken = 0;
    storage_ops = 0;
    calls_made = 0;
    last_pc = 0;
    last_opcode = 0;

    for (int i = 0; i < 256; i++) edge_hashes[i] = 0;
    for (int i = 0; i < 64; i++) branch_hashes[i] = 0;
    for (int i = 0; i < 64; i++) storage_hashes[i] = 0;
}

__host__ __device__ void instance_coverage_t::record_pc(uint32_t pc) {
    pcs_hit++;
    last_pc = pc;
}

__host__ __device__ void instance_coverage_t::record_edge(uint32_t from_pc, uint32_t to_pc) {
    // AFL-style edge hashing
    uint32_t hash = (from_pc >> 1) ^ to_pc;
    edge_hashes[edge_hash_idx & 255] = hash;
    edge_hash_idx++;
    edges_hit++;
}

__host__ __device__ void instance_coverage_t::record_branch(uint32_t pc, bool taken, uint64_t distance) {
    uint32_t hash = pc | (taken ? 0x80000000 : 0);
    branch_hashes[branch_hash_idx & 63] = hash;
    branch_hash_idx++;
    branches_taken++;
}

__host__ __device__ void instance_coverage_t::record_storage(uint32_t pc, uint32_t slot_hash, bool is_write) {
    uint32_t hash = (pc << 16) ^ slot_hash ^ (is_write ? 0x1 : 0x0);
    storage_hashes[storage_hash_idx & 63] = hash;
    storage_hash_idx++;
    storage_ops++;
}

__host__ __device__ void instance_coverage_t::record_call(uint32_t pc, uint32_t target_hash, uint8_t opcode, bool success) {
    calls_made++;
}

// ============================================================================
// Coverage Instrumentation Implementation
// ============================================================================

__host__ __device__ CoverageInstrumentation::CoverageInstrumentation(
    gpu_coverage_map_t* global_map, instance_coverage_t* instance)
    : global_map_(global_map), instance_(instance) {}

__host__ __device__ void CoverageInstrumentation::on_instruction_start(uint32_t pc, uint8_t opcode) {
    instance_->record_pc(pc);

    // Track edge from last PC
    if (instance_->last_pc != 0) {
        instance_->record_edge(instance_->last_pc, pc);
    }

    instance_->last_opcode = opcode;
}

__host__ __device__ void CoverageInstrumentation::on_instruction_end(uint32_t pc, uint8_t opcode, uint32_t error_code) {
    instance_->last_pc = pc;

    // Update global statistics atomically
#ifdef __CUDA_ARCH__
    atomicAdd(&global_map_->total_instructions_executed, 1ULL);
#else
    global_map_->total_instructions_executed++;
#endif
}

__host__ __device__ void CoverageInstrumentation::on_jump(uint32_t from_pc, uint32_t to_pc) {
    instance_->record_edge(from_pc, to_pc);

    // Update edge bitmap
    uint32_t edge_hash = hash_edge(from_pc, to_pc);
    uint32_t index = edge_hash % EDGE_COVERAGE_SIZE;

#ifdef __CUDA_ARCH__
    atomicAdd((unsigned char*)&global_map_->edge_bitmap[index], 1);
#else
    if (global_map_->edge_bitmap[index] < 255) {
        global_map_->edge_bitmap[index]++;
    }
#endif
}

__host__ __device__ void CoverageInstrumentation::on_jumpi(uint32_t pc, uint32_t target, bool taken,
                                                           const evm_word_t& condition) {
    uint64_t distance = compute_branch_distance(condition);
    instance_->record_branch(pc, taken, distance);

    // Update global branch counter
#ifdef __CUDA_ARCH__
    atomicAdd(&global_map_->total_branches_executed, 1ULL);
#else
    global_map_->total_branches_executed++;
#endif

    // Track branch in detailed entries if space available
    uint32_t entry_idx;
#ifdef __CUDA_ARCH__
    entry_idx = atomicAdd(&global_map_->num_branch_entries, 1);
#else
    entry_idx = global_map_->num_branch_entries++;
#endif

    if (entry_idx < BRANCH_COVERAGE_SIZE) {
        branch_coverage_entry_t* entry = &global_map_->branch_entries[entry_idx];
        entry->pc = pc;
        entry->distance_bucket = quantize_distance(distance);
        if (taken) {
            entry->taken_true = 1;
            entry->true_target = target;
        } else {
            entry->taken_false = 1;
            entry->false_target = target;
        }
        if (entry->min_distance == 0 || distance < entry->min_distance) {
            entry->min_distance = distance;
        }
    }
}

__host__ __device__ void CoverageInstrumentation::on_sload(uint32_t pc, const evm_word_t& slot, bool warm) {
    uint32_t slot_hash = hash_slot(slot);
    instance_->record_storage(pc, slot_hash, false);

#ifdef __CUDA_ARCH__
    atomicAdd(&global_map_->total_storage_ops, 1ULL);
#else
    global_map_->total_storage_ops++;
#endif
}

__host__ __device__ void CoverageInstrumentation::on_sstore(uint32_t pc, const evm_word_t& slot,
                                                            const evm_word_t& old_value, const evm_word_t& new_value) {
    uint32_t slot_hash = hash_slot(slot);
    instance_->record_storage(pc, slot_hash, true);

#ifdef __CUDA_ARCH__
    atomicAdd(&global_map_->total_storage_ops, 1ULL);
#else
    global_map_->total_storage_ops++;
#endif

    // Track in detailed storage entries
    uint32_t entry_idx;
#ifdef __CUDA_ARCH__
    entry_idx = atomicAdd(&global_map_->num_storage_entries, 1);
#else
    entry_idx = global_map_->num_storage_entries++;
#endif

    if (entry_idx < STORAGE_COVERAGE_SIZE) {
        storage_coverage_entry_t* entry = &global_map_->storage_entries[entry_idx];
        entry->pc = pc;
        entry->slot_hash = slot_hash;
        entry->is_read = 0;
        entry->is_write = 1;
        // Check if value changed
        bool changed = false;
        for (int i = 0; i < 8; i++) {
            if (old_value._limbs[i] != new_value._limbs[i]) {
                changed = true;
                break;
            }
        }
        entry->value_changed = changed ? 1 : 0;
    }
}

__host__ __device__ void CoverageInstrumentation::on_call(uint32_t pc, uint8_t opcode, const evm_word_t& target,
                                                          const evm_word_t& value, bool success) {
    uint32_t target_hash = hash_slot(target);
    instance_->record_call(pc, target_hash, opcode, success);

#ifdef __CUDA_ARCH__
    atomicAdd(&global_map_->total_calls, 1ULL);
#else
    global_map_->total_calls++;
#endif

    // Track in detailed call entries
    uint32_t entry_idx;
#ifdef __CUDA_ARCH__
    entry_idx = atomicAdd(&global_map_->num_call_entries, 1);
#else
    entry_idx = global_map_->num_call_entries++;
#endif

    if (entry_idx < CALL_COVERAGE_SIZE) {
        call_coverage_entry_t* entry = &global_map_->call_entries[entry_idx];
        entry->pc = pc;
        entry->opcode = opcode;
        entry->callee_address_hash = target_hash;
        entry->success = success ? 1 : 0;
        // Check if precompile (addresses 0x01-0x09)
        bool is_precompile = true;
        for (int i = 1; i < 8; i++) {
            if (target._limbs[i] != 0) {
                is_precompile = false;
                break;
            }
        }
        if (is_precompile && target._limbs[0] >= 1 && target._limbs[0] <= 9) {
            entry->is_precompile = 1;
        } else {
            entry->is_precompile = 0;
        }
        // Check if value transferred
        bool has_value = false;
        for (int i = 0; i < 8; i++) {
            if (value._limbs[i] != 0) {
                has_value = true;
                break;
            }
        }
        entry->value_transferred = has_value ? 1 : 0;
    }
}

__host__ __device__ void CoverageInstrumentation::on_memory_access(uint32_t pc, uint32_t offset, uint32_t size, bool is_write) {
    // Memory coverage tracking - hash-based for efficiency
    uint32_t mem_hash = (pc << 16) ^ (offset >> 5) ^ (is_write ? 0x1 : 0x0);
    uint32_t index = mem_hash % PC_COVERAGE_SIZE;

#ifdef __CUDA_ARCH__
    atomicAdd((unsigned char*)&global_map_->pc_bitmap[index], 1);
#else
    if (global_map_->pc_bitmap[index] < 255) {
        global_map_->pc_bitmap[index]++;
    }
#endif
}

__host__ __device__ void CoverageInstrumentation::on_comparison(uint32_t pc, uint8_t opcode,
                                                                 const evm_word_t& a, const evm_word_t& b,
                                                                 const evm_word_t& result) {
    // Compute comparison distance for gradient guidance
    // This helps the fuzzer understand how close we are to flipping the comparison
    uint64_t distance = 0;

    // Simple distance: XOR of first 8 bytes
    uint64_t a_val = 0, b_val = 0;
    for (int i = 0; i < 2; i++) {
        a_val |= ((uint64_t)a._limbs[i] << (i * 32));
        b_val |= ((uint64_t)b._limbs[i] << (i * 32));
    }

    if (a_val > b_val) {
        distance = a_val - b_val;
    } else {
        distance = b_val - a_val;
    }

    // Record distance bucket for branch guidance
    uint8_t bucket = quantize_distance(distance);

    // Update coverage with distance info
    uint32_t comp_hash = (pc << 8) ^ opcode ^ bucket;
    uint32_t index = comp_hash % EDGE_COVERAGE_SIZE;

#ifdef __CUDA_ARCH__
    atomicAdd((unsigned char*)&global_map_->edge_bitmap[index], 1);
#else
    if (global_map_->edge_bitmap[index] < 255) {
        global_map_->edge_bitmap[index]++;
    }
#endif
}

__host__ __device__ void CoverageInstrumentation::on_return(uint32_t pc, bool success, uint32_t return_size) {
    // Track return/revert patterns
    uint32_t ret_hash = (pc << 1) ^ (success ? 1 : 0) ^ (return_size & 0xFFFF);
    uint32_t index = ret_hash % PC_COVERAGE_SIZE;

#ifdef __CUDA_ARCH__
    atomicAdd((unsigned char*)&global_map_->pc_bitmap[index], 1);
#else
    if (global_map_->pc_bitmap[index] < 255) {
        global_map_->pc_bitmap[index]++;
    }
#endif
}

__host__ __device__ void CoverageInstrumentation::finalize() {
    // Merge instance edge hashes to global bitmap
    for (uint32_t i = 0; i < instance_->edge_hash_idx && i < 256; i++) {
        uint32_t hash = instance_->edge_hashes[i];
        uint32_t index = hash % EDGE_COVERAGE_SIZE;
#ifdef __CUDA_ARCH__
        atomicAdd((unsigned char*)&global_map_->edge_bitmap[index], 1);
#else
        if (global_map_->edge_bitmap[index] < 255) {
            global_map_->edge_bitmap[index]++;
        }
#endif
    }

    // Update PC bitmap from instance
    // Note: In production, we'd track actual PCs, but for efficiency we use hashing
}

__host__ __device__ uint32_t CoverageInstrumentation::hash_edge(uint32_t from, uint32_t to) {
    // AFL-style edge hashing
    return ((from >> 1) ^ to) & (EDGE_COVERAGE_SIZE - 1);
}

__host__ __device__ uint32_t CoverageInstrumentation::hash_slot(const evm_word_t& slot) {
    // Simple hash of 256-bit storage slot
    uint32_t hash = 0;
    for (int i = 0; i < 8; i++) {
        hash ^= slot._limbs[i];
        hash = (hash << 5) | (hash >> 27);  // Rotate
    }
    return hash;
}

__host__ __device__ uint8_t CoverageInstrumentation::quantize_distance(uint64_t distance) {
    for (uint8_t i = 0; i < DISTANCE_BUCKETS; i++) {
        if (distance <= DISTANCE_THRESHOLDS[i]) {
            return i;
        }
    }
    return DISTANCE_BUCKETS - 1;
}

__host__ __device__ uint64_t CoverageInstrumentation::compute_branch_distance(const evm_word_t& condition) {
    // Distance to zero (for ISZERO-based branches)
    uint64_t distance = 0;
    for (int i = 0; i < 2; i++) {
        distance |= ((uint64_t)condition._limbs[i] << (i * 32));
    }
    return distance;
}

// ============================================================================
// Coverage Map Allocator Implementation
// ============================================================================

__host__ gpu_coverage_map_t* CoverageMapAllocator::allocate_global(uint32_t num_contracts) {
    gpu_coverage_map_t* map = nullptr;

    cudaMallocManaged(&map, sizeof(gpu_coverage_map_t));
    cudaMallocManaged(&map->pc_bitmap, PC_COVERAGE_SIZE * sizeof(coverage_counter_t));
    cudaMallocManaged(&map->edge_bitmap, EDGE_COVERAGE_SIZE * sizeof(coverage_counter_t));
    cudaMallocManaged(&map->opcode_counters, OPCODE_COVERAGE_SIZE * sizeof(coverage_counter_t));
    cudaMallocManaged(&map->branch_entries, BRANCH_COVERAGE_SIZE * sizeof(branch_coverage_entry_t));
    cudaMallocManaged(&map->storage_entries, STORAGE_COVERAGE_SIZE * sizeof(storage_coverage_entry_t));
    cudaMallocManaged(&map->call_entries, CALL_COVERAGE_SIZE * sizeof(call_coverage_entry_t));
    cudaMallocManaged(&map->opcode_stats, OPCODE_COVERAGE_SIZE * sizeof(opcode_stats_t));
    cudaMallocManaged(&map->contract_coverage, num_contracts * sizeof(contract_coverage_t));
    cudaMallocManaged(&map->virgin_bits, (COVERAGE_MAP_SIZE / 32) * sizeof(coverage_bitmap_t));

    // Initialize
    cudaMemset(map->pc_bitmap, 0, PC_COVERAGE_SIZE * sizeof(coverage_counter_t));
    cudaMemset(map->edge_bitmap, 0, EDGE_COVERAGE_SIZE * sizeof(coverage_counter_t));
    cudaMemset(map->opcode_counters, 0, OPCODE_COVERAGE_SIZE * sizeof(coverage_counter_t));
    cudaMemset(map->branch_entries, 0, BRANCH_COVERAGE_SIZE * sizeof(branch_coverage_entry_t));
    cudaMemset(map->storage_entries, 0, STORAGE_COVERAGE_SIZE * sizeof(storage_coverage_entry_t));
    cudaMemset(map->call_entries, 0, CALL_COVERAGE_SIZE * sizeof(call_coverage_entry_t));
    cudaMemset(map->opcode_stats, 0, OPCODE_COVERAGE_SIZE * sizeof(opcode_stats_t));
    cudaMemset(map->virgin_bits, 0xFF, (COVERAGE_MAP_SIZE / 32) * sizeof(coverage_bitmap_t));  // All virgin

    map->num_contracts = num_contracts;
    map->init();

    return map;
}

__host__ instance_coverage_t* CoverageMapAllocator::allocate_instances(uint32_t num_instances) {
    instance_coverage_t* instances = nullptr;
    cudaMallocManaged(&instances, num_instances * sizeof(instance_coverage_t));

    for (uint32_t i = 0; i < num_instances; i++) {
        instances[i].init();
    }

    return instances;
}

__host__ void CoverageMapAllocator::free_global(gpu_coverage_map_t* map) {
    if (map) {
        cudaFree(map->pc_bitmap);
        cudaFree(map->edge_bitmap);
        cudaFree(map->opcode_counters);
        cudaFree(map->branch_entries);
        cudaFree(map->storage_entries);
        cudaFree(map->call_entries);
        cudaFree(map->opcode_stats);
        cudaFree(map->contract_coverage);
        cudaFree(map->virgin_bits);
        cudaFree(map);
    }
}

__host__ void CoverageMapAllocator::free_instances(instance_coverage_t* instances) {
    if (instances) {
        cudaFree(instances);
    }
}

__host__ gpu_coverage_map_t* CoverageMapAllocator::allocate_pinned() {
    gpu_coverage_map_t* map = nullptr;
    cudaMallocHost(&map, sizeof(gpu_coverage_map_t));
    cudaMallocHost(&map->pc_bitmap, PC_COVERAGE_SIZE * sizeof(coverage_counter_t));
    cudaMallocHost(&map->edge_bitmap, EDGE_COVERAGE_SIZE * sizeof(coverage_counter_t));
    return map;
}

__host__ void CoverageMapAllocator::copy_to_host(gpu_coverage_map_t* host_map, const gpu_coverage_map_t* device_map) {
    cudaMemcpy(host_map, device_map, sizeof(gpu_coverage_map_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_map->pc_bitmap, device_map->pc_bitmap,
               PC_COVERAGE_SIZE * sizeof(coverage_counter_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_map->edge_bitmap, device_map->edge_bitmap,
               EDGE_COVERAGE_SIZE * sizeof(coverage_counter_t), cudaMemcpyDeviceToHost);
}

// ============================================================================
// Coverage Snapshot Implementation
// ============================================================================

__host__ void coverage_snapshot_t::serialize(void* buffer, size_t* size) {
    uint8_t* ptr = (uint8_t*)buffer;

    // Write header
    memcpy(ptr, &unique_pcs, sizeof(uint32_t)); ptr += sizeof(uint32_t);
    memcpy(ptr, &unique_edges, sizeof(uint32_t)); ptr += sizeof(uint32_t);
    memcpy(ptr, &unique_branches, sizeof(uint32_t)); ptr += sizeof(uint32_t);
    memcpy(ptr, &coverage_score, sizeof(float)); ptr += sizeof(float);
    memcpy(ptr, &timestamp, sizeof(uint64_t)); ptr += sizeof(uint64_t);

    // Write bitmap sizes
    memcpy(ptr, &pc_bitmap_size, sizeof(uint32_t)); ptr += sizeof(uint32_t);
    memcpy(ptr, &edge_bitmap_size, sizeof(uint32_t)); ptr += sizeof(uint32_t);

    // Write bitmaps
    memcpy(ptr, pc_bitmap_data, pc_bitmap_size); ptr += pc_bitmap_size;
    memcpy(ptr, edge_bitmap_data, edge_bitmap_size); ptr += edge_bitmap_size;

    *size = ptr - (uint8_t*)buffer;
}

__host__ coverage_snapshot_t coverage_snapshot_t::deserialize(const void* buffer, size_t size) {
    coverage_snapshot_t snapshot;
    const uint8_t* ptr = (const uint8_t*)buffer;

    memcpy(&snapshot.unique_pcs, ptr, sizeof(uint32_t)); ptr += sizeof(uint32_t);
    memcpy(&snapshot.unique_edges, ptr, sizeof(uint32_t)); ptr += sizeof(uint32_t);
    memcpy(&snapshot.unique_branches, ptr, sizeof(uint32_t)); ptr += sizeof(uint32_t);
    memcpy(&snapshot.coverage_score, ptr, sizeof(float)); ptr += sizeof(float);
    memcpy(&snapshot.timestamp, ptr, sizeof(uint64_t)); ptr += sizeof(uint64_t);

    memcpy(&snapshot.pc_bitmap_size, ptr, sizeof(uint32_t)); ptr += sizeof(uint32_t);
    memcpy(&snapshot.edge_bitmap_size, ptr, sizeof(uint32_t)); ptr += sizeof(uint32_t);

    snapshot.pc_bitmap_data = (uint8_t*)malloc(snapshot.pc_bitmap_size);
    snapshot.edge_bitmap_data = (uint8_t*)malloc(snapshot.edge_bitmap_size);

    memcpy(snapshot.pc_bitmap_data, ptr, snapshot.pc_bitmap_size); ptr += snapshot.pc_bitmap_size;
    memcpy(snapshot.edge_bitmap_data, ptr, snapshot.edge_bitmap_size);

    return snapshot;
}

__host__ bool coverage_snapshot_t::has_new_coverage(const coverage_snapshot_t& baseline) {
    return unique_pcs > baseline.unique_pcs ||
           unique_edges > baseline.unique_edges ||
           unique_branches > baseline.unique_branches;
}

__host__ float coverage_snapshot_t::novelty_score(const coverage_snapshot_t& baseline) {
    float pc_novelty = (unique_pcs - baseline.unique_pcs) / (float)(baseline.unique_pcs + 1);
    float edge_novelty = (unique_edges - baseline.unique_edges) / (float)(baseline.unique_edges + 1);
    float branch_novelty = (unique_branches - baseline.unique_branches) / (float)(baseline.unique_branches + 1);
    return (pc_novelty + edge_novelty * 2 + branch_novelty * 3) / 6.0f;
}

// ============================================================================
// Bitmap Operations
// ============================================================================

namespace bitmap_ops {

__host__ __device__ uint32_t hash_pc(uint32_t pc, uint32_t prev_pc) {
    return ((prev_pc >> 1) ^ pc) & (EDGE_COVERAGE_SIZE - 1);
}

__host__ __device__ void increment_counter(coverage_counter_t* bitmap, uint32_t index) {
#ifdef __CUDA_ARCH__
    atomicAdd((unsigned char*)&bitmap[index], 1);
#else
    if (bitmap[index] < 255) {
        bitmap[index]++;
    }
#endif
}

__host__ __device__ bool check_virgin(coverage_bitmap_t* virgin, uint32_t index) {
    uint32_t word_idx = index / 32;
    uint32_t bit_idx = index % 32;
    return (virgin[word_idx] & (1U << bit_idx)) != 0;
}

__host__ __device__ void mark_virgin(coverage_bitmap_t* virgin, uint32_t index) {
    uint32_t word_idx = index / 32;
    uint32_t bit_idx = index % 32;
#ifdef __CUDA_ARCH__
    atomicAnd(&virgin[word_idx], ~(1U << bit_idx));
#else
    virgin[word_idx] &= ~(1U << bit_idx);
#endif
}

__host__ uint32_t count_bits(const coverage_counter_t* bitmap, uint32_t size) {
    uint32_t count = 0;
    for (uint32_t i = 0; i < size; i++) {
        if (bitmap[i] > 0) count++;
    }
    return count;
}

__host__ uint32_t count_nonzero(const coverage_counter_t* bitmap, uint32_t size) {
    return count_bits(bitmap, size);
}

__host__ void merge_bitmaps(coverage_counter_t* dst, const coverage_counter_t* src, uint32_t size) {
    for (uint32_t i = 0; i < size; i++) {
        uint16_t sum = (uint16_t)dst[i] + (uint16_t)src[i];
        dst[i] = (sum > 255) ? 255 : (coverage_counter_t)sum;
    }
}

__host__ bool has_new_bits(const coverage_counter_t* current, const coverage_counter_t* virgin, uint32_t size) {
    for (uint32_t i = 0; i < size; i++) {
        if (current[i] > 0 && virgin[i] == 0xFF) {
            return true;
        }
    }
    return false;
}

}  // namespace bitmap_ops

// ============================================================================
// CUDA Kernels
// ============================================================================

__global__ void kernel_merge_coverage(
    gpu_coverage_map_t* global_map,
    instance_coverage_t* instances,
    uint32_t num_instances
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_instances) return;

    instance_coverage_t* inst = &instances[idx];

    // Merge edge hashes
    for (uint32_t i = 0; i < inst->edge_hash_idx && i < 256; i++) {
        uint32_t hash = inst->edge_hashes[i];
        uint32_t index = hash % EDGE_COVERAGE_SIZE;
        atomicAdd((unsigned char*)&global_map->edge_bitmap[index], 1);
    }

    // Update global stats
    atomicAdd(&global_map->total_instructions_executed, (unsigned long long)inst->pcs_hit);
    atomicAdd(&global_map->total_branches_executed, (unsigned long long)inst->branches_taken);
    atomicAdd(&global_map->total_storage_ops, (unsigned long long)inst->storage_ops);
    atomicAdd(&global_map->total_calls, (unsigned long long)inst->calls_made);
}

__global__ void kernel_compute_coverage_stats(
    gpu_coverage_map_t* map,
    uint32_t* unique_pcs,
    uint32_t* unique_edges,
    float* coverage_score
) {
    __shared__ uint32_t shared_pc_count;
    __shared__ uint32_t shared_edge_count;

    if (threadIdx.x == 0) {
        shared_pc_count = 0;
        shared_edge_count = 0;
    }
    __syncthreads();

    // Count PCs in parallel
    uint32_t local_pc_count = 0;
    for (uint32_t i = threadIdx.x; i < PC_COVERAGE_SIZE; i += blockDim.x) {
        if (map->pc_bitmap[i] > 0) local_pc_count++;
    }
    atomicAdd(&shared_pc_count, local_pc_count);

    // Count edges in parallel
    uint32_t local_edge_count = 0;
    for (uint32_t i = threadIdx.x; i < EDGE_COVERAGE_SIZE; i += blockDim.x) {
        if (map->edge_bitmap[i] > 0) local_edge_count++;
    }
    atomicAdd(&shared_edge_count, local_edge_count);

    __syncthreads();

    if (threadIdx.x == 0) {
        *unique_pcs = shared_pc_count;
        *unique_edges = shared_edge_count;
        *coverage_score = (float)shared_edge_count / (float)EDGE_COVERAGE_SIZE;
    }
}

__global__ void kernel_detect_new_coverage(
    gpu_coverage_map_t* current,
    gpu_coverage_map_t* baseline,
    uint32_t* new_coverage_flags,
    uint32_t num_instances
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= EDGE_COVERAGE_SIZE) return;

    // Check if this edge is new
    if (current->edge_bitmap[idx] > 0 && baseline->edge_bitmap[idx] == 0) {
        // Mark virgin bit
        uint32_t word_idx = idx / 32;
        uint32_t bit_idx = idx % 32;
        atomicAnd(&baseline->virgin_bits[word_idx], ~(1U << bit_idx));

        // Set flag
        new_coverage_flags[0] = 1;
    }
}

}  // namespace fuzzing
}  // namespace CuEVM
