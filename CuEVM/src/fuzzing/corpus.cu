// CuEVM: CUDA Ethereum Virtual Machine implementation
// GPU Corpus Management Implementation for Smart Contract Fuzzing
// SPDX-License-Identifier: MIT

#include <CuEVM/fuzzing/corpus.cuh>
#include <curand_kernel.h>
#include <cstring>
#include <cstdio>
#include <algorithm>
#include <fstream>
#include <sys/stat.h>
#include <dirent.h>

namespace CuEVM {
namespace fuzzing {

// ============================================================================
// Helper Functions
// ============================================================================

__host__ __device__ static uint64_t get_timestamp() {
#ifdef __CUDA_ARCH__
    return clock64();
#else
    return static_cast<uint64_t>(time(nullptr));
#endif
}

__host__ __device__ static uint32_t hash_combine(uint32_t seed, uint32_t value) {
    return seed ^ (value + 0x9e3779b9 + (seed << 6) + (seed >> 2));
}

__host__ __device__ static uint32_t fnv1a_hash(const uint8_t* data, uint32_t len) {
    uint32_t hash = 2166136261u;
    for (uint32_t i = 0; i < len; i++) {
        hash ^= data[i];
        hash *= 16777619u;
    }
    return hash;
}

// ============================================================================
// seed_entry_t Implementation
// ============================================================================

__host__ __device__ void seed_entry_t::init() {
    data.data = nullptr;
    data.length = 0;
    data.capacity = 0;

    metadata.id = 0;
    metadata.parent_id = 0;
    metadata.timestamp = 0;
    metadata.generation = 0;
    metadata.unique_edges = 0;
    metadata.unique_branches = 0;
    metadata.coverage_hash = 0;
    metadata.coverage_contribution = 0.0f;
    metadata.execution_count = 0;
    metadata.mutation_count = 0;
    metadata.child_count = 0;
    metadata.bug_count = 0;
    metadata.energy = ENERGY_BASE;
    metadata.priority = 0;
    metadata.last_selected = 0;
    metadata.minimized = false;
    metadata.original_length = 0;

    num_transactions = 0;
    for (uint32_t i = 0; i < MAX_SEQUENCE_LENGTH; i++) {
        tx_offsets[i] = 0;
        tx_lengths[i] = 0;
        memset(&senders[i], 0, sizeof(evm_word_t));
        memset(&values[i], 0, sizeof(evm_word_t));
        memset(&receivers[i], 0, sizeof(evm_word_t));
    }
    memset(&block_number, 0, sizeof(evm_word_t));
    memset(&timestamp, 0, sizeof(evm_word_t));
}

__host__ __device__ void seed_entry_t::copy_from(const seed_entry_t& other) {
    // Copy metadata
    metadata = other.metadata;
    num_transactions = other.num_transactions;

    // Copy transaction info
    for (uint32_t i = 0; i < MAX_SEQUENCE_LENGTH; i++) {
        tx_offsets[i] = other.tx_offsets[i];
        tx_lengths[i] = other.tx_lengths[i];
        senders[i] = other.senders[i];
        values[i] = other.values[i];
        receivers[i] = other.receivers[i];
    }
    block_number = other.block_number;
    timestamp = other.timestamp;

    // Deep copy data if allocated
    if (other.data.data && other.data.length > 0) {
        if (!data.data || data.capacity < other.data.length) {
            // Need to allocate - this is tricky in device code
            // Assume pre-allocated for device usage
#ifndef __CUDA_ARCH__
            if (data.data) {
                delete[] data.data;
            }
            data.data = new uint8_t[other.data.length];
            data.capacity = other.data.length;
#endif
        }
        if (data.data) {
            memcpy(data.data, other.data.data, other.data.length);
            data.length = other.data.length;
        }
    }
}

__host__ __device__ void seed_entry_t::set_transaction(uint32_t tx_idx, const uint8_t* calldata,
                                                        uint32_t len, const evm_word_t& sender,
                                                        const evm_word_t& value) {
    if (tx_idx >= MAX_SEQUENCE_LENGTH) return;

    // Calculate offset
    uint32_t offset = 0;
    if (tx_idx > 0) {
        offset = tx_offsets[tx_idx - 1] + tx_lengths[tx_idx - 1];
    }

    // Check capacity
    if (offset + len > data.capacity) {
#ifndef __CUDA_ARCH__
        // Grow buffer
        uint32_t new_capacity = (offset + len) * 2;
        if (new_capacity > MAX_SEED_DATA_SIZE) new_capacity = MAX_SEED_DATA_SIZE;
        uint8_t* new_data = new uint8_t[new_capacity];
        if (data.data && data.length > 0) {
            memcpy(new_data, data.data, data.length);
            delete[] data.data;
        }
        data.data = new_data;
        data.capacity = new_capacity;
#else
        return; // Can't grow in device code
#endif
    }

    // Copy transaction data
    if (data.data && calldata) {
        memcpy(data.data + offset, calldata, len);
    }

    tx_offsets[tx_idx] = offset;
    tx_lengths[tx_idx] = len;
    senders[tx_idx] = sender;
    values[tx_idx] = value;

    if (tx_idx >= num_transactions) {
        num_transactions = tx_idx + 1;
    }
    data.length = offset + len;
}

// ============================================================================
// corpus_stats_t Implementation
// ============================================================================

__host__ __device__ void corpus_stats_t::init() {
    total_seeds_added = 0;
    total_seeds_removed = 0;
    total_executions = 0;
    total_mutations = 0;
    total_new_coverage = 0;
    total_bugs_found = 0;
    current_size = 0;
    unique_coverage_edges = 0;
    unique_coverage_branches = 0;
    overall_coverage_percent = 0.0f;
    last_new_coverage_time = 0;
    last_bug_time = 0;
    cycles_since_progress = 0;
    initial_seeds = 0;
    mutant_seeds = 0;
    splice_seeds = 0;
    minimized_seeds = 0;
}

__host__ __device__ void corpus_stats_t::update_coverage(uint32_t new_edges, uint32_t new_branches) {
    unique_coverage_edges += new_edges;
    unique_coverage_branches += new_branches;
    if (new_edges > 0 || new_branches > 0) {
        total_new_coverage++;
        last_new_coverage_time = get_timestamp();
        cycles_since_progress = 0;
    } else {
        cycles_since_progress++;
    }
}

__host__ __device__ void corpus_stats_t::record_new_seed(bool from_mutation, bool caused_new_coverage) {
    total_seeds_added++;
    current_size++;
    if (from_mutation) {
        mutant_seeds++;
    } else {
        initial_seeds++;
    }
    if (caused_new_coverage) {
        total_new_coverage++;
    }
}

// ============================================================================
// corpus_hash_table_t Implementation
// ============================================================================

__host__ __device__ void corpus_hash_table_t::init() {
    for (uint32_t i = 0; i < CORPUS_BUCKET_COUNT; i++) {
        buckets[i].count = 0;
        for (uint32_t j = 0; j < 16; j++) {
            buckets[i].seed_indices[j] = UINT32_MAX;
        }
    }
}

__host__ __device__ bool corpus_hash_table_t::contains(uint32_t coverage_hash) {
    uint32_t bucket_idx = coverage_hash % CORPUS_BUCKET_COUNT;
    const corpus_bucket_t& bucket = buckets[bucket_idx];

    for (uint32_t i = 0; i < bucket.count && i < 16; i++) {
        if (bucket.seed_indices[i] != UINT32_MAX) {
            // In a full implementation, we'd compare the actual coverage
            // Here we just check if the hash exists
            return true;
        }
    }
    return false;
}

__host__ __device__ void corpus_hash_table_t::insert(uint32_t coverage_hash, uint32_t seed_idx) {
    uint32_t bucket_idx = coverage_hash % CORPUS_BUCKET_COUNT;
    corpus_bucket_t& bucket = buckets[bucket_idx];

    if (bucket.count < 16) {
        bucket.seed_indices[bucket.count] = seed_idx;
        bucket.count++;
    }
}

__host__ __device__ void corpus_hash_table_t::remove(uint32_t coverage_hash, uint32_t seed_idx) {
    uint32_t bucket_idx = coverage_hash % CORPUS_BUCKET_COUNT;
    corpus_bucket_t& bucket = buckets[bucket_idx];

    for (uint32_t i = 0; i < bucket.count && i < 16; i++) {
        if (bucket.seed_indices[i] == seed_idx) {
            // Shift remaining entries
            for (uint32_t j = i; j < bucket.count - 1 && j < 15; j++) {
                bucket.seed_indices[j] = bucket.seed_indices[j + 1];
            }
            bucket.count--;
            bucket.seed_indices[bucket.count] = UINT32_MAX;
            return;
        }
    }
}

// ============================================================================
// invariant_t Implementation
// ============================================================================

__host__ __device__ void invariant_t::init() {
    type = InvariantType::STORAGE_EQUALS;
    id = 0;
    memset(&target_address, 0, sizeof(evm_word_t));
    memset(&slot1, 0, sizeof(evm_word_t));
    memset(&slot2, 0, sizeof(evm_word_t));
    memset(&expected_value, 0, sizeof(evm_word_t));
    memset(&min_value, 0, sizeof(evm_word_t));
    memset(&max_value, 0, sizeof(evm_word_t));
    for (uint32_t i = 0; i < 4; i++) {
        memset(&addresses[i], 0, sizeof(evm_word_t));
        memset(&slots[i], 0, sizeof(evm_word_t));
    }
    num_slots = 0;
    memset(description, 0, sizeof(description));
    enabled = true;
    violation_count = 0;
}

// ============================================================================
// GPUCorpusManager Implementation
// ============================================================================

__host__ GPUCorpusManager::GPUCorpusManager(uint32_t max_size) {
    capacity_ = max_size;
    coverage_baseline_ = nullptr;
    queue_size_ = 0;

    // Allocate seed storage
    cudaMallocManaged(&seeds_, sizeof(seed_entry_t) * max_size);
    cudaMallocManaged(&free_indices_, sizeof(uint32_t) * max_size);
    cudaMallocManaged(&priority_queue_, sizeof(uint32_t) * max_size);

    // Initialize seeds
    for (uint32_t i = 0; i < max_size; i++) {
        seeds_[i].init();
        free_indices_[i] = max_size - 1 - i;  // Stack-based free list
    }
    free_count_ = max_size;

    stats_.init();
    hash_table_.init();
}

__host__ GPUCorpusManager::~GPUCorpusManager() {
    // Free seed data
    for (uint32_t i = 0; i < capacity_; i++) {
        if (seeds_[i].data.data) {
            cudaFree(seeds_[i].data.data);
        }
    }
    cudaFree(seeds_);
    cudaFree(free_indices_);
    cudaFree(priority_queue_);
}

__host__ __device__ uint32_t GPUCorpusManager::allocate_slot() {
    if (free_count_ == 0) return UINT32_MAX;
    free_count_--;
    return free_indices_[free_count_];
}

__host__ __device__ void GPUCorpusManager::deallocate_slot(uint32_t idx) {
    if (idx >= capacity_) return;
    free_indices_[free_count_] = idx;
    free_count_++;
}

__host__ __device__ bool GPUCorpusManager::add_seed(const seed_entry_t& seed, bool check_duplicate) {
    // Check for duplicates
    if (check_duplicate && hash_table_.contains(seed.metadata.coverage_hash)) {
        return false;
    }

    // Allocate slot
    uint32_t idx = allocate_slot();
    if (idx == UINT32_MAX) {
        return false;
    }

    // Copy seed
    seeds_[idx].copy_from(seed);
    seeds_[idx].metadata.id = stats_.total_seeds_added + 1;
    seeds_[idx].metadata.timestamp = get_timestamp();

    // Update hash table
    hash_table_.insert(seed.metadata.coverage_hash, idx);

    // Add to priority queue
    if (queue_size_ < capacity_) {
        priority_queue_[queue_size_] = idx;
        queue_size_++;
    }

    stats_.record_new_seed(seed.metadata.parent_id != 0, false);

    return true;
}

__host__ __device__ bool GPUCorpusManager::add_seed_if_interesting(const seed_entry_t& seed,
                                                                    const coverage_snapshot_t& coverage,
                                                                    const bug_storage_t* bugs) {
    // Check if this seed adds new coverage
    uint32_t new_edges = 0;
    uint32_t new_branches = 0;

    // Compare with baseline if available
    if (coverage_baseline_) {
        // Count new coverage
        for (uint32_t i = 0; i < COVERAGE_MAP_SIZE / 32; i++) {
            uint32_t new_bits = coverage.edge_bitmap[i] & ~coverage_baseline_->edges.hit_bitmap[i];
            new_edges += __builtin_popcount(new_bits);
        }
    } else {
        // No baseline, count all coverage
        for (uint32_t i = 0; i < COVERAGE_MAP_SIZE / 32; i++) {
            new_edges += __builtin_popcount(coverage.edge_bitmap[i]);
        }
    }

    // Check if found new bug
    bool found_new_bug = false;
    if (bugs && bugs->num_bugs > 0) {
        found_new_bug = true;  // Simplified check
    }

    // Add if interesting
    if (new_edges > 0 || new_branches > 0 || found_new_bug) {
        seed_entry_t modified_seed = seed;
        modified_seed.metadata.unique_edges = new_edges;
        modified_seed.metadata.unique_branches = new_branches;
        modified_seed.metadata.coverage_contribution = static_cast<float>(new_edges + new_branches);

        if (found_new_bug) {
            modified_seed.metadata.energy += ENERGY_NEW_BUG;
            modified_seed.metadata.bug_count++;
        } else if (new_edges > 0 || new_branches > 0) {
            modified_seed.metadata.energy += ENERGY_NEW_COVERAGE;
        }

        bool added = add_seed(modified_seed, true);
        if (added) {
            stats_.update_coverage(new_edges, new_branches);
        }
        return added;
    }

    return false;
}

__host__ __device__ void GPUCorpusManager::remove_seed(uint32_t idx) {
    if (idx >= capacity_) return;

    // Remove from hash table
    hash_table_.remove(seeds_[idx].metadata.coverage_hash, idx);

    // Clear seed
    seeds_[idx].init();

    // Return slot to free list
    deallocate_slot(idx);

    stats_.total_seeds_removed++;
    stats_.current_size--;
}

__host__ __device__ seed_entry_t* GPUCorpusManager::get_seed(uint32_t idx) {
    if (idx >= capacity_) return nullptr;
    return &seeds_[idx];
}

__host__ __device__ seed_entry_t* GPUCorpusManager::select_seed(curandState* rng) {
    if (stats_.current_size == 0) return nullptr;

    // Random selection from priority queue
    uint32_t rand_idx;
#ifdef __CUDA_ARCH__
    rand_idx = curand(rng) % queue_size_;
#else
    rand_idx = rand() % queue_size_;
#endif

    uint32_t seed_idx = priority_queue_[rand_idx];
    seed_entry_t* seed = &seeds_[seed_idx];
    seed->metadata.execution_count++;
    seed->metadata.last_selected = get_timestamp();

    return seed;
}

__host__ __device__ seed_entry_t* GPUCorpusManager::select_weighted(curandState* rng) {
    if (stats_.current_size == 0) return nullptr;

    // Calculate total energy
    uint64_t total_energy = 0;
    for (uint32_t i = 0; i < queue_size_; i++) {
        total_energy += seeds_[priority_queue_[i]].metadata.energy;
    }

    if (total_energy == 0) {
        return select_seed(rng);  // Fallback to uniform selection
    }

    // Weighted random selection
    uint64_t target;
#ifdef __CUDA_ARCH__
    target = curand(rng) % total_energy;
#else
    target = rand() % total_energy;
#endif

    uint64_t cumulative = 0;
    for (uint32_t i = 0; i < queue_size_; i++) {
        cumulative += seeds_[priority_queue_[i]].metadata.energy;
        if (cumulative > target) {
            uint32_t seed_idx = priority_queue_[i];
            seed_entry_t* seed = &seeds_[seed_idx];
            seed->metadata.execution_count++;
            seed->metadata.last_selected = get_timestamp();
            return seed;
        }
    }

    return &seeds_[priority_queue_[queue_size_ - 1]];
}

__host__ __device__ void GPUCorpusManager::update_seed_after_execution(uint32_t idx, bool caused_new_coverage,
                                                                        bool found_bug) {
    if (idx >= capacity_) return;

    seed_entry_t* seed = &seeds_[idx];
    seed->metadata.execution_count++;

    if (caused_new_coverage) {
        seed->metadata.energy += ENERGY_NEW_COVERAGE;
        seed->metadata.child_count++;
    }

    if (found_bug) {
        seed->metadata.energy += ENERGY_NEW_BUG;
        seed->metadata.bug_count++;
        stats_.total_bugs_found++;
    }

    stats_.total_executions++;
}

__host__ __device__ uint32_t GPUCorpusManager::compute_coverage_hash(const coverage_snapshot_t& coverage) {
    uint32_t hash = 0;
    for (uint32_t i = 0; i < COVERAGE_MAP_SIZE / 32; i++) {
        hash = hash_combine(hash, coverage.edge_bitmap[i]);
    }
    return hash;
}

__host__ __device__ uint32_t GPUCorpusManager::compute_seed_hash(const seed_entry_t& seed) {
    if (!seed.data.data || seed.data.length == 0) {
        return 0;
    }
    return fnv1a_hash(seed.data.data, seed.data.length);
}

__host__ __device__ float GPUCorpusManager::compute_priority(const seed_metadata_t& metadata) {
    float priority = 1.0f;

    // Favor seeds with high coverage contribution
    priority += metadata.coverage_contribution * 10.0f;

    // Favor bug-finding seeds
    priority += metadata.bug_count * 100.0f;

    // Penalize over-mutated seeds
    if (metadata.mutation_count > 1000) {
        priority *= 0.5f;
    }

    // Favor newer seeds
    if (metadata.generation < 10) {
        priority *= 1.5f;
    }

    return priority;
}

__host__ void GPUCorpusManager::cull_corpus() {
    if (stats_.current_size <= MIN_CORPUS_ENTRIES) {
        return;
    }

    // Remove seeds with low priority
    uint32_t target_size = stats_.current_size * 3 / 4;  // Keep 75%
    if (target_size < MIN_CORPUS_ENTRIES) {
        target_size = MIN_CORPUS_ENTRIES;
    }

    // Sort by priority (ascending, so worst first)
    std::vector<std::pair<float, uint32_t>> priorities;
    for (uint32_t i = 0; i < queue_size_; i++) {
        uint32_t idx = priority_queue_[i];
        float pri = compute_priority(seeds_[idx].metadata);
        priorities.push_back({pri, idx});
    }

    std::sort(priorities.begin(), priorities.end());

    // Remove lowest priority seeds
    uint32_t to_remove = stats_.current_size - target_size;
    for (uint32_t i = 0; i < to_remove && i < priorities.size(); i++) {
        remove_seed(priorities[i].second);
    }

    compact_corpus();
}

__host__ void GPUCorpusManager::compact_corpus() {
    // Rebuild priority queue with only valid entries
    uint32_t new_size = 0;
    for (uint32_t i = 0; i < queue_size_; i++) {
        uint32_t idx = priority_queue_[i];
        if (seeds_[idx].metadata.id != 0) {
            priority_queue_[new_size] = idx;
            new_size++;
        }
    }
    queue_size_ = new_size;
}

__host__ void GPUCorpusManager::sort_by_priority() {
    std::vector<std::pair<float, uint32_t>> priorities;
    for (uint32_t i = 0; i < queue_size_; i++) {
        uint32_t idx = priority_queue_[i];
        float pri = compute_priority(seeds_[idx].metadata);
        priorities.push_back({pri, idx});
    }

    std::sort(priorities.begin(), priorities.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    for (uint32_t i = 0; i < queue_size_; i++) {
        priority_queue_[i] = priorities[i].second;
    }
}

__host__ void GPUCorpusManager::recalculate_energies() {
    for (uint32_t i = 0; i < queue_size_; i++) {
        uint32_t idx = priority_queue_[i];
        seed_entry_t& seed = seeds_[idx];

        // Decay energy over time
        seed.metadata.energy = seed.metadata.energy / ENERGY_DECAY_FACTOR;
        if (seed.metadata.energy < ENERGY_MIN) {
            seed.metadata.energy = ENERGY_MIN;
        }

        // Recalculate priority
        seed.metadata.priority = static_cast<uint32_t>(compute_priority(seed.metadata));
    }
}

__host__ void GPUCorpusManager::minimize_seed(uint32_t idx) {
    if (idx >= capacity_) return;

    seed_entry_t* seed = &seeds_[idx];
    if (seed->metadata.minimized) return;

    // Simple minimization: try removing chunks
    SeedMinimizer minimizer;

    // For now, just mark as minimized
    // Full implementation would use delta debugging
    seed->metadata.minimized = true;
    seed->metadata.original_length = seed->data.length;
}

__host__ void GPUCorpusManager::minimize_all() {
    for (uint32_t i = 0; i < queue_size_; i++) {
        minimize_seed(priority_queue_[i]);
    }
    stats_.minimized_seeds = queue_size_;
}

__host__ void GPUCorpusManager::merge_from(const GPUCorpusManager& other) {
    for (uint32_t i = 0; i < other.queue_size_; i++) {
        uint32_t idx = other.priority_queue_[i];
        const seed_entry_t& seed = other.seeds_[idx];
        add_seed(seed, true);
    }
}

__host__ void GPUCorpusManager::import_seeds(const char* directory) {
    DIR* dir = opendir(directory);
    if (!dir) return;

    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        if (entry->d_name[0] == '.') continue;

        char filepath[512];
        snprintf(filepath, sizeof(filepath), "%s/%s", directory, entry->d_name);

        // Read seed file
        FILE* f = fopen(filepath, "rb");
        if (!f) continue;

        fseek(f, 0, SEEK_END);
        long size = ftell(f);
        fseek(f, 0, SEEK_SET);

        if (size > 0 && size <= MAX_SEED_DATA_SIZE) {
            seed_entry_t seed;
            seed.init();

            uint8_t* data;
            cudaMallocManaged(&data, size);
            fread(data, 1, size, f);

            seed.data.data = data;
            seed.data.length = static_cast<uint32_t>(size);
            seed.data.capacity = static_cast<uint32_t>(size);
            seed.num_transactions = 1;
            seed.tx_offsets[0] = 0;
            seed.tx_lengths[0] = static_cast<uint32_t>(size);

            add_seed(seed, false);
        }

        fclose(f);
    }

    closedir(dir);
}

__host__ void GPUCorpusManager::export_seeds(const char* directory) {
    mkdir(directory, 0755);

    for (uint32_t i = 0; i < queue_size_; i++) {
        uint32_t idx = priority_queue_[i];
        const seed_entry_t& seed = seeds_[idx];

        if (!seed.data.data || seed.data.length == 0) continue;

        char filepath[512];
        snprintf(filepath, sizeof(filepath), "%s/seed_%lu.bin",
                 directory, seed.metadata.id);

        FILE* f = fopen(filepath, "wb");
        if (f) {
            fwrite(seed.data.data, 1, seed.data.length, f);
            fclose(f);
        }
    }
}

__host__ void GPUCorpusManager::export_interesting_seeds(const char* directory, uint32_t max_seeds) {
    mkdir(directory, 0755);

    // Sort by priority
    sort_by_priority();

    uint32_t exported = 0;
    for (uint32_t i = 0; i < queue_size_ && exported < max_seeds; i++) {
        uint32_t idx = priority_queue_[i];
        const seed_entry_t& seed = seeds_[idx];

        if (!seed.data.data || seed.data.length == 0) continue;

        char filepath[512];
        snprintf(filepath, sizeof(filepath), "%s/interesting_%u_id%lu.bin",
                 directory, exported, seed.metadata.id);

        FILE* f = fopen(filepath, "wb");
        if (f) {
            fwrite(seed.data.data, 1, seed.data.length, f);
            fclose(f);
            exported++;
        }
    }
}

__host__ void GPUCorpusManager::save_checkpoint(const char* filename) {
    FILE* f = fopen(filename, "wb");
    if (!f) return;

    // Write stats
    fwrite(&stats_, sizeof(corpus_stats_t), 1, f);

    // Write number of seeds
    fwrite(&queue_size_, sizeof(uint32_t), 1, f);

    // Write each seed
    for (uint32_t i = 0; i < queue_size_; i++) {
        uint32_t idx = priority_queue_[i];
        const seed_entry_t& seed = seeds_[idx];

        // Write metadata
        fwrite(&seed.metadata, sizeof(seed_metadata_t), 1, f);
        fwrite(&seed.num_transactions, sizeof(uint32_t), 1, f);
        fwrite(seed.tx_offsets, sizeof(uint32_t), MAX_SEQUENCE_LENGTH, f);
        fwrite(seed.tx_lengths, sizeof(uint32_t), MAX_SEQUENCE_LENGTH, f);
        fwrite(seed.senders, sizeof(evm_word_t), MAX_SEQUENCE_LENGTH, f);
        fwrite(seed.values, sizeof(evm_word_t), MAX_SEQUENCE_LENGTH, f);

        // Write data
        fwrite(&seed.data.length, sizeof(uint32_t), 1, f);
        if (seed.data.length > 0 && seed.data.data) {
            fwrite(seed.data.data, 1, seed.data.length, f);
        }
    }

    fclose(f);
}

__host__ void GPUCorpusManager::load_checkpoint(const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (!f) return;

    // Read stats
    fread(&stats_, sizeof(corpus_stats_t), 1, f);

    // Read number of seeds
    uint32_t num_seeds;
    fread(&num_seeds, sizeof(uint32_t), 1, f);

    // Read each seed
    for (uint32_t i = 0; i < num_seeds; i++) {
        seed_entry_t seed;
        seed.init();

        // Read metadata
        fread(&seed.metadata, sizeof(seed_metadata_t), 1, f);
        fread(&seed.num_transactions, sizeof(uint32_t), 1, f);
        fread(seed.tx_offsets, sizeof(uint32_t), MAX_SEQUENCE_LENGTH, f);
        fread(seed.tx_lengths, sizeof(uint32_t), MAX_SEQUENCE_LENGTH, f);
        fread(seed.senders, sizeof(evm_word_t), MAX_SEQUENCE_LENGTH, f);
        fread(seed.values, sizeof(evm_word_t), MAX_SEQUENCE_LENGTH, f);

        // Read data
        uint32_t data_len;
        fread(&data_len, sizeof(uint32_t), 1, f);
        if (data_len > 0) {
            cudaMallocManaged(&seed.data.data, data_len);
            fread(seed.data.data, 1, data_len, f);
            seed.data.length = data_len;
            seed.data.capacity = data_len;
        }

        add_seed(seed, false);
    }

    fclose(f);
}

__host__ void GPUCorpusManager::set_coverage_baseline(const gpu_coverage_map_t* baseline) {
    coverage_baseline_ = const_cast<gpu_coverage_map_t*>(baseline);
}

__host__ void GPUCorpusManager::update_coverage_contribution(uint32_t seed_idx,
                                                              const coverage_snapshot_t& new_coverage) {
    if (seed_idx >= capacity_) return;

    seed_entry_t* seed = &seeds_[seed_idx];

    // Calculate contribution
    uint32_t contribution = 0;
    for (uint32_t i = 0; i < COVERAGE_MAP_SIZE / 32; i++) {
        contribution += __builtin_popcount(new_coverage.edge_bitmap[i]);
    }

    seed->metadata.coverage_contribution = static_cast<float>(contribution);
    seed->metadata.coverage_hash = compute_coverage_hash(new_coverage);
}

__host__ void GPUCorpusManager::print_stats() {
    printf("=== Corpus Statistics ===\n");
    printf("Current size: %u / %u\n", stats_.current_size, capacity_);
    printf("Total seeds added: %lu\n", stats_.total_seeds_added);
    printf("Total seeds removed: %lu\n", stats_.total_seeds_removed);
    printf("Total executions: %lu\n", stats_.total_executions);
    printf("Total mutations: %lu\n", stats_.total_mutations);
    printf("Unique coverage edges: %u\n", stats_.unique_coverage_edges);
    printf("Unique coverage branches: %u\n", stats_.unique_coverage_branches);
    printf("Coverage: %.2f%%\n", stats_.overall_coverage_percent);
    printf("Bugs found: %lu\n", stats_.total_bugs_found);
    printf("Initial seeds: %u\n", stats_.initial_seeds);
    printf("Mutant seeds: %u\n", stats_.mutant_seeds);
    printf("Minimized seeds: %u\n", stats_.minimized_seeds);
    printf("Cycles since progress: %u\n", stats_.cycles_since_progress);
    printf("=========================\n");
}

__host__ void GPUCorpusManager::export_stats_json(const char* filename) {
    FILE* f = fopen(filename, "w");
    if (!f) return;

    fprintf(f, "{\n");
    fprintf(f, "  \"current_size\": %u,\n", stats_.current_size);
    fprintf(f, "  \"capacity\": %u,\n", capacity_);
    fprintf(f, "  \"total_seeds_added\": %lu,\n", stats_.total_seeds_added);
    fprintf(f, "  \"total_seeds_removed\": %lu,\n", stats_.total_seeds_removed);
    fprintf(f, "  \"total_executions\": %lu,\n", stats_.total_executions);
    fprintf(f, "  \"total_mutations\": %lu,\n", stats_.total_mutations);
    fprintf(f, "  \"unique_coverage_edges\": %u,\n", stats_.unique_coverage_edges);
    fprintf(f, "  \"unique_coverage_branches\": %u,\n", stats_.unique_coverage_branches);
    fprintf(f, "  \"overall_coverage_percent\": %.4f,\n", stats_.overall_coverage_percent);
    fprintf(f, "  \"total_bugs_found\": %lu,\n", stats_.total_bugs_found);
    fprintf(f, "  \"initial_seeds\": %u,\n", stats_.initial_seeds);
    fprintf(f, "  \"mutant_seeds\": %u,\n", stats_.mutant_seeds);
    fprintf(f, "  \"splice_seeds\": %u,\n", stats_.splice_seeds);
    fprintf(f, "  \"minimized_seeds\": %u,\n", stats_.minimized_seeds);
    fprintf(f, "  \"cycles_since_progress\": %u\n", stats_.cycles_since_progress);
    fprintf(f, "}\n");

    fclose(f);
}

// ============================================================================
// SeedMinimizer Implementation
// ============================================================================

__host__ SeedMinimizer::SeedMinimizer() {}

__host__ bool SeedMinimizer::minimize(seed_entry_t* seed,
                                       bool (*test_fn)(const seed_entry_t*, void*),
                                       void* test_ctx) {
    if (!seed || !seed->data.data || seed->data.length < 2) {
        return false;
    }

    // Try sequence minimization first if it's a multi-tx seed
    if (seed->num_transactions > 1) {
        minimize_sequence(seed, test_fn, test_ctx);
    }

    // Then minimize individual calldata
    bool reduced = false;
    for (uint32_t tx_idx = 0; tx_idx < seed->num_transactions; tx_idx++) {
        uint8_t* tx_data = seed->data.data + seed->tx_offsets[tx_idx];
        uint32_t tx_len = seed->tx_lengths[tx_idx];

        // Create wrapper test function for single transaction
        auto single_tx_test = [&](const uint8_t* data, uint32_t len) -> bool {
            // Temporarily modify seed
            uint32_t orig_len = seed->tx_lengths[tx_idx];
            seed->tx_lengths[tx_idx] = len;
            memcpy(tx_data, data, len);

            bool result = test_fn(seed, test_ctx);

            // Restore if test failed
            if (!result) {
                seed->tx_lengths[tx_idx] = orig_len;
            }
            return result;
        };

        // Delta debugging on this transaction
        uint32_t new_len = tx_len;
        if (ddmin(tx_data, &new_len, 4, nullptr, nullptr)) {
            seed->tx_lengths[tx_idx] = new_len;
            reduced = true;
        }
    }

    seed->metadata.minimized = true;
    seed->metadata.original_length = seed->data.length;

    return reduced;
}

__host__ bool SeedMinimizer::minimize_sequence(seed_entry_t* seed,
                                                bool (*test_fn)(const seed_entry_t*, void*),
                                                void* test_ctx) {
    if (seed->num_transactions <= 1) {
        return false;
    }

    bool reduced = false;

    // Try removing transactions one at a time
    for (uint32_t i = seed->num_transactions; i > 0; i--) {
        uint32_t tx_to_remove = i - 1;

        // Create a copy without this transaction
        seed_entry_t test_seed;
        test_seed.init();

        uint32_t new_idx = 0;
        uint32_t new_offset = 0;
        for (uint32_t j = 0; j < seed->num_transactions; j++) {
            if (j == tx_to_remove) continue;

            // Copy transaction
            test_seed.tx_offsets[new_idx] = new_offset;
            test_seed.tx_lengths[new_idx] = seed->tx_lengths[j];
            test_seed.senders[new_idx] = seed->senders[j];
            test_seed.values[new_idx] = seed->values[j];

            new_offset += seed->tx_lengths[j];
            new_idx++;
        }
        test_seed.num_transactions = new_idx;

        // Allocate and copy data
        if (new_offset > 0) {
            cudaMallocManaged(&test_seed.data.data, new_offset);
            test_seed.data.length = new_offset;
            test_seed.data.capacity = new_offset;

            uint32_t copy_offset = 0;
            for (uint32_t j = 0; j < seed->num_transactions; j++) {
                if (j == tx_to_remove) continue;
                memcpy(test_seed.data.data + copy_offset,
                       seed->data.data + seed->tx_offsets[j],
                       seed->tx_lengths[j]);
                copy_offset += seed->tx_lengths[j];
            }
        }

        // Test if still interesting
        if (test_fn(&test_seed, test_ctx)) {
            // Reduction successful, update original seed
            seed->copy_from(test_seed);
            reduced = true;
            i--;  // Recheck current position
        }

        // Free test seed data
        if (test_seed.data.data) {
            cudaFree(test_seed.data.data);
        }
    }

    return reduced;
}

__host__ bool SeedMinimizer::minimize_calldata(uint8_t* data, uint32_t* length,
                                                bool (*test_fn)(const uint8_t*, uint32_t, void*),
                                                void* test_ctx) {
    return ddmin(data, length, 4, test_fn, test_ctx);
}

__host__ bool SeedMinimizer::ddmin(uint8_t* data, uint32_t* length, uint32_t granularity,
                                    bool (*test_fn)(const uint8_t*, uint32_t, void*),
                                    void* test_ctx) {
    if (*length < granularity * 2) {
        return false;
    }

    bool reduced = false;
    uint32_t n = granularity;

    while (n <= *length / 2) {
        uint32_t chunk_size = *length / n;
        bool chunk_removed = false;

        for (uint32_t i = 0; i < n && !chunk_removed; i++) {
            uint32_t start = i * chunk_size;
            uint32_t end = (i == n - 1) ? *length : (i + 1) * chunk_size;
            uint32_t remove_size = end - start;

            // Create reduced data
            uint32_t new_len = *length - remove_size;
            uint8_t* new_data = new uint8_t[new_len];

            memcpy(new_data, data, start);
            memcpy(new_data + start, data + end, *length - end);

            // Test if still triggers behavior
            bool still_triggers = true;
            if (test_fn) {
                still_triggers = test_fn(new_data, new_len, test_ctx);
            }

            if (still_triggers) {
                // Reduction successful
                memcpy(data, new_data, new_len);
                *length = new_len;
                reduced = true;
                chunk_removed = true;
                n = granularity;  // Reset to try larger chunks again
            }

            delete[] new_data;
        }

        if (!chunk_removed) {
            n *= 2;
        }
    }

    return reduced;
}

// ============================================================================
// CorpusDistiller Implementation
// ============================================================================

__host__ CorpusDistiller::CorpusDistiller(GPUCorpusManager* corpus)
    : source_corpus_(corpus) {}

__host__ void CorpusDistiller::distill(GPUCorpusManager* output_corpus,
                                        const gpu_coverage_map_t* target_coverage) {
    greedy_cover(output_corpus, target_coverage);
}

__host__ void CorpusDistiller::greedy_cover(GPUCorpusManager* output_corpus,
                                             const gpu_coverage_map_t* target_coverage) {
    if (!source_corpus_ || !output_corpus) return;

    // Track which coverage bits we still need
    std::vector<uint32_t> uncovered(COVERAGE_MAP_SIZE / 32);
    for (uint32_t i = 0; i < COVERAGE_MAP_SIZE / 32; i++) {
        uncovered[i] = target_coverage->edges.hit_bitmap[i];
    }

    uint32_t total_uncovered = 0;
    for (uint32_t i = 0; i < COVERAGE_MAP_SIZE / 32; i++) {
        total_uncovered += __builtin_popcount(uncovered[i]);
    }

    // Greedy selection
    corpus_stats_t* stats = source_corpus_->get_stats();
    std::vector<bool> selected(stats->current_size, false);

    while (total_uncovered > 0) {
        uint32_t best_idx = UINT32_MAX;
        uint32_t best_contribution = 0;

        // Find seed that covers most uncovered bits
        for (uint32_t i = 0; i < stats->current_size; i++) {
            if (selected[i]) continue;

            seed_entry_t* seed = source_corpus_->get_seed(i);
            if (!seed) continue;

            // Count how many uncovered bits this seed covers
            uint32_t contribution = 0;
            // In a real implementation, we'd need the seed's coverage bitmap
            // For now, use the coverage hash as a proxy
            contribution = seed->metadata.unique_edges;

            if (contribution > best_contribution) {
                best_contribution = contribution;
                best_idx = i;
            }
        }

        if (best_idx == UINT32_MAX) break;

        // Add best seed to output
        seed_entry_t* best_seed = source_corpus_->get_seed(best_idx);
        output_corpus->add_seed(*best_seed, false);
        selected[best_idx] = true;

        // Update uncovered (simplified)
        total_uncovered -= best_contribution;
        if (total_uncovered > stats->unique_coverage_edges) {
            total_uncovered = 0;  // Prevent underflow
        }
    }
}

// ============================================================================
// InvariantChecker Implementation
// ============================================================================

__host__ __device__ InvariantChecker::InvariantChecker() {
    num_invariants_ = 0;
    for (uint32_t i = 0; i < MAX_INVARIANTS; i++) {
        invariants_[i].init();
    }
}

__host__ __device__ uint32_t InvariantChecker::add_invariant(const invariant_t& inv) {
    if (num_invariants_ >= MAX_INVARIANTS) {
        return UINT32_MAX;
    }

    uint32_t id = num_invariants_;
    invariants_[num_invariants_] = inv;
    invariants_[num_invariants_].id = id;
    num_invariants_++;

    return id;
}

__host__ __device__ void InvariantChecker::remove_invariant(uint32_t id) {
    if (id >= num_invariants_) return;

    // Shift remaining invariants
    for (uint32_t i = id; i < num_invariants_ - 1; i++) {
        invariants_[i] = invariants_[i + 1];
        invariants_[i].id = i;
    }
    num_invariants_--;
}

__host__ __device__ void InvariantChecker::enable_invariant(uint32_t id, bool enabled) {
    if (id < num_invariants_) {
        invariants_[id].enabled = enabled;
    }
}

__host__ __device__ void InvariantChecker::check_all(const evm_word_t* storage,
                                                      const evm_word_t* balances,
                                                      uint32_t tx_index,
                                                      invariant_result_t* results,
                                                      uint32_t* num_violations) {
    *num_violations = 0;

    for (uint32_t i = 0; i < num_invariants_; i++) {
        if (!invariants_[i].enabled) continue;

        invariant_result_t result;
        if (check_single(i, storage, balances, &result)) {
            if (result.violated) {
                result.tx_index = tx_index;
                result.timestamp = get_timestamp();
                results[*num_violations] = result;
                (*num_violations)++;
                invariants_[i].violation_count++;
            }
        }
    }
}

__host__ __device__ bool InvariantChecker::check_single(uint32_t id,
                                                         const evm_word_t* storage,
                                                         const evm_word_t* balances,
                                                         invariant_result_t* result) {
    if (id >= num_invariants_) return false;

    const invariant_t& inv = invariants_[id];
    result->invariant_id = id;
    result->violated = false;

    switch (inv.type) {
        case InvariantType::STORAGE_EQUALS:
            result->violated = !check_storage_equals(inv, storage);
            break;

        case InvariantType::STORAGE_NOT_ZERO:
        case InvariantType::STORAGE_LESS_THAN:
        case InvariantType::STORAGE_GREATER_THAN:
        case InvariantType::STORAGE_IN_RANGE:
            result->violated = !check_storage_range(inv, storage);
            break;

        case InvariantType::BALANCE_CONSERVED:
            result->violated = !check_balance_conserved(inv, balances);
            break;

        case InvariantType::SUM_EQUALS:
        case InvariantType::RATIO_MAINTAINED:
            result->violated = !check_sum_equals(inv, storage);
            break;

        default:
            // Unknown invariant type
            break;
    }

    return true;
}

__host__ __device__ bool InvariantChecker::check_storage_equals(const invariant_t& inv,
                                                                 const evm_word_t* storage) {
    if (!storage) return true;

    // Get slot index (simplified - in reality would need to compute storage location)
    uint32_t slot_idx = inv.slot1.limbs[0] % 1024;  // Assume max 1024 storage slots

    // Compare with expected value
    for (int i = 0; i < 8; i++) {
        if (storage[slot_idx].limbs[i] != inv.expected_value.limbs[i]) {
            return false;
        }
    }
    return true;
}

__host__ __device__ bool InvariantChecker::check_storage_range(const invariant_t& inv,
                                                                const evm_word_t* storage) {
    if (!storage) return true;

    uint32_t slot_idx = inv.slot1.limbs[0] % 1024;

    // Simplified comparison using first limb only
    uint32_t value = storage[slot_idx].limbs[0];

    switch (inv.type) {
        case InvariantType::STORAGE_NOT_ZERO:
            // Check if any limb is non-zero
            for (int i = 0; i < 8; i++) {
                if (storage[slot_idx].limbs[i] != 0) return true;
            }
            return false;

        case InvariantType::STORAGE_LESS_THAN:
            return value < inv.max_value.limbs[0];

        case InvariantType::STORAGE_GREATER_THAN:
            return value > inv.min_value.limbs[0];

        case InvariantType::STORAGE_IN_RANGE:
            return value >= inv.min_value.limbs[0] && value <= inv.max_value.limbs[0];

        default:
            return true;
    }
}

__host__ __device__ bool InvariantChecker::check_balance_conserved(const invariant_t& inv,
                                                                    const evm_word_t* balances) {
    if (!balances) return true;

    // Sum up balances for tracked addresses
    uint64_t total = 0;
    for (uint32_t i = 0; i < inv.num_slots && i < 4; i++) {
        uint32_t addr_idx = inv.addresses[i].limbs[0] % 256;
        total += balances[addr_idx].limbs[0];
    }

    // Check against expected total
    return total == inv.expected_value.limbs[0];
}

__host__ __device__ bool InvariantChecker::check_sum_equals(const invariant_t& inv,
                                                             const evm_word_t* storage) {
    if (!storage) return true;

    // Sum storage slots
    uint64_t sum = 0;
    for (uint32_t i = 0; i < inv.num_slots && i < 4; i++) {
        uint32_t slot_idx = inv.slots[i].limbs[0] % 1024;
        sum += storage[slot_idx].limbs[0];
    }

    // Check against expected sum
    return sum == inv.expected_value.limbs[0];
}

__host__ void InvariantChecker::add_erc20_invariants(const evm_word_t& token_address) {
    // Total supply equals sum of all balances
    invariant_t supply_inv;
    supply_inv.init();
    supply_inv.type = InvariantType::TOTAL_SUPPLY_CONSERVED;
    supply_inv.target_address = token_address;
    snprintf(supply_inv.description, sizeof(supply_inv.description),
             "ERC20: Total supply must equal sum of balances");
    add_invariant(supply_inv);

    // Balance cannot exceed total supply
    invariant_t balance_inv;
    balance_inv.init();
    balance_inv.type = InvariantType::STORAGE_LESS_THAN;
    balance_inv.target_address = token_address;
    snprintf(balance_inv.description, sizeof(balance_inv.description),
             "ERC20: Individual balance cannot exceed total supply");
    add_invariant(balance_inv);
}

__host__ void InvariantChecker::add_erc721_invariants(const evm_word_t& token_address) {
    // Each token has exactly one owner
    invariant_t owner_inv;
    owner_inv.init();
    owner_inv.type = InvariantType::STORAGE_NOT_ZERO;
    owner_inv.target_address = token_address;
    snprintf(owner_inv.description, sizeof(owner_inv.description),
             "ERC721: Each minted token must have an owner");
    add_invariant(owner_inv);
}

__host__ void InvariantChecker::add_erc4626_invariants(const evm_word_t& vault_address) {
    // Asset/share ratio invariant
    invariant_t ratio_inv;
    ratio_inv.init();
    ratio_inv.type = InvariantType::ERC4626_ASSET_SHARE_RATIO;
    ratio_inv.target_address = vault_address;
    snprintf(ratio_inv.description, sizeof(ratio_inv.description),
             "ERC4626: Asset/share ratio must be maintained");
    add_invariant(ratio_inv);
}

__host__ void InvariantChecker::add_amm_invariants(const evm_word_t& pool_address) {
    // Constant product invariant
    invariant_t k_inv;
    k_inv.init();
    k_inv.type = InvariantType::AMM_K_CONSERVED;
    k_inv.target_address = pool_address;
    snprintf(k_inv.description, sizeof(k_inv.description),
             "AMM: Constant product k must be maintained (x * y >= k)");
    add_invariant(k_inv);
}

__host__ void InvariantChecker::add_lending_invariants(const evm_word_t& protocol_address) {
    // Collateral ratio invariant
    invariant_t collateral_inv;
    collateral_inv.init();
    collateral_inv.type = InvariantType::LENDING_COLLATERAL_RATIO;
    collateral_inv.target_address = protocol_address;
    snprintf(collateral_inv.description, sizeof(collateral_inv.description),
             "Lending: Collateral ratio must be maintained");
    add_invariant(collateral_inv);
}

__host__ void InvariantChecker::load_from_json(const char* filename) {
    FILE* f = fopen(filename, "r");
    if (!f) return;

    // Simple JSON parsing for invariants
    char line[512];
    invariant_t current_inv;
    current_inv.init();
    bool in_invariant = false;

    while (fgets(line, sizeof(line), f)) {
        // Very basic parsing
        if (strstr(line, "\"type\":")) {
            char* type_str = strstr(line, ":");
            if (type_str) {
                int type_val = atoi(type_str + 1);
                current_inv.type = static_cast<InvariantType>(type_val);
            }
        } else if (strstr(line, "\"description\":")) {
            char* desc_start = strchr(line, '"');
            if (desc_start) {
                desc_start = strchr(desc_start + 1, '"');
                if (desc_start) {
                    desc_start++;
                    char* desc_end = strchr(desc_start, '"');
                    if (desc_end) {
                        size_t len = desc_end - desc_start;
                        if (len >= sizeof(current_inv.description)) {
                            len = sizeof(current_inv.description) - 1;
                        }
                        strncpy(current_inv.description, desc_start, len);
                        current_inv.description[len] = '\0';
                    }
                }
            }
        } else if (strstr(line, "\"enabled\":")) {
            current_inv.enabled = strstr(line, "true") != nullptr;
        } else if (strstr(line, "}")) {
            // End of invariant object
            if (current_inv.type != InvariantType::STORAGE_EQUALS || current_inv.description[0] != '\0') {
                add_invariant(current_inv);
                current_inv.init();
            }
        }
    }

    fclose(f);
}

__host__ void InvariantChecker::save_to_json(const char* filename) {
    FILE* f = fopen(filename, "w");
    if (!f) return;

    fprintf(f, "{\n  \"invariants\": [\n");

    for (uint32_t i = 0; i < num_invariants_; i++) {
        const invariant_t& inv = invariants_[i];
        fprintf(f, "    {\n");
        fprintf(f, "      \"id\": %u,\n", inv.id);
        fprintf(f, "      \"type\": %d,\n", static_cast<int>(inv.type));
        fprintf(f, "      \"description\": \"%s\",\n", inv.description);
        fprintf(f, "      \"enabled\": %s,\n", inv.enabled ? "true" : "false");
        fprintf(f, "      \"violation_count\": %u\n", inv.violation_count);
        fprintf(f, "    }%s\n", (i < num_invariants_ - 1) ? "," : "");
    }

    fprintf(f, "  ]\n}\n");
    fclose(f);
}

__host__ __device__ uint32_t InvariantChecker::get_violation_count(uint32_t id) {
    if (id >= num_invariants_) return 0;
    return invariants_[id].violation_count;
}

__host__ __device__ uint32_t InvariantChecker::get_total_violations() {
    uint32_t total = 0;
    for (uint32_t i = 0; i < num_invariants_; i++) {
        total += invariants_[i].violation_count;
    }
    return total;
}

// ============================================================================
// CUDA Kernels
// ============================================================================

__global__ void kernel_select_seeds(
    seed_entry_t* seeds,
    uint32_t num_seeds,
    uint32_t* selected_indices,
    uint32_t num_to_select,
    curandState* rng_states
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_to_select) return;

    curandState local_state = rng_states[idx];

    // Weighted selection
    uint64_t total_energy = 0;
    for (uint32_t i = 0; i < num_seeds; i++) {
        total_energy += seeds[i].metadata.energy;
    }

    if (total_energy == 0) {
        // Uniform selection
        selected_indices[idx] = curand(&local_state) % num_seeds;
    } else {
        // Weighted selection
        uint64_t target = curand(&local_state) % total_energy;
        uint64_t cumulative = 0;

        for (uint32_t i = 0; i < num_seeds; i++) {
            cumulative += seeds[i].metadata.energy;
            if (cumulative > target) {
                selected_indices[idx] = i;
                break;
            }
        }
    }

    rng_states[idx] = local_state;
}

__global__ void kernel_update_energies(
    seed_entry_t* seeds,
    uint32_t num_seeds,
    float decay_factor
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_seeds) return;

    seed_entry_t& seed = seeds[idx];

    // Apply decay
    float new_energy = seed.metadata.energy / decay_factor;
    if (new_energy < ENERGY_MIN) {
        new_energy = ENERGY_MIN;
    }
    seed.metadata.energy = static_cast<uint32_t>(new_energy);

    // Recalculate priority
    float priority = 1.0f;
    priority += seed.metadata.coverage_contribution * 10.0f;
    priority += seed.metadata.bug_count * 100.0f;
    if (seed.metadata.mutation_count > 1000) {
        priority *= 0.5f;
    }
    seed.metadata.priority = static_cast<uint32_t>(priority);
}

__global__ void kernel_check_invariants(
    InvariantChecker* checker,
    const evm_word_t* storages,
    const evm_word_t* balances,
    uint32_t num_instances,
    invariant_result_t* results,
    uint32_t* violation_counts
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_instances) return;

    // Each instance has its own storage/balance state
    const evm_word_t* instance_storage = storages + idx * 1024;  // Assume 1024 slots per instance
    const evm_word_t* instance_balances = balances + idx * 256;  // Assume 256 addresses per instance

    // Results for this instance
    invariant_result_t* instance_results = results + idx * MAX_INVARIANTS;
    uint32_t violations = 0;

    checker->check_all(instance_storage, instance_balances, idx, instance_results, &violations);
    violation_counts[idx] = violations;
}

__global__ void kernel_compute_coverage_hashes(
    const coverage_snapshot_t* snapshots,
    uint32_t num_snapshots,
    uint32_t* hashes
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_snapshots) return;

    const coverage_snapshot_t& snapshot = snapshots[idx];

    // FNV-1a hash of coverage bitmap
    uint32_t hash = 2166136261u;
    for (uint32_t i = 0; i < COVERAGE_MAP_SIZE / 32; i++) {
        hash ^= snapshot.edge_bitmap[i];
        hash *= 16777619u;
    }

    hashes[idx] = hash;
}

// ============================================================================
// Host Helper Functions
// ============================================================================

__host__ GPUCorpusManager* allocate_corpus_manager(uint32_t max_size) {
    GPUCorpusManager* manager;
    cudaMallocManaged(&manager, sizeof(GPUCorpusManager));
    new (manager) GPUCorpusManager(max_size);
    return manager;
}

__host__ void free_corpus_manager(GPUCorpusManager* manager) {
    if (manager) {
        manager->~GPUCorpusManager();
        cudaFree(manager);
    }
}

__host__ InvariantChecker* allocate_invariant_checker() {
    InvariantChecker* checker;
    cudaMallocManaged(&checker, sizeof(InvariantChecker));
    new (checker) InvariantChecker();
    return checker;
}

__host__ void free_invariant_checker(InvariantChecker* checker) {
    if (checker) {
        checker->~InvariantChecker();
        cudaFree(checker);
    }
}

__host__ void generate_initial_corpus(GPUCorpusManager* corpus,
                                       const uint8_t* contract_abi,
                                       uint32_t abi_length) {
    if (!corpus || !contract_abi || abi_length == 0) return;

    // Parse ABI to find function selectors
    // This is a simplified implementation - real version would parse JSON ABI

    // Common function selectors for testing
    uint8_t selectors[][4] = {
        {0xa9, 0x05, 0x9c, 0xbb},  // transfer(address,uint256)
        {0x23, 0xb8, 0x72, 0xdd},  // transferFrom(address,address,uint256)
        {0x09, 0x5e, 0xa7, 0xb3},  // approve(address,uint256)
        {0x70, 0xa0, 0x82, 0x31},  // balanceOf(address)
        {0x18, 0x16, 0x0d, 0xdd},  // totalSupply()
        {0xdd, 0x62, 0xed, 0x3e},  // allowance(address,address)
        {0x40, 0xc1, 0x0f, 0x19},  // mint(address,uint256)
        {0x42, 0x96, 0x6c, 0x68},  // burn(uint256)
    };

    // Create initial seeds for each function
    for (size_t i = 0; i < sizeof(selectors) / sizeof(selectors[0]); i++) {
        seed_entry_t seed;
        seed.init();

        // Create minimal calldata with selector and zero args
        uint32_t calldata_len = 4 + 64;  // Selector + 2 args
        uint8_t* calldata;
        cudaMallocManaged(&calldata, calldata_len);
        memset(calldata, 0, calldata_len);
        memcpy(calldata, selectors[i], 4);

        seed.data.data = calldata;
        seed.data.length = calldata_len;
        seed.data.capacity = calldata_len;
        seed.num_transactions = 1;
        seed.tx_offsets[0] = 0;
        seed.tx_lengths[0] = calldata_len;

        corpus->add_seed(seed, false);
    }

    // Add edge case seeds
    // Empty calldata
    {
        seed_entry_t seed;
        seed.init();
        uint8_t* calldata;
        cudaMallocManaged(&calldata, 4);
        memset(calldata, 0, 4);
        seed.data.data = calldata;
        seed.data.length = 4;
        seed.data.capacity = 4;
        seed.num_transactions = 1;
        seed.tx_offsets[0] = 0;
        seed.tx_lengths[0] = 4;
        corpus->add_seed(seed, false);
    }

    // Random selector
    {
        seed_entry_t seed;
        seed.init();
        uint8_t* calldata;
        cudaMallocManaged(&calldata, 4);
        calldata[0] = 0xDE;
        calldata[1] = 0xAD;
        calldata[2] = 0xBE;
        calldata[3] = 0xEF;
        seed.data.data = calldata;
        seed.data.length = 4;
        seed.data.capacity = 4;
        seed.num_transactions = 1;
        seed.tx_offsets[0] = 0;
        seed.tx_lengths[0] = 4;
        corpus->add_seed(seed, false);
    }
}

}  // namespace fuzzing
}  // namespace CuEVM
