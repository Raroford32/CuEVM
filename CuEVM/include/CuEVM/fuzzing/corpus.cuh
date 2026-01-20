// CuEVM: CUDA Ethereum Virtual Machine implementation
// GPU Corpus Management for Smart Contract Fuzzing
// SPDX-License-Identifier: MIT

#ifndef _CUEVM_FUZZING_CORPUS_H_
#define _CUEVM_FUZZING_CORPUS_H_

#include <cuda_runtime.h>
#include <cstdint>
#include <CuEVM/fuzzing/coverage.cuh>
#include <CuEVM/fuzzing/mutation.cuh>
#include <CuEVM/fuzzing/oracle.cuh>

namespace CuEVM {
namespace fuzzing {

// ============================================================================
// Corpus Configuration
// ============================================================================

constexpr uint32_t MAX_CORPUS_SIZE = 65536;             // Max seeds in corpus
constexpr uint32_t MAX_SEED_DATA_SIZE = 8192;           // Max bytes per seed
constexpr uint32_t MAX_SEQUENCE_LENGTH = 32;            // Max transactions per sequence
constexpr uint32_t CORPUS_BUCKET_COUNT = 256;           // Hash buckets for dedup
constexpr uint32_t MIN_CORPUS_ENTRIES = 64;             // Minimum seeds to maintain

// Energy assignment for seed scheduling
constexpr uint32_t ENERGY_BASE = 100;
constexpr uint32_t ENERGY_NEW_COVERAGE = 500;
constexpr uint32_t ENERGY_NEW_BUG = 1000;
constexpr uint32_t ENERGY_DECAY_FACTOR = 2;
constexpr uint32_t ENERGY_MIN = 10;

// ============================================================================
// Seed Entry
// ============================================================================

struct seed_data_t {
    uint8_t* data;                      // Raw calldata bytes
    uint32_t length;                    // Data length
    uint32_t capacity;                  // Allocated capacity
};

struct seed_metadata_t {
    uint64_t id;                        // Unique seed ID
    uint64_t parent_id;                 // Parent seed (0 if from initial corpus)
    uint64_t timestamp;                 // When this seed was added
    uint32_t generation;                // Mutation generation count

    // Coverage information
    uint32_t unique_edges;              // Edges this seed covers
    uint32_t unique_branches;           // Branches this seed covers
    uint32_t coverage_hash;             // Hash of coverage bitmap for dedup
    float coverage_contribution;        // How much new coverage this seed added

    // Quality metrics
    uint32_t execution_count;           // How many times this seed was used
    uint32_t mutation_count;            // How many mutants were derived
    uint32_t child_count;               // How many children added to corpus
    uint32_t bug_count;                 // Bugs found from this seed

    // Scheduling
    uint32_t energy;                    // Current energy for scheduling
    uint32_t priority;                  // Priority score (higher = more likely to pick)
    uint32_t last_selected;             // Timestamp of last selection

    // Minimization
    bool minimized;                     // Whether this seed has been minimized
    uint32_t original_length;           // Length before minimization
};

struct seed_entry_t {
    seed_data_t data;
    seed_metadata_t metadata;

    // For sequence seeds
    uint32_t num_transactions;
    uint32_t tx_offsets[MAX_SEQUENCE_LENGTH];   // Offset of each tx in data
    uint32_t tx_lengths[MAX_SEQUENCE_LENGTH];   // Length of each tx

    // Transaction context
    evm_word_t senders[MAX_SEQUENCE_LENGTH];
    evm_word_t values[MAX_SEQUENCE_LENGTH];
    evm_word_t receivers[MAX_SEQUENCE_LENGTH];

    // Block context for sequence
    evm_word_t block_number;
    evm_word_t timestamp;

    __host__ __device__ void init();
    __host__ __device__ void copy_from(const seed_entry_t& other);
    __host__ __device__ void set_transaction(uint32_t tx_idx, const uint8_t* calldata,
                                              uint32_t len, const evm_word_t& sender,
                                              const evm_word_t& value);
};

// ============================================================================
// Corpus Statistics
// ============================================================================

struct corpus_stats_t {
    uint64_t total_seeds_added;
    uint64_t total_seeds_removed;
    uint64_t total_executions;
    uint64_t total_mutations;
    uint64_t total_new_coverage;
    uint64_t total_bugs_found;

    uint32_t current_size;
    uint32_t unique_coverage_edges;
    uint32_t unique_coverage_branches;
    float overall_coverage_percent;

    uint64_t last_new_coverage_time;
    uint64_t last_bug_time;
    uint32_t cycles_since_progress;

    // Per-category counts
    uint32_t initial_seeds;
    uint32_t mutant_seeds;
    uint32_t splice_seeds;
    uint32_t minimized_seeds;

    __host__ __device__ void init();
    __host__ __device__ void update_coverage(uint32_t new_edges, uint32_t new_branches);
    __host__ __device__ void record_new_seed(bool from_mutation, bool caused_new_coverage);
};

// ============================================================================
// Corpus Hash Table (for deduplication)
// ============================================================================

struct corpus_bucket_t {
    uint32_t seed_indices[16];          // Indices of seeds in this bucket
    uint32_t count;
};

struct corpus_hash_table_t {
    corpus_bucket_t buckets[CORPUS_BUCKET_COUNT];

    __host__ __device__ void init();
    __host__ __device__ bool contains(uint32_t coverage_hash);
    __host__ __device__ void insert(uint32_t coverage_hash, uint32_t seed_idx);
    __host__ __device__ void remove(uint32_t coverage_hash, uint32_t seed_idx);
};

// ============================================================================
// GPU Corpus Manager
// ============================================================================

class GPUCorpusManager {
public:
    __host__ GPUCorpusManager(uint32_t max_size = MAX_CORPUS_SIZE);
    __host__ ~GPUCorpusManager();

    // Seed management
    __host__ __device__ bool add_seed(const seed_entry_t& seed, bool check_duplicate = true);
    __host__ __device__ bool add_seed_if_interesting(const seed_entry_t& seed,
                                                      const coverage_snapshot_t& coverage,
                                                      const bug_storage_t* bugs);
    __host__ __device__ void remove_seed(uint32_t idx);
    __host__ __device__ seed_entry_t* get_seed(uint32_t idx);
    __host__ __device__ uint32_t size() const { return stats_.current_size; }

    // Seed selection for fuzzing
    __host__ __device__ seed_entry_t* select_seed(curandState* rng);
    __host__ __device__ seed_entry_t* select_weighted(curandState* rng);
    __host__ __device__ void update_seed_after_execution(uint32_t idx, bool caused_new_coverage,
                                                          bool found_bug);

    // Corpus maintenance
    __host__ void cull_corpus();                    // Remove low-quality seeds
    __host__ void compact_corpus();                 // Remove gaps in storage
    __host__ void sort_by_priority();               // Sort seeds by priority
    __host__ void recalculate_energies();           // Recalculate all seed energies

    // Minimization
    __host__ void minimize_seed(uint32_t idx);
    __host__ void minimize_all();

    // Merging (for parallel fuzzing)
    __host__ void merge_from(const GPUCorpusManager& other);

    // Import/Export
    __host__ void import_seeds(const char* directory);
    __host__ void export_seeds(const char* directory);
    __host__ void export_interesting_seeds(const char* directory, uint32_t max_seeds);
    __host__ void load_checkpoint(const char* filename);
    __host__ void save_checkpoint(const char* filename);

    // Coverage integration
    __host__ void set_coverage_baseline(const gpu_coverage_map_t* baseline);
    __host__ void update_coverage_contribution(uint32_t seed_idx,
                                                const coverage_snapshot_t& new_coverage);

    // Statistics
    __host__ __device__ corpus_stats_t* get_stats() { return &stats_; }
    __host__ void print_stats();
    __host__ void export_stats_json(const char* filename);

private:
    seed_entry_t* seeds_;               // GPU-accessible seed array
    uint32_t capacity_;
    corpus_stats_t stats_;
    corpus_hash_table_t hash_table_;
    gpu_coverage_map_t* coverage_baseline_;

    // Free list for removed seeds
    uint32_t* free_indices_;
    uint32_t free_count_;

    // Priority queue for selection
    uint32_t* priority_queue_;
    uint32_t queue_size_;

    __host__ __device__ uint32_t compute_coverage_hash(const coverage_snapshot_t& coverage);
    __host__ __device__ uint32_t compute_seed_hash(const seed_entry_t& seed);
    __host__ __device__ float compute_priority(const seed_metadata_t& metadata);
    __host__ __device__ uint32_t allocate_slot();
    __host__ __device__ void deallocate_slot(uint32_t idx);
};

// ============================================================================
// Seed Minimizer
// ============================================================================

class SeedMinimizer {
public:
    __host__ SeedMinimizer();

    // Delta-debugging based minimization
    __host__ bool minimize(seed_entry_t* seed,
                           bool (*test_fn)(const seed_entry_t*, void*),
                           void* test_ctx);

    // Minimize transaction sequence
    __host__ bool minimize_sequence(seed_entry_t* seed,
                                    bool (*test_fn)(const seed_entry_t*, void*),
                                    void* test_ctx);

    // Minimize individual calldata
    __host__ bool minimize_calldata(uint8_t* data, uint32_t* length,
                                    bool (*test_fn)(const uint8_t*, uint32_t, void*),
                                    void* test_ctx);

private:
    // Delta debugging helpers
    __host__ bool ddmin(uint8_t* data, uint32_t* length, uint32_t granularity,
                        bool (*test_fn)(const uint8_t*, uint32_t, void*),
                        void* test_ctx);
};

// ============================================================================
// Corpus Distillation (create minimal corpus)
// ============================================================================

class CorpusDistiller {
public:
    __host__ CorpusDistiller(GPUCorpusManager* corpus);

    // Create minimal corpus that maintains coverage
    __host__ void distill(GPUCorpusManager* output_corpus,
                          const gpu_coverage_map_t* target_coverage);

    // Greedy set cover algorithm
    __host__ void greedy_cover(GPUCorpusManager* output_corpus,
                               const gpu_coverage_map_t* target_coverage);

private:
    GPUCorpusManager* source_corpus_;
};

// ============================================================================
// Invariant System
// ============================================================================

enum class InvariantType : uint8_t {
    // Value invariants
    STORAGE_EQUALS = 0,
    STORAGE_NOT_ZERO = 1,
    STORAGE_LESS_THAN = 2,
    STORAGE_GREATER_THAN = 3,
    STORAGE_IN_RANGE = 4,

    // Balance invariants
    BALANCE_MIN = 10,
    BALANCE_MAX = 11,
    BALANCE_EQUALS = 12,
    BALANCE_CONSERVED = 13,

    // Supply invariants (tokens)
    TOTAL_SUPPLY_CONSERVED = 20,
    TOTAL_SUPPLY_MAX = 21,

    // Access control invariants
    OWNER_UNCHANGED = 30,
    ADMIN_ONLY = 31,

    // State machine invariants
    STATE_VALID = 40,
    STATE_TRANSITION_VALID = 41,

    // Relationship invariants
    SUM_EQUALS = 50,
    RATIO_MAINTAINED = 51,

    // Protocol-specific
    AMM_K_CONSERVED = 60,
    LENDING_COLLATERAL_RATIO = 61,
    ERC4626_ASSET_SHARE_RATIO = 62,

    // Custom
    CUSTOM = 100
};

struct invariant_t {
    InvariantType type;
    uint32_t id;

    // Target storage slots/addresses
    evm_word_t target_address;
    evm_word_t slot1;
    evm_word_t slot2;

    // Expected values
    evm_word_t expected_value;
    evm_word_t min_value;
    evm_word_t max_value;

    // For relationship invariants
    evm_word_t addresses[4];
    evm_word_t slots[4];
    uint32_t num_slots;

    // Metadata
    char description[128];
    bool enabled;
    uint32_t violation_count;

    __host__ __device__ void init();
};

struct invariant_result_t {
    uint32_t invariant_id;
    bool violated;
    evm_word_t actual_value;
    evm_word_t expected_value;
    uint32_t tx_index;
    uint64_t timestamp;
};

// ============================================================================
// Invariant Checker
// ============================================================================

constexpr uint32_t MAX_INVARIANTS = 256;

class InvariantChecker {
public:
    __host__ __device__ InvariantChecker();

    // Add invariants
    __host__ __device__ uint32_t add_invariant(const invariant_t& inv);
    __host__ __device__ void remove_invariant(uint32_t id);
    __host__ __device__ void enable_invariant(uint32_t id, bool enabled);

    // Check invariants
    __host__ __device__ void check_all(const evm_word_t* storage,
                                        const evm_word_t* balances,
                                        uint32_t tx_index,
                                        invariant_result_t* results,
                                        uint32_t* num_violations);

    __host__ __device__ bool check_single(uint32_t id,
                                          const evm_word_t* storage,
                                          const evm_word_t* balances,
                                          invariant_result_t* result);

    // Pre-built invariant templates
    __host__ void add_erc20_invariants(const evm_word_t& token_address);
    __host__ void add_erc721_invariants(const evm_word_t& token_address);
    __host__ void add_erc4626_invariants(const evm_word_t& vault_address);
    __host__ void add_amm_invariants(const evm_word_t& pool_address);
    __host__ void add_lending_invariants(const evm_word_t& protocol_address);

    // Import from config
    __host__ void load_from_json(const char* filename);
    __host__ void save_to_json(const char* filename);

    // Statistics
    __host__ __device__ uint32_t get_violation_count(uint32_t id);
    __host__ __device__ uint32_t get_total_violations();

private:
    invariant_t invariants_[MAX_INVARIANTS];
    uint32_t num_invariants_;

    __host__ __device__ bool check_storage_equals(const invariant_t& inv,
                                                   const evm_word_t* storage);
    __host__ __device__ bool check_storage_range(const invariant_t& inv,
                                                  const evm_word_t* storage);
    __host__ __device__ bool check_balance_conserved(const invariant_t& inv,
                                                      const evm_word_t* balances);
    __host__ __device__ bool check_sum_equals(const invariant_t& inv,
                                               const evm_word_t* storage);
};

// ============================================================================
// CUDA Kernels
// ============================================================================

__global__ void kernel_select_seeds(
    seed_entry_t* seeds,
    uint32_t num_seeds,
    uint32_t* selected_indices,
    uint32_t num_to_select,
    curandState* rng_states
);

__global__ void kernel_update_energies(
    seed_entry_t* seeds,
    uint32_t num_seeds,
    float decay_factor
);

__global__ void kernel_check_invariants(
    InvariantChecker* checker,
    const evm_word_t* storages,          // Storage state per instance
    const evm_word_t* balances,          // Balance state per instance
    uint32_t num_instances,
    invariant_result_t* results,
    uint32_t* violation_counts
);

__global__ void kernel_compute_coverage_hashes(
    const coverage_snapshot_t* snapshots,
    uint32_t num_snapshots,
    uint32_t* hashes
);

// ============================================================================
// Host Helper Functions
// ============================================================================

__host__ GPUCorpusManager* allocate_corpus_manager(uint32_t max_size);
__host__ void free_corpus_manager(GPUCorpusManager* manager);

__host__ InvariantChecker* allocate_invariant_checker();
__host__ void free_invariant_checker(InvariantChecker* checker);

__host__ void generate_initial_corpus(GPUCorpusManager* corpus,
                                       const uint8_t* contract_abi,
                                       uint32_t abi_length);

}  // namespace fuzzing
}  // namespace CuEVM

#endif  // _CUEVM_FUZZING_CORPUS_H_
