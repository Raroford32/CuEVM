// CuEVM: CUDA Ethereum Virtual Machine implementation
// GPU Fuzzer Orchestrator for NVIDIA B300 Smart Contract Fuzzing
// SPDX-License-Identifier: MIT

#ifndef _CUEVM_GPU_FUZZER_H_
#define _CUEVM_GPU_FUZZER_H_

#include <cuda_runtime.h>
#include <cstdint>
#include <chrono>

#include <CuEVM/fuzzing/coverage.cuh>
#include <CuEVM/fuzzing/mutation.cuh>
#include <CuEVM/fuzzing/oracle.cuh>
#include <CuEVM/fuzzing/corpus.cuh>

namespace CuEVM {
namespace fuzzing {

// ============================================================================
// B300 Optimization Constants
// ============================================================================

// B300 GPU specifications (SM 103, Blackwell architecture)
constexpr uint32_t B300_SM_COUNT = 192;                 // Streaming multiprocessors
constexpr uint32_t B300_CUDA_CORES = 24576;             // Total CUDA cores
constexpr uint32_t B300_MEMORY_GB = 192;                // HBM3e memory
constexpr uint32_t B300_MEMORY_BANDWIDTH_TB = 8;        // Memory bandwidth TB/s
constexpr uint32_t B300_L2_CACHE_MB = 128;              // L2 cache size

// Optimal batch sizes for B300
constexpr uint32_t DEFAULT_BATCH_SIZE = 65536;          // Default instances per batch
constexpr uint32_t MIN_BATCH_SIZE = 1024;
constexpr uint32_t MAX_BATCH_SIZE = 524288;             // 512K max

// Thread configuration for B300
constexpr uint32_t THREADS_PER_BLOCK = 256;
constexpr uint32_t WARPS_PER_SM = 64;

// Memory pool sizes
constexpr size_t INPUT_POOL_SIZE = 512 * 1024 * 1024;   // 512MB for inputs
constexpr size_t STATE_POOL_SIZE = 1024 * 1024 * 1024;  // 1GB for state
constexpr size_t TRACE_POOL_SIZE = 256 * 1024 * 1024;   // 256MB for traces

// ============================================================================
// Fuzzer Configuration
// ============================================================================

struct fuzzer_config_t {
    // Batch sizing
    uint32_t num_instances;             // Instances per batch
    uint32_t sequence_length;           // Transactions per sequence
    bool auto_tune_batch_size;          // Enable auto-tuning

    // Mutation configuration
    uint32_t mutations_per_seed;        // Mutations per selected seed
    uint32_t havoc_iterations;          // Havoc mutation depth
    bool abi_aware_mutation;            // Enable ABI-aware mutation
    bool dictionary_mutation;           // Enable dictionary-based mutation

    // Coverage configuration
    bool track_edge_coverage;
    bool track_branch_coverage;
    bool track_storage_coverage;
    bool gradient_guided;               // Enable gradient-guided fuzzing

    // Oracle configuration
    oracle_config_t oracle_config;

    // Corpus configuration
    uint32_t max_corpus_size;
    uint32_t min_corpus_size;
    bool minimize_seeds;
    uint32_t cull_interval;             // Cull corpus every N iterations

    // Scheduling
    uint32_t seed_schedule;             // 0=random, 1=weighted, 2=round-robin
    uint32_t energy_decay_iterations;

    // Reporting
    uint32_t stats_interval;            // Print stats every N iterations
    uint32_t checkpoint_interval;       // Save checkpoint every N iterations
    bool verbose;

    // Timeouts
    uint32_t max_iterations;            // 0 = unlimited
    uint32_t max_time_seconds;          // 0 = unlimited
    uint32_t stall_threshold;           // Stop if no progress for N iterations

    // GPU configuration
    int gpu_device_id;
    bool use_pinned_memory;
    bool use_unified_memory;

    __host__ void set_default();
    __host__ void set_for_b300();       // Optimized settings for B300
    __host__ void load_from_json(const char* filename);
    __host__ void save_to_json(const char* filename);
};

// ============================================================================
// Fuzzer Statistics
// ============================================================================

struct fuzzer_stats_t {
    // Execution counts
    uint64_t total_iterations;
    uint64_t total_executions;          // Total EVM executions
    uint64_t total_transactions;        // Total transactions executed

    // Coverage metrics
    uint32_t unique_edges;
    uint32_t unique_branches;
    uint32_t unique_pcs;
    float edge_coverage_percent;
    float branch_coverage_percent;

    // Bug metrics
    uint32_t total_bugs_found;
    uint32_t unique_bugs;
    uint32_t critical_bugs;
    uint32_t high_bugs;
    uint32_t medium_bugs;
    uint32_t low_bugs;

    // Corpus metrics
    uint32_t corpus_size;
    uint32_t seeds_added;
    uint32_t seeds_removed;
    uint32_t interesting_seeds;

    // Performance metrics
    double total_time_seconds;
    double executions_per_second;
    double transactions_per_second;
    double gpu_utilization;
    double memory_usage_gb;

    // Timing breakdown
    double mutation_time_percent;
    double execution_time_percent;
    double coverage_time_percent;
    double oracle_time_percent;

    // Progress tracking
    uint64_t last_new_coverage_iter;
    uint64_t last_bug_iter;
    uint32_t iterations_since_progress;

    __host__ void init();
    __host__ void update(const corpus_stats_t& corpus_stats,
                         const bug_storage_t& bugs,
                         const gpu_coverage_map_t& coverage);
    __host__ void print();
    __host__ void print_summary();
    __host__ void export_json(const char* filename);
};

// ============================================================================
// B300 Batch Optimizer
// ============================================================================

class B300BatchOptimizer {
public:
    __host__ B300BatchOptimizer();

    // Auto-tune batch size for optimal throughput
    __host__ uint32_t optimize_batch_size(uint32_t current_batch_size,
                                          double current_throughput,
                                          double gpu_utilization);

    // Compute optimal configuration
    __host__ void compute_optimal_config(uint32_t contract_size,
                                         uint32_t avg_tx_size,
                                         fuzzer_config_t* config);

    // Memory estimation
    __host__ size_t estimate_memory_usage(uint32_t batch_size,
                                          uint32_t sequence_length,
                                          uint32_t avg_tx_size);

    // Profiling
    __host__ void start_profiling();
    __host__ void end_profiling();
    __host__ void record_iteration(double iteration_time, uint32_t batch_size);
    __host__ void print_profile_stats();

private:
    // Historical data for optimization
    double throughput_history_[64];
    uint32_t batch_size_history_[64];
    uint32_t history_idx_;
    uint32_t history_count_;

    // Profiling
    bool profiling_enabled_;
    std::chrono::high_resolution_clock::time_point profile_start_;
    double total_profile_time_;
    uint64_t total_profile_executions_;
};

// ============================================================================
// GPU Memory Pool Manager
// ============================================================================

class GPUMemoryPool {
public:
    __host__ GPUMemoryPool(size_t input_pool_size = INPUT_POOL_SIZE,
                           size_t state_pool_size = STATE_POOL_SIZE,
                           size_t trace_pool_size = TRACE_POOL_SIZE);
    __host__ ~GPUMemoryPool();

    // Allocate from pools
    __host__ void* allocate_input(size_t size);
    __host__ void* allocate_state(size_t size);
    __host__ void* allocate_trace(size_t size);

    // Free back to pools
    __host__ void free_input(void* ptr);
    __host__ void free_state(void* ptr);
    __host__ void free_trace(void* ptr);

    // Reset pools (for new batch)
    __host__ void reset_input_pool();
    __host__ void reset_trace_pool();

    // Statistics
    __host__ size_t get_input_pool_used();
    __host__ size_t get_state_pool_used();
    __host__ size_t get_trace_pool_used();

private:
    uint8_t* input_pool_;
    uint8_t* state_pool_;
    uint8_t* trace_pool_;
    size_t input_pool_size_;
    size_t state_pool_size_;
    size_t trace_pool_size_;
    size_t input_pool_offset_;
    size_t state_pool_offset_;
    size_t trace_pool_offset_;
};

// ============================================================================
// Execution Batch
// ============================================================================

struct execution_batch_t {
    // Inputs
    mutation_input_t* inputs;           // [num_instances]
    sequence_t* sequences;              // [num_instances] (if sequence mode)

    // Instance coverage tracking
    instance_coverage_t* coverage;      // [num_instances]

    // State trackers for oracles
    execution_state_tracker_t* trackers;// [num_instances]

    // Results
    bool* execution_success;            // [num_instances]
    uint8_t* return_data;               // [num_instances * MAX_RETURN_SIZE]
    uint32_t* return_sizes;             // [num_instances]
    uint64_t* gas_used;                 // [num_instances]

    // Batch metadata
    uint32_t num_instances;
    uint32_t sequence_length;
    bool is_sequence_mode;

    __host__ void allocate(uint32_t instances, uint32_t seq_len, bool sequence_mode);
    __host__ void free();
    __host__ void reset();
};

// ============================================================================
// GPU Fuzzer Main Class
// ============================================================================

class GPUFuzzer {
public:
    __host__ GPUFuzzer(const char* contract_source,
                       const char* contract_name = nullptr,
                       const fuzzer_config_t* config = nullptr);
    __host__ ~GPUFuzzer();

    // Initialization
    __host__ bool initialize();
    __host__ bool load_contract(const char* bytecode, uint32_t bytecode_len);
    __host__ bool load_contract_from_file(const char* filename);

    // Configuration
    __host__ void set_config(const fuzzer_config_t& config);
    __host__ fuzzer_config_t* get_config() { return &config_; }

    // Invariants
    __host__ void add_invariant(const invariant_t& inv);
    __host__ void load_invariants(const char* filename);

    // Initial corpus
    __host__ void add_seed(const uint8_t* calldata, uint32_t len);
    __host__ void add_sequence_seed(const sequence_t& seq);
    __host__ void load_initial_corpus(const char* directory);
    __host__ void generate_initial_seeds();

    // Main fuzzing loop
    __host__ void run();
    __host__ void run_iterations(uint32_t num_iterations);
    __host__ void stop();

    // Single iteration (for fine-grained control)
    __host__ void prepare_batch();
    __host__ void execute_batch();
    __host__ void analyze_batch();
    __host__ void update_corpus();

    // Results
    __host__ fuzzer_stats_t* get_stats() { return &stats_; }
    __host__ bug_storage_t* get_bugs() { return bugs_; }
    __host__ GPUCorpusManager* get_corpus() { return corpus_; }
    __host__ gpu_coverage_map_t* get_coverage() { return global_coverage_; }

    // Reporting
    __host__ void print_stats();
    __host__ void print_bugs();
    __host__ void export_results(const char* directory);
    __host__ void save_checkpoint(const char* filename);
    __host__ void load_checkpoint(const char* filename);

    // Callbacks
    using progress_callback_t = void(*)(const fuzzer_stats_t*, void*);
    using bug_callback_t = void(*)(const detected_bug_t*, void*);
    __host__ void set_progress_callback(progress_callback_t cb, void* ctx);
    __host__ void set_bug_callback(bug_callback_t cb, void* ctx);

private:
    // Configuration
    fuzzer_config_t config_;
    char* contract_source_;
    char* contract_name_;
    uint8_t* contract_bytecode_;
    uint32_t bytecode_len_;

    // Core components
    GPUMutationEngine* mutation_engine_;
    GPUCorpusManager* corpus_;
    InvariantChecker* invariant_checker_;
    CompositeOracle* oracle_;
    B300BatchOptimizer* batch_optimizer_;
    GPUMemoryPool* memory_pool_;

    // Coverage tracking
    gpu_coverage_map_t* global_coverage_;
    coverage_snapshot_t baseline_coverage_;

    // Bug storage
    bug_storage_t* bugs_;

    // Execution batch
    execution_batch_t batch_;

    // Statistics
    fuzzer_stats_t stats_;
    std::chrono::high_resolution_clock::time_point start_time_;

    // Control
    bool running_;
    bool initialized_;

    // Callbacks
    progress_callback_t progress_callback_;
    void* progress_callback_ctx_;
    bug_callback_t bug_callback_;
    void* bug_callback_ctx_;

    // CUDA streams for overlap
    cudaStream_t mutation_stream_;
    cudaStream_t execution_stream_;
    cudaStream_t analysis_stream_;

    // RNG state
    gpu_rng_state_t rng_state_;

    // Internal methods
    __host__ void select_seeds_for_batch();
    __host__ void mutate_batch();
    __host__ void execute_evm_batch();
    __host__ void collect_coverage();
    __host__ void check_oracles();
    __host__ void check_invariants();
    __host__ void process_interesting_inputs();
    __host__ void update_statistics();
    __host__ void report_progress();
    __host__ void maybe_cull_corpus();
    __host__ void maybe_checkpoint();
    __host__ bool should_stop();
};

// ============================================================================
// Convenience Functions
// ============================================================================

// Quick fuzz function for simple usage
__host__ fuzzer_stats_t quick_fuzz(
    const char* contract_source,
    const char* contract_name,
    uint32_t num_iterations = 10000,
    uint32_t num_instances = DEFAULT_BATCH_SIZE
);

// Fuzz with custom configuration
__host__ fuzzer_stats_t fuzz_with_config(
    const char* contract_source,
    const char* contract_name,
    const fuzzer_config_t& config
);

// Multi-contract fuzzing
__host__ void fuzz_multi_contract(
    const char** contract_sources,
    const char** contract_names,
    uint32_t num_contracts,
    const fuzzer_config_t& config,
    fuzzer_stats_t* combined_stats
);

// ============================================================================
// CUDA Kernels
// ============================================================================

// Main fuzzing kernel that executes EVM instances
__global__ void kernel_execute_batch(
    void* evm_instances,                // CuEVM instances
    mutation_input_t* inputs,
    instance_coverage_t* coverage,
    execution_state_tracker_t* trackers,
    bool* success,
    uint8_t* return_data,
    uint32_t* return_sizes,
    uint64_t* gas_used,
    uint32_t num_instances
);

// Coverage merge kernel
__global__ void kernel_merge_batch_coverage(
    instance_coverage_t* instance_coverage,
    gpu_coverage_map_t* global_coverage,
    uint32_t num_instances,
    uint32_t* new_coverage_flags
);

// Oracle checking kernel
__global__ void kernel_run_oracles(
    CompositeOracle* oracle,
    execution_state_tracker_t* trackers,
    uint32_t num_instances,
    bug_storage_t* bugs
);

// Corpus selection kernel
__global__ void kernel_weighted_selection(
    seed_entry_t* seeds,
    uint32_t num_seeds,
    uint32_t* cumulative_weights,
    uint32_t* selected_indices,
    uint32_t num_to_select,
    curandState* rng
);

}  // namespace fuzzing
}  // namespace CuEVM

#endif  // _CUEVM_GPU_FUZZER_H_
