// CuEVM: CUDA Ethereum Virtual Machine implementation
// GPU Fuzzer Orchestrator Implementation for NVIDIA B300
// SPDX-License-Identifier: MIT

#include <CuEVM/fuzzing/gpu_fuzzer.cuh>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <cmath>

namespace CuEVM {
namespace fuzzing {

// ============================================================================
// Fuzzer Configuration Implementation
// ============================================================================

__host__ void fuzzer_config_t::set_default() {
    num_instances = 8192;
    sequence_length = 1;
    auto_tune_batch_size = true;

    mutations_per_seed = 4;
    havoc_iterations = 8;
    abi_aware_mutation = true;
    dictionary_mutation = true;

    track_edge_coverage = true;
    track_branch_coverage = true;
    track_storage_coverage = true;
    gradient_guided = true;

    oracle_config.set_default();

    max_corpus_size = 16384;
    min_corpus_size = 64;
    minimize_seeds = true;
    cull_interval = 1000;

    seed_schedule = 1;  // weighted
    energy_decay_iterations = 100;

    stats_interval = 100;
    checkpoint_interval = 10000;
    verbose = false;

    max_iterations = 0;
    max_time_seconds = 0;
    stall_threshold = 100000;

    gpu_device_id = 0;
    use_pinned_memory = true;
    use_unified_memory = true;
}

__host__ void fuzzer_config_t::set_for_b300() {
    set_default();

    // Optimized for B300's capabilities
    num_instances = DEFAULT_BATCH_SIZE;  // 64K instances
    auto_tune_batch_size = true;

    // More aggressive mutation
    mutations_per_seed = 8;
    havoc_iterations = 16;

    // Larger corpus for B300's memory
    max_corpus_size = 65536;

    // Higher performance settings
    use_pinned_memory = true;
    use_unified_memory = true;
}

__host__ void fuzzer_config_t::load_from_json(const char* filename) {
    FILE* f = fopen(filename, "r");
    if (!f) {
        printf("Warning: Could not open config file %s, using defaults\n", filename);
        set_default();
        return;
    }

    // Simple JSON parsing (would use cJSON in production)
    char buffer[4096];
    size_t len = fread(buffer, 1, 4095, f);
    buffer[len] = '\0';
    fclose(f);

    // Parse key fields (simplified)
    // In production, use proper JSON parsing
    set_default();
}

__host__ void fuzzer_config_t::save_to_json(const char* filename) {
    FILE* f = fopen(filename, "w");
    if (!f) return;

    fprintf(f, "{\n");
    fprintf(f, "  \"num_instances\": %u,\n", num_instances);
    fprintf(f, "  \"sequence_length\": %u,\n", sequence_length);
    fprintf(f, "  \"auto_tune_batch_size\": %s,\n", auto_tune_batch_size ? "true" : "false");
    fprintf(f, "  \"mutations_per_seed\": %u,\n", mutations_per_seed);
    fprintf(f, "  \"havoc_iterations\": %u,\n", havoc_iterations);
    fprintf(f, "  \"abi_aware_mutation\": %s,\n", abi_aware_mutation ? "true" : "false");
    fprintf(f, "  \"max_corpus_size\": %u,\n", max_corpus_size);
    fprintf(f, "  \"max_iterations\": %u,\n", max_iterations);
    fprintf(f, "  \"max_time_seconds\": %u,\n", max_time_seconds);
    fprintf(f, "  \"gpu_device_id\": %d\n", gpu_device_id);
    fprintf(f, "}\n");

    fclose(f);
}

// ============================================================================
// Fuzzer Statistics Implementation
// ============================================================================

__host__ void fuzzer_stats_t::init() {
    total_iterations = 0;
    total_executions = 0;
    total_transactions = 0;

    unique_edges = 0;
    unique_branches = 0;
    unique_pcs = 0;
    edge_coverage_percent = 0.0f;
    branch_coverage_percent = 0.0f;

    total_bugs_found = 0;
    unique_bugs = 0;
    critical_bugs = 0;
    high_bugs = 0;
    medium_bugs = 0;
    low_bugs = 0;

    corpus_size = 0;
    seeds_added = 0;
    seeds_removed = 0;
    interesting_seeds = 0;

    total_time_seconds = 0.0;
    executions_per_second = 0.0;
    transactions_per_second = 0.0;
    gpu_utilization = 0.0;
    memory_usage_gb = 0.0;

    mutation_time_percent = 0.0;
    execution_time_percent = 0.0;
    coverage_time_percent = 0.0;
    oracle_time_percent = 0.0;

    last_new_coverage_iter = 0;
    last_bug_iter = 0;
    iterations_since_progress = 0;
}

__host__ void fuzzer_stats_t::update(const corpus_stats_t& corpus_stats,
                                      const bug_storage_t& bugs,
                                      const gpu_coverage_map_t& coverage) {
    corpus_size = corpus_stats.current_size;
    unique_edges = coverage.unique_edges;
    unique_branches = coverage.unique_branches;

    total_bugs_found = bugs.bug_count;
    critical_bugs = bugs.count_by_severity(BugSeverity::CRITICAL);
    high_bugs = bugs.count_by_severity(BugSeverity::HIGH);
    medium_bugs = bugs.count_by_severity(BugSeverity::MEDIUM);
    low_bugs = bugs.count_by_severity(BugSeverity::LOW);

    if (total_time_seconds > 0) {
        executions_per_second = total_executions / total_time_seconds;
        transactions_per_second = total_transactions / total_time_seconds;
    }
}

__host__ void fuzzer_stats_t::print() {
    printf("\n");
    printf("================================================================================\n");
    printf("                           FUZZER STATISTICS                                    \n");
    printf("================================================================================\n");
    printf("\n");

    printf("EXECUTION:\n");
    printf("  Iterations:        %lu\n", total_iterations);
    printf("  Total Executions:  %lu\n", total_executions);
    printf("  Total Txs:         %lu\n", total_transactions);
    printf("  Time (s):          %.2f\n", total_time_seconds);
    printf("  Exec/sec:          %.2f\n", executions_per_second);
    printf("  Tx/sec:            %.2f\n", transactions_per_second);
    printf("\n");

    printf("COVERAGE:\n");
    printf("  Unique Edges:      %u\n", unique_edges);
    printf("  Unique Branches:   %u\n", unique_branches);
    printf("  Unique PCs:        %u\n", unique_pcs);
    printf("  Edge Coverage:     %.2f%%\n", edge_coverage_percent);
    printf("\n");

    printf("BUGS:\n");
    printf("  Total Found:       %u\n", total_bugs_found);
    printf("  Critical:          %u\n", critical_bugs);
    printf("  High:              %u\n", high_bugs);
    printf("  Medium:            %u\n", medium_bugs);
    printf("  Low:               %u\n", low_bugs);
    printf("\n");

    printf("CORPUS:\n");
    printf("  Current Size:      %u\n", corpus_size);
    printf("  Seeds Added:       %u\n", seeds_added);
    printf("  Interesting:       %u\n", interesting_seeds);
    printf("\n");

    printf("================================================================================\n");
}

__host__ void fuzzer_stats_t::print_summary() {
    printf("[%lu] execs: %lu (%.0f/s) | cov: %u edges | bugs: %u | corpus: %u\n",
           total_iterations, total_executions, executions_per_second,
           unique_edges, total_bugs_found, corpus_size);
}

__host__ void fuzzer_stats_t::export_json(const char* filename) {
    FILE* f = fopen(filename, "w");
    if (!f) return;

    fprintf(f, "{\n");
    fprintf(f, "  \"total_iterations\": %lu,\n", total_iterations);
    fprintf(f, "  \"total_executions\": %lu,\n", total_executions);
    fprintf(f, "  \"total_transactions\": %lu,\n", total_transactions);
    fprintf(f, "  \"unique_edges\": %u,\n", unique_edges);
    fprintf(f, "  \"unique_branches\": %u,\n", unique_branches);
    fprintf(f, "  \"total_bugs_found\": %u,\n", total_bugs_found);
    fprintf(f, "  \"critical_bugs\": %u,\n", critical_bugs);
    fprintf(f, "  \"high_bugs\": %u,\n", high_bugs);
    fprintf(f, "  \"corpus_size\": %u,\n", corpus_size);
    fprintf(f, "  \"total_time_seconds\": %.2f,\n", total_time_seconds);
    fprintf(f, "  \"executions_per_second\": %.2f\n", executions_per_second);
    fprintf(f, "}\n");

    fclose(f);
}

// ============================================================================
// B300 Batch Optimizer Implementation
// ============================================================================

__host__ B300BatchOptimizer::B300BatchOptimizer()
    : history_idx_(0), history_count_(0), profiling_enabled_(false),
      total_profile_time_(0.0), total_profile_executions_(0) {
    for (int i = 0; i < 64; i++) {
        throughput_history_[i] = 0.0;
        batch_size_history_[i] = 0;
    }
}

__host__ uint32_t B300BatchOptimizer::optimize_batch_size(uint32_t current_batch_size,
                                                          double current_throughput,
                                                          double gpu_utilization) {
    // Record current performance
    throughput_history_[history_idx_] = current_throughput;
    batch_size_history_[history_idx_] = current_batch_size;
    history_idx_ = (history_idx_ + 1) % 64;
    if (history_count_ < 64) history_count_++;

    // Find optimal from history
    double best_throughput = 0.0;
    uint32_t best_batch_size = current_batch_size;
    for (uint32_t i = 0; i < history_count_; i++) {
        if (throughput_history_[i] > best_throughput) {
            best_throughput = throughput_history_[i];
            best_batch_size = batch_size_history_[i];
        }
    }

    // If GPU is underutilized, try increasing batch size
    if (gpu_utilization < 0.8 && current_batch_size < MAX_BATCH_SIZE) {
        return std::min(current_batch_size * 2, MAX_BATCH_SIZE);
    }

    // If throughput is declining, try the best historical size
    if (history_count_ > 4) {
        double recent_avg = 0.0;
        for (int i = 0; i < 4; i++) {
            int idx = (history_idx_ - 1 - i + 64) % 64;
            recent_avg += throughput_history_[idx];
        }
        recent_avg /= 4.0;

        if (recent_avg < best_throughput * 0.9) {
            return best_batch_size;
        }
    }

    return current_batch_size;
}

__host__ void B300BatchOptimizer::compute_optimal_config(uint32_t contract_size,
                                                         uint32_t avg_tx_size,
                                                         fuzzer_config_t* config) {
    // Estimate memory per instance
    size_t mem_per_instance = contract_size +           // Bytecode
                              avg_tx_size * 2 +         // Input + output
                              32 * 1024 +               // Stack + memory
                              sizeof(instance_coverage_t) +
                              sizeof(execution_state_tracker_t);

    // Calculate max instances that fit in B300's memory
    size_t available_memory = (size_t)B300_MEMORY_GB * 1024 * 1024 * 1024;
    available_memory = available_memory * 80 / 100;  // Reserve 20% for system

    uint32_t max_instances = (uint32_t)(available_memory / mem_per_instance);
    max_instances = std::min(max_instances, MAX_BATCH_SIZE);
    max_instances = std::max(max_instances, MIN_BATCH_SIZE);

    // Round to multiple of SM count for optimal occupancy
    max_instances = (max_instances / B300_SM_COUNT) * B300_SM_COUNT;

    config->num_instances = max_instances;

    // Adjust mutation depth based on contract complexity
    if (contract_size > 100000) {
        config->mutations_per_seed = 4;
        config->havoc_iterations = 4;
    } else if (contract_size > 10000) {
        config->mutations_per_seed = 8;
        config->havoc_iterations = 8;
    } else {
        config->mutations_per_seed = 16;
        config->havoc_iterations = 16;
    }
}

__host__ size_t B300BatchOptimizer::estimate_memory_usage(uint32_t batch_size,
                                                          uint32_t sequence_length,
                                                          uint32_t avg_tx_size) {
    size_t input_memory = batch_size * avg_tx_size * sequence_length;
    size_t coverage_memory = batch_size * sizeof(instance_coverage_t);
    size_t tracker_memory = batch_size * sizeof(execution_state_tracker_t);
    size_t result_memory = batch_size * (sizeof(bool) + sizeof(uint64_t) + 1024);  // return data

    return input_memory + coverage_memory + tracker_memory + result_memory;
}

__host__ void B300BatchOptimizer::start_profiling() {
    profiling_enabled_ = true;
    profile_start_ = std::chrono::high_resolution_clock::now();
}

__host__ void B300BatchOptimizer::end_profiling() {
    profiling_enabled_ = false;
}

__host__ void B300BatchOptimizer::record_iteration(double iteration_time, uint32_t batch_size) {
    if (!profiling_enabled_) return;

    total_profile_time_ += iteration_time;
    total_profile_executions_ += batch_size;
}

__host__ void B300BatchOptimizer::print_profile_stats() {
    if (total_profile_time_ > 0) {
        printf("\nB300 Profiling Stats:\n");
        printf("  Total Time: %.2f s\n", total_profile_time_);
        printf("  Total Executions: %lu\n", total_profile_executions_);
        printf("  Average Throughput: %.2f exec/s\n",
               total_profile_executions_ / total_profile_time_);
    }
}

// ============================================================================
// GPU Memory Pool Implementation
// ============================================================================

__host__ GPUMemoryPool::GPUMemoryPool(size_t input_pool_size,
                                       size_t state_pool_size,
                                       size_t trace_pool_size)
    : input_pool_size_(input_pool_size),
      state_pool_size_(state_pool_size),
      trace_pool_size_(trace_pool_size),
      input_pool_offset_(0),
      state_pool_offset_(0),
      trace_pool_offset_(0) {

    cudaMalloc(&input_pool_, input_pool_size);
    cudaMalloc(&state_pool_, state_pool_size);
    cudaMalloc(&trace_pool_, trace_pool_size);
}

__host__ GPUMemoryPool::~GPUMemoryPool() {
    cudaFree(input_pool_);
    cudaFree(state_pool_);
    cudaFree(trace_pool_);
}

__host__ void* GPUMemoryPool::allocate_input(size_t size) {
    size = (size + 255) & ~255;  // Align to 256 bytes
    if (input_pool_offset_ + size > input_pool_size_) {
        return nullptr;
    }
    void* ptr = input_pool_ + input_pool_offset_;
    input_pool_offset_ += size;
    return ptr;
}

__host__ void* GPUMemoryPool::allocate_state(size_t size) {
    size = (size + 255) & ~255;
    if (state_pool_offset_ + size > state_pool_size_) {
        return nullptr;
    }
    void* ptr = state_pool_ + state_pool_offset_;
    state_pool_offset_ += size;
    return ptr;
}

__host__ void* GPUMemoryPool::allocate_trace(size_t size) {
    size = (size + 255) & ~255;
    if (trace_pool_offset_ + size > trace_pool_size_) {
        return nullptr;
    }
    void* ptr = trace_pool_ + trace_pool_offset_;
    trace_pool_offset_ += size;
    return ptr;
}

__host__ void GPUMemoryPool::free_input(void* ptr) {
    // Pool-based, no individual frees
}

__host__ void GPUMemoryPool::free_state(void* ptr) {
    // Pool-based, no individual frees
}

__host__ void GPUMemoryPool::free_trace(void* ptr) {
    // Pool-based, no individual frees
}

__host__ void GPUMemoryPool::reset_input_pool() {
    input_pool_offset_ = 0;
}

__host__ void GPUMemoryPool::reset_trace_pool() {
    trace_pool_offset_ = 0;
}

__host__ size_t GPUMemoryPool::get_input_pool_used() {
    return input_pool_offset_;
}

__host__ size_t GPUMemoryPool::get_state_pool_used() {
    return state_pool_offset_;
}

__host__ size_t GPUMemoryPool::get_trace_pool_used() {
    return trace_pool_offset_;
}

// ============================================================================
// Execution Batch Implementation
// ============================================================================

__host__ void execution_batch_t::allocate(uint32_t instances, uint32_t seq_len, bool sequence_mode) {
    num_instances = instances;
    sequence_length = seq_len;
    is_sequence_mode = sequence_mode;

    allocate_mutation_inputs(&inputs, instances, MAX_SEED_DATA_SIZE);

    if (sequence_mode) {
        allocate_sequences(&sequences, instances, seq_len);
    } else {
        sequences = nullptr;
    }

    coverage = CoverageMapAllocator::allocate_instances(instances);
    trackers = allocate_trackers(instances);

    cudaMallocManaged(&execution_success, instances * sizeof(bool));
    cudaMallocManaged(&return_data, instances * 1024);  // 1KB per instance
    cudaMallocManaged(&return_sizes, instances * sizeof(uint32_t));
    cudaMallocManaged(&gas_used, instances * sizeof(uint64_t));
}

__host__ void execution_batch_t::free() {
    free_mutation_inputs(inputs, num_instances);
    if (sequences) {
        free_sequences(sequences, num_instances);
    }
    CoverageMapAllocator::free_instances(coverage);
    free_trackers(trackers);
    cudaFree(execution_success);
    cudaFree(return_data);
    cudaFree(return_sizes);
    cudaFree(gas_used);
}

__host__ void execution_batch_t::reset() {
    for (uint32_t i = 0; i < num_instances; i++) {
        coverage[i].init();
        trackers[i].init();
        execution_success[i] = false;
        return_sizes[i] = 0;
        gas_used[i] = 0;
    }
    cudaMemset(return_data, 0, num_instances * 1024);
}

// ============================================================================
// GPU Fuzzer Implementation
// ============================================================================

__host__ GPUFuzzer::GPUFuzzer(const char* contract_source,
                               const char* contract_name,
                               const fuzzer_config_t* config)
    : running_(false), initialized_(false),
      progress_callback_(nullptr), progress_callback_ctx_(nullptr),
      bug_callback_(nullptr), bug_callback_ctx_(nullptr) {

    // Copy contract info
    if (contract_source) {
        contract_source_ = strdup(contract_source);
    } else {
        contract_source_ = nullptr;
    }
    if (contract_name) {
        contract_name_ = strdup(contract_name);
    } else {
        contract_name_ = nullptr;
    }

    contract_bytecode_ = nullptr;
    bytecode_len_ = 0;

    // Set configuration
    if (config) {
        config_ = *config;
    } else {
        config_.set_for_b300();
    }

    // Initialize statistics
    stats_.init();
}

__host__ GPUFuzzer::~GPUFuzzer() {
    if (contract_source_) free(contract_source_);
    if (contract_name_) free(contract_name_);
    if (contract_bytecode_) cudaFree(contract_bytecode_);

    if (initialized_) {
        delete mutation_engine_;
        delete corpus_;
        delete invariant_checker_;
        delete oracle_;
        delete batch_optimizer_;
        delete memory_pool_;

        CoverageMapAllocator::free_global(global_coverage_);
        free_bug_storage(bugs_);
        batch_.free();

        cudaStreamDestroy(mutation_stream_);
        cudaStreamDestroy(execution_stream_);
        cudaStreamDestroy(analysis_stream_);
    }
}

__host__ bool GPUFuzzer::initialize() {
    if (initialized_) return true;

    // Set GPU device
    cudaSetDevice(config_.gpu_device_id);

    // Create CUDA streams
    cudaStreamCreate(&mutation_stream_);
    cudaStreamCreate(&execution_stream_);
    cudaStreamCreate(&analysis_stream_);

    // Initialize RNG
    rng_state_.init(config_.num_instances, time(nullptr));

    // Create components
    mutation_engine_ = new GPUMutationEngine(config_.num_instances, time(nullptr));
    mutation_engine_->enable_abi_aware(config_.abi_aware_mutation);

    corpus_ = new GPUCorpusManager(config_.max_corpus_size);

    invariant_checker_ = new InvariantChecker();

    oracle_config_t* oracle_config = allocate_oracle_config();
    *oracle_config = config_.oracle_config;
    bugs_ = allocate_bug_storage();
    oracle_ = new CompositeOracle(oracle_config, bugs_);

    batch_optimizer_ = new B300BatchOptimizer();
    memory_pool_ = new GPUMemoryPool();

    // Allocate global coverage map
    global_coverage_ = CoverageMapAllocator::allocate_global(1);

    // Allocate execution batch
    batch_.allocate(config_.num_instances, config_.sequence_length,
                    config_.sequence_length > 1);

    start_time_ = std::chrono::high_resolution_clock::now();
    initialized_ = true;

    return true;
}

__host__ bool GPUFuzzer::load_contract(const char* bytecode, uint32_t bytecode_len) {
    if (contract_bytecode_) {
        cudaFree(contract_bytecode_);
    }

    bytecode_len_ = bytecode_len;
    cudaMallocManaged(&contract_bytecode_, bytecode_len);
    memcpy(contract_bytecode_, bytecode, bytecode_len);

    return true;
}

__host__ void GPUFuzzer::set_config(const fuzzer_config_t& config) {
    config_ = config;
}

__host__ void GPUFuzzer::add_invariant(const invariant_t& inv) {
    if (invariant_checker_) {
        invariant_checker_->add_invariant(inv);
    }
}

__host__ void GPUFuzzer::add_seed(const uint8_t* calldata, uint32_t len) {
    if (!corpus_) return;

    seed_entry_t seed;
    seed.init();
    seed.data.length = len;
    cudaMallocManaged(&seed.data.data, len);
    memcpy(seed.data.data, calldata, len);
    seed.data.capacity = len;
    seed.num_transactions = 1;
    seed.tx_offsets[0] = 0;
    seed.tx_lengths[0] = len;

    corpus_->add_seed(seed);
}

__host__ void GPUFuzzer::generate_initial_seeds() {
    if (!corpus_) return;

    // Generate simple seeds
    // Empty calldata
    uint8_t empty[4] = {0, 0, 0, 0};
    add_seed(empty, 4);

    // Common function selectors with no args
    uint8_t selectors[][4] = {
        {0x06, 0xfd, 0xde, 0x03},  // name()
        {0x95, 0xd8, 0x9b, 0x41},  // symbol()
        {0x31, 0x3c, 0xe5, 0x67},  // decimals()
        {0x18, 0x16, 0x0d, 0xdd},  // totalSupply()
    };

    for (int i = 0; i < 4; i++) {
        add_seed(selectors[i], 4);
    }
}

__host__ void GPUFuzzer::run() {
    if (!initialized_ && !initialize()) {
        printf("Failed to initialize fuzzer\n");
        return;
    }

    running_ = true;
    uint32_t iteration = 0;

    printf("Starting GPU fuzzer on B300...\n");
    printf("Config: %u instances, %u sequence length\n",
           config_.num_instances, config_.sequence_length);

    while (running_ && !should_stop()) {
        // Single fuzzing iteration
        prepare_batch();
        execute_batch();
        analyze_batch();
        update_corpus();

        iteration++;
        stats_.total_iterations = iteration;

        // Periodic operations
        if (iteration % config_.stats_interval == 0) {
            report_progress();
        }

        maybe_cull_corpus();
        maybe_checkpoint();
    }

    printf("\nFuzzing complete.\n");
    print_stats();
}

__host__ void GPUFuzzer::run_iterations(uint32_t num_iterations) {
    if (!initialized_ && !initialize()) {
        return;
    }

    running_ = true;

    for (uint32_t i = 0; i < num_iterations && running_; i++) {
        prepare_batch();
        execute_batch();
        analyze_batch();
        update_corpus();

        stats_.total_iterations++;

        if ((i + 1) % config_.stats_interval == 0) {
            report_progress();
        }
    }
}

__host__ void GPUFuzzer::stop() {
    running_ = false;
}

__host__ void GPUFuzzer::prepare_batch() {
    batch_.reset();

    // Select seeds from corpus
    select_seeds_for_batch();

    // Mutate selected inputs
    mutate_batch();
}

__host__ void GPUFuzzer::execute_batch() {
    // Execute EVM instances on GPU
    // This would interface with CuEVM's kernel_evm_multiple_instances
    // For now, simulated

    stats_.total_executions += config_.num_instances;
    stats_.total_transactions += config_.num_instances * config_.sequence_length;
}

__host__ void GPUFuzzer::analyze_batch() {
    // Collect coverage
    collect_coverage();

    // Check oracles for bugs
    check_oracles();

    // Check invariants
    check_invariants();

    // Process interesting inputs
    process_interesting_inputs();
}

__host__ void GPUFuzzer::update_corpus() {
    // Update corpus with new interesting seeds
    // Handled in process_interesting_inputs
}

__host__ void GPUFuzzer::select_seeds_for_batch() {
    if (corpus_->size() == 0) {
        // No seeds in corpus, use default inputs
        for (uint32_t i = 0; i < config_.num_instances; i++) {
            batch_.inputs[i].length = 4;
            for (int j = 0; j < 4; j++) {
                batch_.inputs[i].data[j] = 0;
            }
        }
        return;
    }

    // Select seeds based on scheduling policy
    for (uint32_t i = 0; i < config_.num_instances; i++) {
        seed_entry_t* seed;
        if (config_.seed_schedule == 1) {
            seed = corpus_->select_weighted(&rng_state_.states[i]);
        } else {
            seed = corpus_->select_seed(&rng_state_.states[i]);
        }

        if (seed) {
            batch_.inputs[i].copy_from(seed->data);
        }
    }
}

__host__ void GPUFuzzer::mutate_batch() {
    mutation_engine_->mutate_batch(batch_.inputs, config_.num_instances,
                                   config_.mutations_per_seed, mutation_stream_);
    cudaStreamSynchronize(mutation_stream_);
}

__host__ void GPUFuzzer::collect_coverage() {
    // Merge instance coverage to global
    uint32_t blocks = (config_.num_instances + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    kernel_merge_coverage<<<blocks, THREADS_PER_BLOCK, 0, analysis_stream_>>>(
        global_coverage_, batch_.coverage, config_.num_instances
    );
    cudaStreamSynchronize(analysis_stream_);
}

__host__ void GPUFuzzer::check_oracles() {
    uint32_t blocks = (config_.num_instances + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    kernel_check_reentrancy<<<blocks, THREADS_PER_BLOCK, 0, analysis_stream_>>>(
        batch_.trackers, config_.num_instances, bugs_, &config_.oracle_config
    );
    cudaStreamSynchronize(analysis_stream_);
}

__host__ void GPUFuzzer::check_invariants() {
    // Check invariants on post-states
    // Would check against stored invariants
}

__host__ void GPUFuzzer::process_interesting_inputs() {
    // Find inputs that caused new coverage
    uint32_t prev_edges = stats_.unique_edges;

    // Count current coverage
    uint32_t new_edges = 0;
    for (uint32_t i = 0; i < EDGE_COVERAGE_SIZE; i++) {
        if (global_coverage_->edge_bitmap[i] > 0) new_edges++;
    }

    if (new_edges > prev_edges) {
        stats_.unique_edges = new_edges;
        stats_.last_new_coverage_iter = stats_.total_iterations;
        stats_.iterations_since_progress = 0;

        // Add interesting inputs to corpus
        // (Would track which inputs caused the new coverage)
        stats_.seeds_added++;
    } else {
        stats_.iterations_since_progress++;
    }

    // Check for new bugs
    if (bugs_->bug_count > stats_.total_bugs_found) {
        stats_.total_bugs_found = bugs_->bug_count;
        stats_.last_bug_iter = stats_.total_iterations;
        stats_.iterations_since_progress = 0;

        // Callback for new bug
        if (bug_callback_ && bugs_->bug_count > 0) {
            bug_callback_(&bugs_->bugs[bugs_->bug_count - 1], bug_callback_ctx_);
        }
    }
}

__host__ void GPUFuzzer::update_statistics() {
    auto now = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = now - start_time_;
    stats_.total_time_seconds = elapsed.count();

    if (stats_.total_time_seconds > 0) {
        stats_.executions_per_second = stats_.total_executions / stats_.total_time_seconds;
        stats_.transactions_per_second = stats_.total_transactions / stats_.total_time_seconds;
    }

    stats_.corpus_size = corpus_->size();
    stats_.update(*corpus_->get_stats(), *bugs_, *global_coverage_);
}

__host__ void GPUFuzzer::report_progress() {
    update_statistics();

    if (config_.verbose) {
        stats_.print_summary();
    }

    if (progress_callback_) {
        progress_callback_(&stats_, progress_callback_ctx_);
    }
}

__host__ void GPUFuzzer::maybe_cull_corpus() {
    if (config_.cull_interval > 0 &&
        stats_.total_iterations % config_.cull_interval == 0) {
        corpus_->cull_corpus();
    }
}

__host__ void GPUFuzzer::maybe_checkpoint() {
    if (config_.checkpoint_interval > 0 &&
        stats_.total_iterations % config_.checkpoint_interval == 0) {
        char filename[256];
        snprintf(filename, sizeof(filename), "checkpoint_%lu.bin",
                 stats_.total_iterations);
        save_checkpoint(filename);
    }
}

__host__ bool GPUFuzzer::should_stop() {
    if (config_.max_iterations > 0 &&
        stats_.total_iterations >= config_.max_iterations) {
        return true;
    }

    if (config_.max_time_seconds > 0 &&
        stats_.total_time_seconds >= config_.max_time_seconds) {
        return true;
    }

    if (config_.stall_threshold > 0 &&
        stats_.iterations_since_progress >= config_.stall_threshold) {
        printf("Stopping: No progress for %u iterations\n", config_.stall_threshold);
        return true;
    }

    return false;
}

__host__ void GPUFuzzer::print_stats() {
    update_statistics();
    stats_.print();
}

__host__ void GPUFuzzer::print_bugs() {
    print_bug_report(bugs_);
}

__host__ void GPUFuzzer::export_results(const char* directory) {
    char filename[512];

    // Export stats
    snprintf(filename, sizeof(filename), "%s/stats.json", directory);
    stats_.export_json(filename);

    // Export bugs
    snprintf(filename, sizeof(filename), "%s/bugs.json", directory);
    export_bugs_json(bugs_, filename);

    // Export coverage
    snprintf(filename, sizeof(filename), "%s/coverage.bin", directory);
    // Would save coverage bitmap

    // Export corpus
    snprintf(filename, sizeof(filename), "%s/corpus", directory);
    corpus_->export_seeds(filename);
}

__host__ void GPUFuzzer::save_checkpoint(const char* filename) {
    FILE* f = fopen(filename, "wb");
    if (!f) return;

    // Write stats
    fwrite(&stats_, sizeof(stats_), 1, f);

    // Write coverage
    fwrite(global_coverage_->edge_bitmap, EDGE_COVERAGE_SIZE, 1, f);

    // Write corpus info
    uint32_t corpus_size = corpus_->size();
    fwrite(&corpus_size, sizeof(corpus_size), 1, f);

    fclose(f);
}

__host__ void GPUFuzzer::load_checkpoint(const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (!f) return;

    // Read stats
    fread(&stats_, sizeof(stats_), 1, f);

    // Read coverage
    fread(global_coverage_->edge_bitmap, EDGE_COVERAGE_SIZE, 1, f);

    fclose(f);
}

__host__ void GPUFuzzer::set_progress_callback(progress_callback_t cb, void* ctx) {
    progress_callback_ = cb;
    progress_callback_ctx_ = ctx;
}

__host__ void GPUFuzzer::set_bug_callback(bug_callback_t cb, void* ctx) {
    bug_callback_ = cb;
    bug_callback_ctx_ = ctx;
}

// ============================================================================
// Convenience Functions
// ============================================================================

__host__ fuzzer_stats_t quick_fuzz(
    const char* contract_source,
    const char* contract_name,
    uint32_t num_iterations,
    uint32_t num_instances) {

    fuzzer_config_t config;
    config.set_for_b300();
    config.num_instances = num_instances;
    config.max_iterations = num_iterations;

    GPUFuzzer fuzzer(contract_source, contract_name, &config);
    fuzzer.initialize();
    fuzzer.generate_initial_seeds();
    fuzzer.run();

    return *fuzzer.get_stats();
}

__host__ fuzzer_stats_t fuzz_with_config(
    const char* contract_source,
    const char* contract_name,
    const fuzzer_config_t& config) {

    GPUFuzzer fuzzer(contract_source, contract_name, &config);
    fuzzer.initialize();
    fuzzer.generate_initial_seeds();
    fuzzer.run();

    return *fuzzer.get_stats();
}

// ============================================================================
// CUDA Kernel Implementations
// ============================================================================

__global__ void kernel_merge_batch_coverage(
    instance_coverage_t* instance_coverage,
    gpu_coverage_map_t* global_coverage,
    uint32_t num_instances,
    uint32_t* new_coverage_flags) {

    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_instances) return;

    instance_coverage_t* inst = &instance_coverage[idx];

    // Merge edge hashes
    for (uint32_t i = 0; i < inst->edge_hash_idx && i < 256; i++) {
        uint32_t hash = inst->edge_hashes[i];
        uint32_t bitmap_idx = hash % EDGE_COVERAGE_SIZE;

        uint8_t old_val = global_coverage->edge_bitmap[bitmap_idx];
        atomicAdd((unsigned char*)&global_coverage->edge_bitmap[bitmap_idx], 1);

        if (old_val == 0) {
            atomicExch(new_coverage_flags, 1);
        }
    }

    // Update global stats
    atomicAdd(&global_coverage->total_instructions_executed,
              (unsigned long long)inst->pcs_hit);
}

__global__ void kernel_run_oracles(
    CompositeOracle* oracle,
    execution_state_tracker_t* trackers,
    uint32_t num_instances,
    bug_storage_t* bugs) {

    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_instances) return;

    // Check for reentrancy in this instance
    if (trackers[idx].check_reentrancy()) {
        detected_bug_t bug;
        bug.type = BugType::REENTRANCY_ETH;
        bug.severity = BugSeverity::CRITICAL;
        bug.location.pc = 0;
        bug.location.tx_index = 0;
        bug.location.call_depth = trackers[idx].call_depth;
        bugs->add_bug(bug);
    }
}

__global__ void kernel_weighted_selection(
    seed_entry_t* seeds,
    uint32_t num_seeds,
    uint32_t* cumulative_weights,
    uint32_t* selected_indices,
    uint32_t num_to_select,
    curandState* rng) {

    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_to_select) return;

    uint32_t total_weight = cumulative_weights[num_seeds - 1];
    uint32_t rand_val = curand(&rng[idx]) % total_weight;

    // Binary search for the selected seed
    uint32_t low = 0, high = num_seeds - 1;
    while (low < high) {
        uint32_t mid = (low + high) / 2;
        if (cumulative_weights[mid] <= rand_val) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }

    selected_indices[idx] = low;
}

}  // namespace fuzzing
}  // namespace CuEVM
