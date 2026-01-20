// CuEVM: CUDA Ethereum Virtual Machine implementation
// Comprehensive Oracle and Bug Detection for Smart Contract Fuzzing
// SPDX-License-Identifier: MIT

#ifndef _CUEVM_FUZZING_ORACLE_H_
#define _CUEVM_FUZZING_ORACLE_H_

#include <cuda_runtime.h>
#include <cstdint>
#include <CuEVM/utils/arith.cuh>
#include <CuEVM/fuzzing/coverage.cuh>

namespace CuEVM {
namespace fuzzing {

// ============================================================================
// Bug Types and Severity Levels
// ============================================================================

enum class BugType : uint8_t {
    // Arithmetic vulnerabilities
    INTEGER_OVERFLOW = 0,
    INTEGER_UNDERFLOW = 1,
    DIVISION_BY_ZERO = 2,
    MODULO_BY_ZERO = 3,
    EXPONENT_OVERFLOW = 4,

    // Access control vulnerabilities
    UNAUTHORIZED_CALL = 10,
    UNAUTHORIZED_SELFDESTRUCT = 11,
    UNAUTHORIZED_DELEGATECALL = 12,
    TX_ORIGIN_AUTH = 13,
    MISSING_ACCESS_CONTROL = 14,

    // Reentrancy vulnerabilities
    REENTRANCY_ETH = 20,
    REENTRANCY_ERC20 = 21,
    REENTRANCY_CROSS_FUNCTION = 22,
    REENTRANCY_CROSS_CONTRACT = 23,
    READ_ONLY_REENTRANCY = 24,

    // State manipulation
    UNINITIALIZED_STORAGE = 30,
    STORAGE_COLLISION = 31,
    DIRTY_HIGH_BITS = 32,
    UNCHECKED_RETURN = 33,

    // Token vulnerabilities
    ERC20_APPROVAL_RACE = 40,
    ERC20_TRANSFER_TO_ZERO = 41,
    ERC20_BURN_WITHOUT_APPROVAL = 42,
    ERC721_UNAUTHORIZED_TRANSFER = 43,
    TOKEN_BALANCE_MANIPULATION = 44,

    // Oracle/price manipulation
    ORACLE_MANIPULATION = 50,
    FLASHLOAN_ATTACK = 51,
    SANDWICH_VULNERABLE = 52,
    SLIPPAGE_VULNERABILITY = 53,

    // Gas vulnerabilities
    BLOCK_GAS_LIMIT = 60,
    UNBOUNDED_LOOP = 61,
    GAS_GRIEFING = 62,
    OUT_OF_GAS_CALL = 63,

    // Fund safety
    ETHER_LEAK = 70,
    STUCK_ETHER = 71,
    UNEXPECTED_ETH_BALANCE = 72,
    FORCE_FEED_VULNERABLE = 73,
    SELFDESTRUCT_ETH_LEAK = 74,

    // Logic bugs
    ASSERTION_VIOLATION = 80,
    INVARIANT_VIOLATION = 81,
    STATE_INCONSISTENCY = 82,
    UNEXPECTED_REVERT = 83,

    // External interaction issues
    EXTERNAL_CALL_FAILURE = 90,
    UNTRUSTED_EXTERNAL_CALL = 91,
    RETURN_DATA_MANIPULATION = 92,

    // Signature/crypto issues
    SIGNATURE_REPLAY = 100,
    SIGNATURE_MALLEABILITY = 101,
    WEAK_RANDOMNESS = 102,

    // Proxy pattern issues
    UNINITIALIZED_PROXY = 110,
    STORAGE_SLOT_COLLISION = 111,
    IMPLEMENTATION_DESTROYED = 112,

    // Custom/unknown
    CUSTOM_ORACLE_VIOLATION = 200,
    UNKNOWN = 255
};

enum class BugSeverity : uint8_t {
    INFORMATIONAL = 0,
    LOW = 1,
    MEDIUM = 2,
    HIGH = 3,
    CRITICAL = 4
};

// ============================================================================
// Bug Detection Result
// ============================================================================

struct bug_location_t {
    uint32_t pc;                    // Program counter where bug occurred
    uint32_t tx_index;              // Transaction index in sequence
    uint32_t call_depth;            // Call stack depth
    uint32_t contract_id;           // Contract identifier
    uint8_t opcode;                 // Opcode that triggered the bug
};

struct bug_context_t {
    evm_word_t operand1;            // First operand (for arithmetic bugs)
    evm_word_t operand2;            // Second operand
    evm_word_t result;              // Result value
    evm_word_t expected;            // Expected value (for invariant checks)
    evm_word_t caller;              // msg.sender
    evm_word_t callee;              // Call target
    evm_word_t value;               // msg.value
    uint8_t context_data[256];      // Additional context
    uint32_t context_length;
};

struct detected_bug_t {
    BugType type;
    BugSeverity severity;
    bug_location_t location;
    bug_context_t context;
    uint64_t timestamp;             // When the bug was detected
    uint64_t input_hash;            // Hash of input that triggered the bug
    uint32_t sequence_id;           // Sequence that triggered the bug
    bool confirmed;                 // Whether bug was confirmed on replay
    char description[256];          // Human-readable description
};

// ============================================================================
// Oracle Configuration
// ============================================================================

struct oracle_config_t {
    // Arithmetic checks
    bool check_overflow;
    bool check_underflow;
    bool check_div_zero;

    // Access control checks
    bool check_unauthorized_access;
    bool check_tx_origin;
    bool check_selfdestruct;

    // Reentrancy checks
    bool check_reentrancy;
    bool check_cross_function_reentrancy;
    bool check_read_only_reentrancy;

    // Token checks
    bool check_erc20_issues;
    bool check_erc721_issues;

    // Fund safety checks
    bool check_ether_leak;
    bool check_stuck_ether;
    bool check_force_feed;

    // Gas checks
    bool check_gas_issues;

    // Severity threshold (only report bugs >= this severity)
    BugSeverity min_severity;

    // Maximum bugs to track per type
    uint32_t max_bugs_per_type;

    // Deduplication window
    uint32_t dedup_window_size;

    __host__ __device__ void set_default();
    __host__ __device__ void enable_all();
    __host__ __device__ void set_minimal();
};

// ============================================================================
// Bug Storage
// ============================================================================

constexpr uint32_t MAX_BUGS_TOTAL = 4096;
constexpr uint32_t MAX_BUGS_PER_TYPE = 256;

struct bug_storage_t {
    detected_bug_t bugs[MAX_BUGS_TOTAL];
    uint32_t bug_count;

    // Deduplication - track recent bug signatures
    uint64_t recent_signatures[1024];
    uint32_t signature_idx;

    // Per-type counts
    uint32_t type_counts[(uint32_t)BugType::UNKNOWN + 1];

    __host__ __device__ void init();
    __host__ __device__ bool add_bug(const detected_bug_t& bug);
    __host__ __device__ bool is_duplicate(uint64_t signature);
    __host__ __device__ uint32_t count_by_type(BugType type);
    __host__ __device__ uint32_t count_by_severity(BugSeverity severity);
    __host__ __device__ void clear();
};

// ============================================================================
// Execution State Tracker (for reentrancy detection)
// ============================================================================

constexpr uint32_t MAX_CALL_DEPTH = 64;
constexpr uint32_t MAX_STORAGE_WRITES = 256;

struct call_frame_t {
    evm_word_t caller;
    evm_word_t callee;
    evm_word_t value;
    uint32_t pc;
    uint8_t opcode;                 // CALL, CALLCODE, DELEGATECALL, STATICCALL
    bool has_state_change;          // Whether state was modified before call
    bool is_external;               // Whether call is to external contract
};

struct storage_write_t {
    evm_word_t address;
    evm_word_t slot;
    evm_word_t old_value;
    evm_word_t new_value;
    uint32_t pc;
    uint32_t call_depth;
};

struct execution_state_tracker_t {
    // Call stack
    call_frame_t call_stack[MAX_CALL_DEPTH];
    uint32_t call_depth;

    // Storage writes (for reentrancy detection)
    storage_write_t storage_writes[MAX_STORAGE_WRITES];
    uint32_t num_storage_writes;

    // Balance tracking
    evm_word_t initial_balances[64];    // Track initial balances
    evm_word_t current_balances[64];    // Current balances
    uint32_t num_tracked_addresses;

    // Reentrancy detection
    bool in_external_call;
    bool state_modified_before_call;
    uint32_t reentrancy_guard_slot;     // If we detect a reentrancy guard

    // Gas tracking
    uint64_t initial_gas;
    uint64_t gas_used;

    // Return value tracking
    bool last_call_success;
    bool last_call_checked;

    __host__ __device__ void init();
    __host__ __device__ void push_call(const call_frame_t& frame);
    __host__ __device__ void pop_call();
    __host__ __device__ void record_storage_write(const storage_write_t& write);
    __host__ __device__ bool check_reentrancy();
    __host__ __device__ void track_balance(const evm_word_t& address, const evm_word_t& balance);
};

// ============================================================================
// Oracle Detector Base Class
// ============================================================================

class OracleDetector {
public:
    __host__ __device__ OracleDetector(oracle_config_t* config, bug_storage_t* storage);

    // Pre-execution hooks
    __host__ __device__ void on_transaction_start(const evm_word_t& sender, const evm_word_t& receiver,
                                                   const evm_word_t& value, const uint8_t* calldata, uint32_t calldata_len);

    // Instruction-level hooks
    __host__ __device__ void on_instruction(uint32_t pc, uint8_t opcode,
                                            const evm_word_t* stack, uint32_t stack_size,
                                            execution_state_tracker_t* tracker);

    // Arithmetic operation hooks
    __host__ __device__ void check_add(uint32_t pc, const evm_word_t& a, const evm_word_t& b,
                                       const evm_word_t& result);
    __host__ __device__ void check_sub(uint32_t pc, const evm_word_t& a, const evm_word_t& b,
                                       const evm_word_t& result);
    __host__ __device__ void check_mul(uint32_t pc, const evm_word_t& a, const evm_word_t& b,
                                       const evm_word_t& result);
    __host__ __device__ void check_div(uint32_t pc, const evm_word_t& a, const evm_word_t& b);
    __host__ __device__ void check_mod(uint32_t pc, const evm_word_t& a, const evm_word_t& b);
    __host__ __device__ void check_exp(uint32_t pc, const evm_word_t& base, const evm_word_t& exp,
                                       const evm_word_t& result);

    // Storage hooks
    __host__ __device__ void on_sload(uint32_t pc, const evm_word_t& slot, const evm_word_t& value,
                                      execution_state_tracker_t* tracker);
    __host__ __device__ void on_sstore(uint32_t pc, const evm_word_t& slot,
                                       const evm_word_t& old_value, const evm_word_t& new_value,
                                       execution_state_tracker_t* tracker);

    // Call hooks
    __host__ __device__ void on_call_start(uint32_t pc, uint8_t opcode,
                                           const evm_word_t& target, const evm_word_t& value,
                                           const evm_word_t& gas,
                                           execution_state_tracker_t* tracker);
    __host__ __device__ void on_call_end(uint32_t pc, bool success, const uint8_t* return_data,
                                         uint32_t return_size, execution_state_tracker_t* tracker);

    // Balance hooks
    __host__ __device__ void on_balance_change(const evm_word_t& address,
                                               const evm_word_t& old_balance, const evm_word_t& new_balance);

    // Special instruction hooks
    __host__ __device__ void on_selfdestruct(uint32_t pc, const evm_word_t& beneficiary,
                                             const evm_word_t& balance);
    __host__ __device__ void on_create(uint32_t pc, const evm_word_t& value,
                                       const evm_word_t& new_address);
    __host__ __device__ void on_origin(uint32_t pc);

    // Post-execution hooks
    __host__ __device__ void on_transaction_end(bool success, const uint8_t* return_data,
                                                uint32_t return_size, uint64_t gas_used,
                                                execution_state_tracker_t* tracker);

    // Invariant checking
    __host__ __device__ void check_custom_invariant(uint32_t invariant_id, bool condition,
                                                    const char* description);

    // Get results
    __host__ __device__ bug_storage_t* get_bugs() { return storage_; }
    __host__ __device__ uint32_t get_bug_count() { return storage_->bug_count; }

protected:
    oracle_config_t* config_;
    bug_storage_t* storage_;
    uint32_t current_tx_index_;
    uint32_t current_sequence_id_;
    evm_word_t current_sender_;
    evm_word_t current_receiver_;

    __host__ __device__ void report_bug(BugType type, BugSeverity severity,
                                        const bug_location_t& location,
                                        const bug_context_t& context,
                                        const char* description);

    __host__ __device__ uint64_t compute_bug_signature(BugType type, uint32_t pc,
                                                       const evm_word_t& key_value);

    __host__ __device__ BugSeverity determine_severity(BugType type, const bug_context_t& context);

private:
    // Reentrancy detection helpers
    __host__ __device__ bool is_reentrancy_safe_call(uint8_t opcode, const evm_word_t& target);
    __host__ __device__ bool is_reentrancy_guard_pattern(const evm_word_t& slot,
                                                         const evm_word_t& old_value,
                                                         const evm_word_t& new_value);

    // Arithmetic overflow detection helpers
    __host__ __device__ bool check_add_overflow(const evm_word_t& a, const evm_word_t& b);
    __host__ __device__ bool check_mul_overflow(const evm_word_t& a, const evm_word_t& b);
    __host__ __device__ bool check_sub_underflow(const evm_word_t& a, const evm_word_t& b);
};

// ============================================================================
// Specialized Oracles
// ============================================================================

/**
 * Integer overflow/underflow detector
 */
class ArithmeticOracle : public OracleDetector {
public:
    __host__ __device__ ArithmeticOracle(oracle_config_t* config, bug_storage_t* storage);

    // Safe math verification
    __host__ __device__ void verify_safe_add(uint32_t pc, const evm_word_t& a, const evm_word_t& b,
                                             const evm_word_t& result);
    __host__ __device__ void verify_safe_sub(uint32_t pc, const evm_word_t& a, const evm_word_t& b,
                                             const evm_word_t& result);
    __host__ __device__ void verify_safe_mul(uint32_t pc, const evm_word_t& a, const evm_word_t& b,
                                             const evm_word_t& result);
};

/**
 * Reentrancy vulnerability detector
 */
class ReentrancyOracle : public OracleDetector {
public:
    __host__ __device__ ReentrancyOracle(oracle_config_t* config, bug_storage_t* storage);

    __host__ __device__ void track_external_call(uint32_t pc, const evm_word_t& target,
                                                 execution_state_tracker_t* tracker);
    __host__ __device__ void track_state_modification(uint32_t pc, const evm_word_t& slot,
                                                      execution_state_tracker_t* tracker);
    __host__ __device__ void check_reentrancy_pattern(execution_state_tracker_t* tracker);

private:
    // Known reentrancy guard patterns
    bool has_reentrancy_guard_;
    evm_word_t guard_slot_;
};

/**
 * Access control vulnerability detector
 */
class AccessControlOracle : public OracleDetector {
public:
    __host__ __device__ AccessControlOracle(oracle_config_t* config, bug_storage_t* storage);

    // Track privileged operations
    __host__ __device__ void on_privileged_operation(uint32_t pc, uint8_t opcode,
                                                     const evm_word_t& sender);

    // Track authorization checks
    __host__ __device__ void on_authorization_check(uint32_t pc, const evm_word_t& checked_address);

    // Verify access control
    __host__ __device__ void verify_access_control(uint32_t pc, uint8_t operation);

private:
    bool authorization_checked_;
    evm_word_t authorized_addresses_[16];
    uint32_t num_authorized_;
};

/**
 * ERC20/Token vulnerability detector
 */
class TokenOracle : public OracleDetector {
public:
    __host__ __device__ TokenOracle(oracle_config_t* config, bug_storage_t* storage);

    // ERC20 specific checks
    __host__ __device__ void check_transfer(uint32_t pc, const evm_word_t& from,
                                            const evm_word_t& to, const evm_word_t& amount);
    __host__ __device__ void check_approve(uint32_t pc, const evm_word_t& owner,
                                           const evm_word_t& spender, const evm_word_t& amount);
    __host__ __device__ void check_transferFrom(uint32_t pc, const evm_word_t& from,
                                                const evm_word_t& to, const evm_word_t& amount,
                                                const evm_word_t& allowance);

    // Balance consistency
    __host__ __device__ void track_balance_change(const evm_word_t& address,
                                                  const evm_word_t& old_balance,
                                                  const evm_word_t& new_balance);
    __host__ __device__ void check_total_supply_consistency();

private:
    evm_word_t tracked_total_supply_;
    uint32_t total_supply_slot_;
};

/**
 * Fund safety oracle (Ether leak detection)
 */
class FundSafetyOracle : public OracleDetector {
public:
    __host__ __device__ FundSafetyOracle(oracle_config_t* config, bug_storage_t* storage);

    // Track ETH flow
    __host__ __device__ void on_eth_received(const evm_word_t& from, const evm_word_t& amount);
    __host__ __device__ void on_eth_sent(uint32_t pc, const evm_word_t& to, const evm_word_t& amount);

    // Check for stuck ETH
    __host__ __device__ void check_stuck_ether(const evm_word_t& contract_balance);

    // Check for unexpected ETH
    __host__ __device__ void check_unexpected_eth(const evm_word_t& expected, const evm_word_t& actual);

    // Selfdestruct checks
    __host__ __device__ void check_selfdestruct_safety(uint32_t pc, const evm_word_t& beneficiary);

private:
    evm_word_t total_eth_received_;
    evm_word_t total_eth_sent_;
    bool has_withdrawal_function_;
};

/**
 * Gas-related vulnerability detector
 */
class GasOracle : public OracleDetector {
public:
    __host__ __device__ GasOracle(oracle_config_t* config, bug_storage_t* storage);

    // Track gas usage
    __host__ __device__ void on_gas_usage(uint32_t pc, uint64_t gas_used, uint64_t gas_remaining);

    // Detect potential DoS
    __host__ __device__ void check_unbounded_loop(uint32_t pc, uint32_t iteration_count);
    __host__ __device__ void check_block_gas_limit(uint64_t total_gas);

    // External call gas checks
    __host__ __device__ void check_call_gas(uint32_t pc, uint64_t gas_forwarded);

private:
    uint64_t max_gas_observed_;
    uint32_t loop_iteration_counts_[64];
    uint32_t loop_pcs_[64];
    uint32_t num_loops_;
};

// ============================================================================
// Composite Oracle (combines all detectors)
// ============================================================================

class CompositeOracle {
public:
    __host__ __device__ CompositeOracle(oracle_config_t* config, bug_storage_t* storage);

    // Initialize all sub-oracles
    __host__ __device__ void init();

    // Forward hooks to all active oracles
    __host__ __device__ void on_transaction_start(const evm_word_t& sender, const evm_word_t& receiver,
                                                   const evm_word_t& value, const uint8_t* calldata,
                                                   uint32_t calldata_len);
    __host__ __device__ void on_instruction(uint32_t pc, uint8_t opcode,
                                            const evm_word_t* stack, uint32_t stack_size,
                                            execution_state_tracker_t* tracker);
    __host__ __device__ void on_transaction_end(bool success, const uint8_t* return_data,
                                                uint32_t return_size, uint64_t gas_used,
                                                execution_state_tracker_t* tracker);

    // Get combined results
    __host__ __device__ bug_storage_t* get_bugs() { return storage_; }

private:
    oracle_config_t* config_;
    bug_storage_t* storage_;

    ArithmeticOracle arithmetic_;
    ReentrancyOracle reentrancy_;
    AccessControlOracle access_control_;
    TokenOracle token_;
    FundSafetyOracle fund_safety_;
    GasOracle gas_;
};

// ============================================================================
// CUDA Kernels for Batch Oracle Checking
// ============================================================================

__global__ void kernel_check_arithmetic(
    uint8_t opcode,
    const evm_word_t* operands_a,
    const evm_word_t* operands_b,
    const evm_word_t* results,
    uint32_t* pcs,
    uint32_t num_operations,
    bug_storage_t* bug_storage,
    oracle_config_t* config
);

__global__ void kernel_check_reentrancy(
    execution_state_tracker_t* trackers,
    uint32_t num_instances,
    bug_storage_t* bug_storage,
    oracle_config_t* config
);

__global__ void kernel_check_invariants(
    const evm_word_t* pre_state,
    const evm_word_t* post_state,
    const uint32_t* invariant_types,
    uint32_t num_invariants,
    bug_storage_t* bug_storage
);

// ============================================================================
// Host Helper Functions
// ============================================================================

__host__ oracle_config_t* allocate_oracle_config();
__host__ bug_storage_t* allocate_bug_storage();
__host__ execution_state_tracker_t* allocate_trackers(uint32_t num_instances);
__host__ void free_oracle_config(oracle_config_t* config);
__host__ void free_bug_storage(bug_storage_t* storage);
__host__ void free_trackers(execution_state_tracker_t* trackers);

__host__ void copy_bugs_to_host(detected_bug_t* host_bugs, const bug_storage_t* device_storage);
__host__ void print_bug_report(const bug_storage_t* storage);
__host__ void export_bugs_json(const bug_storage_t* storage, const char* filename);

}  // namespace fuzzing
}  // namespace CuEVM

#endif  // _CUEVM_FUZZING_ORACLE_H_
