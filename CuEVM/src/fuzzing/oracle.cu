// CuEVM: CUDA Ethereum Virtual Machine implementation
// Comprehensive Oracle and Bug Detection Implementation
// SPDX-License-Identifier: MIT

#include <CuEVM/fuzzing/oracle.cuh>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>

namespace CuEVM {
namespace fuzzing {

// EVM Opcodes for reference
constexpr uint8_t OP_ADD = 0x01;
constexpr uint8_t OP_MUL = 0x02;
constexpr uint8_t OP_SUB = 0x03;
constexpr uint8_t OP_DIV = 0x04;
constexpr uint8_t OP_SDIV = 0x05;
constexpr uint8_t OP_MOD = 0x06;
constexpr uint8_t OP_SMOD = 0x07;
constexpr uint8_t OP_EXP = 0x0A;
constexpr uint8_t OP_SLOAD = 0x54;
constexpr uint8_t OP_SSTORE = 0x55;
constexpr uint8_t OP_CALL = 0xF1;
constexpr uint8_t OP_CALLCODE = 0xF2;
constexpr uint8_t OP_DELEGATECALL = 0xF4;
constexpr uint8_t OP_STATICCALL = 0xFA;
constexpr uint8_t OP_CREATE = 0xF0;
constexpr uint8_t OP_CREATE2 = 0xF5;
constexpr uint8_t OP_SELFDESTRUCT = 0xFF;
constexpr uint8_t OP_ORIGIN = 0x32;
constexpr uint8_t OP_CALLER = 0x33;

// ============================================================================
// Helper Functions for 256-bit Arithmetic
// ============================================================================

__host__ __device__ bool is_zero(const evm_word_t& val) {
    for (int i = 0; i < 8; i++) {
        if (val.limbs[i] != 0) return false;
    }
    return true;
}

__host__ __device__ bool equals(const evm_word_t& a, const evm_word_t& b) {
    for (int i = 0; i < 8; i++) {
        if (a.limbs[i] != b.limbs[i]) return false;
    }
    return true;
}

__host__ __device__ bool less_than(const evm_word_t& a, const evm_word_t& b) {
    for (int i = 7; i >= 0; i--) {
        if (a.limbs[i] < b.limbs[i]) return true;
        if (a.limbs[i] > b.limbs[i]) return false;
    }
    return false;
}

__host__ __device__ bool greater_than(const evm_word_t& a, const evm_word_t& b) {
    for (int i = 7; i >= 0; i--) {
        if (a.limbs[i] > b.limbs[i]) return true;
        if (a.limbs[i] < b.limbs[i]) return false;
    }
    return false;
}

__host__ __device__ void copy_word(evm_word_t& dst, const evm_word_t& src) {
    for (int i = 0; i < 8; i++) {
        dst.limbs[i] = src.limbs[i];
    }
}

__host__ __device__ void zero_word(evm_word_t& val) {
    for (int i = 0; i < 8; i++) {
        val.limbs[i] = 0;
    }
}

__host__ __device__ uint64_t hash_word(const evm_word_t& val) {
    uint64_t hash = 0;
    for (int i = 0; i < 8; i++) {
        hash ^= ((uint64_t)val.limbs[i]) << ((i & 1) * 32);
        hash = (hash << 7) | (hash >> 57);
    }
    return hash;
}

// ============================================================================
// Oracle Configuration Implementation
// ============================================================================

__host__ __device__ void oracle_config_t::set_default() {
    check_overflow = true;
    check_underflow = true;
    check_div_zero = true;
    check_unauthorized_access = true;
    check_tx_origin = true;
    check_selfdestruct = true;
    check_reentrancy = true;
    check_cross_function_reentrancy = true;
    check_read_only_reentrancy = false;
    check_erc20_issues = true;
    check_erc721_issues = false;
    check_ether_leak = true;
    check_stuck_ether = true;
    check_force_feed = true;
    check_gas_issues = true;
    min_severity = BugSeverity::LOW;
    max_bugs_per_type = MAX_BUGS_PER_TYPE;
    dedup_window_size = 1024;
}

__host__ __device__ void oracle_config_t::enable_all() {
    check_overflow = true;
    check_underflow = true;
    check_div_zero = true;
    check_unauthorized_access = true;
    check_tx_origin = true;
    check_selfdestruct = true;
    check_reentrancy = true;
    check_cross_function_reentrancy = true;
    check_read_only_reentrancy = true;
    check_erc20_issues = true;
    check_erc721_issues = true;
    check_ether_leak = true;
    check_stuck_ether = true;
    check_force_feed = true;
    check_gas_issues = true;
    min_severity = BugSeverity::INFORMATIONAL;
    max_bugs_per_type = MAX_BUGS_PER_TYPE;
    dedup_window_size = 1024;
}

__host__ __device__ void oracle_config_t::set_minimal() {
    check_overflow = true;
    check_underflow = true;
    check_div_zero = false;
    check_unauthorized_access = false;
    check_tx_origin = false;
    check_selfdestruct = true;
    check_reentrancy = true;
    check_cross_function_reentrancy = false;
    check_read_only_reentrancy = false;
    check_erc20_issues = false;
    check_erc721_issues = false;
    check_ether_leak = true;
    check_stuck_ether = false;
    check_force_feed = false;
    check_gas_issues = false;
    min_severity = BugSeverity::HIGH;
    max_bugs_per_type = 64;
    dedup_window_size = 256;
}

// ============================================================================
// Bug Storage Implementation
// ============================================================================

__host__ __device__ void bug_storage_t::init() {
    bug_count = 0;
    signature_idx = 0;
    for (int i = 0; i <= (int)BugType::UNKNOWN; i++) {
        type_counts[i] = 0;
    }
    for (int i = 0; i < 1024; i++) {
        recent_signatures[i] = 0;
    }
}

__host__ __device__ bool bug_storage_t::is_duplicate(uint64_t signature) {
    for (uint32_t i = 0; i < 1024; i++) {
        if (recent_signatures[i] == signature) {
            return true;
        }
    }
    return false;
}

__host__ __device__ bool bug_storage_t::add_bug(const detected_bug_t& bug) {
    // Compute signature for deduplication
    uint64_t signature = hash_word(bug.context.operand1) ^
                         ((uint64_t)bug.type << 56) ^
                         ((uint64_t)bug.location.pc << 32);

    // Check for duplicate
    if (is_duplicate(signature)) {
        return false;
    }

    // Check if we have space
    if (bug_count >= MAX_BUGS_TOTAL) {
        return false;
    }

    // Check per-type limit
    if (type_counts[(int)bug.type] >= MAX_BUGS_PER_TYPE) {
        return false;
    }

    // Add bug
#ifdef __CUDA_ARCH__
    uint32_t idx = atomicAdd(&bug_count, 1);
    if (idx >= MAX_BUGS_TOTAL) {
        atomicSub(&bug_count, 1);
        return false;
    }
    atomicAdd(&type_counts[(int)bug.type], 1);
#else
    uint32_t idx = bug_count++;
    type_counts[(int)bug.type]++;
#endif

    bugs[idx] = bug;

    // Add to dedup window
    recent_signatures[signature_idx % 1024] = signature;
    signature_idx++;

    return true;
}

__host__ __device__ uint32_t bug_storage_t::count_by_type(BugType type) {
    return type_counts[(int)type];
}

__host__ __device__ uint32_t bug_storage_t::count_by_severity(BugSeverity severity) {
    uint32_t count = 0;
    for (uint32_t i = 0; i < bug_count; i++) {
        if (bugs[i].severity >= severity) count++;
    }
    return count;
}

__host__ __device__ void bug_storage_t::clear() {
    init();
}

// ============================================================================
// Execution State Tracker Implementation
// ============================================================================

__host__ __device__ void execution_state_tracker_t::init() {
    call_depth = 0;
    num_storage_writes = 0;
    num_tracked_addresses = 0;
    in_external_call = false;
    state_modified_before_call = false;
    reentrancy_guard_slot = 0;
    initial_gas = 0;
    gas_used = 0;
    last_call_success = false;
    last_call_checked = true;
}

__host__ __device__ void execution_state_tracker_t::push_call(const call_frame_t& frame) {
    if (call_depth < MAX_CALL_DEPTH) {
        call_stack[call_depth] = frame;
        call_depth++;
        if (frame.is_external) {
            in_external_call = true;
        }
    }
}

__host__ __device__ void execution_state_tracker_t::pop_call() {
    if (call_depth > 0) {
        call_depth--;
        if (call_depth == 0) {
            in_external_call = false;
        }
    }
}

__host__ __device__ void execution_state_tracker_t::record_storage_write(const storage_write_t& write) {
    if (num_storage_writes < MAX_STORAGE_WRITES) {
        storage_writes[num_storage_writes++] = write;
        state_modified_before_call = true;
    }
}

__host__ __device__ bool execution_state_tracker_t::check_reentrancy() {
    // Check if we're in an external call and state was modified before
    if (in_external_call && state_modified_before_call) {
        // Check if any storage was written before the call and after
        for (uint32_t i = 0; i < num_storage_writes; i++) {
            if (storage_writes[i].call_depth < call_depth) {
                // Storage write happened before current call depth
                return true;  // Potential reentrancy
            }
        }
    }
    return false;
}

__host__ __device__ void execution_state_tracker_t::track_balance(const evm_word_t& address,
                                                                   const evm_word_t& balance) {
    // Find existing or add new
    for (uint32_t i = 0; i < num_tracked_addresses; i++) {
        if (equals(initial_balances[i], address)) {
            copy_word(current_balances[i], balance);
            return;
        }
    }
    if (num_tracked_addresses < 64) {
        copy_word(initial_balances[num_tracked_addresses], address);
        copy_word(current_balances[num_tracked_addresses], balance);
        num_tracked_addresses++;
    }
}

// ============================================================================
// Oracle Detector Implementation
// ============================================================================

__host__ __device__ OracleDetector::OracleDetector(oracle_config_t* config, bug_storage_t* storage)
    : config_(config), storage_(storage), current_tx_index_(0), current_sequence_id_(0) {
    zero_word(current_sender_);
    zero_word(current_receiver_);
}

__host__ __device__ void OracleDetector::on_transaction_start(
    const evm_word_t& sender, const evm_word_t& receiver,
    const evm_word_t& value, const uint8_t* calldata, uint32_t calldata_len) {
    copy_word(current_sender_, sender);
    copy_word(current_receiver_, receiver);
}

__host__ __device__ void OracleDetector::on_instruction(
    uint32_t pc, uint8_t opcode,
    const evm_word_t* stack, uint32_t stack_size,
    execution_state_tracker_t* tracker) {

    // Handle different opcodes
    switch (opcode) {
        case OP_ADD:
            if (stack_size >= 2 && config_->check_overflow) {
                check_add(pc, stack[stack_size - 1], stack[stack_size - 2], stack[stack_size - 1]);
            }
            break;
        case OP_SUB:
            if (stack_size >= 2 && config_->check_underflow) {
                check_sub(pc, stack[stack_size - 1], stack[stack_size - 2], stack[stack_size - 1]);
            }
            break;
        case OP_MUL:
            if (stack_size >= 2 && config_->check_overflow) {
                check_mul(pc, stack[stack_size - 1], stack[stack_size - 2], stack[stack_size - 1]);
            }
            break;
        case OP_DIV:
        case OP_SDIV:
            if (stack_size >= 2 && config_->check_div_zero) {
                check_div(pc, stack[stack_size - 1], stack[stack_size - 2]);
            }
            break;
        case OP_MOD:
        case OP_SMOD:
            if (stack_size >= 2 && config_->check_div_zero) {
                check_mod(pc, stack[stack_size - 1], stack[stack_size - 2]);
            }
            break;
        case OP_ORIGIN:
            if (config_->check_tx_origin) {
                on_origin(pc);
            }
            break;
        default:
            break;
    }
}

__host__ __device__ void OracleDetector::check_add(uint32_t pc, const evm_word_t& a, const evm_word_t& b,
                                                   const evm_word_t& result) {
    if (check_add_overflow(a, b)) {
        bug_location_t location;
        location.pc = pc;
        location.tx_index = current_tx_index_;
        location.call_depth = 0;
        location.contract_id = 0;
        location.opcode = OP_ADD;

        bug_context_t context;
        copy_word(context.operand1, a);
        copy_word(context.operand2, b);
        copy_word(context.result, result);
        context.context_length = 0;

        report_bug(BugType::INTEGER_OVERFLOW, BugSeverity::HIGH, location, context,
                   "Integer overflow in ADD operation");
    }
}

__host__ __device__ void OracleDetector::check_sub(uint32_t pc, const evm_word_t& a, const evm_word_t& b,
                                                   const evm_word_t& result) {
    if (check_sub_underflow(a, b)) {
        bug_location_t location;
        location.pc = pc;
        location.tx_index = current_tx_index_;
        location.call_depth = 0;
        location.contract_id = 0;
        location.opcode = OP_SUB;

        bug_context_t context;
        copy_word(context.operand1, a);
        copy_word(context.operand2, b);
        copy_word(context.result, result);
        context.context_length = 0;

        report_bug(BugType::INTEGER_UNDERFLOW, BugSeverity::HIGH, location, context,
                   "Integer underflow in SUB operation");
    }
}

__host__ __device__ void OracleDetector::check_mul(uint32_t pc, const evm_word_t& a, const evm_word_t& b,
                                                   const evm_word_t& result) {
    if (check_mul_overflow(a, b)) {
        bug_location_t location;
        location.pc = pc;
        location.tx_index = current_tx_index_;
        location.call_depth = 0;
        location.contract_id = 0;
        location.opcode = OP_MUL;

        bug_context_t context;
        copy_word(context.operand1, a);
        copy_word(context.operand2, b);
        copy_word(context.result, result);
        context.context_length = 0;

        report_bug(BugType::INTEGER_OVERFLOW, BugSeverity::HIGH, location, context,
                   "Integer overflow in MUL operation");
    }
}

__host__ __device__ void OracleDetector::check_div(uint32_t pc, const evm_word_t& a, const evm_word_t& b) {
    if (is_zero(b)) {
        bug_location_t location;
        location.pc = pc;
        location.tx_index = current_tx_index_;
        location.call_depth = 0;
        location.contract_id = 0;
        location.opcode = OP_DIV;

        bug_context_t context;
        copy_word(context.operand1, a);
        copy_word(context.operand2, b);
        context.context_length = 0;

        report_bug(BugType::DIVISION_BY_ZERO, BugSeverity::MEDIUM, location, context,
                   "Division by zero");
    }
}

__host__ __device__ void OracleDetector::check_mod(uint32_t pc, const evm_word_t& a, const evm_word_t& b) {
    if (is_zero(b)) {
        bug_location_t location;
        location.pc = pc;
        location.tx_index = current_tx_index_;
        location.call_depth = 0;
        location.contract_id = 0;
        location.opcode = OP_MOD;

        bug_context_t context;
        copy_word(context.operand1, a);
        copy_word(context.operand2, b);
        context.context_length = 0;

        report_bug(BugType::MODULO_BY_ZERO, BugSeverity::MEDIUM, location, context,
                   "Modulo by zero");
    }
}

__host__ __device__ void OracleDetector::check_exp(uint32_t pc, const evm_word_t& base, const evm_word_t& exp,
                                                   const evm_word_t& result) {
    // Check if exponentiation would overflow
    // Simplified check: if base > 1 and exp is large
    if (!is_zero(base) && !is_zero(exp)) {
        bool base_gt_1 = false;
        for (int i = 7; i >= 0; i--) {
            if (base.limbs[i] > 0) {
                if (base.limbs[i] > 1 || i > 0) {
                    base_gt_1 = true;
                }
                break;
            }
        }
        if (base_gt_1 && exp.limbs[0] > 255) {
            bug_location_t location;
            location.pc = pc;
            location.tx_index = current_tx_index_;
            location.opcode = OP_EXP;

            bug_context_t context;
            copy_word(context.operand1, base);
            copy_word(context.operand2, exp);
            copy_word(context.result, result);
            context.context_length = 0;

            report_bug(BugType::EXPONENT_OVERFLOW, BugSeverity::MEDIUM, location, context,
                       "Potential overflow in EXP operation");
        }
    }
}

__host__ __device__ void OracleDetector::on_sload(uint32_t pc, const evm_word_t& slot, const evm_word_t& value,
                                                   execution_state_tracker_t* tracker) {
    // Track storage reads for reentrancy detection
}

__host__ __device__ void OracleDetector::on_sstore(uint32_t pc, const evm_word_t& slot,
                                                    const evm_word_t& old_value, const evm_word_t& new_value,
                                                    execution_state_tracker_t* tracker) {
    if (tracker) {
        storage_write_t write;
        copy_word(write.slot, slot);
        copy_word(write.old_value, old_value);
        copy_word(write.new_value, new_value);
        write.pc = pc;
        write.call_depth = tracker->call_depth;
        tracker->record_storage_write(write);
    }
}

__host__ __device__ void OracleDetector::on_call_start(uint32_t pc, uint8_t opcode,
                                                        const evm_word_t& target, const evm_word_t& value,
                                                        const evm_word_t& gas,
                                                        execution_state_tracker_t* tracker) {
    if (!config_->check_reentrancy || !tracker) return;

    call_frame_t frame;
    copy_word(frame.caller, current_sender_);
    copy_word(frame.callee, target);
    copy_word(frame.value, value);
    frame.pc = pc;
    frame.opcode = opcode;
    frame.has_state_change = tracker->num_storage_writes > 0;
    frame.is_external = !is_reentrancy_safe_call(opcode, target);

    tracker->push_call(frame);

    // Check for reentrancy pattern
    if (frame.is_external && frame.has_state_change) {
        // State was modified before external call - potential reentrancy
        if (config_->check_reentrancy) {
            bug_location_t location;
            location.pc = pc;
            location.tx_index = current_tx_index_;
            location.call_depth = tracker->call_depth;
            location.opcode = opcode;

            bug_context_t context;
            copy_word(context.callee, target);
            copy_word(context.value, value);
            context.context_length = 0;

            report_bug(BugType::REENTRANCY_ETH, BugSeverity::CRITICAL, location, context,
                       "Potential reentrancy: state modified before external call");
        }
    }
}

__host__ __device__ void OracleDetector::on_call_end(uint32_t pc, bool success, const uint8_t* return_data,
                                                      uint32_t return_size, execution_state_tracker_t* tracker) {
    if (tracker) {
        tracker->last_call_success = success;
        tracker->last_call_checked = false;
        tracker->pop_call();
    }

    // Check for unchecked return value
    if (!success && tracker && !tracker->last_call_checked) {
        // Will be checked on next ISZERO or comparison
    }
}

__host__ __device__ void OracleDetector::on_balance_change(const evm_word_t& address,
                                                            const evm_word_t& old_balance,
                                                            const evm_word_t& new_balance) {
    // Track for ether leak detection
}

__host__ __device__ void OracleDetector::on_selfdestruct(uint32_t pc, const evm_word_t& beneficiary,
                                                          const evm_word_t& balance) {
    if (!config_->check_selfdestruct) return;

    bug_location_t location;
    location.pc = pc;
    location.tx_index = current_tx_index_;
    location.opcode = OP_SELFDESTRUCT;

    bug_context_t context;
    copy_word(context.callee, beneficiary);
    copy_word(context.value, balance);
    context.context_length = 0;

    // Check if selfdestruct is called with non-trivial value
    if (!is_zero(balance)) {
        report_bug(BugType::SELFDESTRUCT_ETH_LEAK, BugSeverity::HIGH, location, context,
                   "SELFDESTRUCT with ETH balance");
    }
}

__host__ __device__ void OracleDetector::on_create(uint32_t pc, const evm_word_t& value,
                                                    const evm_word_t& new_address) {
    // Track contract creation
}

__host__ __device__ void OracleDetector::on_origin(uint32_t pc) {
    if (!config_->check_tx_origin) return;

    bug_location_t location;
    location.pc = pc;
    location.tx_index = current_tx_index_;
    location.opcode = OP_ORIGIN;

    bug_context_t context;
    context.context_length = 0;

    report_bug(BugType::TX_ORIGIN_AUTH, BugSeverity::MEDIUM, location, context,
               "tx.origin used (potential phishing vulnerability)");
}

__host__ __device__ void OracleDetector::on_transaction_end(
    bool success, const uint8_t* return_data, uint32_t return_size,
    uint64_t gas_used, execution_state_tracker_t* tracker) {
    current_tx_index_++;
}

__host__ __device__ void OracleDetector::check_custom_invariant(uint32_t invariant_id, bool condition,
                                                                 const char* description) {
    if (!condition) {
        bug_location_t location;
        location.pc = 0;
        location.tx_index = current_tx_index_;

        bug_context_t context;
        context.context_length = 0;

        report_bug(BugType::INVARIANT_VIOLATION, BugSeverity::HIGH, location, context, description);
    }
}

__host__ __device__ void OracleDetector::report_bug(BugType type, BugSeverity severity,
                                                     const bug_location_t& location,
                                                     const bug_context_t& context,
                                                     const char* description) {
    if ((int)severity < (int)config_->min_severity) return;

    detected_bug_t bug;
    bug.type = type;
    bug.severity = severity;
    bug.location = location;
    bug.context = context;
    bug.timestamp = 0;  // Would use real timestamp in production
    bug.input_hash = hash_word(context.operand1);
    bug.sequence_id = current_sequence_id_;
    bug.confirmed = false;

    // Copy description
    for (int i = 0; i < 255 && description[i]; i++) {
        bug.description[i] = description[i];
        bug.description[i + 1] = '\0';
    }

    storage_->add_bug(bug);
}

__host__ __device__ uint64_t OracleDetector::compute_bug_signature(BugType type, uint32_t pc,
                                                                    const evm_word_t& key_value) {
    return ((uint64_t)type << 56) ^ ((uint64_t)pc << 32) ^ hash_word(key_value);
}

__host__ __device__ BugSeverity OracleDetector::determine_severity(BugType type, const bug_context_t& context) {
    switch (type) {
        case BugType::REENTRANCY_ETH:
        case BugType::UNAUTHORIZED_SELFDESTRUCT:
            return BugSeverity::CRITICAL;
        case BugType::INTEGER_OVERFLOW:
        case BugType::INTEGER_UNDERFLOW:
        case BugType::ETHER_LEAK:
            return BugSeverity::HIGH;
        case BugType::TX_ORIGIN_AUTH:
        case BugType::DIVISION_BY_ZERO:
            return BugSeverity::MEDIUM;
        default:
            return BugSeverity::LOW;
    }
}

__host__ __device__ bool OracleDetector::is_reentrancy_safe_call(uint8_t opcode, const evm_word_t& target) {
    // STATICCALL is always safe (no state changes)
    if (opcode == OP_STATICCALL) return true;

    // Check if target is a known safe address (precompiles)
    bool is_precompile = true;
    for (int i = 1; i < 8; i++) {
        if (target.limbs[i] != 0) {
            is_precompile = false;
            break;
        }
    }
    if (is_precompile && target.limbs[0] >= 1 && target.limbs[0] <= 9) {
        return true;
    }

    return false;
}

__host__ __device__ bool OracleDetector::is_reentrancy_guard_pattern(
    const evm_word_t& slot, const evm_word_t& old_value, const evm_word_t& new_value) {
    // Common pattern: slot changes from 1->2 (enter) or 2->1 (exit)
    if (is_zero(old_value) && !is_zero(new_value)) {
        return true;  // Entering critical section
    }
    if (!is_zero(old_value) && is_zero(new_value)) {
        return true;  // Exiting critical section
    }
    return false;
}

__host__ __device__ bool OracleDetector::check_add_overflow(const evm_word_t& a, const evm_word_t& b) {
    // Overflow if a + b < a (when both are non-negative)
    uint64_t carry = 0;
    for (int i = 0; i < 8; i++) {
        uint64_t sum = (uint64_t)a.limbs[i] + (uint64_t)b.limbs[i] + carry;
        carry = sum >> 32;
    }
    return carry > 0;
}

__host__ __device__ bool OracleDetector::check_mul_overflow(const evm_word_t& a, const evm_word_t& b) {
    // Simplified check: if both have high bits set, likely overflow
    // More accurate would require full 512-bit multiplication
    int a_high = -1, b_high = -1;
    for (int i = 7; i >= 0; i--) {
        if (a.limbs[i] != 0 && a_high < 0) a_high = i;
        if (b.limbs[i] != 0 && b_high < 0) b_high = i;
    }
    // If a_high + b_high >= 8, result needs more than 256 bits
    if (a_high >= 0 && b_high >= 0 && a_high + b_high >= 7) {
        return true;
    }
    return false;
}

__host__ __device__ bool OracleDetector::check_sub_underflow(const evm_word_t& a, const evm_word_t& b) {
    // Underflow if a < b
    return less_than(a, b);
}

// ============================================================================
// Specialized Oracle Implementations
// ============================================================================

__host__ __device__ ArithmeticOracle::ArithmeticOracle(oracle_config_t* config, bug_storage_t* storage)
    : OracleDetector(config, storage) {}

__host__ __device__ void ArithmeticOracle::verify_safe_add(uint32_t pc, const evm_word_t& a,
                                                           const evm_word_t& b, const evm_word_t& result) {
    check_add(pc, a, b, result);
}

__host__ __device__ void ArithmeticOracle::verify_safe_sub(uint32_t pc, const evm_word_t& a,
                                                           const evm_word_t& b, const evm_word_t& result) {
    check_sub(pc, a, b, result);
}

__host__ __device__ void ArithmeticOracle::verify_safe_mul(uint32_t pc, const evm_word_t& a,
                                                           const evm_word_t& b, const evm_word_t& result) {
    check_mul(pc, a, b, result);
}

__host__ __device__ ReentrancyOracle::ReentrancyOracle(oracle_config_t* config, bug_storage_t* storage)
    : OracleDetector(config, storage), has_reentrancy_guard_(false) {
    zero_word(guard_slot_);
}

__host__ __device__ void ReentrancyOracle::track_external_call(uint32_t pc, const evm_word_t& target,
                                                                execution_state_tracker_t* tracker) {
    check_reentrancy_pattern(tracker);
}

__host__ __device__ void ReentrancyOracle::track_state_modification(uint32_t pc, const evm_word_t& slot,
                                                                     execution_state_tracker_t* tracker) {
    if (tracker) {
        tracker->state_modified_before_call = true;
    }
}

__host__ __device__ void ReentrancyOracle::check_reentrancy_pattern(execution_state_tracker_t* tracker) {
    if (!tracker || !config_->check_reentrancy) return;

    if (tracker->check_reentrancy()) {
        bug_location_t location;
        location.pc = 0;
        location.tx_index = current_tx_index_;
        location.call_depth = tracker->call_depth;

        bug_context_t context;
        context.context_length = 0;

        report_bug(BugType::REENTRANCY_ETH, BugSeverity::CRITICAL, location, context,
                   "Reentrancy detected: state modified before and during external call");
    }
}

__host__ __device__ AccessControlOracle::AccessControlOracle(oracle_config_t* config, bug_storage_t* storage)
    : OracleDetector(config, storage), authorization_checked_(false), num_authorized_(0) {}

__host__ __device__ void AccessControlOracle::on_privileged_operation(uint32_t pc, uint8_t opcode,
                                                                       const evm_word_t& sender) {
    if (!config_->check_unauthorized_access) return;

    if (!authorization_checked_) {
        bug_location_t location;
        location.pc = pc;
        location.tx_index = current_tx_index_;
        location.opcode = opcode;

        bug_context_t context;
        copy_word(context.caller, sender);
        context.context_length = 0;

        report_bug(BugType::MISSING_ACCESS_CONTROL, BugSeverity::HIGH, location, context,
                   "Privileged operation without authorization check");
    }
}

__host__ __device__ void AccessControlOracle::on_authorization_check(uint32_t pc,
                                                                      const evm_word_t& checked_address) {
    authorization_checked_ = true;
    if (num_authorized_ < 16) {
        copy_word(authorized_addresses_[num_authorized_++], checked_address);
    }
}

__host__ __device__ void AccessControlOracle::verify_access_control(uint32_t pc, uint8_t operation) {
    on_privileged_operation(pc, operation, current_sender_);
}

__host__ __device__ TokenOracle::TokenOracle(oracle_config_t* config, bug_storage_t* storage)
    : OracleDetector(config, storage), total_supply_slot_(0) {
    zero_word(tracked_total_supply_);
}

__host__ __device__ void TokenOracle::check_transfer(uint32_t pc, const evm_word_t& from,
                                                      const evm_word_t& to, const evm_word_t& amount) {
    if (!config_->check_erc20_issues) return;

    // Check transfer to zero address
    if (is_zero(to)) {
        bug_location_t location;
        location.pc = pc;
        location.tx_index = current_tx_index_;

        bug_context_t context;
        copy_word(context.operand1, from);
        copy_word(context.operand2, to);
        copy_word(context.result, amount);
        context.context_length = 0;

        report_bug(BugType::ERC20_TRANSFER_TO_ZERO, BugSeverity::MEDIUM, location, context,
                   "Token transfer to zero address");
    }
}

__host__ __device__ void TokenOracle::check_approve(uint32_t pc, const evm_word_t& owner,
                                                     const evm_word_t& spender, const evm_word_t& amount) {
    // Check for approval race condition (non-zero to non-zero)
    // Would need to track previous allowance
}

__host__ __device__ void TokenOracle::check_transferFrom(uint32_t pc, const evm_word_t& from,
                                                          const evm_word_t& to, const evm_word_t& amount,
                                                          const evm_word_t& allowance) {
    if (!config_->check_erc20_issues) return;

    // Check if transfer exceeds allowance
    if (greater_than(amount, allowance)) {
        bug_location_t location;
        location.pc = pc;
        location.tx_index = current_tx_index_;

        bug_context_t context;
        copy_word(context.operand1, amount);
        copy_word(context.operand2, allowance);
        context.context_length = 0;

        report_bug(BugType::ERC20_BURN_WITHOUT_APPROVAL, BugSeverity::HIGH, location, context,
                   "Transfer amount exceeds allowance");
    }
}

__host__ __device__ void TokenOracle::track_balance_change(const evm_word_t& address,
                                                            const evm_word_t& old_balance,
                                                            const evm_word_t& new_balance) {
    // Track for total supply consistency checking
}

__host__ __device__ void TokenOracle::check_total_supply_consistency() {
    // Check that sum of balances equals total supply
}

__host__ __device__ FundSafetyOracle::FundSafetyOracle(oracle_config_t* config, bug_storage_t* storage)
    : OracleDetector(config, storage), has_withdrawal_function_(false) {
    zero_word(total_eth_received_);
    zero_word(total_eth_sent_);
}

__host__ __device__ void FundSafetyOracle::on_eth_received(const evm_word_t& from, const evm_word_t& amount) {
    // Add to total received
    uint64_t carry = 0;
    for (int i = 0; i < 8; i++) {
        uint64_t sum = (uint64_t)total_eth_received_.limbs[i] + (uint64_t)amount.limbs[i] + carry;
        total_eth_received_.limbs[i] = (uint32_t)sum;
        carry = sum >> 32;
    }
}

__host__ __device__ void FundSafetyOracle::on_eth_sent(uint32_t pc, const evm_word_t& to,
                                                        const evm_word_t& amount) {
    if (!config_->check_ether_leak) return;

    // Add to total sent
    uint64_t carry = 0;
    for (int i = 0; i < 8; i++) {
        uint64_t sum = (uint64_t)total_eth_sent_.limbs[i] + (uint64_t)amount.limbs[i] + carry;
        total_eth_sent_.limbs[i] = (uint32_t)sum;
        carry = sum >> 32;
    }

    // Check if sent more than received (potential leak)
    if (greater_than(total_eth_sent_, total_eth_received_)) {
        bug_location_t location;
        location.pc = pc;
        location.tx_index = current_tx_index_;

        bug_context_t context;
        copy_word(context.operand1, total_eth_received_);
        copy_word(context.operand2, total_eth_sent_);
        copy_word(context.callee, to);
        copy_word(context.value, amount);
        context.context_length = 0;

        report_bug(BugType::ETHER_LEAK, BugSeverity::HIGH, location, context,
                   "More ETH sent than received");
    }
}

__host__ __device__ void FundSafetyOracle::check_stuck_ether(const evm_word_t& contract_balance) {
    if (!config_->check_stuck_ether) return;

    // Check if contract has balance but no withdrawal mechanism detected
    if (!is_zero(contract_balance) && !has_withdrawal_function_) {
        bug_location_t location;
        location.pc = 0;
        location.tx_index = current_tx_index_;

        bug_context_t context;
        copy_word(context.value, contract_balance);
        context.context_length = 0;

        report_bug(BugType::STUCK_ETHER, BugSeverity::MEDIUM, location, context,
                   "Contract has ETH balance but no withdrawal function detected");
    }
}

__host__ __device__ void FundSafetyOracle::check_unexpected_eth(const evm_word_t& expected,
                                                                 const evm_word_t& actual) {
    if (!config_->check_force_feed) return;

    if (!equals(expected, actual)) {
        bug_location_t location;
        location.pc = 0;
        location.tx_index = current_tx_index_;

        bug_context_t context;
        copy_word(context.expected, expected);
        copy_word(context.result, actual);
        context.context_length = 0;

        report_bug(BugType::UNEXPECTED_ETH_BALANCE, BugSeverity::MEDIUM, location, context,
                   "Unexpected ETH balance (possible force-feeding)");
    }
}

__host__ __device__ void FundSafetyOracle::check_selfdestruct_safety(uint32_t pc,
                                                                      const evm_word_t& beneficiary) {
    // Already handled in base class on_selfdestruct
}

__host__ __device__ GasOracle::GasOracle(oracle_config_t* config, bug_storage_t* storage)
    : OracleDetector(config, storage), max_gas_observed_(0), num_loops_(0) {}

__host__ __device__ void GasOracle::on_gas_usage(uint32_t pc, uint64_t gas_used, uint64_t gas_remaining) {
    if (gas_used > max_gas_observed_) {
        max_gas_observed_ = gas_used;
    }
}

__host__ __device__ void GasOracle::check_unbounded_loop(uint32_t pc, uint32_t iteration_count) {
    if (!config_->check_gas_issues) return;

    // Find or create loop entry
    int loop_idx = -1;
    for (uint32_t i = 0; i < num_loops_; i++) {
        if (loop_pcs_[i] == pc) {
            loop_idx = i;
            break;
        }
    }
    if (loop_idx < 0 && num_loops_ < 64) {
        loop_idx = num_loops_++;
        loop_pcs_[loop_idx] = pc;
        loop_iteration_counts_[loop_idx] = 0;
    }

    if (loop_idx >= 0) {
        loop_iteration_counts_[loop_idx] = iteration_count;

        // Check for potentially unbounded loop (> 1000 iterations)
        if (iteration_count > 1000) {
            bug_location_t location;
            location.pc = pc;
            location.tx_index = current_tx_index_;

            bug_context_t context;
            context.operand1.limbs[0] = iteration_count;
            for (int i = 1; i < 8; i++) context.operand1.limbs[i] = 0;
            context.context_length = 0;

            report_bug(BugType::UNBOUNDED_LOOP, BugSeverity::MEDIUM, location, context,
                       "Potentially unbounded loop detected");
        }
    }
}

__host__ __device__ void GasOracle::check_block_gas_limit(uint64_t total_gas) {
    if (!config_->check_gas_issues) return;

    // Ethereum block gas limit is around 30 million
    if (total_gas > 30000000) {
        bug_location_t location;
        location.pc = 0;
        location.tx_index = current_tx_index_;

        bug_context_t context;
        context.operand1.limbs[0] = (uint32_t)(total_gas & 0xFFFFFFFF);
        context.operand1.limbs[1] = (uint32_t)(total_gas >> 32);
        for (int i = 2; i < 8; i++) context.operand1.limbs[i] = 0;
        context.context_length = 0;

        report_bug(BugType::BLOCK_GAS_LIMIT, BugSeverity::HIGH, location, context,
                   "Transaction exceeds block gas limit");
    }
}

__host__ __device__ void GasOracle::check_call_gas(uint32_t pc, uint64_t gas_forwarded) {
    // Check if 1/64th rule is violated or gas is unexpectedly low
}

// ============================================================================
// Composite Oracle Implementation
// ============================================================================

__host__ __device__ CompositeOracle::CompositeOracle(oracle_config_t* config, bug_storage_t* storage)
    : config_(config), storage_(storage),
      arithmetic_(config, storage),
      reentrancy_(config, storage),
      access_control_(config, storage),
      token_(config, storage),
      fund_safety_(config, storage),
      gas_(config, storage) {}

__host__ __device__ void CompositeOracle::init() {
    storage_->init();
}

__host__ __device__ void CompositeOracle::on_transaction_start(
    const evm_word_t& sender, const evm_word_t& receiver,
    const evm_word_t& value, const uint8_t* calldata, uint32_t calldata_len) {

    arithmetic_.on_transaction_start(sender, receiver, value, calldata, calldata_len);
    reentrancy_.on_transaction_start(sender, receiver, value, calldata, calldata_len);
    access_control_.on_transaction_start(sender, receiver, value, calldata, calldata_len);
    token_.on_transaction_start(sender, receiver, value, calldata, calldata_len);
    fund_safety_.on_transaction_start(sender, receiver, value, calldata, calldata_len);
    gas_.on_transaction_start(sender, receiver, value, calldata, calldata_len);
}

__host__ __device__ void CompositeOracle::on_instruction(
    uint32_t pc, uint8_t opcode,
    const evm_word_t* stack, uint32_t stack_size,
    execution_state_tracker_t* tracker) {

    arithmetic_.on_instruction(pc, opcode, stack, stack_size, tracker);
    // Other oracles hook into specific opcodes via their own mechanisms
}

__host__ __device__ void CompositeOracle::on_transaction_end(
    bool success, const uint8_t* return_data, uint32_t return_size,
    uint64_t gas_used, execution_state_tracker_t* tracker) {

    arithmetic_.on_transaction_end(success, return_data, return_size, gas_used, tracker);
    reentrancy_.on_transaction_end(success, return_data, return_size, gas_used, tracker);
    gas_.on_transaction_end(success, return_data, return_size, gas_used, tracker);
}

// ============================================================================
// CUDA Kernel Implementations
// ============================================================================

__global__ void kernel_check_arithmetic(
    uint8_t opcode,
    const evm_word_t* operands_a,
    const evm_word_t* operands_b,
    const evm_word_t* results,
    uint32_t* pcs,
    uint32_t num_operations,
    bug_storage_t* bug_storage,
    oracle_config_t* config) {

    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_operations) return;

    ArithmeticOracle oracle(config, bug_storage);

    switch (opcode) {
        case OP_ADD:
            oracle.verify_safe_add(pcs[idx], operands_a[idx], operands_b[idx], results[idx]);
            break;
        case OP_SUB:
            oracle.verify_safe_sub(pcs[idx], operands_a[idx], operands_b[idx], results[idx]);
            break;
        case OP_MUL:
            oracle.verify_safe_mul(pcs[idx], operands_a[idx], operands_b[idx], results[idx]);
            break;
    }
}

__global__ void kernel_check_reentrancy(
    execution_state_tracker_t* trackers,
    uint32_t num_instances,
    bug_storage_t* bug_storage,
    oracle_config_t* config) {

    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_instances) return;

    ReentrancyOracle oracle(config, bug_storage);
    oracle.check_reentrancy_pattern(&trackers[idx]);
}

__global__ void kernel_check_invariants(
    const evm_word_t* pre_state,
    const evm_word_t* post_state,
    const uint32_t* invariant_types,
    uint32_t num_invariants,
    bug_storage_t* bug_storage) {

    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_invariants) return;

    // Check specific invariant based on type
    uint32_t type = invariant_types[idx];

    bool violated = false;
    switch (type) {
        case 0:  // EQUALS
            violated = !equals(pre_state[idx], post_state[idx]);
            break;
        case 1:  // NOT_LESS_THAN
            violated = less_than(post_state[idx], pre_state[idx]);
            break;
        case 2:  // NOT_GREATER_THAN
            violated = greater_than(post_state[idx], pre_state[idx]);
            break;
        case 3:  // NON_ZERO
            violated = is_zero(post_state[idx]);
            break;
    }

    if (violated) {
        detected_bug_t bug;
        bug.type = BugType::INVARIANT_VIOLATION;
        bug.severity = BugSeverity::HIGH;
        bug.location.pc = 0;
        bug.location.tx_index = 0;
        copy_word(bug.context.expected, pre_state[idx]);
        copy_word(bug.context.result, post_state[idx]);
        bug_storage->add_bug(bug);
    }
}

// ============================================================================
// Host Helper Functions
// ============================================================================

__host__ oracle_config_t* allocate_oracle_config() {
    oracle_config_t* config;
    cudaMallocManaged(&config, sizeof(oracle_config_t));
    config->set_default();
    return config;
}

__host__ bug_storage_t* allocate_bug_storage() {
    bug_storage_t* storage;
    cudaMallocManaged(&storage, sizeof(bug_storage_t));
    storage->init();
    return storage;
}

__host__ execution_state_tracker_t* allocate_trackers(uint32_t num_instances) {
    execution_state_tracker_t* trackers;
    cudaMallocManaged(&trackers, num_instances * sizeof(execution_state_tracker_t));
    for (uint32_t i = 0; i < num_instances; i++) {
        trackers[i].init();
    }
    return trackers;
}

__host__ void free_oracle_config(oracle_config_t* config) {
    if (config) cudaFree(config);
}

__host__ void free_bug_storage(bug_storage_t* storage) {
    if (storage) cudaFree(storage);
}

__host__ void free_trackers(execution_state_tracker_t* trackers) {
    if (trackers) cudaFree(trackers);
}

__host__ void copy_bugs_to_host(detected_bug_t* host_bugs, const bug_storage_t* device_storage) {
    cudaMemcpy(host_bugs, device_storage->bugs,
               device_storage->bug_count * sizeof(detected_bug_t),
               cudaMemcpyDeviceToHost);
}

__host__ void print_bug_report(const bug_storage_t* storage) {
    printf("\n========== BUG REPORT ==========\n");
    printf("Total bugs found: %u\n\n", storage->bug_count);

    const char* severity_names[] = {"INFO", "LOW", "MEDIUM", "HIGH", "CRITICAL"};
    const char* type_names[] = {
        "INTEGER_OVERFLOW", "INTEGER_UNDERFLOW", "DIVISION_BY_ZERO", "MODULO_BY_ZERO",
        "EXPONENT_OVERFLOW", "", "", "", "", "",
        "UNAUTHORIZED_CALL", "UNAUTHORIZED_SELFDESTRUCT", "UNAUTHORIZED_DELEGATECALL",
        "TX_ORIGIN_AUTH", "MISSING_ACCESS_CONTROL", "", "", "", "", "",
        "REENTRANCY_ETH", "REENTRANCY_ERC20", "REENTRANCY_CROSS_FUNCTION",
        "REENTRANCY_CROSS_CONTRACT", "READ_ONLY_REENTRANCY"
    };

    for (uint32_t i = 0; i < storage->bug_count; i++) {
        const detected_bug_t& bug = storage->bugs[i];
        printf("Bug #%u:\n", i + 1);
        printf("  Type: %s\n", ((int)bug.type < 25) ? type_names[(int)bug.type] : "UNKNOWN");
        printf("  Severity: %s\n", severity_names[(int)bug.severity]);
        printf("  PC: %u\n", bug.location.pc);
        printf("  TX Index: %u\n", bug.location.tx_index);
        printf("  Description: %s\n", bug.description);
        printf("\n");
    }
}

__host__ void export_bugs_json(const bug_storage_t* storage, const char* filename) {
    FILE* f = fopen(filename, "w");
    if (!f) return;

    fprintf(f, "{\n  \"bug_count\": %u,\n  \"bugs\": [\n", storage->bug_count);

    for (uint32_t i = 0; i < storage->bug_count; i++) {
        const detected_bug_t& bug = storage->bugs[i];
        fprintf(f, "    {\n");
        fprintf(f, "      \"type\": %u,\n", (unsigned)bug.type);
        fprintf(f, "      \"severity\": %u,\n", (unsigned)bug.severity);
        fprintf(f, "      \"pc\": %u,\n", bug.location.pc);
        fprintf(f, "      \"tx_index\": %u,\n", bug.location.tx_index);
        fprintf(f, "      \"description\": \"%s\"\n", bug.description);
        fprintf(f, "    }%s\n", (i < storage->bug_count - 1) ? "," : "");
    }

    fprintf(f, "  ]\n}\n");
    fclose(f);
}

}  // namespace fuzzing
}  // namespace CuEVM
