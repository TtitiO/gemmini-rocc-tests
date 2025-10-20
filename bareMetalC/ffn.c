// See LICENSE for license details.

#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini.h"
#include "include/gemmini_nn.h"
#include "include/gemmini_testutils.h"

//================================================================================
// START: Updated code for counter instrumentation
//================================================================================

/**
 * @brief A helper function to print the values of our configured performance counters.
 * @param step_name A string describing the measured computation step.
 */
static void print_step_counters(const char* step_name) {
    // Read the configured counters
    uint64_t load_cycles = counter_read(0);
    uint64_t store_cycles = counter_read(1);
    uint64_t exec_cycles = counter_read(2);
    uint64_t load_dma_wait_cycles = counter_read(3);
    uint64_t exec_preload_haz_cycles = counter_read(4);
    uint64_t all_pipelines_active_cycles = counter_read(5);

    printf("  -- Counter Stats for [%s] --\n", step_name);
    printf("     -- Active Cycles --\n");
    printf("     LOAD_ACTIVE_CYCLE:      %llu\n", load_cycles);
    printf("     STORE_ACTIVE_CYCLE:     %llu\n", store_cycles);
    printf("     EXE_ACTIVE_CYCLE:       %llu\n", exec_cycles);
    printf("\n     -- Stall & Overlap Analysis --\n");
    printf("     LOAD_DMA_WAIT_CYCLE:    %llu (Load unit waiting for DRAM)\n", load_dma_wait_cycles);
    printf("     EXE_PRELOAD_HAZ_CYCLE:  %llu (Exec unit waiting for Load unit)\n", exec_preload_haz_cycles);
    printf("     MAIN_LD_ST_EX_CYCLES:   %llu (All 3 pipelines active)\n", all_pipelines_active_cycles);

    // Calculate and print derived metrics for better insight
    uint64_t total_exec_pipeline_cycles = exec_cycles + exec_preload_haz_cycles;
    if (total_exec_pipeline_cycles > 0) {
        float exec_utilization = 100.0f * exec_cycles / total_exec_pipeline_cycles;
        printf("     => Execute Unit Utilization: %.2f%%\n", exec_utilization);
    }

    printf("  ---------------------------------------\n");
}

//================================================================================
// END: Updated code for counter instrumentation
//================================================================================


/**
 * @brief Implements a Feed-Forward Network (FFN) block for a Transformer model.
 * ... (rest of the detailed comment is unchanged) ...
 */
void ffn(int hidden_dim, int expansion_dim, int seq_len,
        const elem_t * input, elem_t * out,
        const elem_t * ff1_w, const elem_t * ff2_w,
        const acc_t * ff1_b, const acc_t * ff2_b,
        elem_t * out_buf, acc_t * out_buf_acc)
{
    // Configure counters at the start of the function.
    // Gemmini has a limited number of hardware counters. Here, we configure 6.
    counter_configure(0, LOAD_ACTIVE_CYCLE);
    counter_configure(1, STORE_ACTIVE_CYCLE);
    counter_configure(2, EXE_ACTIVE_CYCLE);
    counter_configure(3, LOAD_DMA_WAIT_CYCLE);      // New: See how often we wait for main memory
    counter_configure(4, EXE_PRELOAD_HAZ_CYCLE);    // New: See pipeline stalls between load and execute
    counter_configure(5, MAIN_LD_ST_EX_CYCLES);     // New: See how often all units are running in parallel

    // Step 1 & 2: First linear layer (input * ff1_w + ff1_b) followed by GELU activation.
    printf("\nStarting first linear layer with GELU activation...\n");
    counter_reset(); // Reset counters for this step
    tiled_matmul_auto(seq_len, expansion_dim, hidden_dim,
        /*A=*/ input, /*B=*/ ff1_w,
        /*D=*/ ff1_b, /*C=*/ out_buf,
        /*stride_A=*/hidden_dim, /*stride_B=*/expansion_dim, /*stride_D=*/0, /*stride_C=*/expansion_dim,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        IGELU, /*scale=*/ ACC_SCALE_IDENTITY, /*bert_scale=*/ ACC_SCALE_IDENTITY,
        /*repeating_bias=*/ false,
        false, /*transpose_B=*/ false,
        false, false,
        0,
        WS);

    gemmini_fence();
    print_step_counters("Matmul1+GELU");

    // Step 3: Second linear layer (out_buf * ff2_w + ff2_b).
    printf("Starting second linear layer...\n");
    counter_reset(); // Reset counters for this step
    tiled_matmul_auto(seq_len, hidden_dim, expansion_dim,
        /*A=*/ out_buf, /*B=*/ ff2_w,
        /*D=*/ ff2_b, /*C=*/ out_buf_acc,
        /*stride_A=*/expansion_dim, /*stride_B=*/hidden_dim, /*stride_D=*/0, /*stride_C=*/hidden_dim,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, /*scale=*/ ACC_SCALE_IDENTITY, /*bert_scale=*/ 0,
        /*repeating_bias=*/ false,
        false, /*transpose_B=*/ false,
        true, false, // Output is in accumulator
        0,
        WS);

    gemmini_fence();
    print_step_counters("Matmul2");

    // Step 4: Layer Normalization.
    printf("Starting layer normalization...\n");
    counter_reset(); // Reset counters for this step
    tiled_norm_auto(seq_len, hidden_dim,
        (acc_t*)out_buf_acc, (elem_t*)out,
        ACC_SCALE_IDENTITY,
        LAYERNORM, WS);

    gemmini_fence();
    print_step_counters("LayerNorm");

    // Step 5: Residual Connection (out = out + input).
    printf("Starting residual connection...\n");
    counter_reset(); // Reset counters for this step
    tiled_resadd_auto(seq_len, hidden_dim,
        MVIN_SCALE_IDENTITY,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        out,   // The post-LN FFN output
        input, // The original input to the block
        out,   // The final output
        /*relu=*/ false,
        WS);

    gemmini_fence();
    print_step_counters("ResAdd");
}

// FIX: The FFN_BENCH macro has been removed. Its logic is now inside PRINT_FFN_BENCH.

// A helper macro to print the benchmark results for a given FFN configuration
// FIX: All large arrays are now declared 'static' here to avoid stack overflow.
#define PRINT_FFN_BENCH(name, hidden_dim, expansion_dim, seq_len) { \
    /* Allocate ALL large arrays statically in the BSS segment */ \
    static elem_t input[seq_len][hidden_dim]; \
    static elem_t output[seq_len][hidden_dim]; \
    static elem_t ff1_w[hidden_dim][expansion_dim]; \
    static elem_t ff2_w[expansion_dim][hidden_dim]; \
    static acc_t ff1_b[expansion_dim]; \
    static acc_t ff2_b[hidden_dim]; \
    static elem_t out_buf[seq_len][expansion_dim]; \
    static acc_t out_buf_acc[seq_len][hidden_dim]; \
    \
    uint64_t start_cycle = read_cycles(); \
    \
    ffn(hidden_dim, expansion_dim, seq_len, \
        (elem_t*)input, (elem_t*)output, \
        (elem_t*)ff1_w, (elem_t*)ff2_w, \
        (acc_t*)ff1_b, (acc_t*)ff2_b, \
        (elem_t*)out_buf, (acc_t*)out_buf_acc); \
    \
    uint64_t end_cycle = read_cycles(); \
    uint64_t cycles = end_cycle - start_cycle; \
    \
    printf("\n%s FFN stats: hidden_dim=%d, expansion_dim=%d, seq_len=%d\n", \
           name, hidden_dim, expansion_dim, seq_len); \
    printf("%s FFN total CPU cycles: %llu\n\n", name, cycles); \
}

int main (int argc, char * argv[]) {

#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif

    gemmini_flush(0);
    printf("========== Running FFN Benchmarks ==========\n");

    // Benchmark for a BERT-base-like FFN
    PRINT_FFN_BENCH("bert-base",
            /*hidden_dim=*/768,
            /*expansion_dim=*/3072, // 768 * 4
            /*seq_len=*/128);

    // Benchmark for a smaller Transformer FFN
    PRINT_FFN_BENCH("transformer-small",
            /*hidden_dim=*/512,
            /*expansion_dim=*/2048, // 512 * 4
            /*seq_len=*/256);

    printf("=============================================\n");

    exit(0);
}
