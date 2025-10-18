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

/**
 * @brief Implements a Feed-Forward Network (FFN) block for a Transformer model.
 *
 * This block consists of two linear layers with a GELU activation function,
 * followed by a layer normalization and a residual connection.
 * This is a common component in models like BERT and other Transformers.
 *
 * The computation flow is:
 * 1. res = input
 * 2. x = GELU(input @ W1 + b1)
 * 3. x = x @ W2 + b2
 * 4. x = LayerNorm(x)
 * 5. output = x + res
 *
 * @param hidden_dim The dimension of the input and output features.
 * @param expansion_dim The dimension of the intermediate layer (usually hidden_dim * 4).
 * @param seq_len The length of the input sequence.
 * @param input Pointer to the input tensor of shape (seq_len, hidden_dim).
 * @param out Pointer to the output tensor of shape (seq_len, hidden_dim).
 * @param ff1_w Pointer to the weights of the first linear layer, shape (hidden_dim, expansion_dim).
 * @param ff2_w Pointer to the weights of the second linear layer, shape (expansion_dim, hidden_dim).
 * @param ff1_b Pointer to the bias of the first linear layer, shape (expansion_dim).
 * @param ff2_b Pointer to the bias of the second linear layer, shape (hidden_dim).
 * @param out_buf A temporary buffer for the output of the first linear layer, shape (seq_len, expansion_dim).
 * @param out_buf_acc A temporary accumulator buffer for the output of the second linear layer, shape (seq_len, hidden_dim).
 */
void ffn(int hidden_dim, int expansion_dim, int seq_len,
        const elem_t * input, elem_t * out,
        const elem_t * ff1_w, const elem_t * ff2_w,
        const acc_t * ff1_b, const acc_t * ff2_b,
        elem_t * out_buf, acc_t * out_buf_acc)
{
    // Step 1 & 2: First linear layer (input * ff1_w + ff1_b) followed by GELU activation.
    // The result is stored in `out_buf`.
    // The dimensions are: (seq_len, hidden_dim) @ (hidden_dim, expansion_dim) -> (seq_len, expansion_dim)
    printf("Starting first linear layer with GELU activation\n");
    tiled_matmul_auto(seq_len, expansion_dim, hidden_dim,
        /*A=*/ input, /*B=*/ ff1_w,
        /*D=*/ ff1_b, /*C=*/ out_buf,
        /*stride_A=*/hidden_dim, /*stride_B=*/expansion_dim, /*stride_D=*/0, /*stride_C=*/expansion_dim,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        IGELU, /*scale=*/ ACC_SCALE_IDENTITY, /*bert_scale=*/ ACC_SCALE_IDENTITY,
        /*repeating_bias=*/ false, // Bias is not repeated, as it's added per-row on the output
        false, /*transpose_B=*/ false,
        false, false,
        0,
        WS);

    gemmini_fence();

    // Step 3: Second linear layer (out_buf * ff2_w + ff2_b).
    // The result is stored in the accumulator `out_buf_acc` to maintain precision.
    // The dimensions are: (seq_len, expansion_dim) @ (expansion_dim, hidden_dim) -> (seq_len, hidden_dim)
    printf("Starting second linear layer\n");
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

    // Step 4: Layer Normalization.
    // The normalized result of `out_buf_acc` is stored in the final `out` buffer.
    printf("Starting layer normalization\n");
    tiled_norm_auto(seq_len, hidden_dim,
        (acc_t*)out_buf_acc, (elem_t*)out,
        ACC_SCALE_IDENTITY,
        LAYERNORM, WS);

    gemmini_fence();

    // Step 5: Residual Connection (out = out + input).
    // The original input is added to the output of the FFN block.
    // The result is stored back into `out`.
    printf("Starting residual connection\n");
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
}

// A macro to simplify calling and instrumenting the FFN function
#define FFN_BENCH(hidden_dim, expansion_dim, seq_len, input, output) ({ \
    \
    /* Statically allocate weights, biases, and buffers */ \
    static elem_t ff1_w[hidden_dim][expansion_dim]; \
    static elem_t ff2_w[expansion_dim][hidden_dim]; \
    static acc_t ff1_b[expansion_dim]; \
    static acc_t ff2_b[hidden_dim]; \
    \
    /* Temporary buffers needed by the FFN function */ \
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
    \
    end_cycle - start_cycle; \
})

// A helper macro to print the benchmark results for a given FFN configuration
#define PRINT_FFN_BENCH(name, hidden_dim, expansion_dim, seq_len) { \
    static elem_t input[seq_len][hidden_dim]; \
    static elem_t output[seq_len][hidden_dim]; \
    \
    uint64_t cycles = FFN_BENCH(hidden_dim, expansion_dim, seq_len, input, output); \
    \
    printf("%s FFN stats: hidden_dim=%d, expansion_dim=%d, seq_len=%d\n", \
           name, hidden_dim, expansion_dim, seq_len); \
    printf("%s FFN cycles: %llu\n\n", name, cycles); \
}

int main (int argc, char * argv[]) {
#if defined(FAST) || !defined(HAS_NORMALIZATIONS)
    exit(0);
#endif

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

