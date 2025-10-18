#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

#ifndef BAREMETAL
#include <sys/mman.h>
#endif

#include "include/gemmini.h"
#include "include/gemmini_nn.h"
#include "include/gemmini_testutils.h"

// Define some constants for clarity to manage on-chip memory addresses.
// These addresses are logical offsets within the scratchpad or accumulator space.
// Let's assume the scratchpad is split, with inputs/weights at the bottom and
// intermediate activations at the top.
#define SCRATCHPAD_A_OFFSET 0
#define SCRATCHPAD_B_OFFSET (BANK_NUM * BANK_ROWS / 2) // Put B in the second half of spad
#define INTERMEDIATE_A_SPAD_ADDR (SCRATCHPAD_A_OFFSET) // We can reuse the A buffer

// The accumulator address space for the final output.
// See gemmini.h: The 30th bit being set usually denotes the accumulator space.
#define ACC_C_OFFSET (3 << (ADDR_LEN - 2))

// Note: For self-attention, "enc_out" should be the same as "input".
void attention(int hidden_dim, int expansion_dim, int num_heads, int seq_len,
        int compression_factor,

        const elem_t * input, const elem_t * enc_out,
        elem_t * out, elem_t * resadd_out,
        const elem_t * Wq, const elem_t * Wk, const elem_t * Wv, const elem_t * Wo,

        const acc_t * Wq_b, const acc_t * Wk_b, const acc_t * Wv_b,
        const acc_t * Wo_b,

        elem_t * Q_buf, elem_t * K_buf, elem_t * V_buf,
        elem_t * attn_buf, elem_t * out_buf, acc_t * out_buf_acc)
{
    int hidden_dim_compressed = hidden_dim / compression_factor;
    int hidden_dim_per_head = hidden_dim_compressed / num_heads;

    if (compression_factor < 0) {
        hidden_dim_compressed = hidden_dim;
        hidden_dim_per_head = (hidden_dim_compressed / 12) * (-compression_factor);
    }

    // Q = input @ Wq, K = enc_out @ Wk, V = enc_out @ Wv
    const int qkv_matmuls_n = 3;
    {
        const elem_t * qkv_weights[] = {Wq, Wk, Wv};
        const elem_t * qkv_ins[] = {input, enc_out, enc_out};
        const acc_t * qkv_bs[] = {Wq_b, Wk_b, Wv_b};
        elem_t * qkv_outs[] = {Q_buf, K_buf, V_buf};

        for (int i = 0; i < qkv_matmuls_n; i++) {
            tiled_matmul_auto(seq_len, hidden_dim_compressed, hidden_dim,
                /*A=*/ qkv_ins[i], /*B=*/ qkv_weights[i],
                /*D=*/ qkv_bs[i], /*C=*/ qkv_outs[i],
                /*stride_A=*/hidden_dim, /*stride_B=*/hidden_dim, /*stride_D=*/0, /*stride_C=*/hidden_dim_compressed,
                MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
                NO_ACTIVATION, /*scale=*/ ACC_SCALE_IDENTITY, /*bert_scale=*/ 0,
                /*repeating_bias=*/ false,
                false, /*transpose_B=*/ false,
                false, false,
                0,
                WS);
        }
    }

    gemmini_fence();

    // attn = softmax(Q @ K^T)
    for (int head = 0; head < num_heads; head++) {
        const elem_t * A = Q_buf + head * hidden_dim_per_head;
        const elem_t * B = K_buf + head * hidden_dim_per_head;
        elem_t * C = attn_buf; // Reuse the same attention buffer

        tiled_matmul_auto(seq_len, seq_len, hidden_dim_per_head,
            /*A=*/ A, /*B=*/ B,
            /*D=*/ NULL, /*C=*/ C,
            /*stride_A=*/hiddean_dim_compressed, /*stride_B=*/hidden_dim_compressed, /*stride_D=*/0, /*stride_C=*/seq_len,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            SOFTMAX, /*scale=*/ ACC_SCALE_IDENTITY, /*bert_scale=*/ 0,
            /*repeating_bias=*/ false,
            false, /*transpose_B=*/ true,
            false, false,
            0,
            WS);

        gemmini_fence();

        // out_buf_partial = attn @ V_partial
        elem_t * C_out = out_buf + head * hidden_dim_per_head;
        const elem_t * A_attn = C;
        const elem_t * B_v = V_buf + head * hidden_dim_per_head;

        tiled_matmul_auto(seq_len, hidden_dim_per_head, seq_len,
            /*A=*/ A_attn, /*B=*/ B_v,
            /*D=*/ NULL, /*C=*/ C_out,
            /*stride_A=*/seq_len, /*stride_B=*/hidden_dim_compressed, /*stride_D=*/0, /*stride_C=*/hidden_dim_compressed,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            NO_ACTIVATION, /*scale=*/ ACC_SCALE_IDENTITY, /*bert_scale=*/ 0,
            /*repeating_bias=*/ false,
            false, /*transpose_B=*/ false,
            false, false,
            0,
            WS);

        gemmini_fence();
    }

    // out_buf_acc = out_buf @ Wo
    tiled_matmul_auto(seq_len, hidden_dim, hidden_dim_compressed,
        /*A=*/ out_buf, /*B=*/ Wo,
        /*D=*/ Wo_b, /*C=*/ out_buf_acc,
        /*stride_A=*/hidden_dim_compressed, /*stride_B=*/hidden_dim, /*stride_D=*/0, /*stride_C=*/hidden_dim,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, /*scale=*/ ACC_SCALE_IDENTITY, /*bert_scale=*/ 0,
        /*repeating_bias=*/ false,
        false, /*transpose_B=*/ false,
        true, false,
        0,
        WS);

    gemmini_fence();

    // out = LN(out_buf_acc)
    tiled_norm_auto(seq_len, hidden_dim,
        (acc_t*)out_buf_acc, (elem_t*)out,
        ACC_SCALE_IDENTITY,
        LAYERNORM, WS);

    gemmini_fence();

    // resadd_out = input + out
    tiled_resadd_auto(seq_len, hidden_dim,
        MVIN_SCALE_IDENTITY,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        input,
        out,
        resadd_out,
        /*relu=*/ false,
        WS);

    gemmini_fence();
}

void ffn(int hidden_dim, int expansion_dim, int seq_len,
        const elem_t * input, elem_t * out,
        const elem_t * ff1_w, const elem_t * ff2_w,
        const acc_t * ff1_b, const acc_t * ff2_b,

        elem_t * out_buf, acc_t * out_buf_acc)
{
    // out_buf = GELU(input @ ff1_w + ff1_b)
    tiled_matmul_auto(seq_len, expansion_dim, hidden_dim,
        /*A=*/ input, /*B=*/ ff1_w,
        /*D=*/ ff1_b, /*C=*/ out_buf,
        /*stride_A=*/hidden_dim, /*stride_B=*/expansion_dim, /*stride_D=*/0, /*stride_C=*/expansion_dim,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        IGELU, /*scale=*/ ACC_SCALE_IDENTITY, /*bert_scale=*/ ACC_SCALE_IDENTITY,
        /*repeating_bias=*/ false, // FIX: Was true
        false, /*transpose_B=*/ false,
        false, false,
        0,
        WS);

    gemmini_fence();

    // out_buf_acc = out_buf @ ff2_w + ff2_b
    tiled_matmul_auto(seq_len, hidden_dim, expansion_dim,
        /*A=*/ out_buf, /*B=*/ ff2_w,
        /*D=*/ ff2_b, /*C=*/ out_buf_acc,
        /*stride_A=*/expansion_dim, /*stride_B=*/hidden_dim, /*stride_D=*/0, /*stride_C=*/hidden_dim,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, /*scale=*/ ACC_SCALE_IDENTITY, /*bert_scale=*/ 0,
        /*repeating_bias=*/ false, // FIX: Was true
        false, /*transpose_B=*/ false,
        true, false,
        0,
        WS);

    gemmini_fence();

    // out = LN(out_buf_acc)
    tiled_norm_auto(seq_len, hidden_dim,
        (acc_t*)out_buf_acc, (elem_t*)out,
        ACC_SCALE_IDENTITY,
        LAYERNORM, WS);

    gemmini_fence();

    // out = out + input (final residual connection)
    tiled_resadd_auto(seq_len, hidden_dim,
        MVIN_SCALE_IDENTITY,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        out,
        input,
        out,
        /*relu=*/ false,
        WS);

    gemmini_fence();
}


uint64_t encoder_decoder(
        int hidden_dim, int expansion_dim, int num_heads, int cross_num_heads,
        int seq_len, int compression_factor,

        const elem_t * input, const elem_t * enc_out, elem_t * out,
        const elem_t * Wq, const elem_t * Wk, const elem_t * Wv, const elem_t * Wo,
        const elem_t * Wq_cross, const elem_t * Wk_cross, const elem_t * Wv_cross, const elem_t * Wo_cross,

        const acc_t * Wq_b, const acc_t * Wk_b, const acc_t * Wv_b,
        const acc_t * Wo_b,
        const acc_t * Wq_cross_b, const acc_t * Wk_cross_b, const acc_t * Wv_cross_b,
        const acc_t * Wo_cross_b,

        const elem_t * ff1_w, const elem_t * ff2_w,
        const acc_t * ff1_b, const acc_t * ff2_b,

        elem_t * Q_buf, elem_t * K_buf, elem_t * V_buf,
        elem_t * attn_buf, elem_t * att_out_buf, acc_t * out_buf_acc,
        elem_t * ffn_out_buf,
        elem_t * resadd1_buf, elem_t * resadd2_buf)
{
    uint64_t start = read_cycles();

    const bool is_encoder = enc_out == NULL;
    const elem_t* ffn_input = NULL;

    // Self-Attention block
    attention(hidden_dim, expansion_dim, num_heads, seq_len, compression_factor,
        input, input, // Self-attention input is the same for Q, K, V
        out, resadd1_buf, // out = LN(Attn(input)), resadd1_buf = input + out
        Wq, Wk, Wv, Wo,
        Wq_b, Wk_b, Wv_b, Wo_b,
        Q_buf, K_buf, V_buf, attn_buf, att_out_buf, out_buf_acc);

    if (is_encoder) {
        ffn_input = resadd1_buf;
    } else {
        // Cross-Attention block (for decoders)
        attention(hidden_dim, expansion_dim, cross_num_heads, seq_len, compression_factor,
            resadd1_buf, enc_out, // Query from previous block, Key/Value from encoder output
            out, resadd2_buf, // out = LN(CrossAttn(resadd1_buf)), resadd2_buf = resadd1_buf + out
            Wq_cross, Wk_cross, Wv_cross, Wo_cross,
            Wq_cross_b, Wk_cross_b, Wv_cross_b, Wo_cross_b,
            Q_buf, K_buf, V_buf, attn_buf, att_out_buf, out_buf_acc);
        ffn_input = resadd2_buf;
    }

    // Feed-Forward block
    ffn(hidden_dim, expansion_dim, seq_len,
        ffn_input,
        out, // Final output of the layer
        ff1_w, ff2_w,
        ff1_b, ff2_b,
        ffn_out_buf, out_buf_acc);

    uint64_t end = read_cycles();
    return end - start;
}

// Rewritten macro for clarity and correctness
#define ENCODER_DECODER(hidden_dim, expansion_dim, num_heads, cross_num_heads, seq_len, compression_factor, input, enc_out, output) ({ \
    static elem_t Wq[hidden_dim][hidden_dim]; \
    static elem_t Wk[hidden_dim][hidden_dim]; \
    static elem_t Wv[hidden_dim][hidden_dim]; \
    static elem_t Wo[hidden_dim][hidden_dim]; \
    static elem_t Wq_cross[hidden_dim][hidden_dim]; \
    static elem_t Wk_cross[hidden_dim][hidden_dim]; \
    static elem_t Wv_cross[hidden_dim][hidden_dim]; \
    static elem_t Wo_cross[hidden_dim][hidden_dim]; \
    static acc_t Wq_b[hidden_dim], Wk_b[hidden_dim], Wv_b[hidden_dim], Wo_b[hidden_dim]; \
    static acc_t Wq_cross_b[hidden_dim], Wk_cross_b[hidden_dim], Wv_cross_b[hidden_dim], Wo_cross_b[hidden_dim]; \
    static elem_t ff1_w[hidden_dim][expansion_dim]; \
    static elem_t ff2_w[expansion_dim][hidden_dim]; \
    static acc_t ff1_b[expansion_dim]; \
    static acc_t ff2_b[hidden_dim]; \
    static elem_t Q_buf[seq_len][hidden_dim]; \
    static elem_t K_buf[seq_len][hidden_dim]; \
    static elem_t V_buf[seq_len][hidden_dim]; \
    static elem_t attn_buf[seq_len][seq_len]; \
    static elem_t att_out_buf[seq_len][hidden_dim]; \
    static elem_t ffn_out_buf[seq_len][expansion_dim]; \
    static acc_t out_buf_acc[seq_len][hidden_dim]; \
    static elem_t resadd1_buf[seq_len][hidden_dim]; \
    static elem_t resadd2_buf[seq_len][hidden_dim]; \
    uint64_t cycles = encoder_decoder( \
            hidden_dim, expansion_dim, num_heads, cross_num_heads, seq_len, compression_factor, \
            input, enc_out, output, \
            (elem_t*)Wq, (elem_t*)Wk, (elem_t*)Wv, (elem_t*)Wo, \
            (elem_t*)Wq_cross, (elem_t*)Wk_cross, (elem_t*)Wv_cross, (elem_t*)Wo_cross, \
            (acc_t*)Wq_b, (acc_t*)Wk_b, (acc_t*)Wv_b, (acc_t*)Wo_b, \
            (acc_t*)Wq_cross_b, (acc_t*)Wk_cross_b, (acc_t*)Wv_cross_b, (acc_t*)Wo_cross_b, \
            (elem_t*)ff1_w, (elem_t*)ff2_w, \
            (acc_t*)ff1_b, (acc_t*)ff2_b, \
            (elem_t*)Q_buf, (elem_t*)K_buf, (elem_t*)V_buf, \
            (elem_t*)attn_buf, (elem_t*)att_out_buf, (acc_t*)out_buf_acc, \
            (elem_t*)ffn_out_buf, \
            (elem_t*)resadd1_buf, (elem_t*)resadd2_buf \
    ); \
    cycles; \
})

#define PRINT_ENCODER_DECODER(name, is_encoder, hidden_dim, expansion_dim, num_heads, cross_num_heads, seq_len, compression_factor) { \
    static elem_t input[seq_len][hidden_dim]; \
    static elem_t enc_out[seq_len][hidden_dim]; \
    static elem_t output[seq_len][hidden_dim]; \
    char * type_str = is_encoder ? "encoder" : "decoder"; \
    uint64_t cycles = ENCODER_DECODER(hidden_dim, expansion_dim, num_heads, cross_num_heads, seq_len, compression_factor, (elem_t*)input, is_encoder ? NULL : (elem_t*)enc_out, (elem_t*)output); \
    printf("%s stats: %s, hidden_dim=%d, expansion_dim=%d, num_heads=%d, cross_num_heads=%d, seq_len=%d, compression_factor=%d\n", \
            name, type_str, hidden_dim, expansion_dim, num_heads, cross_num_heads, seq_len, compression_factor); \
    printf("%s cycles: %llu\n\n", name, cycles); \
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
    printf("========== Running Transformer Benchmarks ==========\n");

    PRINT_ENCODER_DECODER("bert-base", /*is_encoder=*/true,
            /*hidden_dim=*/768, /*expansion_dim=*/3072, /*num_heads=*/12, /*cross_num_heads=*/12, /*seq_len=*/128, /*compression_factor=*/1);

    PRINT_ENCODER_DECODER("transformer-small", /*is_encoder=*/true,
            /*hidden_dim=*/512, /*expansion_dim=*/2048, /*num_heads=*/8, /*cross_num_heads=*/8, /*seq_len=*/128, /*compression_factor=*/1);

    printf("====================================================\n");

    exit(0);
}

