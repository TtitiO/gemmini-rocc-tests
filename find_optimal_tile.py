from math import ceil

DIM = 16
SP_ROWS = 4 * 4096
ACC_ROWS = 1024

def fits(S, H, E, hidden, seq):
    sp = ceil(S*hidden/DIM) + ceil(hidden*E/DIM) + ceil(S*E/DIM)
    acc = ceil(S*H/DIM)
    return sp <= SP_ROWS and acc <= ACC_ROWS, sp, acc

def compute_iterations(S, H, E, hidden, expansion, seq):
    """Calculate number of Gemmini kernel launches needed"""
    s_iters = ceil(seq / S)
    h_iters = ceil(hidden / H)
    e_iters = ceil(expansion / E)
    total_iters = s_iters * h_iters * e_iters
    return total_iters, s_iters, h_iters, e_iters

def compute_utilization(S, H, E, hidden, expansion, seq):
    """Calculate how well tiles divide the actual dimensions"""
    s_util = (seq % S) / S if seq % S != 0 else 1.0
    h_util = (hidden % H) / H if hidden % H != 0 else 1.0
    e_util = (expansion % E) / E if expansion % E != 0 else 1.0
    avg_util = (s_util + h_util + e_util) / 3
    return avg_util

def find_optimal_tiles(hidden, expansion, seq, max_S=256, max_E=512, max_H=512):
    valid_configs = []

    for S in range(16, min(max_S + 1, seq + 1), 16):
        for E in range(16, min(max_E + 1, expansion + 1), 16):
            for H in range(16, min(max_H + 1, hidden + 1), 16):
                is_valid, sp_used, acc_used = fits(S, H, E, hidden, seq)

                if is_valid:
                    total_iters, s_iters, h_iters, e_iters = compute_iterations(
                        S, H, E, hidden, expansion, seq
                    )
                    util = compute_utilization(S, H, E, hidden, expansion, seq)

                    # Perfect division bonus (no partial tiles)
                    perfect_div = (seq % S == 0) and (hidden % H == 0) and (expansion % E == 0)

                    valid_configs.append({
                        'S': S, 'H': H, 'E': E,
                        'total_iters': total_iters,
                        's_iters': s_iters,
                        'h_iters': h_iters,
                        'e_iters': e_iters,
                        'sp_used': sp_used,
                        'acc_used': acc_used,
                        'sp_util': sp_used / SP_ROWS,
                        'acc_util': acc_used / ACC_ROWS,
                        'avg_util': util,
                        'perfect_div': perfect_div
                    })

    # Sort by: 1) fewest iterations, 2) perfect division, 3) highest utilization
    valid_configs.sort(key=lambda x: (x['total_iters'], not x['perfect_div'], -x['avg_util']))

    return valid_configs

# Test for bert-base
print("=" * 80)
print("BERT-BASE: hidden=768, expansion=3072, seq=128")
print("=" * 80)
configs = find_optimal_tiles(hidden=768, expansion=3072, seq=128, max_S=128, max_E=320, max_H=256)

print(f"\nTop 10 configurations (out of {len(configs)} valid):\n")
print(f"{'S':>4} {'H':>4} {'E':>4} | {'Iters':>6} {'S×H×E':>9} | {'SP%':>5} {'Acc%':>5} | {'Perfect':>7}")
print("-" * 80)

for i, cfg in enumerate(configs[:10]):
    print(f"{cfg['S']:>4} {cfg['H']:>4} {cfg['E']:>4} | "
          f"{cfg['total_iters']:>6} "
          f"{cfg['s_iters']:>2}×{cfg['h_iters']:>2}×{cfg['e_iters']:>2} | "
          f"{cfg['sp_util']*100:>5.1f} {cfg['acc_util']*100:>5.1f} | "
          f"{'Yes' if cfg['perfect_div'] else 'No':>7}")

# Test for transformer-small
print("\n" + "=" * 80)
print("TRANSFORMER-SMALL: hidden=512, expansion=2048, seq=256")
print("=" * 80)
configs = find_optimal_tiles(hidden=512, expansion=2048, seq=256, max_S=256, max_E=320, max_H=256)

print(f"\nTop 10 configurations (out of {len(configs)} valid):\n")
print(f"{'S':>4} {'H':>4} {'E':>4} | {'Iters':>6} {'S×H×E':>9} | {'SP%':>5} {'Acc%':>5} | {'Perfect':>7}")
print("-" * 80)

for i, cfg in enumerate(configs[:10]):
    print(f"{cfg['S']:>4} {cfg['H']:>4} {cfg['E']:>4} | "
          f"{cfg['total_iters']:>6} "
          f"{cfg['s_iters']:>2}×{cfg['h_iters']:>2}×{cfg['e_iters']:>2} | "
          f"{cfg['sp_util']*100:>5.1f} {cfg['acc_util']*100:>5.1f} | "
          f"{'Yes' if cfg['perfect_div'] else 'No':>7}")
