import pandas as pd
from collections import defaultdict

df = pd.read_parquet("datasets/deepmath_103k/deepmath_103k.parquet")

# Build a nested count tree
tree = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int)))))
counts = defaultdict(int)

for topic in df["topic"]:
    parts = [p.strip() for p in topic.split("->")]
    # accumulate counts at each level
    for depth in range(1, len(parts) + 1):
        key = " -> ".join(parts[:depth])
        counts[key] += 1

# Build hierarchy: level1 -> level2 -> ... (just store full paths by depth)
from collections import Counter

topic_series = df["topic"].str.strip()
level_counts = [Counter(), Counter(), Counter(), Counter(), Counter(), Counter()]

for topic in topic_series:
    parts = [p.strip() for p in topic.split("->")]
    for d, part in enumerate(parts):
        if d < len(level_counts):
            prefix = " -> ".join(parts[: d + 1])
            level_counts[d][prefix] += 1

# Print as tree
def print_tree(level_counts, max_depth=6):
    # Gather level-1 roots
    roots = sorted(level_counts[0].items(), key=lambda x: -x[1])
    for root, root_cnt in roots:
        print(f"\n{'='*70}")
        print(f"[L1] {root}  ({root_cnt:,})")
        print(f"{'='*70}")

        # L2 children
        l2 = sorted(
            [(k, v) for k, v in level_counts[1].items() if k.startswith(root + " -> ")],
            key=lambda x: -x[1],
        )
        for l2_key, l2_cnt in l2:
            l2_name = l2_key.split(" -> ")[-1]
            print(f"  {'─'*2} [L2] {l2_name}  ({l2_cnt:,})")

            # L3 children
            l3 = sorted(
                [(k, v) for k, v in level_counts[2].items() if k.startswith(l2_key + " -> ")],
                key=lambda x: -x[1],
            )
            for l3_key, l3_cnt in l3:
                l3_name = l3_key.split(" -> ")[-1]
                print(f"       {'─'*2} [L3] {l3_name}  ({l3_cnt:,})")

                # L4 children
                l4 = sorted(
                    [(k, v) for k, v in level_counts[3].items() if k.startswith(l3_key + " -> ")],
                    key=lambda x: -x[1],
                )
                for l4_key, l4_cnt in l4:
                    l4_name = l4_key.split(" -> ")[-1]
                    print(f"            {'─'*2} [L4] {l4_name}  ({l4_cnt:,})")

                    # L5 children
                    l5 = sorted(
                        [(k, v) for k, v in level_counts[4].items() if k.startswith(l4_key + " -> ")],
                        key=lambda x: -x[1],
                    )
                    for l5_key, l5_cnt in l5:
                        l5_name = l5_key.split(" -> ")[-1]
                        print(f"                 {'─'*2} [L5] {l5_name}  ({l5_cnt:,})")

                        l6 = sorted(
                            [(k, v) for k, v in level_counts[5].items() if k.startswith(l5_key + " -> ")],
                            key=lambda x: -x[1],
                        )
                        for l6_key, l6_cnt in l6:
                            l6_name = l6_key.split(" -> ")[-1]
                            print(f"                      {'─'*2} [L6] {l6_name}  ({l6_cnt:,})")


print(f"Total rows: {len(df):,}")
print(f"Unique topics: {df['topic'].nunique():,}")

# Depth distribution
depths = df["topic"].apply(lambda t: len(t.split("->")))
print(f"\nTopic depth distribution:")
for d, cnt in sorted(depths.value_counts().items()):
    print(f"  depth {d}: {cnt:,} rows")

print_tree(level_counts)
