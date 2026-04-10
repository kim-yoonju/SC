import pandas as pd

df = pd.read_parquet("datasets/deepmath_16k.parquet")

result = df[df["topic"] == "Mathematics -> Algebra -> Intermediate Algebra -> Complex Numbers"]
print(f"총 {len(result)}개")

sample = result.iloc[1]
for col, val in sample.items():
    print(f"[{col}]\n{val}\n")
