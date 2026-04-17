from datasets import load_dataset

# 대체 경로인 'reasoning-machines/MWPBench'를 시도합니다.
try:
    dataset = load_dataset("reasoning-machines/MWPBench", "CollegeMath")
    print("성공적으로 데이터를 불러왔습니다.")
except Exception as e:
    # 위 경로도 안 될 경우, 원본 논문(MathScale) 데이터가 포함된 경로를 시도합니다.
    dataset = load_dataset("fdqerq22ds/MWPBench", "CollegeMath")
    print("미러 경로를 통해 데이터를 불러왔습니다.")

print(dataset)

import pandas as pd

# 'test' 슬라이스에 2,818개의 문항이 포함되어 있습니다.
df = dataset['test'].to_pandas()

# Parquet 파일로 저장
output_path = "/mnt/yoonju/NRL/S2R/CollegeMath_2818.parquet"
df.to_parquet(output_path, engine="pyarrow")

print(f"저장 경로: {output_path}")
print(f"데이터 개수: {len(df)}개") # 2818이 출력되어야 합니다.