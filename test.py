import os
import json

def process_sc_data_recursive(input_path, output_all, output_wrong):
    # id를 키로 하여 중복을 제거하기 위한 딕셔너리
    unique_data = {}
    found_files_count = 0
    
    if not os.path.exists(input_path):
        print(f"경로를 찾을 수 없습니다: {input_path}")
        return

    print(f"대상 경로: {input_path}")
    print("파일 검색 중...")

    # 1. os.walk를 사용하여 하위 폴더의 모든 jsonl 파일 찾기
    for root, dirs, files in os.walk(input_path):
        for filename in files:
            if filename.endswith('.jsonl'):
                found_files_count += 1
                file_full_path = os.path.join(root, filename)
                
                with open(file_full_path, 'r', encoding='utf-8') as f_in:
                    for line in f_in:
                        line = line.strip()
                        if not line:
                            continue
                        
                        try:
                            data = json.loads(line)
                            item_id = str(data.get("id"))
                            
                            # ID 기준 중복 제거 (먼저 발견된 데이터 우선)
                            if item_id not in unique_data:
                                unique_data[item_id] = data
                        except json.JSONDecodeError:
                            continue

    # 2. 결과 파일 저장
    total_count = 0
    wrong_count = 0

    with open(output_all, 'w', encoding='utf-8') as f_all, \
         open(output_wrong, 'w', encoding='utf-8') as f_wrong:
        
        for item_id in unique_data:
            data = unique_data[item_id]
            json_line = json.dumps(data, ensure_ascii=False)
            
            # 전체 통합 파일 저장
            f_all.write(json_line + '\n')
            total_count += 1
            
            # is_right가 false인 데이터만 추출
            if data.get("is_right") is False:
                f_wrong.write(json_line + '\n')
                wrong_count += 1

    print("-" * 30)
    print(f"처리 결과")
    print(f"- 발견된 총 .jsonl 파일 수: {found_files_count}개")
    print(f"- 중복 제거 후 전체 문항: {total_count}개 -> {output_all}")
    print(f"- 이 중 틀린 문항(is_right: false): {wrong_count}개 -> {output_wrong}")
    print("-" * 30)

# 실행
if __name__ == "__main__":
    # 윤주님의 환경에 맞춘 경로 설정
    target_dir = '/mnt/yoonju/SC/output/base_wrong'
    all_file = 'base_single_reasoning.jsonl'
    wrong_file = 'base_single_reasoning_wrong.jsonl'

    process_sc_data_recursive(target_dir, all_file, wrong_file)