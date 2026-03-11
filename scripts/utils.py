"""
공통 유틸리티: 답 파싱, 정답 확인, 프롬프트 포맷
"""

import re
from pathlib import Path
from typing import Optional, Tuple

# prompts/ 디렉토리는 프로젝트 루트(scripts/ 의 부모)에 위치
_PROMPT_DIR = Path(__file__).parent.parent / "prompts"


def _load_prompt(filename: str) -> str:
    """prompts/ 디렉토리에서 텍스트 파일을 읽어 반환한다."""
    path = _PROMPT_DIR / filename
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


# ---------------------------------------------------------------------------
# 파싱
# ---------------------------------------------------------------------------

def parse_step(text: str) -> Tuple[Optional[str], Optional[str]]:
    """생성된 텍스트에서 action 태그와 내용을 추출한다."""
    for action in ["solve", "correct", "end"]:
        pattern = rf"<{action}>(.*?)</{action}>"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return action, match.group(1).strip()
    return None, None


def parse_boxed(text: str) -> Optional[str]:
    r"""텍스트에서 \boxed{} 안의 내용을 추출한다 (중첩 괄호 처리)."""
    if text is None:
        return None
    pattern = r"\\boxed\{"
    matches = list(re.finditer(pattern, text))
    if not matches:
        return None
    # 마지막 \boxed{} 사용
    match = matches[-1]
    start = match.end()  # '{' 다음 위치
    depth = 1
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i].strip()
    return None


# ---------------------------------------------------------------------------
# 정답 비교
# ---------------------------------------------------------------------------

def normalize_answer(answer: Optional[str]) -> str:
    """비교를 위해 답을 정규화한다."""
    if answer is None:
        return ""
    answer = answer.strip().strip("$").replace(" ", "")
    answer = answer.replace("\\left", "").replace("\\right", "")
    answer = answer.replace("\\!", "").replace("\\,", "")
    return answer


def check_answer_correct(pred_text: Optional[str], gold_answer: str) -> bool:
    """모델 출력 텍스트와 정답이 일치하는지 확인한다."""
    if pred_text is None:
        return False

    # pred_text 에서 \boxed{} 추출 시도
    pred = parse_boxed(pred_text)
    if pred is None:
        pred = pred_text.strip()

    pred_norm = normalize_answer(pred)
    gold_norm = normalize_answer(gold_answer)

    if pred_norm == gold_norm:
        return True

    # 수치 비교
    try:
        pred_float = float(pred_norm)
        gold_float = float(gold_norm)
        return abs(pred_float - gold_float) < 1e-6
    except (ValueError, TypeError):
        pass

    return False


# ---------------------------------------------------------------------------
# 프롬프트 포맷
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = _load_prompt("system_prompt.txt")


def format_messages(problem: str, history: list) -> list:
    """문제와 이전 스텝 이력으로 chat messages 리스트를 구성한다."""
    user_content = f"Problem:\n{problem}"
    if history:
        user_content += "\n\nPrevious steps:\n"
        for i, step in enumerate(history, 1):
            user_content += f"Step {i}: {step}\n"
    user_content += "\nGenerate your next step (one action tag only):"

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def apply_chat_template(tokenizer, messages: list) -> str:
    """토크나이저의 chat template을 적용하거나, 없으면 수동으로 포맷한다."""
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    # Fallback: 수동 포맷
    text = f"System: {messages[0]['content']}\n\nUser: {messages[1]['content']}\n\nAssistant:"
    return text


def extract_first_action(generated: str) -> str:
    """생성된 텍스트에서 첫 번째 완전한 action 태그만 반환한다."""
    for action in ["solve", "correct", "end"]:
        pattern = rf"<{action}>.*?</{action}>"
        match = re.search(pattern, generated, re.DOTALL)
        if match:
            return match.group(0)
    # 완전한 태그가 없으면 원문 반환 (유효하지 않음 처리는 호출부에서)
    return generated.strip()
