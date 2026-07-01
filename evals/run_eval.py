"""oOoBOT 평가셋 러너.

발화 시퀀스를 실제 파이프라인(의도 분석 → 행동 생성 → 결정론적 검증)에 흘려보내고,
각 턴의 결과 배치가 기대한 성질을 만족하는지 점검한다. STT는 건너뛰고 발화 텍스트를 직접 넣는다.

실행 (레포 루트에서, .env에 OPENAI_API_KEY 필요):
    python evals/run_eval.py

비결정성(LLM) 때문에 매번 완전히 같진 않으니, 체크는 좌표 하나하나가 아니라
'인원 유지/변경', '식탁이 실제로 줄었나', 'L+H 조합인가' 같은 성질 수준으로 둔다.
"""
import os
import sys
import json

# 레포 루트를 import 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from STT_sr_button_groq_openai import (
    ask_intent_to_openai,
    ask_action_to_openai,
    validate_robot_action,
    to_output_meters,
)
from openai import OpenAI


# ---------- 상태 조회 헬퍼 ----------
def active_kind(state, kind):
    """활성(active) 로봇 중 kind('L'/'H') 종류만."""
    key = "l_bots" if kind == "L" else "h_bots"
    return [r for r in state.get(key, []) if r.get("active") == "active"]


def actives(state):
    return active_kind(state, "L") + active_kind(state, "H")


# ---------- 평가 케이스 ----------
# 각 turn: {"utterance": 발화, "checks": [(설명, fn(intent, state, prev_state) -> bool), ...]}
CASES = [
    {
        "name": "밥먹자 → 식탁 줄여 → 혼자 독서(새 상황)",
        "turns": [
            {
                "utterance": "친구야 같이 밥 먹자",
                "checks": [
                    ("인원 2명 인식", lambda i, s, p: i.get("number") == 2),
                    ("자세 sitting", lambda i, s, p: i.get("posture") == "sitting"),
                    ("좌석(L BOT) 2대 이상", lambda i, s, p: len(active_kind(s, "L")) >= 2),
                    ("식탁(H BOT) 존재", lambda i, s, p: len(active_kind(s, "H")) >= 1),
                    ("식탁이 과대 구성 아님: H BOT 1대 (2명엔 한 대면 충분)", lambda i, s, p: len(active_kind(s, "H")) == 1),
                ],
            },
            {
                "utterance": "식탁이 너무 큰데",
                "checks": [
                    ("인원 2명 유지(맥락 유지)", lambda i, s, p: i.get("number") == 2),
                    ("직전과 배치가 달라짐(조정이 무시되지 않음)", lambda i, s, p: s != p),
                    ("테이블이 더 커지지는 않음", lambda i, s, p: len(active_kind(s, "H")) <= len(active_kind(p, "H"))),
                ],
            },
            {
                "utterance": "이제 나 혼자 책 읽을래",
                "checks": [
                    ("인원 1명으로 변경(새 상황)", lambda i, s, p: i.get("number") == 1),
                    ("자세 sitting", lambda i, s, p: i.get("posture") == "sitting"),
                    ("독서 의자(L BOT) 존재", lambda i, s, p: len(active_kind(s, "L")) >= 1),
                    ("책 놓을 높은 테이블(H BOT) 존재 [좌석+테이블=L+H 규칙]", lambda i, s, p: len(active_kind(s, "H")) >= 1),
                ],
            },
        ],
    },
]


def run():
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    total = passed = 0
    for case in CASES:
        print("\n=== CASE:", case["name"], "===")
        last_intent = None
        history = []
        prev_state = None
        for ti, turn in enumerate(case["turns"], 1):
            text = turn["utterance"]
            intent = ask_intent_to_openai(client, text, last_intent)
            if not intent:
                print(f"[turn {ti}] 의도 분석 실패 → 스킵")
                continue
            last_intent = intent
            raw = ask_action_to_openai(client, intent, history, text)
            state = validate_robot_action(raw)

            print(f"\n[turn {ti}] 발화: {text}")
            print("  의도 :", json.dumps(intent, ensure_ascii=False))
            print("  출력(m):", json.dumps(to_output_meters(state), ensure_ascii=False))
            for desc, fn in turn["checks"]:
                total += 1
                try:
                    ok = bool(fn(intent, state, prev_state))
                except Exception as e:
                    ok = False
                    desc += f" (예외: {e})"
                passed += 1 if ok else 0
                print(f"    [{'PASS' if ok else 'FAIL'}] {desc}")

            history.append(state)
            prev_state = state

    print(f"\n결과: {passed}/{total} 통과")
    return passed, total


if __name__ == "__main__":
    run()
