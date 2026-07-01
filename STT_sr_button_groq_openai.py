import ctypes
from io import BytesIO
import json
import math
import os
from dotenv import load_dotenv
load_dotenv()
import time

import speech_recognition as sr

from groq import Groq
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# API 키
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Groq Whisper API에서 사용할 모델/언어
GROQ_MODEL = "whisper-large-v3-turbo"
GROQ_LANGUAGE = "ko"

# OpenAI API에서 사용할 모델
OPENAI_MODEL = "gpt-5.5"

# 폴링 (대기시간) 간격 설정
POLL_SECONDS = 0.01

# 스페이스바 눌림 감지 확인 (0x20은 스페이스바의 가상 키 코드)
def is_space_pressed():
    return bool(ctypes.windll.user32.GetAsyncKeyState(0x20) & 0x8000)

# 스페이스바가 눌려있는 동안 음성 녹음
def record_while_space_is_pressed(source):
    frames = []
    print("Recording... release Space to stop.")

    while is_space_pressed():
        frames.append(source.stream.read(source.CHUNK))

    return sr.AudioData(
        b"".join(frames),
        source.SAMPLE_RATE,
        source.SAMPLE_WIDTH,
    )

# Groq API를 사용해 음성 인식
def transcribe_audio(client, audio):
    try:
        wav_file = BytesIO(audio.get_wav_data())
        wav_file.name = "speech.wav"

        transcription = client.audio.transcriptions.create(
            file=wav_file,
            model=GROQ_MODEL,
            language=GROQ_LANGUAGE,
        )
        text = transcription.text
        print("You said: " + text)
        return text
    except sr.UnknownValueError:
        print("Sorry, I could not understand what you said.")
    except Exception as e:
        print("Sorry, an error occurred while processing your request: {0}".format(e))
    return None

# 전사된 텍스트를 OpenAI LLM를 통해 의도 분석
def ask_intent_to_openai(client, usertext, prev_intent=None):
    try:
        response = client.responses.create(
            model=OPENAI_MODEL,
            input=[
                {
                    "role": "developer",
                    "content": "너는 사용자가 하는 말의 의도를 최대한 정확하게 알아듣는 의도 파악 전문가야."
                    "입력은 '직전 상황(prev_intent)'과 이번에 해석할 '새 발화(utterance)'로 나뉜다. prev_intent는 직전까지 파악된 맥락이고, 이번에 해석할 대상은 오직 새 발화다."
                    "number(인원)는 대화 내내 이어지는 맥락이다. 새 발화가 '테이블이 너무 작은데?', '더 크게', '원래대로'처럼 이미 구성된 가구를 평가·조정하는 말이면, 인원수를 새로 세지 말고 prev_intent의 number를 '그대로' 유지하라. 이런 조정성 발화에서 발화에 사람이 명시되지 않았다고 number를 1로 줄이지 마라."
                    "number를 바꾸는 경우는 다음 두 가지뿐이다: (1) 새 발화가 이전과 다른 '새로운 상황/장면'을 묘사할 때(예: '아 졸려', '이제 일하자', '손님 오신다'), (2) 누군가 '합류하거나 떠나는' 정황이 분명할 때(예: '친구 한 명 더 왔어', '다들 갔어', '나 혼자 남았어'). 그 외에는 prev_intent의 number를 유지하라."
                    "posture(자세: standing/sitting/lying)도 number처럼 이어지는 맥락이다. 발화·행동에서 자세를 추론하되(예: '앉아서 책 읽자'→sitting, '누워서 쉴래'→lying, '서서 작업'→standing), 조정성 발화('테이블이 너무 작은데?')에서는 직전 posture를 그대로 유지하라. 자세가 드러나지 않고 prev_intent도 없으면 행동에 가장 자연스러운 자세로 추정하라(기본은 standing)."
                    "situation·activity·furniture도 number·posture와 똑같은 '맥락 필드'다. '테이블이 너무 작은데?', '더 크게', '원래대로' 같은 조정성 발화에서는 이 세 값을 새로 지어내지 말고 prev_intent의 값을 그대로 유지하라 — 발화가 그 내용을 실제로 바꿀 때(새로운 상황·행동·가구가 등장할 때)만 갱신한다. prev_intent가 null이면(첫 발화) 발화에서 직접 인원·자세·상황·행동·가구를 파악하라.",
                },
                {
                    "role": "user",
                    "content": json.dumps({
                        "직전 상황(prev_intent)": prev_intent,
                        "새 발화(utterance)": usertext,
                    }, ensure_ascii=False),
                },
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "intent_result",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "number": {
                                "type": "number",
                                "description": "사용자의 발화를 통해 파악할 수 있는 현재 상황의 인원 수"
                            },
                            "situation": {
                                "type": "string",
                                "description": "사용자의 발화를 통해 알 수 있는 사용자가 처한 상황"
                            },
                            "activity": {
                                "type": "string",
                                "description": "사용자의 발화를 통해 알 수 있는 사용자가 하게 될 행동"
                            },
                            "furniture": {
                                "type": "string",
                                "description": "사용자의 발화를 통해 알 수 있는 사용자가 필요한 가구"
                            },
                            "posture": {
                                "type": "string",
                                "enum": ["standing", "sitting", "lying"],
                                "description": "사용자의 현재 자세. standing=서 있음, sitting=앉아 있음, lying=누워 있음. 발화·행동으로 추론하고, 조정성 발화면 직전 자세를 유지한다."
                            },
                        },
                        "required": ["number", "situation", "activity", "furniture", "posture"],
                        "additionalProperties": False,
                    },
                }
            },
        )

        result = json.loads(response.output_text)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return result
    except Exception as e:
        print("Sorry, an error occurred while asking OpenAI: {0}".format(e))

# 출력된 의도 분석 결과를 OpenAI LLM를 통해 로봇 행동 제안
def ask_action_to_openai(client, intent, history, utterance=""):
    try:
        # 좌표/치수는 아래 상수에서 계산해 프롬프트에 주입 → 프롬프트와 코드가 어긋나지 않도록 단일 출처 유지
        ux, uy = USER_X_CM, USER_Y_CM
        W, D = SPACE_W_CM, SPACE_D_CM
        (lr0, lr1) = ROBOT_RADIUS_CM["L"]
        (hr0, hr1) = ROBOT_RADIUS_CM["H"]
        h_conn = round(hr1 * 2)      # H BOT size100 두 대 연결 시 중심 간격
        l_conn = round(lr1 * 2)      # L BOT size100 두 대 연결 시 중심 간격
        # 자세별 앵커(사용자 중심 기준 오프셋으로 계산 → 원점/공간 크기가 바뀌어도 자동으로 따라 이동)
        A_stand_front = (ux, uy + 50)
        A_stand_R, A_stand_L = (ux + 40, uy + 10), (ux - 40, uy + 10)
        A_seat = (ux, uy)
        A_sit_tableR, A_sit_tableL = (ux + 56, uy), (ux - 56, uy)
        A_foot, A_sit_lowside = (ux, uy + 45), (ux + 44, uy)
        A_bed1, A_bed2 = (ux, uy - 33), (ux, uy + 33)
        A_lie_R, A_lie_L = (ux + 45, uy + 65), (ux - 45, uy + 65)
        A_hconn_L, A_hconn_R = (ux - round(hr1), uy), (ux + round(hr1), uy)
        response = client.responses.create(
            model=OPENAI_MODEL,
            input=[
                {
                    "role": "developer",
                    "content": "너는 사용자의 상황과 행동에 맞는 로봇 가구 환경을 구성하는 전문가야."
                    "각 로봇에는 두 축이 있다 — (가) 높이: L BOT은 항상 낮고(윗면 높이 약 44.5cm — 앉는 면·발받침·낮은 협탁 높이), H BOT은 항상 높다(약 70.5cm — 서서 쓰거나 앉은 사람이 무릎 위에서 쓰는 테이블·작업면 높이). size로 높이는 안 바뀐다. (나) size(0~100): 윗면 형태를 정한다. 0=좁고 평평한 면, 중간(25·50·75)=가운데가 패인 그릇(작을수록 깊은 바구니, 클수록 넓고 얕은 트레이), 100=크고 평평한 원형 면."
                    "size 기준점(앵커): 평평한 면으로 앉거나 작은 물건을 올려두는 용도 → size 0. 안에 물건을 담아 두거나 옮기는 용도 → 중간 size(담을 양·깊이에 따라 25,50,75). 넓고 평평한 면에 여러 물건을 올리거나 테이블로 쓰는 용도 → size 100. 이건 용도에 맞는 형태를 고르는 기준점이니, 특별한 이유가 없으면 이 기준을 따르라. "
                    "size에 따른 윗면 형태: size가 0이면 윗면이 좁고 평평하다. size가 커지면 윗면 가운데가 아래로 패면서 뒤집힌 원뿔 모양의 주머니(안에 물건을 담을 수 있는 그릇)가 생긴다. size가 더 커질수록 이 주머니는 반지름이 넓어지는 대신 깊이는 얕아진다 — 작은 size에서는 좁고 깊은 바구니, 큰 size로 갈수록 넓고 얕은 그릇·트레이가 된다. size 100이면 주머니가 완전히 펴져 하나의 크고 평평한 원형 면(테이블처럼 쓸 수 있는 넓은 윗면)이 된다."
                    "여기서 따라오는 물리적 사실: (1) 높이는 종류로 정해진다 — 낮게 앉거나 발을 올리는 용도엔 낮은 L BOT이, 서서 쓰거나 물건을 높이 올려두는 용도엔 높은 H BOT이 맞다. (2) size 양 끝(0과 100 부근)은 평평한 면이라 앉거나 물건을 올려두기 좋고, 중간 size는 가운데가 패인 그릇이라 물건을 담아 두거나 옮기기 좋다. 같은 중간이라도 작은 size는 깊은 바구니, 큰 size는 넓고 얕은 트레이에 가깝다. 이건 제한이 아니라 형태가 가진 사실일 뿐이니, 용도에 맞는 로봇 종류와 size를 네가 판단할 근거로만 써라."
                    "furniture는 자유 라벨이다. 위 size 기준은 '용도→형태(size)'를 고르기 위한 것이지 이름을 정해진 목록에 가두는 게 아니다. 같은 size라도 사용자의 위치·상황·용도에 따라 이름은 자유롭게 붙여라 — 예를 들어 size 0인 낮은 L BOT은 앉으면 '낮은 의자', 발을 올리면 '발 받침대', 옆에 두면 '낮은 협탁'이 된다. 실제 명령값은 size이고 furniture는 그 형태를 맥락에 맞게 표현한 이름일 뿐이다."
                    "'상하관계'가 필요한 조합에 주의하라: 사람이 앉는 면과, 그 자세에서 위에 올려두고 쓰는 작업면(책·식사·노트북을 놓는 테이블)이 함께 필요하면, 둘은 높이 차이가 있어야 실제로 쓸 수 있다. 앉는 좌석은 낮아야 하니 L BOT, 그 위에서 쓰는 테이블은 앉은 무릎보다 높아야 하니 H BOT으로 구성하라 — 즉 '좌석+테이블'은 L BOT(좌석)+H BOT(테이블) 조합이 기본이다. 좌석과 테이블을 둘 다 같은 높이(예: L BOT 두 대)로 만들면 테이블이 좌석과 같은 높이라 무릎 위에서 쓸 수 없다. (단순히 옆에 물건을 얹어만 두는 낮은 협탁처럼 높이 차가 필요 없는 경우는 예외다.) 필요한 종류의 로봇이 지금 inactive로 놀고 있으면, 다른 종류를 억지로 그 용도로 바꾸지 말고 그 종류를 활성화해서 써라."
                    "inactive(미사용) 로봇은 아무 용도도 배정되지 않은 '빈' 로봇이다. inactive면 furniture는 항상 'none'으로 두고(코드도 강제한다), history에서 inactive 로봇에 남아 있던 옛 라벨(예: 직전 식사 때의 '접힌 식탁')을 보고 '이미 그 가구가 있다'고 오해하지 마라. 그 로봇은 지금 자유롭게 다른 용도로 활성화할 수 있는 빈 자원이다."
                    "한 대로 충분한 상황이면 한 대만 쓰고, 환경을 더 편하고 풍부하게 만들면 좋을 상황이면 여러 대를 조합해 구성하라. 어떤 상황에 어떤 조합이 정답이라는 고정된 세트는 없으니, 매번 그 상황을 보고 필요한 만큼만 자연스럽게 새로 판단하라. 무조건 많이 켜지도, 무조건 한 대만 쓰지도 마라."
                    "가용 로봇은 L BOT 2대(L BOT 1, L BOT 2)와 H BOT 2대(H BOT 1, H BOT 2)로 총 4대뿐이다. 활동에 가구가 더 많이 필요해 보여도 이 4대를 초과해 만들어낼 수 없다. 수요가 4대를 넘으면, 가장 중요한 가구부터 우선순위를 정해 한정된 4대 안에서 최선의 조합을 구성하라."
                    "입력은 세 부분이다 — '과거 기록(history)', 구조화된 '새 요청(new_request)', 그리고 사용자가 이번에 실제로 한 말 '발화 원문(utterance)'. new_request는 대화가 이어지며 유지되는 안정적 맥락(인원·자세·상황 등)이고, utterance는 이번 턴의 진짜 명령이다. 특히 '식탁이 너무 큰데', '더 작게', '더 크게', '원래대로'처럼 직전 구성을 바꾸라는 '상대적 조정'은 new_request만 보면 직전과 똑같아 보일 수 있으니, 반드시 utterance를 읽어 history 최신 상태를 기준으로 그 변화를 실제로 반영하라. history는 상대 표현 해석용 참고이지 그대로 복사하는 게 아니다."
                    "【핵심 작동 원리】 너의 일은 '현재 상태(history 최신 항목)'를 utterance가 요구하는 대로 편집해 새 전체 상태를 내는 '상태 편집기'다. 로봇에서 바꿀 수 있는 축은 딱 네 가지뿐이다: (1) active — 켜기/끄기(대수 늘리고 줄이기), (2) size — 윗면 형태(0=평평, 중간=그릇, 100=넓은 평면), (3) x·y — 위치, (4) 인접 연결 — 여러 대를 붙여 복합 가구로. '줄여/키워/치워/옮겨/새로 만들어/바꿔' 등 세상의 어떤 요청이든 결국 이 네 축의 조합으로 표현된다. 그러니 '이 상황엔 이렇게'라는 정해진 규칙표를 찾지 말고, 매번 그 자리에서 '사용자의 말이 이뤄지려면 이 네 축 중 무엇을 어떻게 바꿔야 하는가'를 추론해 적용하라. 아래에 나오는 구체 예(더 크게→대수↑, 너무 크다→대수↓, 치워→접기 등)는 지켜야 할 규칙이 아니라 이 원리를 보여주는 예시일 뿐이다. 예에 없는 새 요청도 같은 방식으로 네 축을 조절해 처리하라."
                    "history의 가장 최근 항목이 로봇 4대의 현재 상태다. 매번 4대(L BOT 1, L BOT 2, H BOT 1, H BOT 2) 전체 상태를 절대값으로 출력해. 이번 요청과 직접 관련 없는 로봇은 현재 상태(history 최신 항목) 값을 그대로 유지해 출력하라. 멋대로 기본값으로 되돌리거나 끄지 마라."
                    "size는 [0, 25, 50, 75, 100] 다섯 단계 중 하나를 쓴다. furniture는 완전히 자유로운 라벨이며 정해진 이름 목록이 없다. 실제 명령값은 size이고 furniture는 그 형태를 맥락에 맞게 표현한 이름일 뿐이다."
                    "'더 크게/작게', '너무 작다/크다'처럼 크기를 바꾸라는 요청은 두 경우로 나눠 처리하라. (가) 담는 용도(중간 size의 그릇·바구니·트레이)라면 furniture 라벨만 바꾸지 말고 size를 실제로 한 단계 이상 올리거나 내려 깊이·넓이를 조정하라. history 최신 항목의 현재 size를 기준으로 조정하고, 바뀐 형태에 맞는 라벨을 다시 붙여라. (나) 평평한 면으로 쓰는 가구(테이블=size 100, 좌석·발받침=size 0)는 size를 바꾸면 평면이 깨져 용도를 잃는다 — 이런 평면 가구가 '너무 작다/더 크게'면 size는 그대로 두고 같은 용도의 로봇을 한 대 더 붙여(연결) 면적을 넓히고, 반대로 '너무 크다/더 작게'면 그 가구를 이루는 로봇 중 한 대를 접어(inactive) 대수를 줄여 면적을 좁혀라(예: 대형 식탁이 H BOT 두 대면 한 대를 접어 1인용 크기로). 어느 경우든 이 조정은 현재 그 가구로 쓰이는 모든 로봇에 같이 적용하라."
                    "'치워', '그만', '다 접어', '정리하자'처럼 사용 종료·정리를 뜻하는 말에는 해당 로봇들을 active='inactive', size=0으로 되돌려라. (수납가구로 변형하라는 뜻이 아니라, 로봇을 접어 홈으로 보내 쉬게 하라는 뜻이다.) inactive 로봇의 위치는 코드가 홈으로 정리하므로 신경 쓰지 않아도 된다."
                    f"이제 size에 더해 각 로봇의 위치(x, y)도 함께 출력한다. 좌표계의 원점 (0,0)은 로봇들이 대기하는 구석이고, 사용자는 가동 공간 중심 ({ux}, {uy})에 서서 +y(앞)를 바라본다. x가 클수록 사용자 기준 오른쪽, y가 클수록 앞이며 단위는 cm다. 로봇 윗면은 원형이라 방향(회전)은 의미가 없으므로 위치(x,y)만 정하면 된다. "
                    f"가동 공간은 가로 {W}cm x 세로 {D}cm이라 좌표 범위는 x는 0~{W}, y는 0~{D}이다(사용자 중심은 ({ux},{uy})). 로봇은 실제 크기가 있으니(H BOT 반지름 약 {hr0}~{hr1}cm, L BOT 약 {lr0}~{lr1}cm, size가 클수록 큼) 본체가 이 경계를 넘지 않도록 여유를 두고 배치하라(경계를 넘는 좌표는 코드가 안으로 끌어당긴다). "
                    "미사용(inactive) 로봇은 접힌 채 뒷줄(y=0 근처)에 일렬로 각자 자리(dock)에서 대기한다. 로봇을 쓰지 않게 되면 그 자리로 돌려보낸다(코드가 자동 처리하므로 inactive 로봇의 위치는 신경 쓰지 않아도 된다). "
                    f"위치는 사용자의 '자세'(intent.posture: standing/sitting/lying)에 따라 달라진다. 사용자 신체 기준: 키 160cm, 다리 74cm, 팔 54cm, 오른손잡이. 그래서 옆에 두는 사이드 가구(협탁·트레이 등)는 손이 잘 닿는 '오른쪽(사용자 x={ux}보다 큰 쪽)에 먼저' 배정하고, 하나 더 필요하면 그다음 '왼쪽(x가 {ux}보다 작은 쪽)'에 둔다. "
                    "닿아야 하는 작은 사이드 가구(협탁·트레이)는 팔을 끝까지 뻗지 않고도 편히 닿도록, 사용자(앉거나 누웠으면 그 손) 기준 대략 '팔 길이의 절반 정도(약 25~45cm)' 거리에 둔다. 이 25~45cm 범위 안에서 더 가까이(25)로 둘지 더 멀리(45)로 둘지는 상황(다른 가구와의 간섭, 동선)을 보고 네가 판단하라. 단, 부피가 큰 작업 테이블(특히 좌석 바깥에 둬야 하는 H BOT 테이블)은 로봇이 커서 더 멀리 가므로 약 45~55cm로 본다. 가운데에 좌석·침대처럼 큰 로봇이 있으면 사이드 가구는 그 로봇에 겹치지 않는 바깥쪽에 둔다. "
                    f"자세별 배치 가이드라인(원점=구석 홈, 사용자 중심=({ux},{uy}), +y=앞. 아래 좌표는 고정 규칙이 아니라 앵커이니 상황에 맞게 조정하라): "
                    f"[standing 서 있음] 앞에서 서서 쓰는 작업면·테이블: {A_stand_front} 부근. 오른쪽 사이드: {A_stand_R}, 왼쪽 사이드: {A_stand_L}. "
                    f"[sitting 앉음] 좌석은 '낮은 L BOT'을 사용자 바로 아래 {A_seat}에 둔다. 앉아서 책·식사·작업에 쓰는 테이블은 좌석보다 높아야 하므로 'H BOT'(무릎 위 높이)으로, 다리를 막지 않게 손 닿는 옆쪽에 둔다 — 오른쪽 {A_sit_tableR}, 더 필요하면 왼쪽 {A_sit_tableL}(큰 테이블이라 좌석 바깥이라 45~55cm로 살짝 멀다). 발받침이 필요하면 낮은 L BOT을 앞쪽 {A_foot}에 둔다. 단순히 물건만 얹는 낮은 협탁이면 낮은 L BOT을 옆 {A_sit_lowside}에 둬도 된다. "
                    f"[lying 누움] 침대는 낮은 L BOT 두 대를 size 100으로 몸 아래 y축을 따라 이어 붙인다: {A_bed1}과 {A_bed2}(연결 간격 {l_conn}cm). 침대 상판(반지름 약 {lr1}cm)을 벗어난 머리맡(+y쪽)에 사이드: 오른쪽 {A_lie_R}, 왼쪽 {A_lie_L}. "
                    f"핵심: 같은 size라도 '어디에 두느냐'에 따라 가구의 의미가 달라진다. 위치까지 정하면 의자/발받침대/협탁 구분이 라벨이 아니라 좌표에서 자연스럽게 유도된다. 예를 들어 size 0인 낮은 L BOT을 사용자 바로 앞 가까이(y가 사용자 {uy}보다 조금 큰 쪽)에 두면 '발받침대', 사용자 옆(x가 {ux}에서 좌우로 벗어난 쪽)에 두면 '낮은 협탁', 앉을 자리에 두면 '낮은 의자'가 된다. 위치는 사용자의 동선과 손이 닿는 범위를 고려해 자연스럽게 정하고, furniture 라벨도 그 위치·형태에 맞게 붙여라. "
                    "여러 로봇을 독립 가구로 쓸 때는 서로 본체가 겹치지 않게 떨어뜨려라(중심 간 거리가 두 로봇 반지름의 합보다 크도록). 단, 충돌·경계의 '최종' 안전 검증은 코드(결정론적 레이어)가 책임지므로, 너는 물리적으로 그럴듯한 배치를 제안하는 데 집중하면 된다. "
                    "size와 마찬가지로, 이번 요청과 직접 관련 없는 로봇의 위치(x,y)도 history 최신 상태값을 그대로 유지해 출력하라. 멋대로 0으로 되돌리지 마라."
                    "intent의 number(인원)를 적극 반영하라. 단, size는 인원수에 비례해 올리는 값이 '아니다' — size는 용도에 맞는 윗면 형태(평평한 면=0 또는 100, 담는 그릇=중간 25·50·75)를 고르는 값이다. 그러니 인원이 많아 더 넓은 면적이나 더 많은 좌석이 필요할 때는 size를 어중간하게 올리지 말고, ⓐ같은 용도의 로봇을 한 대 더 쓰거나 ⓑ여러 대를 인접 연결해 복합 가구로 넓혀라. 예: 테이블이 좁다고 size를 75로 내리면 평면이 아니라 오목한 트레이가 되어 테이블이 못 된다 — 대신 size 100짜리 H BOT을 한 대 더 옆에 붙여 더 큰 테이블로 만든다. 좌석이 더 필요하면 size 0 L BOT을 한 대 더 둔다. '테이블이 너무 작은데?' 같은 조정 요청도 size를 바꾸는 게 아니라 이렇게 면적을 키우라는 뜻으로 해석하라."
                    f"로봇은 한 대씩 독립된 가구로만 쓰는 게 아니라, 여러 대를 인접하게 붙여(위치 x,y를 맞닿게) 하나의 더 큰 '복합 가구'로 합칠 수 있다. 예: H BOT 두 대를 size 100으로 양옆에 붙이면 다인용 대형 테이블이 되고, L BOT 두 대를 size 100으로 붙이면 침대처럼 넓은 평면이 된다. 이건 정해진 목록이 아니라 가능성의 예시일 뿐이니, 인원·상황에 맞는 새로운 조합도 자유롭게 구성하라. 복합 가구로 붙일 때는 두 로봇이 같은 자리에 포개지지 않고 가장자리만 맞닿도록, 중심 간 거리를 '두 로봇 반지름의 합' 정도로 두어라. 예: H BOT 두 대를 size 100으로 연결하면 중심 간 약 {h_conn}cm(예: {A_hconn_L}과 {A_hconn_R}), L BOT 두 대를 size 100으로 연결하면 약 {l_conn}cm 간격이다. 한 복합 가구를 이루는 로봇들에는 그 사실이 드러나는 일관된 furniture 라벨을 붙여라(예: 둘 다 '대형 테이블')."
                    "입력 받은 사용자의 인원수, 상황, 행동, 필요한 가구에 맞게 로봇의 대수와 위치, 단독/복합 구성, 그 로봇이 수행해야 할 행동을 제안해줘."
                },
                {
                    "role": "user",
                    "content": json.dumps({
                        "과거 기록(history)": history[-3:],
                        "지금 처리할 새 요청(new_request)": intent,
                        "사용자 발화 원문(utterance)": utterance,
                    }, ensure_ascii=False),
                },
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "robot_action_result",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "l_bots": {
                                "type": "array",
                                "description": "낮은 로봇(L BOT)들의 목록. 2대(L BOT 1, L BOT 2)를 모두 출력하되 사용하지 않는 로봇은 active를 'inactive'로 표시할 것",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "robot": {
                                            "type": "string",
                                            "enum": ["L BOT 1", "L BOT 2"],
                                            "description": "행동을 수행할 낮은 로봇의 이름"
                                        },
                                        "active": {
                                            "type": "string",
                                            "enum": ["active", "inactive"],
                                            "description": "로봇의 활성상태 (사용할 로봇은 'active', 사용하지 않을 로봇은 'inactive')"
                                        },
                                        "furniture": {
                                            "type": "string",
                                            "description": "size를 사람이 읽기 쉽게 표현한 가구 라벨 (가이드라인일 뿐 실제 명령값은 size)"
                                        },
                                        "size": {
                                            "type": "number",
                                            "enum": [0, 25, 50, 75, 100],
                                            "description": "로봇의 확장 단계. 0=완전 축소, 25=조금 확장, 50=중간, 75=많이 확장, 100=완전 확장. 이 다섯 단계 중 하나만 사용."
                                        },
                                        "x": {
                                            "type": "number",
                                            "description": "가동 공간의 가로 위치(cm, 0~240). 원점 0은 로봇 홈이 있는 구석, 사용자 중심은 x=120. 값이 클수록 사용자 기준 오른쪽. inactive면 0."
                                        },
                                        "y": {
                                            "type": "number",
                                            "description": "가동 공간의 세로 위치(cm, 0~200). 원점 0은 홈 구석(뒤쪽), 사용자 중심은 y=100. 값이 클수록 앞(사용자 정면). inactive면 0."
                                        }
                                    },
                                    "required": ["robot", "active", "furniture", "size", "x", "y"],
                                    "additionalProperties": False,
                                }
                            },
                            "h_bots": {
                                "type": "array",
                                "description": "높은 로봇(H BOT)들의 목록. 2대(H BOT 1, H BOT 2)를 모두 출력하되 사용하지 않는 로봇은 active를 'inactive'로 표시할 것",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "robot": {
                                            "type": "string",
                                            "enum": ["H BOT 1", "H BOT 2"],
                                            "description": "행동을 수행할 높은 로봇의 이름"
                                        },
                                        "active": {
                                            "type": "string",
                                            "enum": ["active", "inactive"],
                                            "description": "로봇의 활성상태 (사용할 로봇은 'active', 사용하지 않을 로봇은 'inactive')"
                                        },
                                        "furniture": {
                                            "type": "string",
                                            "description": "size를 사람이 읽기 쉽게 표현한 가구 라벨 (가이드라인일 뿐 실제 명령값은 size)"
                                        },
                                        "size": {
                                            "type": "number",
                                            "enum": [0, 25, 50, 75, 100],
                                            "description": "로봇의 확장 단계. 0=완전 축소, 25=조금 확장, 50=중간, 75=많이 확장, 100=완전 확장. 이 다섯 단계 중 하나만 사용."
                                        },
                                        "x": {
                                            "type": "number",
                                            "description": "가동 공간의 가로 위치(cm, 0~240). 원점 0은 로봇 홈이 있는 구석, 사용자 중심은 x=120. 값이 클수록 사용자 기준 오른쪽. inactive면 0."
                                        },
                                        "y": {
                                            "type": "number",
                                            "description": "가동 공간의 세로 위치(cm, 0~200). 원점 0은 홈 구석(뒤쪽), 사용자 중심은 y=100. 값이 클수록 앞(사용자 정면). inactive면 0."
                                        }
                                    },
                                    "required": ["robot", "active", "furniture", "size", "x", "y"],
                                    "additionalProperties": False,
                                }
                            }
                        },
                        "required": ["l_bots", "h_bots"],
                        "additionalProperties": False,
                    },
                }
            },
        )

        result = json.loads(response.output_text)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return result
    except Exception as e:
        print("Sorry, an error occurred while asking OpenAI: {0}".format(e))
        # 6. LLM 응답 실패/파싱 깨짐 시: 로봇이 의도치 않게 움직이지 않도록 4대 모두 inactive 기본값 반환
        return default_inactive_command()

# 과거 명령 history (검증을 통과한 명령 객체만 저장. 발화나 의도는 넣지 않음).
# history의 가장 최근 항목이 곧 로봇 4대의 현재 상태이므로 별도의 현재상태 변수는 두지 않는다.
command_history = []

# 6. LLM 응답 실패 시 사용할 안전 기본값: 4대 모두 inactive (각자 홈 도크에서 대기)
def _inactive_robot(name):
    hx, hy = home_for({"robot": name})
    return {"robot": name, "active": "inactive", "furniture": "none", "size": 0, "x": hx, "y": hy}

def default_inactive_command():
    return {
        "l_bots": [_inactive_robot("L BOT 1"), _inactive_robot("L BOT 2")],
        "h_bots": [_inactive_robot("H BOT 1"), _inactive_robot("H BOT 2")],
    }

# --- 물리 치수 / 공간 상수 (cm) (결정론적 안전 레이어) ---
# 가동 공간은 240(가로) x 200(세로). 원점 (0,0)은 로봇 홈이 있는 구석이고,
# 사용자는 그 중심 (120,100)에 서서 +y(앞)를 바라본다고 가정. 좌표 범위 x∈[0,240], y∈[0,200].
SPACE_W_CM = 240         # 가동 공간 가로 (x: 0 ~ 240)
SPACE_D_CM = 200         # 가동 공간 세로 (y: 0 ~ 200)
USER_X_CM = 120          # 사용자 중심 x
USER_Y_CM = 100          # 사용자 중심 y (사용자는 +y(앞)를 바라봄)

# 미사용(inactive) 로봇은 접힌 채 뒷줄(y=0)에 일렬로 각자 자리(dock)에서 대기한다.
# 4대가 한 점에 겹치지 않도록 x를 벌려 배치(접힌 반지름 ~22cm보다 넉넉히 60cm 간격).
HOME_DOCKS = {
    "L BOT 1": (30, 0),
    "L BOT 2": (90, 0),
    "H BOT 1": (150, 0),
    "H BOT 2": (210, 0),
}

def home_for(robot):
    # 로봇 이름에 맞는 대기 도크 좌표. 이름을 모르면 구석(0,0).
    return HOME_DOCKS.get(robot.get("robot", ""), (0, 0))

# 최종 출력 단위 변환: 내부 계산은 cm, 로봇에 보내는 출력만 m(float)
CM_PER_M = 100.0

# 로봇 종류별 반지름(cm): size 0(base) ~ size 100(top)을 선형 보간.
#   H BOT: 41x41(size0) -> 72x72(size100)  => 반지름 20.5 -> 36.0
#   L BOT: 43x43(size0) -> 65x65(size100)  => 반지름 21.5 -> 32.5
ROBOT_RADIUS_CM = {
    "H": (20.5, 36.0),
    "L": (21.5, 32.5),
}

# 정수 반올림 오차를 흡수할 여유 (cm)
CONNECT_SLACK_CM = 2

def _robot_kind(robot):
    # 이름 앞글자로 종류 판별 ("H BOT 1" -> H, "L BOT 2" -> L)
    return "H" if str(robot.get("robot", "")).upper().startswith("H") else "L"

def robot_radius(robot):
    # 현재 size에서 로봇이 차지하는 반지름(cm). base~top 선형 보간.
    base, top = ROBOT_RADIUS_CM[_robot_kind(robot)]
    size = robot.get("size", 0)
    if not isinstance(size, (int, float)):
        size = 0
    size = max(0, min(100, size))
    return base + (top - base) * (size / 100.0)

def _coerce_number(value, default=0):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    if math.isnan(value) or math.isinf(value):
        return default
    return value

def _iter_robots(action):
    for key in ("l_bots", "h_bots"):
        for robot in action.get(key, []):
            yield robot

# 위치 정규화: 좌표를 정수 cm로 만들고, 로봇 본체가 가동 공간(240x200) 밖으로 나가지 않게 안으로 clamp.
def _normalize_position(robot):
    x = round(_coerce_number(robot.get("x", 0)))
    y = round(_coerce_number(robot.get("y", 0)))
    # 로봇 반지름만큼 여유를 두어 본체 전체가 경계 [r, 크기-r] 안에 들어오도록 중심을 제한한다.
    r = robot_radius(robot)
    x = max(r, min(SPACE_W_CM - r, x))
    y = max(r, min(SPACE_D_CM - r, y))
    robot["x"] = int(round(x))
    robot["y"] = int(round(y))

# 겹치지 않는 위치 찾기: (x,y)에서 시작해, 이미 자리 잡은 로봇과 파고들면 그 로봇에서 바깥으로
# '맞닿는 거리'까지 밀어낸다. 경계 안으로 clamp하며 몇 번 반복. 끝내 못 피하면 None.
def _place_without_overlap(x, y, r, accepted, max_iter=10):
    px, py = float(x), float(y)
    for _ in range(max_iter):
        conflict = None
        for ax, ay, ar in accepted:
            if math.hypot(px - ax, py - ay) < (r + ar - CONNECT_SLACK_CM):
                conflict = (ax, ay, ar)
                break
        if conflict is None:
            # 충돌 없음 → 경계 안으로만 확인하고 확정
            cx = max(r, min(SPACE_W_CM - r, px))
            cy = max(r, min(SPACE_D_CM - r, py))
            if (cx, cy) == (px, py):
                return (int(round(px)), int(round(py)))
            px, py = cx, cy  # 경계 clamp 후 다시 충돌 검사
            continue
        # 충돌 → 상대 중심에서 바깥으로 '거의 맞닿는' 거리까지 밀어낸다
        ax, ay, ar = conflict
        d = math.hypot(px - ax, py - ay)
        dirx, diry = (1.0, 0.0) if d < 1e-6 else ((px - ax) / d, (py - ay) / d)
        target = ar + r - CONNECT_SLACK_CM + 0.5
        px = max(r, min(SPACE_W_CM - r, ax + dirx * target))
        py = max(r, min(SPACE_D_CM - r, ay + diry * target))
    return None

# 충돌 처리(결정론적 안전 레이어): 두 로봇의 본체가 파고들면(중심 거리 < 반지름 합) 나중 로봇을
# 바깥으로 '밀어내' 겹침을 없앤다. 정확히 맞닿는 정도는 허용하므로 복합 가구(붙인 테이블/침대)로의
# 의도적 '연결'은 그대로 통과한다. 밀어내도 자리를 못 찾으면(공간이 꽉 참) 그때만 접어 홈으로 보낸다.
# 안전 검증은 LLM이 아니라 코드가 최종 책임진다.
def _resolve_collisions(action):
    accepted = []  # (x, y, radius)
    # 우선순위: l_bots → h_bots, 목록에 나온 순서대로. 먼저 받아들여진 로봇이 자리를 차지한다.
    for robot in _iter_robots(action):
        if robot.get("active") != "active":
            continue
        r = robot_radius(robot)
        placed = _place_without_overlap(robot.get("x", 0), robot.get("y", 0), r, accepted)
        if placed is None:
            # 밀어내도 겹침을 못 피함 → 안전을 위해 접어 홈 도크로 보낸다.
            hx, hy = home_for(robot)
            print("[validate] 충돌 회피 실패 -> {0} inactive 처리".format(robot.get("robot")))
            robot["active"] = "inactive"
            robot["furniture"] = "none"
            robot["size"] = 0
            robot["x"] = hx
            robot["y"] = hy
        else:
            if (placed[0], placed[1]) != (robot.get("x"), robot.get("y")):
                print("[validate] 겹침 밀어냄 -> {0} {1}".format(robot.get("robot"), placed))
            robot["x"], robot["y"] = placed
            accepted.append((placed[0], placed[1], r))

# 3. 명령 객체 검증 (furniture는 자유 라벨이므로 가두지 않는다)
def validate_robot_action(action):
    if not action:
        return default_inactive_command()

    for robot in _iter_robots(action):
        # size를 0~100 범위로 clamp
        size = robot.get("size", 0)
        if not isinstance(size, (int, float)):
            size = 0
        if size < 0:
            size = 0
        elif size > 100:
            size = 100
        # enum 밖 값(예: 37)이 들어오면 가장 가까운 단계로 스냅 (37 -> 25)
        size = min((0, 25, 50, 75, 100), key=lambda step: abs(step - size))
        robot["size"] = size

        # active인데 furniture가 비어 있거나 공백이면 inactive로 되돌림
        furniture = robot.get("furniture", "")
        if robot.get("active") == "active" and (not furniture or not str(furniture).strip()):
            robot["active"] = "inactive"

        # 위치 정규화 (정수 cm + 가동 공간 240x200 안으로)
        _normalize_position(robot)

        # inactive 로봇은 '빈' 로봇으로 정리: 라벨 none, size 0, 각자 뒷줄 도크에서 대기.
        # (옛 라벨이 history에 남아 다음 판단을 오염시키는 것을 막는다.)
        if robot.get("active") != "active":
            hx, hy = home_for(robot)
            robot["furniture"] = "none"
            robot["size"] = 0
            robot["x"] = hx
            robot["y"] = hy

    # 충돌·도달가능성 검증은 모든 로봇 정규화 후 한 번에 (결정론적 차단)
    _resolve_collisions(action)

    return action

# 로봇에 보낼 최종 출력: 내부/ history는 cm(정수)로 두고, 위치만 m 단위 float으로 변환한 사본을 만든다.
def to_output_meters(action):
    out = json.loads(json.dumps(action))  # 원본을 건드리지 않도록 복사
    for robot in _iter_robots(out):
        robot["x"] = round(robot.get("x", 0) / CM_PER_M, 2)
        robot["y"] = round(robot.get("y", 0) / CM_PER_M, 2)
    return out

def main():
    if OpenAI is None:
        print("OpenAI SDK is not installed. Install it with: pip install openai")
        return

    recognizer = sr.Recognizer()
    groq_client = Groq(api_key=GROQ_API_KEY)
    openai_client = OpenAI(api_key=OPENAI_API_KEY)

    with sr.Microphone() as source:
        print("Adjusting for background noise...")
        # 주변 소음에 맞게 마이크 감도 조절
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print("Hold Space to record. Release Space to transcribe. Press Ctrl+C to stop.")

        # 직전 의도(특히 number 인원수)는 대화 맥락이므로 다음 발화 분석에 넘겨 유지한다.
        last_intent = None

        while True:
            if not is_space_pressed():
                time.sleep(POLL_SECONDS)
                continue

            audio = record_while_space_is_pressed(source)
            if not audio.frame_data:
                continue

            # 1) STT
            text = transcribe_audio(groq_client, audio)
            if not text:
                continue

            # 2) 의도 분석 (직전 의도를 넘겨 number 등 맥락 유지)
            result_intent = ask_intent_to_openai(openai_client, text, last_intent)
            if not result_intent:
                continue
            # 다음 발화의 맥락 유지를 위해 직전 의도를 갱신
            last_intent = result_intent

            # 3) 명령 생성 (history + 구조화 의도 + 원문 발화. 상대적 조정은 원문에서 읽는다)
            raw_robot_action = ask_action_to_openai(
                openai_client, result_intent, command_history, text
            )
            # 4) 검증: size clamp + 빈 furniture 체크 + 위치 정규화/충돌 처리(cm)
            result_robot_action = validate_robot_action(raw_robot_action)
            # 5) 검증 통과한 명령을 history에 추가 (cm 그대로. history 최신 항목이 곧 현재 4대 상태)
            command_history.append(result_robot_action)
            if len(command_history) > 10:
                del command_history[:-10]
            # 6) 로봇에 보낼 최종 출력: 위치를 m 단위 float으로 변환해 출력
            output_command = to_output_meters(result_robot_action)
            print(json.dumps(output_command, ensure_ascii=False))

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
