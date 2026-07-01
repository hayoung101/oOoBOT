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
from openai import OpenAI

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
                    "prev_intent가 null이면(첫 발화) 발화에서 직접 인원·자세를 파악하라. situation·activity·furniture는 새 발화에 맞게 갱신하되, 맥락이 이어지는 조정성 발화라면 prev_intent를 참고해 일관되게 채워라.",
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
def ask_action_to_openai(client, intent, history):
    try:
        response = client.responses.create(
            model=OPENAI_MODEL,
            input=[
                {
                    "role": "developer",
                    "content": "너는 사용자의 상황과 행동에 맞는 로봇 가구 환경을 구성하는 전문가야."
                    "각 로봇에는 두 축이 있다 — (가) 높이: L BOT은 항상 낮고 H BOT은 항상 높다. size로 높이는 안 바뀐다. (나) size(0~100): 윗면 형태를 정한다. 0=좁고 평평한 면, 중간(25·50·75)=가운데가 패인 그릇(작을수록 깊은 바구니, 클수록 넓고 얕은 트레이), 100=크고 평평한 원형 면."
                    "size 기준점(앵커): 평평한 면으로 앉거나 작은 물건을 올려두는 용도 → size 0. 안에 물건을 담아 두거나 옮기는 용도 → 중간 size(담을 양·깊이에 따라 25,50,75). 넓고 평평한 면에 여러 물건을 올리거나 테이블로 쓰는 용도 → size 100. 이건 용도에 맞는 형태를 고르는 기준점이니, 특별한 이유가 없으면 이 기준을 따르라. "
                    "size에 따른 윗면 형태: size가 0이면 윗면이 좁고 평평하다. size가 커지면 윗면 가운데가 아래로 패면서 뒤집힌 원뿔 모양의 주머니(안에 물건을 담을 수 있는 그릇)가 생긴다. size가 더 커질수록 이 주머니는 반지름이 넓어지는 대신 깊이는 얕아진다 — 작은 size에서는 좁고 깊은 바구니, 큰 size로 갈수록 넓고 얕은 그릇·트레이가 된다. size 100이면 주머니가 완전히 펴져 하나의 크고 평평한 원형 면(테이블처럼 쓸 수 있는 넓은 윗면)이 된다."
                    "여기서 따라오는 물리적 사실: (1) 높이는 종류로 정해진다 — 낮게 앉거나 발을 올리는 용도엔 낮은 L BOT이, 서서 쓰거나 물건을 높이 올려두는 용도엔 높은 H BOT이 맞다. (2) size 양 끝(0과 100 부근)은 평평한 면이라 앉거나 물건을 올려두기 좋고, 중간 size는 가운데가 패인 그릇이라 물건을 담아 두거나 옮기기 좋다. 같은 중간이라도 작은 size는 깊은 바구니, 큰 size는 넓고 얕은 트레이에 가깝다. 이건 제한이 아니라 형태가 가진 사실일 뿐이니, 용도에 맞는 로봇 종류와 size를 네가 판단할 근거로만 써라."
                    "furniture는 자유 라벨이다. 위 size 기준은 '용도→형태(size)'를 고르기 위한 것이지 이름을 정해진 목록에 가두는 게 아니다. 같은 size라도 사용자의 위치·상황·용도에 따라 이름은 자유롭게 붙여라 — 예를 들어 size 0인 낮은 L BOT은 앉으면 '낮은 의자', 발을 올리면 '발 받침대', 옆에 두면 '낮은 협탁'이 된다. 실제 명령값은 size이고 furniture는 그 형태를 맥락에 맞게 표현한 이름일 뿐이다."
                    "한 대로 충분한 상황이면 한 대만 쓰고, 환경을 더 편하고 풍부하게 만들면 좋을 상황이면 여러 대를 조합해 구성하라. 어떤 상황에 어떤 조합이 정답이라는 고정된 세트는 없으니, 매번 그 상황을 보고 필요한 만큼만 자연스럽게 새로 판단하라. 무조건 많이 켜지도, 무조건 한 대만 쓰지도 마라."
                    "가용 로봇은 L BOT 2대(L BOT 1, L BOT 2)와 H BOT 2대(H BOT 1, H BOT 2)로 총 4대뿐이다. 활동에 가구가 더 많이 필요해 보여도 이 4대를 초과해 만들어낼 수 없다. 수요가 4대를 넘으면, 가장 중요한 가구부터 우선순위를 정해 한정된 4대 안에서 최선의 조합을 구성하라."
                    "입력은 '과거 기록(history)'와 '지금 처리할 새 요청(new_request)' 두 부분으로 구분돼 있어. 이번에 처리할 명령은 오직 '새 요청'이고, '과거 기록'은 '원래대로', '더 크게' 같은 상대적 표현을 해석하기 위한 참고용일 뿐이니 새 요청으로 착각하지 마."
                    "history의 가장 최근 항목이 로봇 4대의 현재 상태다. 매번 4대(L BOT 1, L BOT 2, H BOT 1, H BOT 2) 전체 상태를 절대값으로 출력해. 이번 요청과 직접 관련 없는 로봇은 현재 상태(history 최신 항목) 값을 그대로 유지해 출력하라. 멋대로 기본값으로 되돌리거나 끄지 마라."
                    "size는 [0, 25, 50, 75, 100] 다섯 단계 중 하나를 쓴다. furniture는 완전히 자유로운 라벨이며 정해진 이름 목록이 없다. 실제 명령값은 size이고 furniture는 그 형태를 맥락에 맞게 표현한 이름일 뿐이다."
                    "'더 크게/작게', '너무 작다/크다'처럼 크기를 바꾸라는 요청은 두 경우로 나눠 처리하라. (가) 담는 용도(중간 size의 그릇·바구니·트레이)라면 furniture 라벨만 바꾸지 말고 size를 실제로 한 단계 이상 올리거나 내려 깊이·넓이를 조정하라. history 최신 항목의 현재 size를 기준으로 조정하고, 바뀐 형태에 맞는 라벨을 다시 붙여라. (나) 평평한 면으로 쓰는 가구(테이블=size 100, 좌석·발받침=size 0)는 size를 바꾸면 평면이 깨져 용도를 잃는다 — 이런 평면 가구가 '너무 작다/더 크게'면 size는 그대로 두고, 같은 용도의 로봇을 한 대 더 붙여(복합 가구로 연결) 면적을 넓혀라. 어느 경우든 이 조정은 현재 그 가구로 쓰이는 모든 로봇에 같이 적용하라."
                    "'치워', '그만', '다 접어', '정리하자'처럼 사용 종료·정리를 뜻하는 말에는 해당 로봇들을 active='inactive', size=0으로 되돌려라. (수납가구로 변형하라는 뜻이 아니라, 로봇을 접어 홈으로 보내 쉬게 하라는 뜻이다.) inactive 로봇의 위치는 코드가 홈으로 정리하므로 신경 쓰지 않아도 된다."
                    "이제 size에 더해 각 로봇의 위치(x, y)도 함께 출력한다. 좌표계는 사용자가 서 있는 중심 (0,0)을 원점으로 한 직교좌표이고, 사용자는 +y(앞)를 바라본다. x는 좌(-)/우(+), y는 앞(+)/뒤(-)이며 단위는 cm다. 로봇 윗면은 원형이라 방향(회전)은 의미가 없으므로 위치(x,y)만 정하면 된다. "
                    "가동 공간은 가로 240cm x 세로 200cm이고 사용자가 그 중심에 있으므로, 좌표 범위는 x는 -120~+120, y는 -100~+100이다. 로봇은 실제 크기가 있으니(H BOT 반지름 약 20.5~36cm, L BOT 약 21.5~32.5cm, size가 클수록 큼) 본체가 이 경계를 넘지 않도록 여유를 두고 배치하라(경계를 넘는 좌표는 코드가 안으로 끌어당긴다). "
                    "미사용(inactive) 로봇은 접힌 채 뒤쪽 좌측 구석 홈(x=-120, y=-100)에서 대기한다. 로봇을 쓰지 않게 되면 그 홈으로 돌려보낸다(코드가 자동 처리하므로 inactive 로봇의 위치는 신경 쓰지 않아도 된다). "
                    "위치는 사용자의 '자세'(intent.posture: standing/sitting/lying)에 따라 달라진다. 사용자 신체 기준: 키 160cm, 다리 74cm, 팔 54cm, 오른손잡이. 그래서 옆에 두는 사이드 가구(협탁·트레이 등)는 손이 잘 닿는 '오른쪽(+x)에 먼저' 배정하고, 하나 더 필요하면 그다음 '왼쪽(-x)'에 둔다. "
                    "닿아야 하는 사이드 가구는 팔을 끝까지 뻗지 않고도 편히 닿도록, 사용자(앉거나 누웠으면 그 손) 기준 대략 '팔 길이의 절반 정도(약 25~45cm)' 거리에 둔다. 이 25~45cm 범위 안에서 더 가까이(25)로 둘지 더 멀리(45)로 둘지는 상황(다른 가구와의 간섭, 동선)을 보고 네가 판단하라. 단, 가운데에 좌석·침대처럼 큰 로봇이 있으면 사이드 가구는 그 로봇에 겹치지 않는 바깥쪽(대체로 이 범위의 먼 쪽)에 둔다. "
                    "자세별 배치 가이드라인(중심 (0,0)=사용자, +y=앞. 아래 좌표는 고정 규칙이 아니라 앵커이니 상황에 맞게 조정하라): "
                    "[standing 서 있음] 앞에서 서서 쓰는 작업면·테이블: (0, +50) 부근. 오른쪽 사이드: (+40, +10), 왼쪽 사이드: (-40, +10). "
                    "[sitting 앉음] 사용자가 앉는 좌석(낮은 L BOT): 사용자 바로 아래 (0, 0). 다리가 앞으로 뻗으므로 앞쪽 낮은 테이블: (0, +55), 발받침: (0, +45) 부근(둘을 동시에 쓰면 겹치지 않게 하나는 더 앞/옆으로). 좌석(반지름 약 21.5cm) 바깥으로 오른쪽 협탁: (+44, 0), 왼쪽: (-44, 0). "
                    "[lying 누움] 침대는 낮은 L BOT 두 대를 size 100으로 몸 아래 y축을 따라 이어 붙인다: (0, -33)과 (0, +33)(연결 간격 65cm). 침대 상판(반지름 약 32.5cm)을 벗어난 머리맡(+y쪽)에 사이드: 오른쪽 (+45, +65), 왼쪽 (-45, +65). "
                    "핵심: 같은 size라도 '어디에 두느냐'에 따라 가구의 의미가 달라진다. 위치까지 정하면 의자/발받침대/협탁 구분이 라벨이 아니라 좌표에서 자연스럽게 유도된다. 예를 들어 size 0인 낮은 L BOT을 사용자 바로 앞 가까이(작은 +y)에 두면 '발받침대', 사용자 옆(±x)에 두면 '낮은 협탁', 앉을 자리에 두면 '낮은 의자'가 된다. 위치는 사용자의 동선과 손이 닿는 범위를 고려해 자연스럽게 정하고, furniture 라벨도 그 위치·형태에 맞게 붙여라. "
                    "여러 로봇을 독립 가구로 쓸 때는 서로 본체가 겹치지 않게 떨어뜨려라(중심 간 거리가 두 로봇 반지름의 합보다 크도록). 단, 충돌·경계의 '최종' 안전 검증은 코드(결정론적 레이어)가 책임지므로, 너는 물리적으로 그럴듯한 배치를 제안하는 데 집중하면 된다. "
                    "size와 마찬가지로, 이번 요청과 직접 관련 없는 로봇의 위치(x,y)도 history 최신 상태값을 그대로 유지해 출력하라. 멋대로 0으로 되돌리지 마라."
                    "intent의 number(인원)를 적극 반영하라. 단, size는 인원수에 비례해 올리는 값이 '아니다' — size는 용도에 맞는 윗면 형태(평평한 면=0 또는 100, 담는 그릇=중간 25·50·75)를 고르는 값이다. 그러니 인원이 많아 더 넓은 면적이나 더 많은 좌석이 필요할 때는 size를 어중간하게 올리지 말고, ⓐ같은 용도의 로봇을 한 대 더 쓰거나 ⓑ여러 대를 인접 연결해 복합 가구로 넓혀라. 예: 테이블이 좁다고 size를 75로 내리면 평면이 아니라 오목한 트레이가 되어 테이블이 못 된다 — 대신 size 100짜리 H BOT을 한 대 더 옆에 붙여 더 큰 테이블로 만든다. 좌석이 더 필요하면 size 0 L BOT을 한 대 더 둔다. '테이블이 너무 작은데?' 같은 조정 요청도 size를 바꾸는 게 아니라 이렇게 면적을 키우라는 뜻으로 해석하라."
                    "로봇은 한 대씩 독립된 가구로만 쓰는 게 아니라, 여러 대를 인접하게 붙여(위치 x,y를 맞닿게) 하나의 더 큰 '복합 가구'로 합칠 수 있다. 예: H BOT 두 대를 size 100으로 양옆에 붙이면 다인용 대형 테이블이 되고, L BOT 두 대를 size 100으로 붙이면 침대처럼 넓은 평면이 된다. 이건 정해진 목록이 아니라 가능성의 예시일 뿐이니, 인원·상황에 맞는 새로운 조합도 자유롭게 구성하라. 복합 가구로 붙일 때는 두 로봇이 같은 자리에 포개지지 않고 가장자리만 맞닿도록, 중심 간 거리를 '두 로봇 반지름의 합' 정도로 두어라. 예: H BOT 두 대를 size 100으로 연결하면 중심 간 약 72cm(예: x=-36과 x=+36), L BOT 두 대를 size 100으로 연결하면 약 65cm 간격이다. 한 복합 가구를 이루는 로봇들에는 그 사실이 드러나는 일관된 furniture 라벨을 붙여라(예: 둘 다 '대형 테이블')."
                    "입력 받은 사용자의 인원수, 상황, 행동, 필요한 가구에 맞게 로봇의 대수와 위치, 단독/복합 구성, 그 로봇이 수행해야 할 행동을 제안해줘."
                },
                {
                    "role": "user",
                    "content": json.dumps({
                        "과거 기록(history)": history[-3:],
                        "지금 처리할 새 요청(new_request)": intent,
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
                                            "description": "배치 중심점 (0,0) 기준 좌우 위치(cm). 좌는 음수(-), 우는 양수(+). inactive면 0."
                                        },
                                        "y": {
                                            "type": "number",
                                            "description": "배치 중심점 (0,0) 기준 앞뒤 위치(cm). 앞(사용자 정면 방향)은 양수(+), 뒤는 음수(-). inactive면 0."
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
                                            "description": "배치 중심점 (0,0) 기준 좌우 위치(cm). 좌는 음수(-), 우는 양수(+). inactive면 0."
                                        },
                                        "y": {
                                            "type": "number",
                                            "description": "배치 중심점 (0,0) 기준 앞뒤 위치(cm). 앞(사용자 정면 방향)은 양수(+), 뒤는 음수(-). inactive면 0."
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

# 6. LLM 응답 실패 시 사용할 안전 기본값: 4대 모두 inactive (홈 위치에서 대기)
def default_inactive_command():
    return {
        "l_bots": [
            {"robot": "L BOT 1", "active": "inactive", "furniture": "none", "size": 0, "x": HOME_X_CM, "y": HOME_Y_CM},
            {"robot": "L BOT 2", "active": "inactive", "furniture": "none", "size": 0, "x": HOME_X_CM, "y": HOME_Y_CM},
        ],
        "h_bots": [
            {"robot": "H BOT 1", "active": "inactive", "furniture": "none", "size": 0, "x": HOME_X_CM, "y": HOME_Y_CM},
            {"robot": "H BOT 2", "active": "inactive", "furniture": "none", "size": 0, "x": HOME_X_CM, "y": HOME_Y_CM},
        ],
    }

# --- 물리 치수 / 공간 상수 (cm) (결정론적 안전 레이어) ---
# 가동 공간은 240(가로) x 200(세로). 사용자는 중심 (0,0)에 서서 +y(앞)를 바라본다고 가정.
WORKSPACE_X_CM = 120     # 중심 기준 좌우 한계 (x ∈ [-120, +120], 전체 240)
WORKSPACE_Y_CM = 100     # 중심 기준 앞뒤 한계 (y ∈ [-100, +100], 전체 200)

# 미사용(inactive) 로봇이 접힌 채 대기하는 홈(뒤쪽 좌측 구석)
HOME_X_CM = -120
HOME_Y_CM = -100

# 로봇 종류별 반지름(cm): size 0(base) ~ size 100(top)을 선형 보간.
#   H BOT: 41x41(size0) -> 72x72(size100)  => 반지름 20.5 -> 36.0
#   L BOT: 43x43(size0) -> 65x65(size100)  => 반지름 21.5 -> 32.5
ROBOT_RADIUS_CM = {
    "H": (20.5, 36.0),
    "L": (21.5, 32.5),
}

# 의도적 '연결'(복합 가구)에서 정수 반올림 오차를 흡수할 여유 (cm)
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
    # 로봇 반지름만큼 여유를 두어 본체 전체가 경계 안에 들어오도록 중심을 사각형 안으로 제한한다.
    r = robot_radius(robot)
    xlim = max(0, WORKSPACE_X_CM - r)
    ylim = max(0, WORKSPACE_Y_CM - r)
    x = max(-xlim, min(xlim, x))
    y = max(-ylim, min(ylim, y))
    robot["x"] = int(round(x))
    robot["y"] = int(round(y))

# 충돌 차단(결정론적 안전 레이어): 두 로봇의 본체가 물리적으로 겹치면 나중 로봇을 접어 홈으로 보낸다.
# 중심 간 거리가 두 로봇 반지름의 합보다 작으면(=서로 파고들면) 충돌이다. 정확히 맞닿는 정도는
# 허용하므로 복합 가구(붙인 테이블/침대)로의 의도적 '연결'은 통과하고, 같은 자리에 포개지는 것만 막는다.
# 안전 검증은 LLM이 아니라 코드가 최종 책임진다.
def _resolve_collisions(action):
    accepted = []  # (x, y, radius)
    # 우선순위: l_bots → h_bots, 목록에 나온 순서대로. 먼저 받아들여진 로봇이 자리를 차지한다.
    for robot in _iter_robots(action):
        if robot.get("active") != "active":
            continue
        x, y, r = robot.get("x", 0), robot.get("y", 0), robot_radius(robot)
        collided = False
        for ax, ay, ar in accepted:
            # 반지름 합에서 약간의 여유(CONNECT_SLACK)를 빼, 맞닿는 연결은 허용하고 실제 파고듦만 차단
            if math.hypot(x - ax, y - ay) < (r + ar - CONNECT_SLACK_CM):
                collided = True
                break
        if collided:
            # 겹치면 안전을 위해 이 로봇을 접어 홈으로 보낸다(충돌 차단).
            print("[validate] 충돌 감지 -> {0} inactive 처리 (위치 {1})".format(
                robot.get("robot"), (x, y)))
            robot["active"] = "inactive"
            robot["size"] = 0
            robot["x"] = HOME_X_CM
            robot["y"] = HOME_Y_CM
        else:
            accepted.append((x, y, r))

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

        # inactive 로봇은 접힌 채 홈(구석)으로 대기시킨다 (history 현재상태를 깔끔하게 유지)
        if robot.get("active") != "active":
            robot["x"] = HOME_X_CM
            robot["y"] = HOME_Y_CM

    # 충돌·도달가능성 검증은 모든 로봇 정규화 후 한 번에 (결정론적 차단)
    _resolve_collisions(action)

    return action

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

            # 3) 명령 생성 (최근 history만 전달. history 최신 항목이 곧 현재 4대 상태)
            raw_robot_action = ask_action_to_openai(
                openai_client, result_intent, command_history
            )
            # 4) 검증: size clamp + 빈 furniture 체크만 한 깨끗한 명령 객체
            result_robot_action = validate_robot_action(raw_robot_action)
            # 5) 검증 통과한 명령을 history에 추가 (오래된 것은 잘라서 최근 위주로 유지)
            command_history.append(result_robot_action)
            if len(command_history) > 10:
                del command_history[:-10]

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
