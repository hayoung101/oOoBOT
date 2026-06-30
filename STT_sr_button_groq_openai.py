import ctypes
from io import BytesIO
import json
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
def ask_intent_to_openai(client, usertext):
    try:
        response = client.responses.create(
            model=OPENAI_MODEL,
            input=[
                {
                    "role": "developer",
                    "content": "너는 사용자가 하는 말의 의도를 최대한 정확하게 알아듣는 의도 파악 전문가야.",
                },
                {
                    "role": "user",
                    "content": usertext,
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
                        },
                        "required": ["number", "situation", "activity", "furniture"],
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
                    "'더 크게/작게', '너무 작다/크다'처럼 크기를 바꾸라는 요청에는 furniture 라벨만 바꾸지 말고 반드시 size를 실제로 한 단계 이상 올리거나 내려라. history 최신 항목의 현재 size를 기준으로 조정하고, 바뀐 형태에 맞는 라벨을 다시 붙여라. 이런 크기 조정은 현재 그 가구로 쓰이는 모든 로봇에 같이 적용하라."
                    "'치워', '그만', '다 접어', '정리하자'처럼 사용 종료·정리를 뜻하는 말에는 해당 로봇들을 active='inactive', size=0으로 되돌려라. (수납가구로 변형하라는 뜻이 아니라, 로봇을 접어 쉬게 하라는 뜻이다.)"
                    "입력 받은 사용자의 인원수, 상황, 행동, 필요한 가구에 맞게 로봇의 대수와 그 로봇이 수행해야 할 행동을 제안해줘."
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
                                        }
                                    },
                                    "required": ["robot", "active", "furniture", "size"],
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
                                        }
                                    },
                                    "required": ["robot", "active", "furniture", "size"],
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

# 6. LLM 응답 실패 시 사용할 안전 기본값: 4대 모두 inactive
def default_inactive_command():
    return {
        "l_bots": [
            {"robot": "L BOT 1", "active": "inactive", "furniture": "none", "size": 0},
            {"robot": "L BOT 2", "active": "inactive", "furniture": "none", "size": 0},
        ],
        "h_bots": [
            {"robot": "H BOT 1", "active": "inactive", "furniture": "none", "size": 0},
            {"robot": "H BOT 2", "active": "inactive", "furniture": "none", "size": 0},
        ],
    }

# 3. 명령 객체 검증 (furniture는 자유 라벨이므로 가두지 않는다)
def validate_robot_action(action):
    if not action:
        return default_inactive_command()

    for key in ("l_bots", "h_bots"):
        for robot in action.get(key, []):
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

            # 2) 의도 분석
            result_intent = ask_intent_to_openai(openai_client, text)
            if not result_intent:
                continue

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
