import requests
import numpy as np
import os
import json
import math
import threading
from openai import AzureOpenAI
import speech_recognition as sr
from urllib.error import URLError
from dotenv import load_dotenv
import time
import tiktoken


# setup
load_dotenv()
sleep_time = 0.1
sampling_frequency = 16000
number_of_samples_per_chunk = 1365
time_between_audio_chunks = number_of_samples_per_chunk / sampling_frequency
max_response_tokens = 250
token_limit = 4096
BODY_URL = "http://localhost:5004"

# Azure OpenAI client
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2024-12-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)


# =====================================================================================================================
# Behavior map: parola chiave → nome comportamento installato sul robot
# =====================================================================================================================

BEHAVIOR_MAP = {
    # Ballo e movimento
    "balla":        "animations/Stand/Waiting/FunnyDancer_1",
    "danza":        "animations/Stand/Waiting/FunnyDancer_1",
    "chitarra":     "animations/Stand/Waiting/AirGuitar_1",
    "headbang":     "animations/Stand/Waiting/Headbang_1",
    "robot":        "animations/Stand/Waiting/Robot_1",
    "zombie":       "animations/Stand/Waiting/Zombie_1",
    "kungfu":       "animations/Stand/Waiting/KungFu_1",
    "elicottero":   "animations/Stand/Waiting/Helicopter_1",
    # Saluti e gesti sociali
    "saluta":       "animations/Stand/Gestures/Hey_1",
    "ciao":         "animations/Stand/Gestures/Hey_1",
    "inchino":      "animations/Stand/Gestures/BowShort_1",
    "saluto":       "animations/Stand/Gestures/Salute_1",
    "tiamo":        "animations/Stand/Waiting/LoveYou_1",
    "bacio":        "animations/Stand/Gestures/Kisses_1",
    # Emozioni positive
    "applaudi":     "animations/Stand/Gestures/Applause_1",
    "vinci":        "animations/Stand/Emotions/Positive/Winner_1",
    "felice":       "animations/Stand/Emotions/Positive/Happy_1",
    "ride":         "animations/Stand/Emotions/Positive/Laugh_1",
    "esulta":       "animations/Stand/Emotions/Positive/Excited_1",
    "muscoli":      "animations/Stand/Waiting/ShowMuscles_1",
    "orgoglioso":   "animations/Stand/Emotions/Positive/Proud_1",
    # Emozioni negative
    "triste":       "animations/Stand/Emotions/Negative/Sad_1",
    "arrabbiato":   "animations/Stand/Emotions/Negative/Angry_1",
    "paura":        "animations/Stand/Emotions/Negative/Fear_1",
    "sorpreso":     "animations/Stand/Emotions/Negative/Surprise_1",
    "deluso":       "animations/Stand/Emotions/Negative/Disappointed_1",
    # Pensiero e riflessione
    "pensa":        "animations/Stand/Gestures/Thinking_1",
    "nonlosso":     "animations/Stand/Gestures/IDontKnow_1",
    "confuso":      "animations/Stand/Gestures/Confused_1",
    # Attesa e idle
    "stanco":       "animations/Stand/Emotions/Negative/Exhausted_1",
    "rilassati":    "animations/Stand/Waiting/Relaxation_1",
    "alza":         "animations/Stand/Waiting/WakeUp_1",
    "fitness":      "animations/Stand/Waiting/Fitness_1",
    "fotografia":   "animations/Stand/Waiting/TakePicture_1",
    "buongiorno":   "animations/Stand/Emotions/Neutral/Hello_1",
}


# =====================================================================================================================
# GPT Tools (function calling)
# =====================================================================================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_behavior",
            "description": (
                "Esegui un'animazione o comportamento fisico del robot NAO quando l'utente lo chiede. "
                "Esempi: ballare, salutare, fare kungfu, fare il robot, essere triste, ridere, applaudire, ecc. "
                "Usa 'none' se nessun comportamento visivo e' richiesto."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "behavior": {
                        "type": "string",
                        "enum": list(BEHAVIOR_MAP.keys()) + ["none"],
                        "description": "Il comportamento da eseguire, oppure 'none'."
                    }
                },
                "required": ["behavior"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "move_robot",
            "description": (
                "Muovi fisicamente il robot NAO nello spazio. "
                "Usa questo tool SEMPRE quando l'utente chiede di: camminare, avanzare, andare avanti, "
                "andare indietro, girare a destra, girare a sinistra, ruotare, spostarsi. "
                "x = distanza in metri (positivo=avanti, negativo=indietro). "
                "y = spostamento laterale in metri (di solito 0). "
                "theta = rotazione in radianti (positivo=sinistra, negativo=destra). "
                "Esempi utili: 90 gradi = 1.5708 rad, 180 gradi = 3.1416 rad, 45 gradi = 0.7854 rad."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {
                        "type": "number",
                        "description": "Metri avanti (positivo) o indietro (negativo). Es: 1.0 = un metro avanti."
                    },
                    "y": {
                        "type": "number",
                        "description": "Metri laterale. Di solito 0."
                    },
                    "theta": {
                        "type": "number",
                        "description": "Rotazione in radianti. Es: 1.5708 = 90 gradi sinistra, -1.5708 = 90 gradi destra."
                    }
                },
                "required": ["x", "y", "theta"]
            }
        }
    }
]


# =====================================================================================================================
# Audio classes
# =====================================================================================================================

class NaoStream:

    def __init__(self, audio_generator):
        self.audio_generator = audio_generator

    def read(self, size=-1):
        try:
            return next(self.audio_generator)
        except StopIteration:
            return b''


class NaoAudioSource(sr.AudioSource):

    def __init__(self, server_url=BODY_URL):
        self.server_url = server_url
        self.stream = None
        self.is_listening = False
        self.CHUNK = 1365
        self.SAMPLE_RATE = 16000
        self.SAMPLE_WIDTH = 2

    def __enter__(self):
        requests.post(f"{self.server_url}/start_listening")
        self.is_listening = True
        self.stream = NaoStream(self.audio_generator())
        return self

    def audio_generator(self):
        while self.is_listening:
            response = requests.get(f"{self.server_url}/get_audio_chunk")
            yield response.content
            current_buffer_length = requests.get(f"{self.server_url}/get_server_buffer_length").json()["length"]
            correcting_factor = 1.0 / (1.0 + np.exp(current_buffer_length - np.pi))
            corrected_time = time_between_audio_chunks * correcting_factor
            time.sleep(corrected_time)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.is_listening = False
        requests.post(f"{self.server_url}/stop_listening")


# =====================================================================================================================
# Core functions
# =====================================================================================================================

def get_user_text():
    """Record audio from NAO and transcribe it to Italian text."""
    recognizer = sr.Recognizer()
    recognizer.pause_threshold = 1
    recognizer.operation_timeout = 4
    audio_data = None
    filename = "input.wav"

    while True:
        if audio_data is None:
            with NaoAudioSource() as source:
                print("Recording...")
                start_time = time.time()
                audio_data = recognizer.listen(source, phrase_time_limit=10, timeout=None)
                with open(filename, "wb") as f:
                    f.write(audio_data.get_wav_data())
                print(f"Recording took {time.time() - start_time:.2f} seconds")

        try:
            print("Transcribing...")
            start_time = time.time()
            text = recognizer.recognize_google(audio_data, language="it-IT")
            print(f"Transcribing took {time.time() - start_time:.2f} seconds")
            print(f"You said: {text}")
            return text
        except (sr.RequestError, URLError, ConnectionResetError) as e:
            print(f"Network error: {e}, retrying...")
            time.sleep(sleep_time)
        except sr.UnknownValueError:
            print("Could not understand audio, retrying...")
            audio_data = None
        except TimeoutError as e:
            print(f"Operation timed out: {e}, retrying...")
            audio_data = None


def execute_behavior_on_robot(behavior_keyword):
    """Send behavior request to body.py in a background thread."""
    if behavior_keyword == "none" or behavior_keyword not in BEHAVIOR_MAP:
        return

    behavior_name = BEHAVIOR_MAP[behavior_keyword]
    print(f"Executing behavior '{behavior_keyword}' → '{behavior_name}'")

    try:
        response = requests.post(
            f"{BODY_URL}/run_behavior",
            json={"behavior_name": behavior_name}
        )
        if response.ok:
            print(f"Behavior '{behavior_name}' executed successfully")
        else:
            print(f"Behavior '{behavior_name}' failed: {response.json()}")
    except Exception as e:
        print(f"Error sending behavior request: {e}")


def move_robot_body(x, y, theta):
    """Send move request to body.py."""
    print(f"Moving robot: x={x} y={y} theta={theta} rad ({math.degrees(theta):.1f} deg)")
    try:
        response = requests.post(
            f"{BODY_URL}/move",
            json={"x": float(x), "y": float(y), "theta": float(theta)}
        )
        if response.ok:
            print("Move executed successfully")
        else:
            print(f"Move failed: {response.json()}")
    except Exception as e:
        print(f"Error sending move request: {e}")


def get_gpt_text(conversation_context):
    """
    Call GPT with function calling enabled.
    Handles both execute_behavior and move_robot tool calls.
    Returns the text response.
    """
    conversation_context = trim_context(conversation_context)

    start = time.time()
    response = client.chat.completions.create(
        model="gpt-5.2-chat",
        messages=conversation_context,
        tools=TOOLS,
        tool_choice="auto",
        max_completion_tokens=max_response_tokens
    )
    print(f"{response.model} took {time.time() - start:.2f} seconds to respond")

    message = response.choices[0].message

    # Process tool calls
    if message.tool_calls:
        for tool_call in message.tool_calls:

            if tool_call.function.name == "execute_behavior":
                try:
                    args = json.loads(tool_call.function.arguments)
                    behavior_keyword = args.get("behavior", "none")
                    print(f"GPT requested behavior: '{behavior_keyword}'")
                    threading.Thread(
                        target=execute_behavior_on_robot,
                        args=(behavior_keyword,),
                        daemon=True
                    ).start()
                except Exception as e:
                    print(f"Error parsing behavior tool call: {e}")

            elif tool_call.function.name == "move_robot":
                try:
                    args = json.loads(tool_call.function.arguments)
                    x     = float(args.get("x", 0.0))
                    y     = float(args.get("y", 0.0))
                    theta = float(args.get("theta", 0.0))
                    print(f"GPT requested move: x={x} y={y} theta={theta}")
                    threading.Thread(
                        target=move_robot_body,
                        args=(x, y, theta),
                        daemon=True
                    ).start()
                except Exception as e:
                    print(f"Error parsing move tool call: {e}")

    # If GPT returned only a tool call and no text, request a follow-up text response
    gpt_message = message.content
    if not gpt_message:
        print("GPT returned no text content, requesting follow-up...")
        tool_results = []
        for tool_call in (message.tool_calls or []):
            tool_results.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": "ok"
            })
        follow_up = client.chat.completions.create(
            model="gpt-5.2-chat",
            messages=conversation_context + [
                {"role": "assistant", "content": None, "tool_calls": message.tool_calls}
            ] + tool_results,
            max_completion_tokens=max_response_tokens
        )
        gpt_message = follow_up.choices[0].message.content or ""

    print(f"Nao: {gpt_message}")
    return gpt_message


def send_gpt_text_to_body(gpt_message):
    """Send GPT response text to the robot for animated speech."""
    clean_message = remove_emojis(gpt_message)
    print(f"Sending to robot: {clean_message}")
    try:
        response = requests.post(f"{BODY_URL}/talk", json={"message": clean_message})
        print(f"Response status: {response.status_code}")
    except Exception as e:
        print(f"Error sending to robot: {e}")


def remove_emojis(text):
    """Remove non-ASCII characters and Markdown formatting for NAO TTS compatibility."""
    cleaned = ''.join(char for char in text if ord(char) < 128)
    cleaned = cleaned.replace('*', '').replace('_', '').replace('`', '')
    cleaned = cleaned.replace('#', '').replace('>', '').replace('-', '')
    cleaned = ' '.join(cleaned.split())
    return cleaned


def save_conversation(context, filename):
    """Save conversation to file for debugging."""
    with open(filename, "w") as f:
        for entry in context:
            role = entry['role'].capitalize()
            content = entry.get('content') or ""
            f.write(f"{role}:\n{content}\n\n")


def trim_context(context):
    """Trim conversation context to stay within token limit."""

    def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        if model in {
            "gpt-3.5-turbo-0613", "gpt-3.5-turbo-16k-0613",
            "gpt-4-0314", "gpt-4-32k-0314", "gpt-4-0613", "gpt-4-32k-0613",
        }:
            tokens_per_message = 3
            tokens_per_name = 1
        elif "gpt-3.5-turbo" in model:
            return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
        elif "gpt-4" in model:
            return num_tokens_from_messages(messages, model="gpt-4-0613")
        else:
            tokens_per_message = 3
            tokens_per_name = 1
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                if value:
                    num_tokens += len(encoding.encode(str(value)))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3
        return num_tokens

    conv_history_tokens = num_tokens_from_messages(context)
    while conv_history_tokens + max_response_tokens >= token_limit:
        del context[1]
        conv_history_tokens = num_tokens_from_messages(context)
    return context


# =====================================================================================================================
# Conversation loop
# =====================================================================================================================

with open("system_prompt.txt", "r") as f:
    system_prompt = f.read()

conversation_context = [{"role": "system", "content": system_prompt}]

running = True
while running:
    user_message = get_user_text()
    conversation_context.append({"role": "user", "content": user_message})

    gpt_message = get_gpt_text(conversation_context)
    send_gpt_text_to_body(gpt_message)

    conversation_context.append({"role": "assistant", "content": gpt_message})
    save_conversation(context=conversation_context, filename="conversation_context.txt")