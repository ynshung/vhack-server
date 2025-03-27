from flask import Flask, request, jsonify
import requests
import os
import json
import base64
import whisper
from dotenv import load_dotenv
import ssl
from transformer import get_similar_command

app = Flask(__name__)
load_dotenv()

def process_speech(audio_url, audio_config):
    response = requests.post(
        "https://speech.googleapis.com/v1/speech:recognize",
        json={
            "audio": {
                "content": audio_url,
            },
            "config": audio_config,
        },
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "X-goog-api-key": os.getenv("GOOGLE_SPEECH_TO_TEXT_API_KEY"),
        },
    )
    speech_results = response.json()
    return speech_results

def process_speech_whisper(audio_base64):
    audio_data = base64.b64decode(audio_base64)
    audio_path = "temp_audio.wav"
    
    with open(audio_path, "wb") as f:
        f.write(audio_data)
    
    ssl._create_default_https_context = ssl._create_unverified_context
    model = whisper.load_model("base")
    
    result = model.transcribe(audio_path)
    
    # Remove temporary audio file
    os.remove(audio_path)
    
    return result

@app.route("/api/process-speech", methods=["POST"])
def process_speech_flask():
    data = request.get_json()
    speech_results = process_speech_whisper(data["audioUrl"]) # Whisper
    # speech_results = process_speech(data["audioUrl"], data["config"]) # Google
    print(speech_results)
    return jsonify(speech_results)


@app.route("/api/get-commands", methods=["POST"])
def get_commands_flask():
    data = request.get_json()
    commands = data["commands"]
    transcript = data["transcript"]

    similar_commands = get_similar_command(transcript, commands)
    return json.dumps(similar_commands, default=str)
    

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=4321, debug=True)
