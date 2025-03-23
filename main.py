from flask import Flask, request, jsonify
import requests
import os
import json
from dotenv import load_dotenv
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


@app.route("/api/process-speech", methods=["POST"])
def process_speech_flask():
    data = request.get_json()
    speech_results = process_speech(data["audioUrl"], data["config"])
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
