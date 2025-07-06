import base64
import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from worker import speech_to_text, text_to_speech, watsonx_process_message

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/speech-to-text', methods=['POST'])
def speech_to_text_route():
    print("processing Speech-to-Text")
    
    # Get the user's audio (binary data)
    audio_binary = request.data
    
    # Transcribe the speech to text using Watson STT
    text = speech_to_text(audio_binary)

    # Return the transcribed text as JSON
    response = app.response_class(
        response=json.dumps({'text': text}),
        status=200,
        mimetype='application/json'
    )
    
    print(response)
    print(response.data)
    return response

@app.route('/process-message', methods=['POST'])
def process_message_route():
    # Get user message and preferred voice from request
    user_message = request.json['userMessage']
    print('user_message', user_message)

    voice = request.json['voice']
    print('voice', voice)

    # Generate response text using Watsonx LLM
    watsonx_response_text = watsonx_process_message(user_message)

    # Remove empty lines from the response
    watsonx_response_text = os.linesep.join(
        [s for s in watsonx_response_text.splitlines() if s]
    )

    # Convert the response text to speech using Watson TTS
    watsonx_response_speech = text_to_speech(watsonx_response_text, voice)

    # Encode the audio content to base64 string for transport
    watsonx_response_speech = base64.b64encode(watsonx_response_speech).decode('utf-8')

    # Return both text and speech (as base64) as JSON response
    response = app.response_class(
        response=json.dumps({
            "watsonxResponseText": watsonx_response_text,
            "watsonxResponseSpeech": watsonx_response_speech
        }),
        status=200,
        mimetype='application/json'
    )
    
    print(response)
    return response

if __name__ == "__main__":
    app.run(port=8000, host='0.0.0.0')
