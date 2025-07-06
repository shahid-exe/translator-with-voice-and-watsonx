import requests
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes, DecodingMethods
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams

# Watsonx credentials
PROJECT_ID = "skills-network"
credentials = {
    "url": "https://us-south.ml.cloud.ibm.com"
    # "apikey": "YOUR_API_KEY"  # Uncomment and use if running outside Cloud IDE
}

# Model setup
model_id = ModelTypes.FLAN_UL2
parameters = {
    GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
    GenParams.MIN_NEW_TOKENS: 1,
    GenParams.MAX_NEW_TOKENS: 1024
}

# Load Watsonx model
model = Model(
    model_id=model_id,
    params=parameters,
    credentials=credentials,
    project_id=PROJECT_ID
)

# LLM processing function
def watsonx_process_message(user_message):
    prompt = f"""You are an assistant helping translate sentences from English into Spanish.
Translate the query to Spanish: ```{user_message}```."""
    
    response_text = model.generate_text(prompt=prompt)
    print("watsonx response:", response_text)
    return response_text

# Speech-to-Text using Watson STT API
def speech_to_text(audio_binary):
    base_url = 'https://sn-watson-stt.labs.skills.network'
    api_url = f'{base_url}/speech-to-text/api/v1/recognize'

    params = {
        'model': 'en-US_Multimedia',
    }

    response = requests.post(api_url, params=params, data=audio_binary).json()

    text = 'null'
    while bool(response.get('results')):
        print('Speech-to-Text response:', response)
        text = response.get('results').pop().get('alternatives').pop().get('transcript')
        print('recognized text: ', text)
        return text

# Text-to-Speech using Watson TTS API
def text_to_speech(text, voice=""):
    base_url = 'http://speech-to-text.1xjggjl1zprm.svc.cluster.local'
    api_url = f'{base_url}/text-to-speech/api/v1/synthesize?output=output_text.wav'

    if voice and voice != "default":
        api_url += f"&voice={voice}"

    headers = {
        'Accept': 'audio/wav',
        'Content-Type': 'application/json',
    }

    json_data = {
        'text': text,
    }

    response = requests.post(api_url, headers=headers, json=json_data)
    print('Text-to-Speech response:', response)
    return response.content
