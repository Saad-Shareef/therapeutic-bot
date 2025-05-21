
from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS
from groq import Groq
from elevenlabs import generate, set_api_key, VoiceSettings
from dotenv import load_dotenv
from transformers import pipeline
import torch, os, base64, uuid, tempfile, requests, random

# Setup
app = Flask(__name__)
CORS(app)
load_dotenv()

# API keys
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
set_api_key(os.getenv("ELEVEN_API_KEY"))
GIPHY_API_KEY = os.getenv("GIPHY_API_KEY")

# Load Whisper model
device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = pipeline("automatic-speech-recognition", model="openai/whisper-base", device=device)

# Therapy response via LLaMA3
def get_therapy_response(prompt):
    res = groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "system", "content": "You are a compassionate and empathetic AI therapist. Respond in a warm, understanding, and non-judgmental tone. Offer emotional support and practical advice. End with GIF: description."},
                  {"role": "user", "content": prompt}],
        temperature=1.0,
        max_tokens=250
    )
    return res.choices[0].message.content

# Text-to-Speech using new ElevenLabs format
def text_to_speech(text):
    try:
        audio = generate(
            text=text,
            voice="Xb7hH8MSUJpSbSDYk0k2",
            model="eleven_turbo_v2_5",
        )
        file = f"temp_{uuid.uuid4()}.mp3"
        with open(file, "wb") as f:
            f.write(audio)
        with open(file, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        os.remove(file)
        return encoded
    except Exception as e:
        print("TTS error:", e)
        return None

# Get GIF
def fetch_gif(prompt):
    try:
        r = requests.get("https://api.giphy.com/v1/gifs/search", params={
            "api_key": GIPHY_API_KEY, "q": prompt, "limit": 10, "rating": "pg"
        }).json()
        return random.choice(r["data"])["images"]["original"]["url"] if r["data"] else ""
    except:
        return "https://media.giphy.com/media/3oEjI6SIIHBdRxXI40/giphy.gif"

# Parse response and GIF
def split_therapy_and_gif(text):
    if 'GIF:' in text:
        response, gif_prompt = text.split('GIF:', 1)
        return response.strip(), gif_prompt.strip()
    return text, "mental health support"

# Routes
@app.route('/roast', methods=['POST'])
def therapy_response():
    data = request.json
    user_input = data.get("text", "")
    prompt = f"User says:\n{user_input}\nOffer a therapeutic and emotionally supportive response. End with GIF: description."
    full_response = get_therapy_response(prompt)
    message, gif_prompt = split_therapy_and_gif(full_response)
    return jsonify({
        "roast": message,
        "gif": fetch_gif(gif_prompt),
        "audio": text_to_speech(message)
    })

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file'}), 400
    audio = request.files['audio']
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        audio.save(f.name)
        text = whisper_model(f.name)["text"]
    os.unlink(f.name)
    return jsonify({'transcription': text})

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                             'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
