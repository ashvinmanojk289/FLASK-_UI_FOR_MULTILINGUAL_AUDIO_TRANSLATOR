from flask import Flask, request, render_template, send_file, jsonify
import os
import speech_recognition as sr
from pydub import AudioSegment
from googletrans import Translator
import webbrowser
import threading
import logging
from pathlib import Path
import torch
import soundfile as sf
from TTS.api import TTS

app = Flask(__name__, static_folder="static")
UPLOAD_FOLDER = Path("uploads")
UPLOAD_FOLDER.mkdir(exist_ok=True)
ALLOWED_EXTENSIONS = {"mp3", "wav", "ogg", "flac", "m4a"}
LANGUAGE_MAP = {
    "en": "English", "fr": "French", "es": "Spanish",
    "de": "German", "hi": "Hindi", "ar": "Arabic",
    "it": "Italian", "ja": "Japanese", "ko": "Korean",
    "ta": "Tamil"
}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
MAX_TEXT_LENGTH = 5000  # Characters
progress = 0
cancel_process = False
logging.basicConfig(level=logging.INFO)
device = "cuda" if torch.cuda.is_available() else "cpu"
tts_model_path = "./models/fastspeech2_model.pth"
vocoder_model_path = "./models/vocoder_model.pth"
config_path = "./configs/config.json"
tts = None
try:
    if Path(tts_model_path).exists() and Path(vocoder_model_path).exists() and Path(config_path).exists():
        tts = TTS(model_path=tts_model_path, vocoder_path=vocoder_model_path, config_path=config_path).to(device)
        logging.info("FastSpeech 2 model loaded successfully.")
    else:
        logging.warning("FastSpeech 2 model files not found. Falling back to gTTS.")
except Exception as e:
    tts = None
    logging.error(f"Error loading FastSpeech 2 model: {e}. Falling back to gTTS.")

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def update_progress(value):
    global progress, cancel_process
    if not cancel_process:
        progress = value
        logging.info(f"🟢 Progress updated to: {progress}")
    else:
        progress = 0

def convert_to_wav(audio_path):
    update_progress(20)
    try:
        if audio_path.lower().endswith(".mp3"):
            audio = AudioSegment.from_mp3(audio_path)
            wav_path = str(Path(audio_path).with_suffix(".wav"))
            audio.export(wav_path, format="wav")
            return wav_path
        return audio_path
    except FileNotFoundError:
        logging.error(f"🔴 File not found: {audio_path}")
        return None
    except Exception as e:
        logging.error(f"🔴 Audio conversion failed: {e}")
        return None

def speech_to_text(audio_path):
    global cancel_process
    update_progress(40)
    if cancel_process:
        return None
    recognizer = sr.Recognizer()
    wav_audio_path = convert_to_wav(audio_path)
    if wav_audio_path is None:
        return None
    try:
        with sr.AudioFile(wav_audio_path) as source:
            audio = recognizer.record(source)
        if cancel_process:
            return None
        text = recognizer.recognize_google(audio)
        update_progress(60)
        return text
    except sr.UnknownValueError:
        logging.error("🔴 Speech recognition could not understand audio")
        return None
    except sr.RequestError as e:
        logging.error(f"🔴 Could not request results from Google Speech Recognition service; {e}")
        return None
    except FileNotFoundError:
        logging.error(f"🔴 File not found: {wav_audio_path}")
        return None

def translate_text(text, target_lang):
    global cancel_process
    update_progress(75)
    if cancel_process:
        return None
    translator = Translator()
    try:
        translated = translator.translate(text, dest=target_lang)
        logging.info(f"🟢 Translated text: {translated.text}")
        return translated.text
    except Exception as e:
        logging.error(f"🔴 Translation failed: {e}")
        return None

def text_to_speech(text, lang, output_path="static/translated_audio.wav"):
    global cancel_process
    if cancel_process:
        return None
    try:
        if tts:
            speaker_map = {
                "en": "en_001", "fr": "fr_001", "es": "es_001",
                "de": "de_001", "hi": "hi_001", "ar": "ar_001",
                "it": "it_001", "ja": "ja_001", "ko": "ko_001",
                "ta": "ta_001"
            }
            speaker = speaker_map.get(lang, "en_001")
            wav = tts.tts(text=text, speaker=speaker, language=lang)
            sf.write(output_path, wav, samplerate=24000)
            update_progress(100)
            return output_path
        else:
            from gtts import gTTS
            tts_gtts = gTTS(text=text, lang=lang)
            tts_gtts.save(output_path.replace(".wav", ".mp3"))
            update_progress(100)
            return output_path.replace(".wav", ".mp3")
    except Exception as e:
        logging.error(f"🔴 Text-to-speech conversion failed: {e}")
        return None

@app.route("/", methods=["GET", "POST"])
def index():
    global progress, cancel_process
    progress = 0
    cancel_process = False
    error_message = None

    if request.method == "POST":
        logging.info("🟢 Received a POST request!")
        print(f"Request Files: {request.files}")  # Debug
        print(f"Request Form: {request.form}")  # Debug
        input_type = request.form.get("inputType")
        print(f"Input Type: {input_type}")  # Debug

        target_lang = request.form.get("language")

        if input_type == "audio":
            audio_file = request.files.get("audio")

            if not audio_file or audio_file.filename == "":
                error_message = "No audio file uploaded!"

            elif not allowed_file(audio_file.filename):
                error_message = "Invalid file format!"

            else:
                try:
                    audio_data = audio_file.read()
                    if len(audio_data) > MAX_FILE_SIZE:
                        error_message = f"File size exceeds {MAX_FILE_SIZE / (1024 * 1024)} MB!"
                    else:
                        audio_file.seek(0)
                        filepath = str(UPLOAD_FOLDER / audio_file.filename)
                        audio_file.save(filepath)
                        transcript = speech_to_text(filepath)
                        if cancel_process or not transcript:
                            error_message = "Speech recognition failed!"
                except Exception as e:
                    logging.error(f"🔴 File processing error: {e}")
                    error_message = "File processing error!"

        elif input_type == "text":
            transcript = request.form.get("text")

            if not transcript:
                error_message = "No text entered!"
            elif len(transcript) > MAX_TEXT_LENGTH:
                error_message = f"Text exceeds {MAX_TEXT_LENGTH} characters!"

        else:
            error_message = "Invalid input type!"

        if error_message:
            return render_template("index.html", languages=LANGUAGE_MAP, error=error_message)

        translated_text = translate_text(transcript, target_lang)
        if cancel_process or not translated_text:
            return render_template("index.html", languages=LANGUAGE_MAP, error="Translation failed!")

        audio_output = text_to_speech(translated_text, target_lang)
        if cancel_process or not audio_output:
            return render_template("index.html", languages=LANGUAGE_MAP, error="Text-to-Speech conversion failed!")

        return render_template("result.html", transcript=transcript, translated_text=translated_text, audio_output=audio_output)

    return render_template("index.html", languages=LANGUAGE_MAP)

@app.route("/progress")
def get_progress():
    global progress
    logging.info(f"🔵 Sending progress: {progress}")
    return jsonify({"progress": progress})

@app.route("/download")
def download():
    if tts:
        return send_file("static/translated_audio.wav", as_attachment=True)
    else:
        return send_file("static/translated_audio.mp3", as_attachment=True)

def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000/")

if __name__ == "__main__":
    threading.Timer(1, open_browser).start()
    app.run(debug=True)
