import os
import csv
import re
import queue
import threading
import subprocess
from datetime import datetime

import socketio
import numpy
import whisper
from dotenv import load_dotenv

from .logger import save_transcription_log
from services.process_registry import active_processes

load_dotenv()
BACKEND_API_URL = os.getenv("BACKEND_API_URL")

LOG_FILE_PATH = "data/whisper_logs.csv"
os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)

def save_transcription_log(text, userId):
    timestamp = datetime.now().isoformat()
    with open(LOG_FILE_PATH, mode="a", newline='', encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, userId, text])

model = whisper.load_model("base", device="cpu")
text_queue = queue.Queue()
socketio = socketio.Client()

@socketio.event(namespace="/whisper")
def connect():
    print("Whisper namespace connected")

@socketio.event(namespace="/whisper")
def disconnect():
    print("Whisper namespace disconnected")

socketio.connect(BACKEND_API_URL, namespaces=["/whisper"])

def transcribe_radio_stream(url, userId, channelId):
  def worker():
    try:
      process = subprocess.Popen(
        [
          "ffmpeg", "-i", url,
          "-af", "highpass=f=200, lowpass=f=3000, afftdn, dynaudnorm",
          "-f", "wav", "-acodec", "pcm_s16le",
          "-ar", "16000", "-ac", "1", "-loglevel", "quiet", "-"
        ],
        stdout=subprocess.PIPE
      )
      active_processes[userId] = process

      sample_rate = 16000
      chunk_duration = 5
      chunk_size = sample_rate * 2 * chunk_duration

      while True:
        data = process.stdout.read(chunk_size)
        if not data:
          break

        audio = numpy.frombuffer(data, numpy.int16).astype(numpy.float32) / 32768.0
        result = model.transcribe(audio, language="ko")
        text = result.get("text", "").strip()

        if text:
          clear_spaces_text = re.sub(r"\s+", "", text)
          save_transcription_log(clear_spaces_text, userId)
          socketio.emit("transcribedRadioText", { "text": clear_spaces_text, "userId": userId, "channelId": channelId }, namespace="/whisper")
          print(f"[전송됨] {text}")

    except Exception as e:
      print(f"[whisper_service] 오류: {e}")

  thread = threading.Thread(target=worker, daemon=True)
  thread.start()
