import socketio
import subprocess
import whisper
import numpy
import threading
import queue
import csv
from datetime import datetime
import os
from .logger import save_transcription_log
import re

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

# TODO: Whisper 서버 배포 시 주소 변경
socketio.connect("http://localhost:3000", namespaces=["/whisper"])

def transcribe_radio_stream(url, userId):
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
          clean_text = re.sub(r"\s+", "", text)
          save_transcription_log(clean_text, userId)
          socketio.emit("transcribedRadioText", { "text": text, "userId": userId }, namespace="/whisper")
          print(f"[전송됨] {text}")

    except Exception as e:
      print(f"[whisper_service] 오류: {e}")

  thread = threading.Thread(target=worker, daemon=True)
  thread.start()
