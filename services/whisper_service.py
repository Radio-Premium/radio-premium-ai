import socketio
import subprocess
import whisper
import numpy
import threading
import queue

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
          socketio.emit("transcribedRadioText", { "text": text, "userId": userId }, namespace="/whisper")
          print(f"[전송됨] {text}")

    except Exception as e:
      print(f"[whisper_service] 오류: {e}")

  thread = threading.Thread(target=worker, daemon=True)
  thread.start()
