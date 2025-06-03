import subprocess
import whisper
import numpy
import threading
import queue

model = whisper.load_model("base", device="cpu")
text_queue = queue.Queue()

def transcribe_radio_stream(url):
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
          text_queue.put(text)

    except Exception as e:
      print(f"[whisper_service] 오류: {e}")

  thread = threading.Thread(target=worker, daemon=True)
  thread.start()
