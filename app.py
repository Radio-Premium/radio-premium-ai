from flask import Flask
from routes.transcribe import transcribe_bp  # 1. transcribe 라우터 가져오기

app = Flask(__name__)
app.register_blueprint(transcribe_bp)

@app.route("/")
def index():
  return "Whisper 서버 초기 실행 성공!"

if __name__ == "__main__":
  app.run(host="0.0.0.0", port=5000)
