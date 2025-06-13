from flask import Flask
from flask_cors import CORS
from routes.transcribe import transcribe_bp
from routes.stop import stop_bp

app = Flask(__name__)

# TODO: 추후 배포 시 변경
CORS(app)

app.register_blueprint(transcribe_bp)
app.register_blueprint(stop_bp)

@app.route("/")
def index():
  return "Whisper 서버 초기 실행 성공!"

if __name__ == "__main__":
  app.run(host="0.0.0.0", port=5000)
