from flask import Blueprint, request, jsonify
from services.whisper_service import transcribe_radio_stream

transcribe_bp = Blueprint("transcribe", __name__)

@transcribe_bp.route("/transcribe", methods=["POST"])
def transcribe():
  try:
    data = request.get_json()

    if not data or "url" not in data:
      return jsonify({"error": "스트리밍 URL이 제공되지 않았습니다."}), 400

    url = data.get("url")
    userId = data.get("userId")
    channelId = data.get("channelId")
    print(f"받은 스트리밍 URL: {url}")
    transcribe_radio_stream(url, userId, channelId)

    return jsonify({"message": "transcribe start"}), 200

  except Exception as e:
    print(f"오류 발생: {e}")
    return jsonify({"error": "서버 오류가 발생했습니다."}), 500
