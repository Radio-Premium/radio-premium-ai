from flask import Blueprint, request, jsonify

transcribe_bp = Blueprint("transcribe", __name__)

@transcribe_bp.route("/transcribe", methods=["POST"])
def transcribe():
  try:
    data = request.get_json()
    url = data.get("url")

    if not data or "url" not in data:
      return jsonify({"error": "스트리밍 URL이 제공되지 않았습니다."}), 400

    print(f"받은 스트리밍 URL: {url}")

  except Exception as e:
    print(f"오류 발생: {e}")
    return jsonify({"error": "서버 오류가 발생했습니다."}), 500
