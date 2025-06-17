from flask import Blueprint, request, jsonify
from services.process_registry import active_processes

stop_bp = Blueprint("stop_transcription", __name__)

@stop_bp.route("/transcription", methods=["DELETE"])
def stop_transcription():
    data = request.get_json()
    userId = data.get("userId")

    if not userId:
        return jsonify({"error": "userId가 필요합니다."}), 400

    process = active_processes.get(userId)
    if process:
        process.terminate()
        process.wait()
        del active_processes[userId]
        print(f"{userId}의 Whisper 스트리밍 종료")
        return jsonify({"message": "Transcription stopped"}), 200
    else:
        return jsonify({"error": "진행 중인 스트리밍이 없습니다."}), 404
