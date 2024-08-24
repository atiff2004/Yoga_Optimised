from gevent import monkey
monkey.patch_all()

from flask import Flask, jsonify
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import base64
from main import YogaAnalyzer

app = Flask(__name__)
socketio = SocketIO(app, async_mode='gevent')

yoga_analyzer = YogaAnalyzer()

@socketio.on('connect')
def on_connect():
    emit('response', {'message': 'Connected to Flask WebSocket'})

@socketio.on('send_frame')
def handle_frame(data):
    try:
        frame_data = base64.b64decode(data)
        np_arr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            emit('analysis_result', {'error': 'Failed to decode image'})
            return

        analyzed_frame = yoga_analyzer.analyze_pose(frame)
        results = yoga_analyzer.get_results()
        print(results)
        emit('analysis_result', results)

    except Exception as e:
        emit('analysis_result', {'error': str(e)})

if __name__ == "__main__":
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
