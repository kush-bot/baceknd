from flask import Flask, request, jsonify
from flask_cors import CORS
import json

app = Flask(__name__)
CORS(app)

reports = []

@app.route('/report', methods=['POST'])
def receive_report():
    data = request.get_json()
    # Here you can process the received data as per your requirements
    print("Received report:")
    print(json.dumps(data, indent=4))  # Print the received data for debugging
    reports.append(data)
    return jsonify({'message': 'Report received successfully'})

@app.route('/reports', methods=['GET'])
def get_reports():
    return jsonify(reports)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)

