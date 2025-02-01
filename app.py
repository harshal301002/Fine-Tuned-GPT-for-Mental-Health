from flask import Flask, request, jsonify
from inference import generate_response

app = Flask(__name__)

@app.route('/query', methods=['POST'])
def query_model():
    data = request.json
    prompt = data.get('prompt', '')
    response = generate_response(prompt)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)