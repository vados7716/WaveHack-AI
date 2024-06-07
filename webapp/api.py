from flask import Flask, jsonify
import asyncio
import numpy as np
from modelAI import predict, fetch_coefficients, parse_coefficients_lucky  # замените на ваш путь
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict_route():
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        prediction, confidence = loop.run_until_complete(predict())

        # Преобразование float32 в стандартный float для JSON сериализации
        prediction = float(prediction)
        confidence = float(confidence)

        return jsonify(result=prediction, confidence=confidence)
    except Exception as e:
        return jsonify(error=str(e)), 500

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
