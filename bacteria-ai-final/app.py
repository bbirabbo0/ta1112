from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

# 학습된 모델 불러오기
with open("risk_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    try:
        X_input = [[
            data["place_type"],
            data["people_level"],
            data["touch_level"],
            data["clean_level"],
            data["humidity_level"],
            float(data["colonies"])
        ]]
    except Exception as e:
        return jsonify({"error": f"입력 형식 오류: {e}"}), 400

    try:
        pred = model.predict(X_input)[0]
        prob = max(model.predict_proba(X_input)[0])
    except Exception as e:
        return jsonify({"error": f"모델 예측 중 오류: {e}"}), 500

    return jsonify({
        "risk_label": str(pred),
        "confidence": float(round(prob, 3))
    })

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0",port=port)
