from flask import Flask, request, jsonify, render_template
from infer_classifier.py import predict as clf_predict
from infer_neutralizer import neutralize

app = Flask(__name__)

@app.get("/")
def index():
    return render_template("index.html")

@app.post("/predict")
def predict():
    text = request.form.get("sentence") or request.json.get("text","")
    cls = clf_predict(text)
    out = {**cls, "neutralized_text": text if cls["bias_type"]=="no_bias" else neutralize(text)}
    return jsonify(out)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
