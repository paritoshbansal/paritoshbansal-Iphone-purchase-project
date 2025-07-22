from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model and label encoder
model = joblib.load("model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            gender = request.form.get('Gender')
            age = int(request.form.get('Age'))
            salary = int(request.form.get('EstimatedSalary'))

            gender_encoded = label_encoder.transform([gender])[0]
            features = np.array([[gender_encoded, age, salary]])
            result = model.predict(features)[0]
            prediction = result  # 0 or 1
        except Exception as e:
            print("Error during prediction:", e)
            prediction = None

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)

