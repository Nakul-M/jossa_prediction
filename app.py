from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load trained pipeline
with open("./models/jossa-dummy.pkl", "rb") as f:
    jossa_model = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['POST'])
def predict_datapoint():
    if request.method == "POST":
        # Collect input values from form
        input_data = {
            "Seat Type": request.form["Seat_Type"],
            "Quota": request.form["Quota"],
            "Institute": request.form["Institute"],
            "Academic Program Name": request.form["Academic_Program_Name"],
            "Gender": int(request.form["Gender"]),
            "Year": int(request.form["Year"]),
            "PWD": int(request.form["PWD"])
        }

        # Convert to DataFrame
        df_input = pd.DataFrame([input_data])

        # Predict
        prediction = jossa_model.predict(df_input)[0]

        return render_template("home.html", prediction_text=f"Predicted Closing Rank: {int(prediction)}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
