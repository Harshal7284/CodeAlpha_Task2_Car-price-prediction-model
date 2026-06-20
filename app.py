import pickle
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template, request
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "car_price_model.pkl"
DATA_PATH = BASE_DIR / "car data (1).csv"
CURRENT_YEAR = 2026

FUEL_MAP = {"Petrol": 0, "Diesel": 1, "CNG": 3}
TRANSMISSION_MAP = {"Manual": 1, "Automatic": 2}

app = Flask(__name__)
MIN_PREDICTED_PRICE = 0.0


def train_and_save_model():
    df = pd.read_csv(DATA_PATH)
    df = df.drop_duplicates()
    df["Car_Age"] = CURRENT_YEAR - df["Year"]
    df["Fuel_Type"] = df["Fuel_Type"].map(FUEL_MAP)
    df["Transmission"] = df["Transmission"].map(TRANSMISSION_MAP)

    X = df[["Car_Age", "Present_Price", "Driven_kms", "Fuel_Type", "Transmission"]]
    y = df["Selling_Price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)

    with open(MODEL_PATH, "wb") as file:
        pickle.dump(model, file)

    return model, accuracy


def load_model():
    if not MODEL_PATH.exists():
        return train_and_save_model()

    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    df = pd.read_csv(DATA_PATH)
    df = df.drop_duplicates()
    df["Car_Age"] = CURRENT_YEAR - df["Year"]
    df["Fuel_Type"] = df["Fuel_Type"].map(FUEL_MAP)
    df["Transmission"] = df["Transmission"].map(TRANSMISSION_MAP)
    X = df[["Car_Age", "Present_Price", "Driven_kms", "Fuel_Type", "Transmission"]]
    y = df["Selling_Price"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    accuracy = model.score(X_test, y_test)

    return model, accuracy


model, model_accuracy = load_model()


@app.route("/")
def home():
    years = list(range(2003, CURRENT_YEAR + 1))
    years.reverse()
    return render_template(
        "index.html",
        years=years,
        fuel_types=list(FUEL_MAP.keys()),
        transmissions=list(TRANSMISSION_MAP.keys()),
        model_accuracy=round(model_accuracy * 100, 2),
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json() if request.is_json else request.form

        year = int(data.get("year"))
        present_price = float(data.get("present_price"))
        driven_kms = float(data.get("driven_kms"))
        fuel_type = data.get("fuel_type")
        transmission = data.get("transmission")

        if year < 2003 or year > CURRENT_YEAR:
            return jsonify({"error": f"Year must be between 2003 and {CURRENT_YEAR}."}), 400

        if present_price <= 0:
            return jsonify({"error": "Present price must be greater than 0."}), 400

        if driven_kms < 0:
            return jsonify({"error": "Kilometers driven cannot be negative."}), 400

        if fuel_type not in FUEL_MAP:
            return jsonify({"error": "Invalid fuel type selected."}), 400

        if transmission not in TRANSMISSION_MAP:
            return jsonify({"error": "Invalid transmission selected."}), 400

        car_age = CURRENT_YEAR - year
        car = pd.DataFrame(
            [[car_age, present_price, driven_kms, FUEL_MAP[fuel_type], TRANSMISSION_MAP[transmission]]],
            columns=["Car_Age", "Present_Price", "Driven_kms", "Fuel_Type", "Transmission"],
        )

        raw_prediction = float(model.predict(car)[0])
        predicted_price = max(MIN_PREDICTED_PRICE, raw_prediction)
        was_adjusted = raw_prediction < MIN_PREDICTED_PRICE
        predicted_price = round(predicted_price, 2)

        response = {
            "predicted_price": predicted_price,
            "raw_prediction": round(raw_prediction, 2),
            "was_adjusted": was_adjusted,
            "car_age": car_age,
            "year": year,
            "present_price": present_price,
            "driven_kms": driven_kms,
            "fuel_type": fuel_type,
            "transmission": transmission,
        }

        if was_adjusted:
            response["message"] = (
                "The model returned a negative price, which is not realistic for a car. "
                f"The result has been adjusted to Rs. {predicted_price:.2f} Lakhs."
            )

        return jsonify(response)

    except (TypeError, ValueError):
        return jsonify({"error": "Please enter valid numeric values in all fields."}), 400
    except Exception:
        return jsonify({"error": "Something went wrong. Please try again."}), 500


if __name__ == "__main__":
    # Disable reloader on Windows to avoid Anaconda/numpy crash on restart
    app.run(debug=True, host="127.0.0.1", port=5000, use_reloader=False)
