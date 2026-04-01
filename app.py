from flask import Flask, render_template, request
import pickle
import re
import os
import matplotlib.pyplot as plt

app = Flask(__name__, template_folder="templates")
model = pickle.load(open("model.pkl", "rb"))

# create static folder if not exists
os.makedirs("static", exist_ok=True)


def convert_hour(hour_text):
    hour_text = hour_text.strip().lower()

    if hour_text.isdigit():
        hour = int(hour_text)
        if 0 <= hour <= 23:
            return hour
        raise ValueError("Hour must be between 0 and 23.")

    match = re.fullmatch(r"(\d{1,2})(am|pm)", hour_text)
    if match:
        hour = int(match.group(1))
        period = match.group(2)

        if not 1 <= hour <= 12:
            raise ValueError("For am/pm format, hour must be between 1 and 12.")

        if period == "am":
            return 0 if hour == 12 else hour
        return 12 if hour == 12 else hour + 12

    raise ValueError("Enter hour like 17, 9, 5pm, or 9am.")


def create_bar_chart(hour, day, weekend, trains_arrival, trains_departure, holiday, festival):
    labels = [
        "Hour", "Day", "Weekend", "Arrivals",
        "Departures", "Holiday", "Festival"
    ]
    values = [hour, day, weekend, trains_arrival, trains_departure, holiday, festival]

    plt.figure(figsize=(8, 4.5))
    plt.bar(labels, values)
    plt.title("Input Feature Analysis")
    plt.xlabel("Features")
    plt.ylabel("Values")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig("static/bar_chart.png")
    plt.close()


def create_line_chart(hour, prediction):
    hours = list(range(max(0, hour - 4), min(23, hour + 4) + 1))
    trend = []

    for h in hours:
        if h < 6:
            value = max(1500, prediction * 0.45)
        elif h < 10:
            value = prediction * 0.75
        elif h < 16:
            value = prediction * 0.6
        elif h < 20:
            value = prediction * 1.0
        else:
            value = prediction * 0.8
        trend.append(int(value))

    plt.figure(figsize=(8, 4.5))
    plt.plot(hours, trend, marker='o')
    plt.title("Estimated Hourly Footfall Trend")
    plt.xlabel("Hour of Day")
    plt.ylabel("Estimated Footfall")
    plt.tight_layout()
    plt.savefig("static/line_chart.png")
    plt.close()


@app.route('/')
def home():
    return render_template(
        'index.html',
        result=None,
        crowd_level=None,
        error=None,
        show_graphs=False
    )


@app.route('/predict', methods=['POST'])
def predict():
    try:
        hour = convert_hour(request.form['hour'])
        day = int(request.form['day'])
        weekend = int(request.form['weekend'])
        trains_arrival = int(request.form['trains_arrival'])
        trains_departure = int(request.form['trains_departure'])
        holiday = int(request.form['holiday'])
        festival = int(request.form['festival'])

        values = [[hour, day, weekend, trains_arrival, trains_departure, holiday, festival]]
        prediction = int(model.predict(values)[0])
        prediction = max(1000, prediction)

        if prediction >= 14000:
            crowd_level = "Heavy Crowd"
        elif prediction >= 9000:
            crowd_level = "Moderate Crowd"
        else:
            crowd_level = "Low Crowd"

        create_bar_chart(hour, day, weekend, trains_arrival, trains_departure, holiday, festival)
        create_line_chart(hour, prediction)

        result = f"Predicted Footfall: {prediction} people"

        return render_template(
            'index.html',
            result=result,
            crowd_level=crowd_level,
            error=None,
            show_graphs=True
        )

    except Exception as e:
        return render_template(
            'index.html',
            result=None,
            crowd_level=None,
            error=str(e),
            show_graphs=False
        )


if __name__ == "__main__":
    app.run(debug=True)