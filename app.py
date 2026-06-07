import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import re

# Read CSV
df = pd.read_csv("data/metrics.csv")

# Ask question
query = input("Ask a question: ")
query = query.lower()

# Detect metric
if "disk" in query:
    metric = "Disk"
elif "cpu" in query:
    metric = "CPU"
elif "memory" in query:
    metric = "Memory"
else:
    print("Metric not found!")
    exit()

# Extract threshold
match = re.search(r"\d+", query)

if match:
    threshold = int(match.group())
else:
    print("Threshold not found!")
    exit()

print("Metric:", metric)
print("Threshold:", threshold)

# Select column
metric_data = df[metric]

# Create ARIMA model
model = ARIMA(metric_data, order=(1,1,1))

# Train model
model_fit = model.fit()

# Forecast next 6 months
forecast = model_fit.forecast(steps=6)

print("\nForecast:")
print(forecast)

# Find threshold crossing
for i, value in enumerate(forecast):
    if value >= threshold:
        print(
            f"\n{metric} reaches {threshold}% in approximately {i+1} month(s)."
        )
        break

# Chart generation
history = metric_data.tolist()
future = forecast.tolist()

history_x = list(range(len(history)))
future_x = list(range(len(history), len(history) + len(future)))

plt.figure(figsize=(8,5))

plt.plot(history_x, history, marker="o", label="Historical")
plt.plot(future_x, future, marker="o", label="Forecast")

plt.axhline(
    y=threshold,
    linestyle="--",
    label=f"{threshold}% Threshold"
)

plt.title(f"{metric} Forecast")
plt.xlabel("Month Index")
plt.ylabel(f"{metric} Usage (%)")
plt.legend()

plt.savefig("forecast.png")

print("\nChart saved as forecast.png")