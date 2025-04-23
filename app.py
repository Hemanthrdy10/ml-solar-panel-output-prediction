# app.py
import pandas as pd
import warnings
from flask import Flask, render_template, request

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

# Suppress specific warning
warnings.filterwarnings("ignore", message="X does not have valid feature names*", module="sklearn")

app = Flask(__name__)

# Load the dataset
dataset = pd.read_csv('solarPower.csv')

# Drop rows with missing values
dataset.dropna(inplace=True)

# Separate features and target variable
X = dataset[['Location', 'Date', 'Time', 'Season', 'Humidity', 'AmbientTemp', 'Wind.Speed', 'Visibility', 'Pressure', 'Cloud.Ceiling']]
y = dataset['PolyPwr']

# Encode categorical variables
label_encoders = {}
for column in ['Location', 'Season']:
    label_encoders[column] = LabelEncoder()
    X[column] = label_encoders[column].fit_transform(X[column])

# Train the model for the specified season
def train_model(target_season):
    # Transform the input season
    target_season_encoded = label_encoders['Season'].transform([target_season])[0]

    # Filter data for the target season
    X_train_season = X[X['Season'] == target_season_encoded].drop(columns=['Season'])
    y_train_season = y[X['Season'] == target_season_encoded]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_train_season, y_train_season, test_size=0.2, random_state=42)

    # Initialize the Random Forest model
    model = RandomForestRegressor(random_state=42)

    # Train the Random Forest model
    model.fit(X_train, y_train)

    # Make predictions on the testing set
    y_pred = model.predict(X_test)

    # Calculate mean squared error
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error:", mse)

    return model

# Train the model for all seasons
models = {}
for season in dataset['Season'].unique():
    models[season] = train_model(season)

@app.route('/')
def main():
    return render_template('main.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get user inputs
        location = request.form['location']
        date = request.form['date']
        time = request.form['time']
        humidity = float(request.form['humidity'])
        ambient_temp = float(request.form['ambient_temp'])
        wind_speed = float(request.form['wind_speed'])
        visibility = float(request.form['visibility'])
        pressure = float(request.form['pressure'])
        cloud_ceiling = float(request.form['cloud_ceiling'])
        season = request.form['season']

        # Encode user input
        location_encoded = label_encoders['Location'].transform([location])[0]

        # Make prediction using the corresponding model for the selected season
        input_data = pd.DataFrame({
            'Location': [location_encoded],
            'Date': [date],
            'Time': [time],
            'Humidity': [humidity],
            'AmbientTemp': [ambient_temp],
            'Wind.Speed': [wind_speed],
            'Visibility': [visibility],
            'Pressure': [pressure],
            'Cloud.Ceiling': [cloud_ceiling]
        })

        predicted_poly_pwr = models[season].predict(input_data)
        return render_template('result.html', prediction=predicted_poly_pwr[0])

@app.route('/documentation')
def documentation():
    return render_template('documentation.html')

if __name__ == '__main__':
    app.run(debug=True)