# Car Price Prediction Model 🚗

An end-to-end Machine Learning web application that predicts the selling price of used cars based on historical sales data. Theapplication uses a Linear Regression model trained on various car features and provides a clean, interactive web interface for users to get instant price estimates.

## 📂 Project Structure

* **`app.py`**: The core Flask application that serves the web interface, processes user inputs, and interacts with the machine learning model to return predictions.
* **`Car Sales Prediction.ipynb`**: A Jupyter Notebook detailing the data loading, preprocessing, and training of the Linear Regression model using pandas and scikit-learn.
* **`car_price_model.pkl`**: The serialized Machine Learning model. 
* **`car data (1).csv`**: The dataset used to train the model, containing attributes like Present Price, Driven Kilometers, Fuel Type, and Transmission.
* **`index.html`**: The frontend user interface built with HTML.
* **`style.css`**: The stylesheet that provides the modern, responsive design for the web interface.
* **`requirements.txt`**: A list of all Python dependencies required to run the project.

## 🚀 Features

* **Machine Learning Model:** Utilizes a Linear Regression model for fast and interpretable predictions.
* **Smart Data Processing:** Automatically calculates the age of the car based on the current year and handles negative price predictions with smart adjustments.
* **Interactive UI:** A modern, styled web form for users to input car details like Age, Showroom Price, Kilometers Driven, Fuel Type, and Transmission.

## 🛠️ Tech Stack

* **Backend:** Python, Flask
* **Data Handling & ML:** pandas, scikit-learn
* **Frontend:** HTML, CSS

## 💻 How to Run Locally

Follow these steps to set up and run the project on your local machine:

**1. Clone the repository**
Download or clone the project files to your local machine.
