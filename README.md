# **CitySafe: Predicting Crime Rates in Indian Cities**

CitySafe is an AI-powered tool designed to predict crime rates across 19 major metropolitan cities in India. It helps law enforcement agencies and policy makers allocate resources efficiently and improve public safety.

## 🔍 **Features**

- **Crime Rate Prediction** for 10 crime categories (e.g., murder, kidnapping, crimes against women)
- **Coverage of 19 Indian Cities** with data from 2014 to 2021
- **Map Visualization** to showcase crime intensity in different areas
- **News Section** providing real-time updates on crime-related news
- **Algorithm Performance Report** showing comparison metrics for model accuracy

## 📊 **Machine Learning Model**

- **Model**: Random Forest Regression
- **Accuracy**: 93.2% on testing data
- **Predicted Categories**: Murder, Kidnapping, Crimes Against Women, Cybercrimes, and more

## 📍 **How it Works**

1. Select the desired **city**, **crime type**, and **year**.
2. Click on "Predict" to get the predicted crime rate.
3. Visualize the results on a **map** and view the **latest news**.
4. Explore the **algorithm report** for model performance insights.

## 🛠️ **Installation**

Follow these steps to set up and run the application locally.

1. **Clone the repository**:

   ```bash
   git clone
   ```

2. **Navigate to the project directory**:

   ```bash
   cd CitySafe
   ```

3. **Install required dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```bash
   python app.py
   ```

## 🚀 **Usage**

- Launch the application by running `app.py`.
- Navigate to the interface to predict the crime rate for specific cities and crime categories.
- View the algorithm report, real-time news, and crime map for better insights.

## 🗺️ **Visualization**

- A **map** is available for users to see predicted crime hotspots in the selected city.
- The map provides color-coded intensity visualization for easy interpretation of data.

## 📄 **Algorithm Report**

The report section includes details on the model's performance, such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and R² score, comparing multiple algorithms.

## 📊 **Dataset**

The dataset used for this project was manually compiled from the **National Crime Rate Bureau (NCRB)**, containing statistics on crimes committed in 19 Indian metropolitan cities between 2014 and 2021. It includes 10 different crime categories, such as:

- Murder
- Kidnapping
- Crime against women
- Crime against children
- Crime by juveniles
- Crime against senior citizens
- Economic offenses
- Cybercrimes

## Model Details

The Random Forest Regression model was chosen for its ability to handle complex, non-linear data. It works by averaging the predictions of multiple decision trees, resulting in a more robust and accurate output. The model has achieved a prediction accuracy of **93.20%**.
