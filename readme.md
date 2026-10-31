# Machine Learning-Based Late Delivery Risk Prediction

## Project Overview
This project develops a machine learning system to predict the risk of late deliveries in supply chain operations using order, product, customer, geographical and shipping information.

The project demonstrates how predictive analytics can support supply chain risk monitoring and decision-making.

## Objectives
- Identify orders with a higher risk of late delivery.
- Explore and preprocess logistics data.
- Train and evaluate machine learning classification models.
- Build an interactive prediction dashboard using Streamlit.

## Dataset
- Records: 180,519
- Input features used: 22
- Target variable: `Late_delivery_risk`
- Target classes:
  - `0`: Not classified as late
  - `1`: Classified as late

*Note: The dataset is a public logistics dataset, not verified APL Logistics operational data. APL Logistics is used as the case-study context.*

## Machine Learning Models
- Logistic Regression
- Random Forest
- Balanced Random Forest

## Best Model Performance

| Metric | Logistic Regression |
|---|---:|
| Accuracy | 71.41% |
| ROC-AUC | 0.7823 |

Logistic Regression achieved the best overall performance among the evaluated models.

## Technology Stack
- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- Streamlit
- Plotly
- Joblib

## Interactive Dashboard
The Streamlit dashboard includes:
- Order information input form
- Late-delivery probability prediction
- Visual risk gauge
- Model performance comparison
- Project overview and limitations

## How to Run

1. Clone this repository.
2. Install dependencies:

   `pip install -r requirements.txt`

3. Start the dashboard:

   `python -m streamlit run app.py`

## Limitations
- The dataset does not contain order-date information, preventing chronological validation.
- Predictions are estimates based on historical dataset patterns.
- Real operational data and further validation are needed before production deployment.

## Future Improvements
- Test additional machine learning algorithms.
- Improve feature selection and model calibration.
- Add more detailed risk analysis.
- Validate predictions using real-world operational data.