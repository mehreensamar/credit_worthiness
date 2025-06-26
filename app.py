import joblib
import gradio as gr
import numpy as np

model = joblib.load("credit_model.pkl")

def predict_credit(features):
    prediction = model.predict([features])
    return "Creditworthy ✅" if prediction[0] == 0 else "High Risk ❌"

iface = gr.Interface(
    fn=predict_credit,
    inputs=[
        gr.Textbox(label="RevolvingUtilizationOfUnsecuredLines"),
        gr.Textbox(label="age"),
        gr.Textbox(label="NumberOfTime30-59DaysPastDueNotWorse"),
        gr.Textbox(label="DebtRatio"),
        gr.Textbox(label="MonthlyIncome"),
        gr.Textbox(label="NumberOfOpenCreditLinesAndLoans"),
        gr.Textbox(label="NumberOfTimes90DaysLate"),
        gr.Textbox(label="NumberRealEstateLoansOrLines"),
        gr.Textbox(label="NumberOfTime60-89DaysPastDueNotWorse"),
        gr.Textbox(label="NumberOfDependents"),
    ],
    outputs="text",
    title="💳 Credit Worthiness Predictor",
    theme="dark"
)

iface.launch()
