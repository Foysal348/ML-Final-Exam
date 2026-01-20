import gradio as gr
import pandas as pd
import pickle

# Load trained pipeline
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

FEATURE_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]

def predict_diabetes(
    pregnancies,
    glucose,
    blood_pressure,
    skin_thickness,
    insulin,
    bmi,
    diabetes_pedigree,
    age
):
    
    input_df = pd.DataFrame(
        [[
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            diabetes_pedigree,
            age
        ]],
        columns=FEATURE_COLUMNS
    )

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    status = "Diabetic" if prediction == 1 else "Non-Diabetic"

    return f"Prediction: {status}\nProbability: {probability:.2f}"

app = gr.Interface(
    fn=predict_diabetes,
    inputs=[
        gr.Slider(0, 20, step=1, label="Pregnancies"),
        gr.Slider(0, 200, step=1, label="Glucose"),
        gr.Slider(0, 140, step=1, label="Blood Pressure"),
        gr.Slider(0, 100, step=1, label="Skin Thickness"),
        gr.Slider(0, 900, step=1, label="Insulin"),
        gr.Slider(0, 70, step=0.1, label="BMI"),
        gr.Slider(0.0, 2.5, step=0.01, label="Diabetes Pedigree Function"),
        gr.Slider(1, 100, step=1, label="Age"),
    ],
    outputs=gr.Textbox(label="Result"),
    title="Diabetes Prediction System"
)

if __name__ == "__main__":
    app.launch()
