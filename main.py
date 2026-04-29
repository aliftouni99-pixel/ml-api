from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all (for now)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
model1 = joblib.load("model1_adherence_prediction_pipeline.pkl")
model2 = joblib.load("model2_pipeline.joblib")


@app.post("/predict")
def predict(data: dict):
    try:
        # -------- Model 1 --------
        input1_dict = data["model1_inputs"]

        # Convert dict → list (must match training order)
        input1 = list(input1_dict.values())
        input1 = np.array(input1).reshape(1, -1)

        adherence = model1.predict(input1)[0]

        # -------- Model 2 --------
        input2_dict = data["model2_inputs"]

        # Inject Model 1 output into existing feature
        input2_dict["m1_predicted_risk_score"] = float(adherence)

        input2 = list(input2_dict.values())
        input2 = np.array(input2).reshape(1, -1)

        final_pred = model2.predict(input2)[0]

        return {
            "adherence": int(adherence),
            "final_prediction": int(final_pred)
        }

    except Exception as e:
        return {"error": str(e)}