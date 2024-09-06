from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load the trained model
model = joblib.load('model.pkl')

# Initialize FastAPI app
app = FastAPI()

# Define the input data model using Pydantic
class InputData(BaseModel):
    id: int
    Area: int
    MajorAxisLength: float
    MinorAxisLength: float
    Eccentricity: float
    ConvexArea: int
    EquivDiameter: float
    Extent: float
    Perimeter: float
    Roundness: float
    AspectRation: float

# Define a prediction endpoint
@app.post("/predict")
def predict(input_data: InputData):
    # Convert input data to a numpy array, excluding the id
    data = np.array([[input_data.Area, input_data.MajorAxisLength, input_data.MinorAxisLength,
                      input_data.Eccentricity, input_data.ConvexArea, input_data.EquivDiameter,
                      input_data.Extent, input_data.Perimeter, input_data.Roundness,
                      input_data.AspectRation]])
    # Use the model to make a prediction
    prediction = model.predict(data)
    # Return the binary prediction (0 or 1)
    return {"id": input_data.id, "prediction": int(prediction[0])}

