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
