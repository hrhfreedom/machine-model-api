import os
import uvicorn
from fastapi import FastAPI
from fastapi_utils.cbv import cbv
from fastapi_utils.inferring_router import InferringRouter
import numpy as np
import pickle

app = FastAPI()
router = InferringRouter()

@cbv(router)
class RunModel():
    def __init__(self):
        pkl_filename = "hthub_model.pkl"
        with open(pkl_filename, "rb") as file:
            self.model = pickle.load(file)

    @router.get("/predict")
    def predict(self, new_transaction:float):
        x = np.array([[new_transaction]])
        res = self.model.predict(x)
        if res == 0:
            return {"Category": "not Anomaly"}
        elif res == 1:
            return {"Category" : "Anomaly"}

    
app.include_router(router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))