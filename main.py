from fastapi import  FastAPI, HTTPException
import pickle
import pandas as pd
from pydantic import BaseModel
from xgboost import XGBClassifier

app = FastAPI()

with open('xgb_model.pkl', 'rb') as model_file:
    xgb_model = pickle.load(model_file)
 
# Define the input schema with all required fields
class TransactionData(BaseModel):
    merchant: int
    category: int
    amt: float
    gender: int  # 0 or 1
    city: int
    state: int
    city_pop: int
    job: int
    unix_time: float
    age: int
    
# Preprocess the transaction data for the model
def preprocess_data(trans_data: TransactionData):
    # Convert input to a dictionary
    input_dict = trans_data.dict()
    
    # Convert dictionary to DataFrame
    input_df = pd.DataFrame([input_dict])
    
    # Any additional data preprocessing as I did in the training notebook
    
    
    return input_df


@app.post('/predict')
def predict_fraud(data: TransactionData):
    try:
        # Preprocess the data
        processed_data = preprocess_data(data)
        
        # Make a prediction
        prediction = xgb_model.predict(processed_data)
        
        # # Interpret the prediction
        # result = 'Fraud' if prediction[0] == 1 else 'Not Fraud'
        probabilities = xgb_model.predict_proba(processed_data)[0]
        # [1]
        
        
        return {
            "prediction": prediction.tolist(),
            "probabilities": probabilities.tolist()
            }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")
    

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)











# # Preprocess the transaction data dictionary 
# def preprocess_data(trans_data_dict):
#     input_dict = {
        
#     }