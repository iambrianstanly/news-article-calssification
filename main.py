import torch
from src.model import ModelOne
from fastapi import FastAPI
from src.data_pipeline import tokenize, create_idx

app = FastAPI()

model = ModelOne(24000, 256)

model_state_dict = torch.load('model//02_Experiment_LSTM__100_epoch.pth', weights_only=True)
model.load_state_dict(model_state_dict)



def predict_fn(sentence: str)->str:
    classes = ['business', 'entertainment', 'politics', 'sport', 'tech']
    
    tokens = tokenize(sentence)
    indices = create_idx(tokens)
    seq_length = len(indices)
    
    input_tensor = torch.tensor([indices], dtype=torch.long)
    seq_length_tensor = torch.tensor([seq_length], dtype=torch.long)
    
    model.eval()
    with torch.inference_mode():
        y_logits = model(input_tensor, seq_length_tensor)
        y_pred = torch.softmax(y_logits, dim=1)
    
    class_name = torch.argmax(y_pred, dim=1).item()
    return classes[class_name]
        


@app.get("/")
def root():
    return {"Message":"Welcome"}



@app.post("/predict")
def create_file(inputs: str):
    class_name = predict_fn(inputs)

    return {'predicted': class_name}


