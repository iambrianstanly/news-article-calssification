import torch
from src.model import ModelOne
from fastapi import FastAPI
from src.data_pipeline import tokenize, create_idx



app = FastAPI()

model = ModelOne(24000, 256)

model_state_dict = torch.load('models/02_Experiment_LSTM/02_Experiment_LSTM__100_epoch.pth', weights_only=True)
model.load_state_dict(model_state_dict)


sentence = """
mauresmo opens with victory in la amelie mauresmo and maria sharapova won their opening matches at the tour championships in los angeles.  france s mauresmo routed vera zvonareva 6-1 6-0  while wimbledon champion sharapova was a 6-1 6-4 winner over fellow russian svetlana kuznetsova. american serena williams also won  edging russian elena dementieva 7-6 7-5 for her second victory of the event. the event is split into two groups of four with the top two from each advancing to the semi-finals.  mauresmo s win was her ninth in a row as she tries to overtake lindsay davenport for the number one spot. mauresmo spent five weeks at number one after the us open before injury ushered davenport back in front.  since then  i feel very confident on court and my game is there. i want to get the ranking back  but it s very different than before i was number one.  it was an obsession  but now i take it in a relaxed way.  mauresmo completed her first match in the season-ending championship in 54 minutes as russia s zvonareva struggled to return her serve and failed to achieve a single break point.   she got mad a little bit and i played some great tennis   said mauresmo  who was runner-up to kim clijsters in last year s final. zvonareva has lost both her games so far  having crashed 6-2 6-4 kuznetsova in the staples centre on wednesday.  sometimes not everything works   she said.  it was lots of pressure. maybe that is why i couldn t do 100%. but i was fighting.  sharapova  who lost 6-2 6-2 to kuznetsova in beijing in september  said:  in beijing  she was coming off such a big winning streak [14 matches] and she was unstoppable.  this time  it was important to start off well and put some pressure on her.  the tournament debutant added:  i love it here. the atmosphere is great.  to be here where the lakers play  you just feel that excitement. i love basketball.  williams admitted she is still some way off her best form but remained positive after two wins in two days.  it s hard to go out there and get it right but i m fighting and i m hoping   said williams.  what makes me happy is the effort. i had a really good effort today.  i m trying to add new dimensions to my game.
"""


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


