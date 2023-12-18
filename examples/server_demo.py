# -*- coding: utf-8 -*-
import argparse
import uvicorn
import sys
import os
from fastapi import FastAPI, Query
from starlette.middleware.cors import CORSMiddleware
from loguru import logger
from scipy.special import softmax
import numpy as np

sys.path.append('..')
from nerpy import NERModel

pwd_path = os.path.abspath(os.path.dirname(__file__))
# Use fine-tuned model
parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", type=str, default="./outputs/cner_bertsoftmax/best_model",
                    help="Model save dir or model name")
args = parser.parse_args()
s_model = NERModel('bert', args.model_name_or_path)

# define the app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])


@app.get('/')
async def index():
    return {"message": "index, docs url: /docs"}


@app.get('/predict/')
async def entity(q: str = Query(..., min_length=1, max_length=128, title='query')):
    try:
        predictions, raw_outputs, entities = s_model.predict([q], split_on_space=False)
        ret = []
        for n, (preds, outs) in enumerate(zip(predictions, raw_outputs)):
            for pred, out in zip(preds, outs):
                key = list(pred.keys())[0]
                preds = list(softmax(np.mean(out[key], axis=0)))
                if pred[key] != 'O': 
                    ret.append(preds[np.argmax(preds)])
            print(np.mean(ret))
            print(entities[0])
            return {
                "entities": [e[0] for e in entities[0]],
                "probs": str(np.mean(ret)),
            }
        
        logger.debug(f"Successfully get sentence entity, q:{q}")
        
    except Exception as e:
        logger.error(e)
        return {'status': False, 'msg': e}, 500


if __name__ == '__main__':
    uvicorn.run(app=app, host='0.0.0.0', port=8002)
