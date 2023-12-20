# -*- coding: utf-8 -*-
import argparse
import uvicorn
import sys
import os
from fastapi import FastAPI, HTTPException, Request
from loguru import logger
from scipy.special import softmax
import numpy as np

sys.path.append('..')
from nerpy import NERModel

pwd_path = os.path.abspath(os.path.dirname(__file__))
# Use fine-tuned model
parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", type=str, default="./examples/outputs/cner_bertsoftmax/best_model",
                    help="Model save dir or model name")
args = parser.parse_args()
s_model = NERModel('bert', args.model_name_or_path)

# define the app
app = FastAPI()

@app.get('/')
async def index():
    return {"message": "index, docs url: /docs"}


@app.post('/predict/')
async def entity(request: Request, response_model=None):
    try:
        data = await request.json()
        q = data.get('query')
        if not q:
            raise HTTPException(status_code=400, detail="Query parameter 'query' is required")

        predictions, raw_outputs, entities = s_model.predict([q], split_on_space=False)
        
        entities_probs = []
        entities_poses = []
        for _, (preds, outs) in enumerate(zip(predictions, raw_outputs)):
            entity_probs = []
            entities_pos = []
            start_flag = False
            for idx, (pred, out) in enumerate(zip(preds, outs)):
                key = list(pred.keys())[0]
                
                if pred[key] == 'B-ORG': 
                    start_flag = True
                    entities_pos.append([idx, 0])
                if pred[key] == 'I-ORG' and start_flag: 
                    entities_pos[-1][1] = idx
                if pred[key] == 'O': 
                    start_flag = False  

                preds = list(softmax(np.mean(out[key], axis=0)))
                prob = preds[np.argmax(preds)]
                if pred[key] != 'O': 
                    entity_probs.append(prob)
            entities_probs.append(entity_probs)
            entities_poses.append(entities_pos)
            
        # Calculating average confidence for each entity
        avg_entity_probs = [np.mean(entity) for entity in entities_probs]
        # Convert numpy.float32 to native Python types
        avg_entity_probs = [round(float(prob), 2) for prob in avg_entity_probs]
        logger.debug(f"Successfully get sentence entity, q:{q}")
        
        filtered_entities = [e for e in [e[0][0] if len(e) > 0 else None for e in entities] if e is not None]
        filtered_probs = [p for p in avg_entity_probs if not np.isnan(p)]
        filtered_start_poses = [e for e in [e[0][0] if len(e) > 0 else None for e in entities_poses] if e is not None]
        filtered_end_poses = [e for e in [e[0][1] if len(e) > 0 else None for e in entities_poses] if e is not None]
        
        return {
            "entities": filtered_entities,
            "probs": filtered_probs,
            "start_poses": filtered_start_poses,
            "end_poses": filtered_end_poses
        }
        
    except Exception as e:
        logger.error(e)
        return {'status': False, 'msg': str(e)}, 500

if __name__ == '__main__':
    uvicorn.run(app=app, host='0.0.0.0', port=8002)
