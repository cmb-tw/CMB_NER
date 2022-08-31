# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import sys
import numpy as np
import pandas as pd
from scipy.special import softmax

sys.path.append('..')
from nerpy.ner_model import NERModel
from nerpy.bertspan import BertSpanDataset


def main():
    # Creating samples
    # Creating samples
    train_samples = [
        [0, "HuggingFace", "B-MISC"],
        [0, "Transformers", "I-MISC"],
        [0, "started", "O"],
        [0, "with", "O"],
        [0, "text", "O"],
        [0, "classification", "B-MISC"],
        [1, "Nerpy", "B-MISC"],
        [1, "Model", "I-MISC"],
        [1, "can", "O"],
        [1, "now", "O"],
        [1, "perform", "O"],
        [1, "NER", "B-MISC"],
    ]
    train_data = pd.DataFrame(train_samples, columns=["sentence_id", "words", "labels"])

    test_samples = [
        [0, "HuggingFace", "B-MISC"],
        [0, "Transformers", "I-MISC"],
        [0, "was", "O"],
        [0, "built", "O"],
        [0, "for", "O"],
        [0, "text", "O"],
        [0, "classification", "B-MISC"],
        [1, "the", "O"],
        [1, "Nerpy", "B-MISC"],
        [1, "Model", "I-MISC"],
        [1, "then", "O"],
        [1, "expanded", "O"],
        [1, "to", "O"],
        [1, "perform", "O"],
        [1, "good", "O"],
        [1, "NER", "B-MISC"],
    ]
    test_data = pd.DataFrame(test_samples, columns=["sentence_id", "words", "labels"])
    labels = ["O", "B-MISC", "I-MISC"]
    # Create a NERModel
    model = NERModel(
        "bertspan",
        "bert-base-uncased",
        args={"overwrite_output_dir": True, "reprocess_input_data": True, "num_train_epochs": 1, },
        use_cuda=False,
        labels=labels,
    )

    # Train the model
    model.train_model(train_data)

    # Evaluate the model
    result, model_outputs, predictions = model.eval_model(test_data)
    print(result, model_outputs, predictions)

    # Predictions on text strings
    sentences = ["Nerpy Model perform sentence NER", "HuggingFace Transformers build for text"]
    predictions, raw_outputs, entities = model.predict(sentences, split_on_space=True)
    print(predictions, raw_outputs, entities)


if __name__ == '__main__':
    main()
