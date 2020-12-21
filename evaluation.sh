#!/bin/bash

#please copy and paste e2e-metrics folder into directory same as run_experiment.py

#the first command will run the prediction, and a prediction_XXX.txt file will be generated in corresponding folders
#the second command will use e2e-metric toolkit to measure BLEU, NIST, CIDEr, ROUGE-L, METEOR scores.

echo "run tranformer baseline prediction and evaluation"
python run_experiment.py Sidi_Lu/Transformer_Baseline/predict_transformer.yaml

python e2e-metrics/measure_scores.py e2e-dataset/testset_w_refs.csv.multi-ref Sidi_Lu/Transformer_Baseline/weights.epoch3.testset.predictions.txt

echo "run PAG prediction and evaluation"
python run_experiment.py Sidi_Lu/PAG/predict_transformer_PAG.yaml

python e2e-metrics/measure_scores.py e2e-dataset/testset_w_refs.csv.multi-ref Sidi_Lu/PAG/weights.epoch7.testset.predictions.txt

echo "run CNE_Enc prediction and evaluation"
python run_experiment.py Sidi_Lu/CNE_Enc/predict_transformer_CNE_Enc.yaml

python e2e-metrics/measure_scores.py e2e-dataset/testset_w_refs.csv.multi-ref Sidi_Lu/CNE_Enc/weights_CNE_Enc.epoch12.testset.predictions.txt

echo "run CNE_Dec prediction and evaluation"
python run_experiment.py Sidi_Lu/CNE_Dec/predict_transformer_CNE_Dec.yaml

python e2e-metrics/measure_scores.py e2e-dataset/testset_w_refs.csv.multi-ref Sidi_Lu/CNE_Dec/weights_CNE_Dec.epoch15.testset.predictions.txt


