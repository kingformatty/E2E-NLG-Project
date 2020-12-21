# CS269 NLG Course Project: Sentence Controllable DA-based E2E Generation in Restaurant Domain

The code of our course project is modified based on
[https://github.com/UKPLab/e2e-nlg-challenge-2017](https://github.com/UKPLab/e2e-nlg-challenge-2017)

Updates:

1. We found a critical error in the original code, which could possibly make the training wrong. (components/model/modules/attention/attn\_bahd.py#68) We have  fixed it in this project.

2. Two more data pre-processing strategies are implemented. ([A General Model for Neural Text Generation from Structured Data](http://www.macs.hw.ac.uk/InteractionLab/E2E/final_papers/E2E-Chen.pdf), [End-to-End Content and Plan Selection for Data-to-Text Generation](https://arxiv.org/abs/1810.04700))

3. Transformer model

4. Sentence Control stratigies:
     * Predict and Generate (PAG)
     * Controllable NOS Embedding (CNE)

## Background info

* Official website: http://www.macs.hw.ac.uk/InteractionLab/E2E/
* Evaluation protocol: automatic metrics for system development
     
## Project structure

We provide basic scripts and their utilities in this repository, along with some output files' content:

* `run_experiment.py`: main script to run (please freeze this script since experiment configurations are declared in YAML file).
* `config/train_XXXX.yaml` and `config/predict_XXXX.yaml`: configuration files to use with the script above.
* `components/`: data_preprocessing (HAV, HIT, UKP), model (MLP, GRU, Transformer...), trainer, evaluator, and necessary utils. 

## Requirements

* 64-bit Linux versions
* Python 3 and dependencies:
    * PyTorch v1.2.0
    * Progressbar2 v3.18.1

## Installation

* Install Python3 dependencies:

```
$ conda install pytorch torchvision cuda80 -c soumith 
$ conda install progressbar2
```

* Python2 dependencies are needed only to run the official evaluation scripts.


## Running the experiments

### Preparation

* Step 1

For your convenience, we have setup multiple configuration files in `config/`. 
1. `train_transformer.YAML` and `predict_transformer.YAML` run experiments using standard Transformer model. Please freeze `nos_option` to be 0. 
2. `train_transformer_CNE_Enc.YAML` and `predict_transformer_CNE_Enc.YAML` run experiments using Transformer model with CNE NOS embedding in encoder input. Please freeze `nos_option` and `nos_position` to maintain the CNE_Enc structure.
3. `train_transformer_CNE_Dec.YAML` and `predict_transformer_CNE_Dec.YAML` run experiments using Transformer model with CNE NOS embedding in decoder input. Please freeze `nos_option` and `nos_position` to maintain the CNE_Dec structure.
4. `train_transformer_PAG.YAML` and `predict_transformer_PAG.YAML` run experiments using Transformer model with PAG NOS. Please freeze `nos_option` to maintain the PAG structure. (`nos_position` is not used in this method).
5. `train_XXXX.YAML` and `predict_XXXX.YAML` files are required to be used in pairs.
6. `train_MLP.YAML` and `predict_MLP.YAML` run experiments using standard MLP model.
7. `train_gru.YAML` and `predict_gru.YAML` run experiments using standard gru model.
8. `train_tfenc.YAML` and `predict_tf.YAML` run experiments using transformer as encoder model.
9. `data_module` can be chosen from `e2e_data_hav`, `e2e_data_hit`, and `e2e_data_MLP`, do not forget to modify corresponding parameters in configuration files following the instruction within them.
10. Constraint: NOS related terms are specifically designed for Transformer model. 
11. For other model configuration parameters including `embedding size`, `hidden size`, `input size`, etc., if your setting confirms the instruction within the configuration file, the script can execute correctly.

* Step2 

Please first download e2e-metrics toolkit from [3]. We are going to use the measure_score.py script to evaluate the quality of the generated sentence.
    



### Training models

1. Adjust data paths and choose a configuration file or use your own defined YAML (*train_transformer.yaml*, as a running example).
    
2. Run the following command:  
        
    ```
    $ python run_experiment.py config/train_transformer.yaml
    ```
    
3. After the experiment, a folder will be created in the directory specified 
    by the *experiments_dir* field of *train_transformer.yaml* file.
    This folder should contain the following files:
        
    * model weights and development set predictions for each training epoch (*weights.epochX*, *predictions.epochX*)   
    * a csv file with scores and train/dev losses for each epoch (*scores.csv*)  
    * configuration dictionary in json format (*config.json*)  
    * pdf files with learning curves (optional)  
    * experiment log (*log.txt*)  
    
4. If you use a model for prediction (setting "predict" as the value for the *mode* field in the config file and 
    specifying model path in *model_fn*), use corresponding predict_transformer.yaml 
    or set the configuration parameters to be the same as your training configuration if you want to use the self-defined configuration.
    the predictions done by the loaded model will be
    stored in:  
        * $model_fn.devset.predictions.txt  
        * $model_fn.testset.predictions.txt  
   
### Evaluation

Evaluate the generated context using following command:

    ```
    $ cd e2e-metrics
    $ python measure_scores.py <ref> <generated>
    $ #Eg.
    $ python measure_scores.py e2e-dataset/testset_w_refs.csv.multi-ref exp/e2e_model_transformer_seed1-emb256-hid512-drop0.05-bs128-lr0.0002_2020-Dec-17_23.53.25/weights.epoch1.testset.predictions.txt.
    ```
    
This script will calculate BLEU, NIST, CIDEr, ROUGE-L, and METEOR. 

[1]: https://www.gnu.org/software/gettext/manual/gettext.html#sh_002dformat
[2]: http://www.macs.hw.ac.uk/InteractionLab/E2E/data/baseline-output.txt
[3]: https://github.com/tuetschek/e2e-metrics
