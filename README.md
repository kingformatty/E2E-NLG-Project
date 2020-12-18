# Notice

The code of our course project is modified based on
[https://github.com/UKPLab/e2e-nlg-challenge-2017](https://github.com/UKPLab/e2e-nlg-challenge-2017)

We found a big error in the original code, which could possibly make the training wrong. (components/model/modules/attention/attn\_bahd.py#68)

We have already fixed it for our course project.

Transformer encoder is updated.

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
    * PyTorch v0.2.0
    * Progressbar2 v3.18.1
* Python 2

## Installation

* Install Python3 dependencies:

```
$ conda install pytorch torchvision cuda80 -c soumith 
$ conda install progressbar2
```

* Python2 dependencies are needed only to run the official evaluation scripts.
See installation instructions [here][3].

## Running the experiments

### Preparation

* Step 1

For your convenience, we have setup multiple configuration files in `config/`. 
1. `train_transformer.YAML` and `predict_transformer.YAML` run experiments using standard Transformer model. Please freeze `nos_option` to be 0. 
2. `train_transformer_CNE_Enc.YAML` and `predict_transformer_CNE_Enc.YAML` run experiments using Transformer model with CNE NOS embedding in encoder input. Please freeze `nos_option` and `nos_position` to maintain the CNE_Enc structure.
3. `train_transformer_CNE_Dec.YAML` and `predict_transformer_CNE_Dec.YAML` run experiments using Transformer model with CNE NOS embedding in decoder input. Please freeze `nos_option` and `nos_position` to maintain the CNE_Dec structure.
4. `train_transformer_PAG.YAML` and `predict_transformer_PAG.YAML` run experiments using Transformer model with PAG NOS. Please freeze `nos_option` to maintain the PAG structure. (`nos_position` is not used in this method).
5. `train_XXXX.YAML` and `predict_XXXX.YAML` files are required to be used in pairs.

* Step2 

Modify `PYTHON2` and `E2E_METRCIS_FOLDER` variables in the following file:

`components/evaluator/eval_scripts/run_eval.sh`

This shell script is calling the [external evaluation tools][3].
`PYTHON2` denotes a specific python environment with all the necessary dependencies installed.
`E2E_METRICS_FOLDER` denotes the cloned repository with the aforementioned tools.

### Training models

* **Model-D**:
    1. Adjust data paths and hyper-parameter values in the config file (*my_config.yaml*, as a running example).
    
    2. Run the following command:  
        
    ```
    $ python run_experiment.py my_config.yaml
    ```
    
    3. After the experiment, a folder will be created in the directory specified 
    by the *experiments_dir* field of *my_config.yaml* file.
    This folder should contain the following files:
        - experiment log (*log.txt*)
        - model weights and development set predictions for each training epoch 
        (*weights.epochX*, *predictions.epochX*)    
        - a csv file with scores and train/dev losses for each epoch (*scores.csv*)
        - configuration dictionary in json format (*config.json*)
	    - pdf files with learning curves (optional)
    
    4. If you use a model for prediction 
    (by setting "predict" as the value for the *mode* field in the config file and 
    specifying model path in *model_fn*), the predictions done by the loaded model will be
    stored in:
        - $model_fn.devset.predictions.txt
        - $model_fn.testset.predictions.txt
   




[1]: https://www.gnu.org/software/gettext/manual/gettext.html#sh_002dformat
[2]: http://www.macs.hw.ac.uk/InteractionLab/E2E/data/baseline-output.txt
[3]: https://github.com/tuetschek/e2e-metrics
