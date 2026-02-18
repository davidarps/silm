# Structure-inducing Language Models

This is the code for the paper **Understanding Syntactic Generalization in Structure-inducing Language Models**.

## Installation

This project was developed with Python 3.10 and GCC 13.2. GCC is required only for compiling GPST-specific code

Install the requirements in `env/requirements.txt`, e.g. in a virtual environment: 

```
python3.10 -m venv .silm
source .silm/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Compile GPST modules:

```
cd silm/gpst/
python setup.py build_ext --inplace
```

## Data

All custom datasets can be downloaded from [here](https://uni-duesseldorf.sciebo.de/s/HNLogZzFqeQpicf).

### English 

English data preprocessing uses the preprocssing scripts from [davidarps/boot-bert](https://github.com/davidarps/boot-bert), a fork of [ltgoslo/boot-bert](https://github.com/ltgoslo/boot-bert).

### Dyck languages

Dyck-k language data has been created from [Hewitt et al. (2020)](https://github.com/john-hewitt/dyckkm-learning/). 
Additional scripts for conversion, as well as data creation scripts and configs for Dyck minimal pair data, are available in `scripts/data-creation`.
The respective files can be unzipped in the `data` repository: 

```
unzip dyckdata.zip -d data/dyckkm/ # Training data
mkdir -p data/blimpfordyck
unzip blimpfordyck.zip -d data/blimpfordyck/ # Dyck minimal pairs evaluation data
```

## Training

Example training commands can be found in `train_commands.sh`

## Evaluation

Evaluation scripts are in [silm-eval](https://github.com/davidarps/silm-eval).

## Acknowledgements

- Source code for SiLM architectures was adapted from the original implementations of StructFormer [[1](https://github.com/google-research/google-research/tree/master/structformer) [2](https://github.com/OmarMomen14/Linguistic-Structure-Induction-from-Language-Models/tree/main)], [UDGN](https://github.com/yikangshen/UDGN/tree/main) and [GPST](https://github.com/ant-research/StructuredLM_RTDT), respectively.


