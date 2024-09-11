# RaFVerAG
Retrieval-and-Fact-Verification-Augmented Generation

This is the implementation of Evidence Retrieval and Truthfulness Score Calculation.

## Table of Contents

- [Requirements](#requirements)
- [Preparation](#preparation)
- [Inference](#inference)

## Requirements

Run the following command to install the required packages:

```bash
pip install -r requirements.txt
```

## Preparation
You need to (1) download the FactCHD data by yourself following the instruction; (2) download the Wikipedia preprocessed data [here](https://rafverag.s3.ap-southeast-1.amazonaws.com/psgs_w100.tsv) and place it in the ```retrieval_model``` directory (3) download the Wikipedia section embeddings by using ```bash download_wikipedia_embeddings.sh```


The project structure should be:
```
.
├── FactCHD
│   ├── raw_test_200.json
│   ├── raw_test_2k.json
│   ├── raw_test.json
│   └── raw_train.json
├── RaFVerAG_release.py
├── retrieval_model
│   ├── passage_retrieval_enlarged.py
│   ├── psgs_w100.tsv
│   ├── src
│   │   ├── contriever.py
│   │   ├── data.py
│   │   ├── dist_utils.py
│   │   ├── index.py
│   │   ├── __init__.py
│   │   ├── normalize_text.py
│   │   ├── slurm.py
│   │   └── utils.py
│   └── wikipedia_embeddings
│       ├── passages_00
│       ├── passages_01
│       ├── passages_02
│       ├── passages_03
│       ├── passages_04
│       ├── passages_05
│       ├── passages_06
│       ├── passages_07
│       ├── passages_08
│       ├── passages_09
│       ├── passages_10
│       ├── passages_11
│       ├── passages_12
│       ├── passages_13
│       ├── passages_14
│       └── passages_15
└── tree.txt

4 directories, 32 files
```

## Inference

Please run ```python RaFVerAG_release.py``` for the prediction and evaluation on the FactCHD dataset.