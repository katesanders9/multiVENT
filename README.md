# MultiVENT: Multilingual Videos of Events and Aligned Natural Text

This repository contains code and data for MultiVENT (Multilingual Videos of Events and aligned Natural Text), a collection of 2,396 multilingual internet news videos with natural language descriptions spanning five target languages. All videos are annotated with ground truth labels and corresponding video text descriptions and current event articles. The repository additionally includes setup code for our MultiCLIP video retrieval approach.

## Overview
```
multiVENT
|   dataset.csv    # MultiVENT dataset in CSV format
|   README.md      # Repository documentation
|   
└───multiCLIP
    |    setup.cfg    # Additional setup file
    |    setup.py     # Setup information for pip install
    |
    └───scripts
    |   |   
    |   └───openclip_xlm
    |   |   |    openclip_featpool_msrvtt_infer.py       # Python code for inference on MSRVTT
    |   |   |    openclip_featpool_msrvtt_infer.sh       # Bash script for inference on MSRVTT
    |   |   |    openclip_featpool_multivent_infer.py    # Python code for inference on MSRVTT
    |   |   |    openclip_featpool_multivent_infer.sh    # Bash script for inference on MSRVTT
    |   |   |    openclip.yaml                           # Config file for model inference
    |   |  
    |   └───utils
    |       |    load_multivent_json.py                  # Generate MultiVENT dataset for inference
    |       |    build_event_ids.py                      # Generate MultiVENT dataset for inference
    |
    └───src
        |   
        └───video_retrieval
            |    __init__.py    
            |    __main__.py    
            |
            └───cli
            |    |    __init__.py                             
            |    |    retrieval_score.py                      # Script to compute evaluation metrics
            |
            └───data
                 |    __init__.py                             
                 |    dataloader_msrvtt_retrieval_laion.py    # Code for MSRVTT data loader
                 |    multivent_retrieval_csv.py              # Code for MultiVENT data loader

```

## Installation
The code in this repository was run on a Python 3.8.6 virtual environment.
### Installation steps:
```
pyenv virtualenv 3.8.6 multiCLIP
pyenv activate multiCLIP
cd multiCLIP
pip install -e .
```
We use the following model weights:

**Tokenizer**: [XLMRobertaTokenizerFast](https://huggingface.co/docs/transformers/model_doc/xlm-roberta)

**OpenCLIP**: [CLIP-ViT-H-14-frozen-xlm-roberta-large-laion5B-s13B-b90k](https://huggingface.co/laion/CLIP-ViT-H-14-frozen-xlm-roberta-large-laion5B-s13B-b90k)

## Data preparation
First, download the videos linked in the dataset.csv file. Then, after converting the CSV file into a JSON file of the format
```
{video_path: {
    language: ...,
    category: ...,
    event: ...,
    description: ...},    # either the video description or event description
...
}
```
the JSON can be converted into the files necessary to run MultiCLIP by running the `load_multivent_json.py` and `build_event_ids.py` files.

## Sources
We draw from the following external repositories:
- https://github.com/roudimit/c2kd
- https://github.com/LAION-AI/temporal-embedding-aggregation
- https://github.com/OpenGVLab/InternVideo/tree/main/Downstream/Video-Text-Retrieval