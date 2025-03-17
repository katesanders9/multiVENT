# MultiVENT: Multilingual Videos of Events and Aligned Natural Text

## 🔥🔥🔥 MultiVENT 2.0 is now on HuggingFace! 🔥🔥🔥

We are happy to announce the full release of MultiVENT 2.0 through Huggingface:

**[https://huggingface.co/datasets/hltcoe/MultiVENT2.0](https://huggingface.co/datasets/hltcoe/MultiVENT2.0)**
 
All train and test videos are provided, subset into tar files of 100 videos. Baseline extracted features are provided as well, including:

- CLIP features of 10 frames extracted via pySceneDetect
- SigLIP features of 16 frames uniformly extracted
- Whisper ASR transcripts
- PaddleOCR output
- ICDAR OCR output (better multilingual OCR from Etter et al., 2023)
- Florence video captions (test only)

The train queries/judgments are also available, along with the test queries. The repository is set for manual approval, so if you do not get approval within a day please let us know.
The leaderboard for final results on test will be up by the end of the week. Please feel free to reach out with any questions!

## Repository information

This repository contains data for:
- **MultiVENT** ([Multilingual Videos of Events and aligned Natural Text](https://arxiv.org/abs/2307.03153))
- **MultiVENT 2.0** ([A Massive Multilingual Benchmark for Event-Centric Video Retrieval](https://arxiv.org/abs/2410.11619))
- **MultiVENT-G** ([Grounding Partially-Defined Events in Multimodal Data](https://arxiv.org/abs/2410.05267))

The repository additionally includes setup code for our MultiCLIP video retrieval approach ([NeurIPS D&B, 2023](https://arxiv.org/abs/2307.03153)).

See the repository overview below to find the correct files for each dataset, and the "**Data Navigation**" section for information on how the datasets are organized within each file.

## Overview
```
multiVENT
|   
|   README.md                  # Repository documentation
|   multivent_datasheet.pdf    # Datasheet PDF providing detailed dataset documentation of MultiVENT 1.0
|
└───data
|   |    multivent_base.csv    # MultiVENT 1.0 dataset in CSV format
|   |    multivent_g.json      # MultiVENT-G video IDs and annotations
|   |
|   └───multivent_2
|       |    multivent_2_ids.csv   # MultiVENT 2.0 (train) video IDs
|       |    multivent_2_q.csv     # MultiVENT 2.0 (train) retrieval queries
|       |    multivent_2_j.jsonl   # MultiVENT 2.0 (train) query-video judgments
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
    |   |   |    openclip_featpool_multivent_infer.py    # Python code for inference on MultiVENT
    |   |   |    openclip_featpool_multivent_infer.sh    # Bash script for inference on MultiVENT
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

## Data Navigation
### MultiVENT 1.0
The data is contained in `multiVENT/data/multivent_base.csv`. The data format is in CSV format, with the following columns:
```
[{
video_URL,
video_description,
language,
event_category,
event_name,
article_url,
en_article_url,
en_article_excerpt
}]
```

### MultiVENT 2.0
The data is contained in three files: `multiVENT/data/multivent_2/multivent_2_ids.csv`, `multiVENT/data/multivent_2/multivent_2_q.csv`, and `multiVENT/data/multivent_2/multivent_2_j.jsonl`.

### MultiVENT-Grounded
The data is contained in a single JSON file, `multiVENT/data/multivent_g.json`. The data format is:
```
{ video_id:
    description: string
    event_type: string
    template: {
        template_field : description
    }
    metadata: dict
    text: {
        role (template field)
        text (annotated text span)
        index (start and end character locations in the description)
    }
    temporal: {
        role (template field)
        time (start and end time stamps)
    }
    spatial: {
        role (template field)
        entity (natural language description)
        frame (video frame number)
        bbox (bounding box coordinates within frame)
        certainty (certainty judgment)
        ocr_flag (whether the entity is text)
    }
}
```


## MultiCLIP Installation
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
