# SplitMix

Public experiment release for the paper:

**SplitMix: Split-Half Consistency and Mean Core Mixing for Multimodal Electroencephalography Visual Representation Learning**

## Project Layout

```text
SplitMix/
  configs/default.yaml
  splitmix/
    data.py
    losses.py
    metrics.py
    model.py
    module.py
  scripts/run_subject_grid.py
  train.py
  evaluate.py
  requirements.txt
```

## Dataset

We used dataset follows the Data availability Section of https://github.com/dongyangli-del/EEG_Image_decode. Please follow their README to download the EEG dataset, thanks to their prior work!

Regarding pretrained models, we have used:

> Open CLIP ViT-H/14：https://github.com/mlfoundations/open_clip
> 
> DepthAnything: https://github.com/LiheYoung/Depth-Anything
> 
> BLIP2: https://huggingface.co/docs/transformers/main/model_doc/blip-2

## Environment

```bash
pip install -r requirements.txt
```

## Run the Experiment

```bash
python scripts/run_subject_grid.py --config configs/default.yaml --subjects sub-01 sub-02 sub-03 sub-04 sub-05 sub-06 sub-07 sub-08 sub-09 sub-10 --seeds 1 2 3
```
