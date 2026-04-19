<div align="center">

# Evaluating Remote Sensing Image Captions Beyond Metric Biases


[Ziyun Chen (陈子赟)](https://multimodality.group/author/%E9%99%88%E5%AD%90%E8%B5%9F/) 
<img src="assets/hhu_logo.png" alt="Logo" width="15">, &nbsp; &nbsp; 
[Fan Liu (刘凡)*](https://multimodality.group/author/%E5%88%98%E5%87%A1/) ✉ 
<img src="assets/hhu_logo.png" alt="Logo" width="15">, &nbsp; &nbsp;
[Liang Yao (姚亮)](https://multimodality.group/author/%E5%A7%9A%E4%BA%AE/)
<img src="assets/hhu_logo.png" alt="Logo" width="15">, &nbsp; &nbsp; 

[Chuanyi Zhang (张传一)](https://ai.hhu.edu.cn/2023/0809/c17670a264073/page.htm) 
<img src="assets/hhu_logo.png" alt="Logo" width="15">, &nbsp; &nbsp;
Yuye Ma (马玉叶)
<img src="assets/hhu_logo.png" alt="Logo" width="15">, &nbsp; &nbsp;
[Wei Zhou (周玮)](https://weizhou-geek.github.io/) 
<img src="assets/Cardiff_University_logo.png" alt="Logo" width="15">

\*  ✉ *Corresponding Author*
</div>

## News
- **2026/4/15**: We publicly released the code of our paper at this repository.

## Introduction
Welcome to the official repository of our paper "Evaluating Remote Sensing Image Captions Beyond Metric Biases"!

![](assets/ReconScore_RemoteDescriber.png)

The core objective of image captioning is to achieve lossless semantic compression from visual signals into textual modalities. However, the reliance on manually curated reference texts for evaluation essentially forces models to mimic specific human annotation styles, thereby masking the true descriptive capabilities of advanced foundation models. This systemic misalignment prompts a critical question: Is task-specific fine-tuning truly necessary for Remote Sensing Image Captioning, or is the perceived performance gap merely an artifact of flawed evaluation criteria? To investigate this discrepancy, we propose ReconScore, a novel reference-free evaluation metric. Rather than computing textual similarities, we assess caption quality by its capability to reconstruct the original visual elements solely from the generated text, effectively neutralizing human annotation biases. Applying this metric, we uncover a profound, counterintuitive truth: inherently powerful, unfine-tuned MLLMs surpass their fine-tuned counterparts in authentic zero-shot RSIC tasks. Driven by this structural discovery, we introduce RemoteDescriber, a completely training-free generation methodology. By employing ReconScore as a self-correction mechanism, we iteratively refine the semantic precision of MLLM outputs without any computational fine-tuning overhead. Comprehensive experiments demonstrate that RemoteDescriber achieves state-of-the-art performance on three datasets. Furthermore, we validate ReconScore's reliability and analyze the flaws of traditional metrics.

## Setting Up

The code has been verified to work with PyTorch v2.5.1 and Python 3.10.
1. Clone this repository.
2. Change directory to root of this repository.

### Package Dependencies
1. Create a new Conda environment with Python 3.10 then activate it:
```shell
conda create -n ReconScore python==3.10
conda activate ReconScore
```

2. Install PyTorch v2.6.0 with a CUDA version that works on your cluster/machine (CUDA 12.4 is used in this example):
```shell
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

3. Install the latest version of diffusers:
```shell
pip install git+https://github.com/huggingface/diffusers
```
4. To accerate the inference of Qwen3-VL, install the latest version of Flash Attention 2:
```shell
pip install -U flash-attn --no-build-isolation
```
5. Install the packages in `requirements.txt` via `pip`:
```shell
pip install -r requirements.txt
```

### The Pre-trained Models for Usage
1. Download Z-Image checkpoints from 🤗[Z-Image](https://huggingface.co/Tongyi-MAI/Z-Image),
and put the files in `./models/Z-Image`. 

2. The first time that you run dreamsim it will automatically download the model weights.
```shell
from dreamsim import dreamsim

device = "cuda"
model, preprocess = dreamsim(pretrained=True, device=device)
```

3. Download Qwen3-VL-8B-Instruct checkpoints from 🤗[Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct),
and put the files in `./models/Qwen3-VL-8B-Instruct`.

## ReconScore

### Single Evaluation
To evaluate the quality of single image caption, run:
```shell
python evaluate_single.py \
    --caption "a dense residential area with multiple tall buildings and intersecting roads" \
    --ref_img_path "/path/to/your/reference_image.jpg"
```
### Batch Evaluation
1. Prepare your dataset folder like this:
```shell
your_dataset_root/
├── captions/
│   ├── image_001.json
│   ├── image_002.json
│   └── ...
├── reference_images/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
└── generated_images/
```
2. The json format of captions should be like this:
```shell
{
    "filename": "image_001.jpg",
    "caption": "A dense residential area with multiple tall buildings..."
}
```
3. To compute the average ReconScore of your captions, edit the settings in 'run_pipeline.sh' and run:
```shell
bash run_pipeline.sh
```

## RemoteDescriber
We designed a training-free remote sensing image catpioning method based on ReconScore. To generate an image caption, run:
```shell
python RemoteDescriber.py \
    --img_path "./data/sample_image.jpg" \
    --output_dir "./temp_candidates" \
    --qwen_model_path "/data/chenziyun/Qwen3-VL-main/checkpoints/Qwen3-VL-8B-Instruct" \
    --zimage_model_path "/data/chenziyun/Z-Image-main/checkpoints" \
    --num_candidates 4 \
    --device "cuda:0"
```

## Acknowledge
- Code in this repository is built on [Z-Image](https://github.com/Tongyi-MAI/Z-Image), [DreamSim](https://github.com/ssundaram21/dreamsim) and [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL). We'd like to thank the authors for open sourcing their project.

## Contact
Please Contact hhu-czy@hhu.edu.cn