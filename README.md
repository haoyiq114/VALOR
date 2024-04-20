## VALOR-Eval: Holistic Coverage and Faithfulness Evaluation of Large Vision-Language Models

<div align="center">
<b><a href = "https://haoyiq114.github.io">Haoyi Qiu</a>*, <a href = "https://gordonhu608.github.io">Wenbo Hu</a>*, <a href = "https://zdou0830.github.io">Zi-Yi Dou</a>, <a href = "https://vnpeng.net">Nanyun Peng</a></b>
</div>
<div align="center">
<b>University of California, Los Angeles</b>
</div>
<div align="center">
*Equal contribution, listed
in alphabetical order by first name.
</div>
<div align="center">
    <a href=""><img src="https://img.shields.io/badge/Paper-Arxiv-orange" ></a>
    <a href="https://gordonhu608.github.io/VALOR-Eval/"><img src="https://img.shields.io/badge/Project-Page-yellow" ></a>
    <a href="https://github.com/haoyiq114/VALOR/blob/main/LICENSE"><img src="https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg"></a>
    <a href= "https://github.com/haoyiq114/VALOR/blob/main/DATA_LICENSE"><img src="https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg"></a>
</div>


## Introduction
🔍 Large Vision Language Models (LVLMs) suffer from hallucination problems, wherein the models generate plausible-sounding but factually incorrect outputs, undermining their reliability.

📚 A comprehensive quantitative evaluation is necessary to identify and understand the extent of hallucinations in these models. However, existing benchmarks are often limited in scope, focusing mainly on object hallucinations. Furthermore, current evaluation methods struggle to effectively address the subtle semantic distinctions between model outputs and reference data, as well as the balance between _hallucination_ and _informativeness_.

![](assets/evaluation_overview.png?v=1&type=image)

📌 To address these issues, we introduce a multi-dimensional benchmark (**VALOR-Bench**) covering objects, attributes, and relations, with challenging images selected based on associative bias. 

⚖️ Moreover, we propose an LLM-based two-stage evaluation framework (**VALOR-Eval**) that generalizes the popular CHAIR metric and incorporates both faithfulness and coverage into the evaluation. 

🚧 We provide a detailed assessment of 10 established LVLMs within our framework and results demonstrate that we provide a more comprehensive and human-correlated evaluation than existing work.

The results of 10 mainstream VLLMs evaluated by VALOR-Eval.

![](assets/results.png?v=1&type=image)

🌟 Through this work, we highlight the critical balance between faithfulness and coverage of model outputs, and we hope our work encourages future progress on addressing hallucinations in LVLMs while keeping their outputs informative.

## Start with Our Code 

1. Under a Linux environment, clone this repository and navigate to VALOR folder
```bash
git clone https://github.com/haoyiq114/VALOR
cd VALOR
```

2. Install Package
```Shell
conda create -n valor python=3.10 -y
conda activate valor
pip install --upgrade pip
pip install -e .
```

## Dataset

Please refer to [datasets](./datasets/Dataset.md) for preparing the images in our benchmarks. 

## Evaluation 

Prepare your OpenAI KEY, and set it up [here](./evaluation/gpt_model.py#L13).

To inference the 10 LVLMs evaluated in VALOR benchmark, please to refer to [model-generation](./evaluation/model_generation/). Notice that some of the models require to be downloaded and installed seperately, for detials please refer to their offical implementation pages. 

For evaluation of your own model on our benchmark. Once obtained the generated captions from your model, format the output file following the name template in our [scripts](./scripts/). For example, to run the evaluation on objects existence, format the output file as "your_model_name_long_caps.json", then simply replace the model name [here](./scripts/evaluate_object_existence.sh#L1) to:

```Shell
evaluated_model="your model name" 
```
Then run,
```Shell
bash scripts/evaluate_object_existence.sh
```

## Citation
If you found this work useful, consider giving this repository a star and citing our paper as followed:
```
@article{qiu2024eval,
  title={Holistic Coverage and Faithfulness Evaluation of Large Vision-Language Models},
  author={Qiu, Haoyi and Hu, Wenbo and Dou, Zi-Yi and Peng, Nanyun},
  journal={},
  year={2024}
}
```
