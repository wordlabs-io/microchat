# microchat
A simple Python SDK to build custom chatbots on commodity hardware

## Why did we build this?
Since 2017 we have been seeing a lot of research work going into large language models based on transformer networks. These models tend to be trained across multiple areas, since this enables them to cross reason across domains. However, this also demands that these LLMs be large (true to their name) with some models already hitting the 540 billion parameters mark. 

Obviously, forget training, even loading a 20B parameter model can be hard on some of the best consumer grade laptops without a good deal of optimisation. For instance, I have a laptop with 16GB RAM and a GTX 1650 (4GB VRAM) and I can only load upto 1B parameter models with optimisation. Most people I know don't have even these specs.

However, many papers are suggesting that finetuning smaller models on specific tasks instead of getting a single model to generalise across everything might yield better results [Teaching Small Language Models to Reason](https://arxiv.org/pdf/2212.08410.pdf). While the paper meant to suggest that ~100B parameter models could be outperformed by ~10B parameter models, I wondered how far we could really take this, so I started testing some extremely small models from the FLAN-T5 family on test datasets, and the results were actually usable. Not great, not perfect, but usable. 

This SDK enables you to fine tune your own chat models on your own data ON YOUR OWN LAPTOPS. Go crazy, and do share your findings with us as well!

## What specs do I need?

On my laptop with the following specs, I was able to finetune up to FLAN-T5 Base on 16000 ip/op pairs for five epochs in two hours. 

| RAM | GPU | CPU | 
|------|------|------|
|16GB | GTX 1650 | i5 10th Gen|

Inference on CPU takes about 2s to generate about 50 tokens

## Supported Models
[FLAN T5 Family](https://huggingface.co/docs/transformers/model_doc/flan-t5) 

## Usage
```python
from microchat import core 

#Create a model object
#Use enable_gpu = True if you want to use CUDA
mchat = core.chat(
  base_model = "google/flan-t5-small",
  enable_gpu = True
)

#Initiate Training on a test dataset, columns: input, output
mchat.train_model("test.csv", "temp")

#Use this if model was trained earlier, otherwise use core object
mchat.peft_model_id = 'temp'

#Load model onto CPU/GPU as defined by enable_gpu
mchat.load_model()

#Generate responses
print(mchat.generate_response("answer: When did the leaning tower of pisa fall?"))
```
