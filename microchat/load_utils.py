import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def load_model(peft_model_id):
    peft_model_id = peft_model_id
    config = PeftConfig.from_pretrained(peft_model_id)

    model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    
    return model, tokenizer  

def generate_response(input, model, tokenizer, enable_gpu = False):
    input_ids = tokenizer(input, return_tensors="pt", truncation=True).input_ids

    if torch.cuda.is_available() == True and enable_gpu == True:
        input_ids = input_ids.to('cuda')
    
    outputs = model.generate(input_ids=input_ids, max_new_tokens=256, do_sample=True, top_p=0.9)

    return tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0] 
