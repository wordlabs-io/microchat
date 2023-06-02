from datasets import load_dataset
from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType
from transformers import AutoModelForSeq2SeqLM
from transformers import DataCollatorForSeq2Seq

def load_dataset_file(data_location):
    return load_dataset("csv", data_files=data_location)

def load_tokenizer(tokenizer_loc):
    return AutoTokenizer.from_pretrained(tokenizer_loc)

def begin_training(data_location, save_model_location, base_model):
    
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model)
    tokenizer = load_tokenizer(base_model)

    def preprocessing_func(sample):
        inputs = tokenizer(sample['input'],  padding="max_length", truncation=True)
        outputs = tokenizer(sample['output'], padding="max_length", truncation=True)
        
        sample = inputs 
        sample['labels'] = outputs.input_ids

        return sample

    dataset = load_dataset_file(data_location)
    dataset = dataset.map(preprocessing_func, remove_columns=["input", "output"])    
    

    lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q", "v"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    )
    model = prepare_model_for_int8_training(model)
    model = get_peft_model(model, lora_config)

    label_pad_token_id = -100
    
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8
    )

    from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

    output_dir="microchat_cache"

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        auto_find_batch_size=True,
        predict_with_generate=True,
        learning_rate=3e-4,
        num_train_epochs=5,
        logging_strategy="steps",
        logging_steps=100,
        save_strategy="epoch",
        save_total_limit=5,
        push_to_hub=False,
        
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
      
    )
    trainer.train()

    peft_model_id = save_model_location
    trainer.model.save_pretrained(peft_model_id)
    tokenizer.save_pretrained(peft_model_id)

    return peft_model_id
