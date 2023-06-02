import train 
import load_utils
import torch

class chat:
    def __init__(self, base_model = None, peft_model = None, enable_gpu = False):
        self.base_model = base_model 
        self.tokenizer = None 
        self.peft_model_id = peft_model
        self.peft_model = None
        if torch.cuda.is_available() == False and enable_gpu == True:
            raise Warning("CUDA not accessible, cannot use GPU, reverting to CPU")
            self.enable_gpu = False
        else:
            self.enable_gpu = enable_gpu

    def train_model(self, data_location =  None, save_model_location = None):
        if self.base_model is None:
            raise Exception("Base Model not provided, please add")
        if data_location is None:
            raise Exception("Data location not provided, please add")
        if save_model_location is None:
            raise Exception("Model saving location not provided, please add")
        
        peft_model = train.begin_training(data_location, save_model_location, self.base_model)
        self.peft_model_id = peft_model

    def load_model(self):
        if self.peft_model_id is None:
            raise Exception("Saved Model ID not provided, please add")
        else:    
            self.peft_model, self.tokenizer = load_utils.load_model(self.peft_model_id)

        if self.enable_gpu == True:
            self.peft_model.to("cuda")

    def generate_response(self, input, enable_gpu = False):
        if self.peft_model is None:
            raise Exception("Saved Model not loaded, please load the model first")
        
        return load_utils.generate_response(input, self.peft_model, self.tokenizer, self.enable_gpu)



