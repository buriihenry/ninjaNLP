import torch
from torch import nn
from transformers import Trainer

class CustomTrainer(Trainer):
    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Convert class weights to Float (32-bit)
        weights = torch.tensor(self.class_weights, dtype=torch.float32).to(device=self.device)
        
        # Compute Custom Loss (ensure consistent types)
        loss_fct = nn.CrossEntropyLoss(weight=weights)
        
        # Make sure logits and labels are both Float32
        logits = logits.to(torch.float32)
        labels = labels.to(torch.int64)  # CrossEntropyLoss expects Long type labels
        
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss
    
    def set_class_weights(self, class_weights):
        self.class_weights = class_weights
    
    def set_device(self, device):
        self.device = device