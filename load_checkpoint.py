import os
import torch
from checkpoint_constants import *

class LoadCheckpoint():
    def __init__(self, model, model_architecture, checkpoint_path, model_optimizer=None):
        self.model = model
        self.architecture = model_architecture
        self.optimizer = model_optimizer
        self.path = checkpoint_path
    
    def load_checkpoint(self):
        try:
            print(f"Loading model from {self.path}")
            model_checkpoint = torch.load(self.path)
            if self.model:
                self.model.classifier = model_checkpoint[checkpoint_model_classifier]
                self.model.input_features = model_checkpoint[checkpoint_input_units]
                self.model.output_features = model_checkpoint[checkpoint_output_units]
                self.model.class_to_idx = model_checkpoint[checkpoint_model_classes_to_indices_map]
                self.model.load_state_dict(model_checkpoint[checkpoint_model_state_dict])
            
            if self.optimizer:
                self.optimizer.load_state_dict(model_checkpoint[checkpoint_optimizer_state_dict])

        except Exception as e:
            print(f"Checkpoint {self.path} was not loaded successfully due to {e}")
        finally:
            return self.model, self.optimizer