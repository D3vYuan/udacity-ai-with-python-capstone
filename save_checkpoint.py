import os
import torch
import time
from checkpoint_constants import *
from time_helper import generate_current_time

class SaveCheckpoint():
    def __init__(self, model, model_architecture, checkpoint_path, model_optimizer, is_gpu):
        self.model = model
        self.architecture = model_architecture
        self.optimizer = model_optimizer
        self.path = checkpoint_path
        self.device = torch.device("cuda" if is_gpu and torch.cuda.is_available() else "cpu")
        self.verify_checkpoint_path()
        
    def verify_checkpoint_path(self):
        if self.path:
            return
        
        checkpoint_timestamp = generate_current_time()
        checkpoint_file = f"/opt/checkpoint_{self.architecture}_{self.device}_{checkpoint_timestamp}.pth"
        self.path = checkpoint_file
        print(f"No checkpoint path provided...defaulting to {self.path}")

    def create_checkpoint(self):
        try:
            checkpoint = { checkpoint_input_units: self.model.input_features,
                           checkpoint_output_units: self.model.output_features,
                           checkpoint_model_state_dict: self.model.state_dict(),
                           checkpoint_model_classifier: self.model.classifier,
                           checkpoint_model_classes_to_indices_map: self.model.class_to_idx,
                           checkpoint_model_training_epochs: self.model.epochs,
                           checkpoint_optimizer_state_dict: self.optimizer.state_dict() }

            print(f"Saving model to {self.path}")
            torch.save(checkpoint, self.path)
            print(f"Model saved to {self.path}")
        except Exception as e:
            print(f"Model NOT saved to {self.path} due to {e}")