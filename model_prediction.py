import torch
import json
from image_file import ImageFile

class ModelPrediction():
    def __init__(self, model, top_k, category_name_file, is_gpu):
        self.model = model
        self.top_k = top_k
        self.device = torch.device("cuda" if is_gpu and torch.cuda.is_available() else "cpu")
        self.index_to_class_dict = { v : k for k, v in self.model.class_to_idx.items() }
        self.load_category_names(category_name_file)

    def load_category_names(self, category_name_file):
        with open(category_name_file, 'r') as f:
            self.class_to_name_dict = json.load(f)

    def index_to_class(self, prediction_list):
        return [self.index_to_class_dict[x] for x in prediction_list]

    def class_to_names(self, prediction_list):
        return [self.class_to_name_dict[x] for x in prediction_list]

    def predict(self, image_file):
        try:
            print(f"== Top {self.top_k} Predictions for {image_file} ==")
            image_processor = ImageFile(image_file)
            image_tensor = image_processor.convert_image_to_tensor()
            image_tensor = image_tensor.unsqueeze_(0)

            self.model.to(self.device)
            with torch.no_grad():
                self.model.eval()
                image_tensor = image_tensor.to(self.device)
                # print(image_tensor.size())
                logps = self.model.forward(image_tensor)

        #         # Calculate accuracy
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(self.top_k, dim=1)

                top_probs_np = top_p[0].detach().cpu().numpy()
                top_class_np = top_class[0].detach().cpu().numpy()

                # print(top_class_np)
                top_class_list = self.index_to_class(top_class_np)
                top_class_names_list = self.class_to_names(top_class_list)

            self.model.train()
            return top_probs_np, top_class_names_list
        except Exception as e:
            print(f"No prediction for {image_file} due to {e}")
            return [], []