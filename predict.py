import argparse

from model_architectures import supported_models, get_pretrained_model
from load_checkpoint import LoadCheckpoint
from model_prediction import ModelPrediction

"""
Predict flower name from an image with predict.py along with the probability of that name. 
That is, you'll pass in a single image /path/to/image and return the flower name and class probability.

Basic usage: python predict.py /path/to/image checkpoint

Options:
Return top KK most likely classes: python predict.py input checkpoint --top_k 3
Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
Use GPU for inference: python predict.py input checkpoint --gpu
"""

parser = argparse.ArgumentParser(description='Predict flower name from an image')

parser.add_argument('input', action="store", metavar='image_path', type=str, help="path where the image is")
parser.add_argument('checkpoint', action="store", metavar='checkpoint_path', type=str, help="path where the checkpoint is")
parser.add_argument('--arch', action="store", default="vgg19_bn", choices=supported_models, dest="model_architecture",  help="architecture to train the model")
# parser.add_argument('--arch', action="store", default="vgg19_bn", dest="model_architecture",  help="architecture to train the model")
parser.add_argument('--top_k', action="store", default=5, type=int, dest="top_k", help="top predictions to return")
parser.add_argument('--category_names', action="store", default="cat_to_name.json", dest="category_names_json",  help="json containing the category to the real names")
parser.add_argument('--gpu', action="store_true", dest="is_gpu", help="use gpu for inference")

def show_predictions(prediction_image, prediction_probabilities, prediction_names):
    print()
    print("The following are the probabilities: ")
    print(f"{prediction_probabilities}")

    print("The following are the prediction names: ")
    print(f"{prediction_names}")
    print(f"{prediction_image} is likely [{prediction_names[0]}] with a probability of [{prediction_probabilities[0]:.4f}]")

if __name__ == '__main__':
    results = parser.parse_args()
    print(results)

    try:
        model = get_pretrained_model(results.model_architecture, freeze=True)
        model_checkpoint = LoadCheckpoint(model, results.model_architecture, results.checkpoint)
        model_loaded, _ = model_checkpoint.load_checkpoint()
        print(f"Model: {model_loaded}")

        model_prediction = ModelPrediction(model_loaded, results.top_k, results.category_names_json, results.is_gpu)
        prediction_probabilities, prediction_names = model_prediction.predict(results.input)
        show_predictions(results.input, prediction_probabilities, prediction_names)
    except Exception as e:
        print("Image may not be inferred successfully due to {e}")