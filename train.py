import argparse
from model_architectures import supported_models, get_pretrained_model, get_model_input_size
from image_folder import ImageFolder
from model_optimizer import AdamOptimizer
from model_criterion import NLLLossCriterion
from model_classifier import ClassifierNetwork
from model_training import ModelTraining
from save_checkpoint import SaveCheckpoint

"""
Train a new network on a data set with train.py

Basic usage: python train.py data_directory
Prints out training loss, validation loss, and validation accuracy as the network trains

Options:
Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
Choose architecture: python train.py data_dir --arch "vgg13"
Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
Use GPU for training: python train.py data_dir --gpu
"""

parser = argparse.ArgumentParser(description='Train a new network on a data set')

parser.add_argument('data_dir', action="store", metavar='data_directory', type=str, help="directory where the images are")
parser.add_argument('--save_dir', action="store", dest="save_directory", help="directory to save checkpoints")
parser.add_argument('--arch', action="store", default="vgg19_bn", choices=supported_models, dest="model_architecture",  help="architecture to train the model")
# parser.add_argument('--arch', action="store", default="vgg19_bn", dest="model_architecture",  help="architecture to train the model")
parser.add_argument('--learning_rate', action="store", dest="learning_rate", default=0.001, type=float, help="learning rate of the architecture")
parser.add_argument('--hidden_units', action="store", dest="hidden_units", default=512, type=int, help="hidden units of the architecture")
parser.add_argument('--epochs', action="store", dest="epochs", default=1, type=int, help="epoch for training")
parser.add_argument('--gpu', action="store_true", dest="is_gpu", help="use gpu for training")

if __name__ == '__main__':
    results = parser.parse_args()
    print(results)

    try:
        model = get_pretrained_model(results.model_architecture, freeze=True)
        input_units = get_model_input_size(results.model_architecture)

        train_data_dir = f"{results.data_dir}/train"
        test_data_dir = f"{results.data_dir}/test"
        image_folder = ImageFolder(train_data_dir, test_data_dir)
        train_loader, test_loader = image_folder.load_batch_data()
        class_to_index_map = image_folder.generate_class_to_index_map()
        output_units = len(class_to_index_map)
        model.class_to_idx = class_to_index_map
        model.input_features = input_units
        model.output_features = output_units
        print(f"Model: {model}")
        # print(f"Class To Index: {model.class_to_idx}")
        # print(f"Input Features: {model.input_features}")
        # print(f"Output Features: {model.output_features}")
        
        classifier_object = ClassifierNetwork(results.model_architecture, input_units, results.hidden_units, output_units)
        classifier = classifier_object.get_classifier()
        model.classifier = classifier
        print(f"Classifier: {model.classifier}")

        optimizer_object = AdamOptimizer(classifier, results.learning_rate)
        adam_optimizer = optimizer_object.get_optimizer()
        print(f"Optimizer: {adam_optimizer}")

        criterion_object = NLLLossCriterion()
        nllloss_criterion = criterion_object.get_criterion()
        # print(f"Criterion: {nllloss_criterion}")

        model_training = ModelTraining(model, adam_optimizer, nllloss_criterion, results.epochs, results.is_gpu)
#         model_training = ModelTraining(model, adam_optimizer, nllloss_criterion, results.epochs, results.is_gpu, 1)
        trained_model, trained_optimizer, _, _ = model_training.train(train_loader, test_loader)
        trained_model.epochs = results.epochs
#         print(f"Trained Model: {trained_model}")
        print(f"Epochs: {trained_model.epochs}")
#         print(f"Trained Optimizer: {trained_optimizer}")
        
        model_checkpoint = SaveCheckpoint(trained_model, results.model_architecture, results.save_directory, trained_optimizer, results.is_gpu)
        model_checkpoint.create_checkpoint()
    except Exception as e:
        print(f"{results.model_architecture} model may not be trained successfully due to {e}")
