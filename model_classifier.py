
from collections import OrderedDict
from torch import nn

class ClassifierNetwork():
    def __init__(self, model_architecture, input_units, hidden_units, output_units, dropout_rate = 0.5) -> None:
        self.architecture = model_architecture
        self.input_units = input_units
        self.hidden_units = hidden_units
        self.output_units = output_units
        self.dropout_rate = dropout_rate

    def get_classifier(self):
        print(f"The architecture for {self.architecture} is: input ({self.input_units}) > hidden ({self.hidden_units}) > output ({self.output_units})")
        return nn.Sequential(
            OrderedDict([
                ('fc1', nn.Linear(self.input_units, self.hidden_units)),
                ('relu1', nn.ReLU()),
                ('dropout1', nn.Dropout(self.dropout_rate)),
                ('fc2', nn.Linear(self.hidden_units, self.output_units)),
                ('output', nn.LogSoftmax(dim=1))
            ])
        )
