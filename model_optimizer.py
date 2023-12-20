from torch import optim

class AdamOptimizer():
    def __init__(self, classifier, learning_rate) -> None:
        self.classifier = classifier
        self.learning_rate = learning_rate

    def get_optimizer(self):
        return optim.Adam(self.classifier.parameters(), lr=self.learning_rate)
