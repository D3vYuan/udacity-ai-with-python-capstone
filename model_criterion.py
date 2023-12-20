from torch import nn

class NLLLossCriterion():
    def __init__(self) -> None:
        pass

    def get_criterion(self):
        return nn.NLLLoss()