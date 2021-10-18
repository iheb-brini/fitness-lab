from abc import ABC, abstractmethod


class AbstractPredictor(ABC):
    @abstractmethod
    def predict(self, model, x, **kwargs):
        pass


class Prediction:
    predictor: AbstractPredictor

    def __init__(self, predictor: AbstractPredictor) -> None:
        self.predictor = predictor

    def run(self, model, x, **kwargs):
        return self.predictor.predict(model, x, **kwargs)
