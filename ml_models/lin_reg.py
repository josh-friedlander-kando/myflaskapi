from abc import ABC

import numpy as np
from sklearn.linear_model import LinearRegression

from .model_template import ModelTemplate


class LinRegTemplate(ModelTemplate, ABC):
    def __init__(self, request):
        self.X = np.array(request.json["X"])
        self.y = np.dot(self.X, np.array(request.json["Y"])) + 3

    def train(self):
        model = LinearRegression()
        model.fit(self.X, self.y)
        return model

    def do_predict(self, model, request):
        pred = model.predict(np.array(request.json["X_test"]))
        return pred
