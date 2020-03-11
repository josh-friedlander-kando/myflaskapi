import os
import time

from dotenv import load_dotenv
from kando import kando_client


class ModelTemplate:
    def __init__(self):
        load_dotenv()
        base_url = "https://kando.herokuapp.com"
        self.client = kando_client.client(base_url, os.getenv('KEY'), os.getenv('SECRET'))

    def train(self, context):
        print('training...')
        start_time = time.time()
        self.do_train(self.client, context)
        print(f'training took {time.time() - start_time} seconds')

    def predict(self, context):
        print('predicting...')
        start_time = time.time()
        pred = self.do_predict(context)
        print(f'predicting took {time.time() - start_time} seconds')
        return pred

    def do_train(self, client, context):
        raise NotImplementedError

    def do_predict(self, context):
        raise NotImplementedError

