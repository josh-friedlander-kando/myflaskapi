import time


class ModelTemplate:
    def train(self, context):
        print('training...')
        start_time = time.time()
        self.do_train(context)
        print(f'training took {time.time() - start_time} seconds')

    def do_train(self, context):
        raise NotImplementedError

    def predict(self, context):
        print('predicting...')
        start_time = time.time()
        pred = self.do_predict(context)
        print(f'predicting took {time.time() - start_time} seconds')
        return pred

    def do_predict(self, context):
        raise NotImplementedError
