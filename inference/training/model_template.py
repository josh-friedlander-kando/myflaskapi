import time


class ModelTemplate:
    def train(self, client, **kwargs):
        print('training...')
        start_time = time.time()
        self.do_train(client, **kwargs)
        print(f'training took {time.time() - start_time} seconds')

    def predict(self, context):
        print('predicting...')
        start_time = time.time()
        pred = self.do_predict(context)
        print(f'predicting took {time.time() - start_time} seconds')
        return pred

    def save_metadata(self):
        print('saving metadata...')
        start_time = time.time()
        self.do_save_metadata()
        print(f'finished, took {time.time() - start_time} seconds')

    def do_train(self, context):
        raise NotImplementedError

    def do_predict(self, context):
        raise NotImplementedError

    def do_save_metadata(self):
        raise NotImplementedError
