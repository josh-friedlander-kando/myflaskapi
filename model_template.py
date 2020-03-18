import time


class ModelTemplate:
    def predict(self, context):
        print('predicting...')
        start_time = time.time()
        pred = self.do_predict(context)
        print(f'predicting took {time.time() - start_time} seconds')
        return pred

    def do_predict(self, context):
        raise NotImplementedError

