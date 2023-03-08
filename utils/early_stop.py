class EarlyStop:
    def __init__(self, patience):
        self.loss = None
        self.epoch = None
        self.patience = patience

    def push(self, epoch, loss):
        if self.loss is None or self.loss > loss:
            self.epoch = epoch
            self.loss = loss
            return False

        if epoch - self.epoch > self.patience:
            return True
