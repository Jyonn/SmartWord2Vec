import argparse

from utils.config_initializer import ConfigInitializer
from utils.random_seed import seeding
from worker import Worker


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--data', type=str)
    parser.add_argument('--cuda', type=int, default=None)

    args = parser.parse_args()
    config = ConfigInitializer.init(
        model_path=args.model,
        data_path=args.data,
    )
    seeding(2022)

    worker = Worker(config=config, cuda=args.cuda)
    worker.run()
