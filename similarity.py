import argparse
import json
import os

from sklearn import metrics


class Result:
    def __init__(self, dataset, n, vq=False):
        if vq:
            save_path = f'{dataset}-vq-n{n}.json'
        else:
            save_path = f'{dataset}-n{n}.json'
        self.save_path = os.path.join('saving/cluster/', save_path)
        self.d = json.load(open(self.save_path))
        keys = list(map(int, self.d.keys()))
        keys.sort()
        self.l = [self.d[str(k)] for k in keys]

        self.reverse_d = dict()
        for k, v in self.d.items():
            if v not in self.reverse_d:
                self.reverse_d[v] = 0
            self.reverse_d[v] += 1

        # print with portion
        portions = []
        for k, v in self.reverse_d.items():
            print(f'{v / len(self.d):.4f}', end=' ')
            portions.append(v / len(self.d))
        print()
        # print variance
        print(f'variance: {sum((p - 1 / n) ** 2 for p in portions) / n:.8f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('-n', type=int)

    args = parser.parse_args()

    r1 = Result(dataset=args.dataset, n=args.n, vq=False)
    r2 = Result(dataset=args.dataset, n=args.n, vq=True)

    print(metrics.adjusted_rand_score(r1.l, r2.l))
