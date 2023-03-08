import argparse
import json
import os

import torch
from UniTok import UniDep, Vocab

from kmeans import kmeans
from utils.gpu import GPU

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('-n', type=int)

    args = parser.parse_args()

    device = GPU.auto_choose(torch_format=True)
    embeddings = torch.load(args.checkpoint)['embeddings.weight']
    print('embeddings', embeddings.shape)

    # kmeans
    cluster_ids, _ = kmeans(
        X=embeddings,
        num_clusters=args.n,
        distance='euclidean',
        device=device,
    )
    cluster_ids = cluster_ids.cpu().long().tolist()

    depot = UniDep(f'data/{args.dataset}')
    item_vocab = depot.vocab_depot[depot.get_vocab('list')]  # type: Vocab

    save_path = f'saving/cluster/'
    os.makedirs(save_path, exist_ok=True)
    save_path = os.path.join(save_path, f'{args.dataset}-n{args.n}.json')

    d = dict()
    for i in range(len(cluster_ids)):
        d[item_vocab.index2obj[i]] = cluster_ids[i]
    json.dump(d, open(save_path, 'w'))

    print(f'Clustered and save on {save_path}')
