import argparse
import json
import os

import torch
from UniTok import UniDep, Vocab

from utils.gpu import GPU

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--checkpoint', type=str)

    args = parser.parse_args()

    device = GPU.auto_choose(torch_format=True)
    state_dict = torch.load(args.checkpoint)

    embeddings = state_dict['embeddings.weight']  # [v, d]
    print('embeddings', embeddings.shape)

    layer = 0
    while f'cascade_codebook.codebooks.{layer}.weight' in state_dict:
        codebook = state_dict[f'cascade_codebook.codebooks.{layer}.weight']  # [n, d]
        print('codebook', codebook.shape)

        n = codebook.shape[0]
        cluster_ids = []
        cluster_counts = [0] * n

        for i in range(embeddings.shape[0]):
            embeds = embeddings[i].unsqueeze(0)
            distances = torch.sum(embeds ** 2, dim=1, keepdim=True) + \
                        torch.sum(codebook ** 2, dim=1) - \
                        2 * torch.matmul(embeds, codebook.t())  # type: torch.Tensor
            # distances = torch.norm(embedding - codebook, dim=1)
            cluster_id = torch.argmin(distances).cpu().long().item()
            cluster_ids.append(cluster_id)
            cluster_counts[cluster_id] += 1

        print('cluster counts', cluster_counts)

        layer += 1

    depot = UniDep(f'data/{args.dataset}')
    item_vocab = depot.vocab_depot[depot.get_vocab('list')]  # type: Vocab

    save_path = f'saving/cluster/'
    os.makedirs(save_path, exist_ok=True)
    save_path = os.path.join(save_path, f'{args.dataset}-vq-n{n}.json')

    d = dict()
    for i in range(len(cluster_ids)):
        d[item_vocab.index2obj[i]] = cluster_ids[i]
    json.dump(d, open(save_path, 'w'))

    print(f'Clustered and save on {save_path}')
