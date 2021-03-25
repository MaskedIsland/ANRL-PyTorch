from __future__ import print_function

import node2vec
from config import *
from evaluation import *
from model import *
from utils import *
from nce import IndexLinear

import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
from tqdm import tqdm

# Device and random seed settings.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

SEED = 996
np.random.seed(SEED)
torch.manual_seed(SEED)
if device == "cuda":
    torch.cuda.manual_seed(SEED)

parser = argparse.ArgumentParser(description="ANRL parameters settings")
parser.add_argument('--datasets', type=str, default='citeseer', help='datasets descriptions')
parser.add_argument('--inputEdgeFile', type=str, default='graph/citeseer.edgelist', help='input graph edge file')
parser.add_argument('--inputFeatureFile', type=str, default='graph/citeseer.feature', help='input graph feature file')
parser.add_argument('--inputLabelFile', type=str, default='graph/citeseer.label', help='input graph label file')
parser.add_argument('--outputEmbedFile', type=str, default='embed/citeseer.embed', help='output embedding result')
parser.add_argument('--dimensions', type=int, default=128, help='embedding dimensions')
parser.add_argument('--feaDims', type=int, default=3703, help='feature dimensions')
parser.add_argument('--walk_length', type=int, default=80, help='walk length')
parser.add_argument('--num_walks', type=int, default=10, help='number of walks')
parser.add_argument('--window_size', type=int, default=10, help='window size')
parser.add_argument('--p', type=float, default=1., help='p value')
parser.add_argument('--q', type=float, default=1., help='q value')
parser.add_argument('--weighted', type=bool, default=False, help='weighted edges')
parser.add_argument('--directed', type=bool, default=False, help='undirected edges')
args = parser.parse_args()


def generate_graph_context_all_pairs(path, window_size):
    # generating graph context pairs
    all_pairs = []
    for k in range(len(path)):
        for i in range(len(path[k])):
            for j in range(i - window_size, i + window_size + 1):
                if i == j or j < 0 or j >= len(path[k]):
                    continue
                else:
                    all_pairs.append([path[k][i], path[k][j]])

    return np.array(all_pairs, dtype=np.int32)


def graph_context_batch_iter(all_pairs, batch_size):
    while True:
        start_idx = np.random.randint(0, len(all_pairs) - batch_size)
        batch_idx = np.array(range(start_idx, start_idx + batch_size))
        batch_idx = np.random.permutation(batch_idx)
        batch = np.zeros(batch_size, dtype=np.int32)
        labels = np.zeros((batch_size, 1), dtype=np.int32)
        batch[:] = all_pairs[batch_idx, 0]
        labels[:, 0] = all_pairs[batch_idx, 1]
        yield batch, labels


def construct_traget_neighbors(nx_G, X, args, mode='WAN'):
    # construct target neighbor feature matrix
    X_target = np.zeros(X.shape)
    nodes = nx_G.nodes()

    if mode == 'OWN':
        # autoencoder for reconstructing itself
        return X
    elif mode == 'EMN':
        # autoencoder for reconstructing Elementwise Median Neighbor
        for node in nodes:
            neighbors = list(nx_G.neighbors(node))
            if len(neighbors) == 0:
                X_target[node] = X[node]
            else:
                temp = np.array(X[node])
                for n in neighbors:
                    if args.weighted:
                        # weighted sum
                        temp = np.vstack((temp, X[n] * nx_G[node][n]['weight']))
                    else:
                        temp = np.vstack((temp, X[n]))
                temp = np.median(temp, axis=0)
                X_target[node] = temp
        return X_target
    elif mode == 'WAN':
        # autoencoder for reconstructing Weighted Average Neighbor
        for node in nodes:
            neighbors = list(nx_G.neighbors(node))
            if len(neighbors) == 0:
                X_target[node] = X[node]
            else:
                temp = np.array(X[node])
                for n in neighbors:
                    if args.weighted:
                        # weighted sum
                        temp = np.vstack((temp, X[n] * nx_G[node][n]['weight']))
                    else:
                        temp = np.vstack((temp, X[n]))
                temp = np.mean(temp, axis=0)
                X_target[node] = temp
        return X_target


MSE = nn.MSELoss(reduction='sum')


def ae_loss(y_hat, y, alpha):
    return alpha * MSE(y_hat, y)


def sg_loss(emb, y, **kwargs):
    dims = kwargs['dims']
    N = kwargs['N']
    noise = torch.FloatTensor([1 / N for _ in range(N)])
    nce_linear = IndexLinear(
        embedding_dim=dims,  # input dim
        num_classes=N,  # output dim
        noise=noise,
        loss_type='sampled'
    ).to(device)
    loss = nce_linear(y, emb).sum()
    return loss


def main():
    inputEdgeFile = args.inputEdgeFile
    inputFeatureFile = args.inputFeatureFile
    inputLabelFile = args.inputLabelFile
    outputEmbedFile = args.outputEmbedFile
    window_size = args.window_size

    # Read graph
    nx_G = read_graph(args, inputEdgeFile)
    print("edges num:", len(list(nx_G.edges)))

    # Perform random walks to generate graph context
    G = node2vec.Graph(nx_G, args.directed, args.p, args.q)

    G.preprocess_transition_probs()
    walks = G.simulate_walks(args.num_walks, args.walk_length)
    print("walks num:", len(walks))

    # Read features
    print('reading features...')
    X = read_feature(inputFeatureFile)
    print("X features dim:", X.shape)

    print('generating graph context pairs...')
    start_time = time.time()
    all_pairs = generate_graph_context_all_pairs(walks, window_size)
    end_time = time.time()
    print('time consumed for constructing graph context: %.2f' % (end_time - start_time))
    print("all pairs num:", all_pairs.shape)

    nodes = nx_G.nodes()

    # X_hat
    X_target = construct_traget_neighbors(nx_G, X, args, mode='WAN')
    print("X_target shape:", X_target.shape)

    # Total number nodes
    N = len(nodes)
    dims = args.dimensions

    config = Config()

    config.struct[0] = args.feaDims
    config.struct[-1] = args.dimensions

    model = ANRL(config.struct).to(device)
    ae_optim = optim.Adam(model.parameters(), lr=config.ae_learning_rate, weight_decay=config.reg)
    sg_optim = optim.Adam(model.encode.parameters(), lr=config.sg_learning_rate)

    batch_size = config.batch_size
    max_iters = config.max_iters

    idx = 0
    print_every_k_iterations = 1000

    loss_sg = 0
    loss_ae = 0

    for _ in tqdm(range(max_iters)):
        idx += 1

        batch_index, batch_labels = next(graph_context_batch_iter(all_pairs, batch_size))

        # train for autoencoder model
        start_idx = np.random.randint(0, N - batch_size)
        batch_idx = np.array(range(start_idx, start_idx + batch_size))
        batch_idx = np.random.permutation(batch_idx)
        batch_X = torch.from_numpy(X[batch_idx]).float().to(device)
        batch_Y = torch.from_numpy(X_target[batch_idx]).float().to(device)
        model.train()
        model.zero_grad()
        y_hat = model.ae_process(batch_X)

        loss_ae_value = ae_loss(y_hat, batch_Y, alpha=config.alpha)
        loss_ae_value.backward()
        ae_optim.step()
        loss_ae += loss_ae_value.item()
        model.zero_grad()

        # train for skip-gram model
        batch_X = torch.from_numpy(X[batch_index]).float().to(device)
        batch_labels = torch.LongTensor(batch_labels).to(device)
        emb = model.sg_process(batch_X)
        loss_sg_value = sg_loss(emb, batch_labels, N=N, dims=dims)

        loss_sg_value.backward()
        sg_optim.step()
        loss_sg += loss_sg_value.item()
        model.zero_grad()

        if idx % print_every_k_iterations == 0:
            model.eval()
            total_loss = loss_sg / idx + loss_ae / idx
            print('loss: %.2f, ' % total_loss, end='')

            y = read_label(inputLabelFile)
            embedding_result = model.sg_process(torch.from_numpy(X).to(device)).detach().cpu()
            macro_f1, micro_f1 = multiclass_node_classification_eval(embedding_result, y, 0.7)
            print('[macro_f1 = %.4f, micro_f1 = %.4f]' % (macro_f1, micro_f1))

    print('optimization finished...')
    y = read_label(inputLabelFile)
    embedding_result = model.sg_process(torch.from_numpy(X).to(device)).detach().cpu()
    print('repeat 10 times for node classification with random split...')
    node_classification_F1(embedding_result, y)
    print('saving embedding result...')
    write_embedding(embedding_result, outputEmbedFile)


if __name__ == '__main__':
    main()
