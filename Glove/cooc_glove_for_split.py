from scipy.sparse import coo_matrix
import numpy as np
import pickle
import pandas as pd
import os
import wandb


def cooc(path):
    with open("./Glove/vocab.pkl", "rb") as f:
        vocab = pickle.load(f)

    data, row, col = [], [], []
    counter = 1
    df = pd.read_json(artifact_dir + '/train.json', lines=True)
    for line in df['sentence1']:
        tokens = [vocab.get(t, -1) for t in line.strip().split()]
        tokens = [t for t in tokens if t >= 0]
        for t in tokens:
            for t2 in tokens:
                data.append(1)
                row.append(t)
                col.append(t2)

        if counter % 10000 == 0:
            print(counter)
        counter += 1
    cooc = coo_matrix((data, (row, col)))
    
    print("summing duplicates (this can take a while)")
    cooc.sum_duplicates()
    if not os.path.exists(f"./Glove/artifacts"):
        os.makedirs(f"./Glove/artifacts")
    with open(f"./Glove/artifacts/cooc.pkl", "wb") as f:
        pickle.dump(cooc, f, pickle.HIGHEST_PROTOCOL)

def glove(path):
    print("loading cooccurrence matrix")
    with open(path, "rb") as f:
        cooc = pickle.load(f)
    print("{} nonzero entries".format(cooc.nnz))

    nmax = 100
    print("using nmax =", nmax, ", cooc.max() =", cooc.max())

    print("initializing embeddings")
    embedding_dim = 20
    xs = np.random.normal(size=(cooc.shape[0], embedding_dim))
    ys = np.random.normal(size=(cooc.shape[1], embedding_dim))

    eta = 0.001
    alpha = 3 / 4

    epochs = 10

    for epoch in range(epochs):
        print("epoch {}".format(epoch))
        for ix, jy, n in zip(cooc.row, cooc.col, cooc.data):
            logn = np.log(n)
            fn = min(1.0, (n / nmax) ** alpha)
            x, y = xs[ix, :], ys[jy, :]
            scale = 2 * eta * fn * (logn - np.dot(x, y))
            xs[ix, :] += scale * y
            ys[jy, :] += scale * x
    np.save(f"./Glove/artifacts/embeddings", xs)


if __name__ == "__main__":
    wandb.login(key='28b1af102774ea945af3ad3deeb378ef8541e375')
    run = wandb.init(name='load_tweet_dataset_1',
                    project='epfl_ml_project2', 
                    tags=['load_dataset'],
                    job_type='for_testing')
    
    artifact = run.use_artifact('hsunyu/epfl_ml_project2/twitter_dataset_1:v0')
    artifact_dir = artifact.download()

    cooc(artifact_dir)
    glove("Glove/artifacts/cooc.pkl")