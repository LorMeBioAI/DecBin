from torch import nn
import torch
from sklearn.cluster import KMeans
from hyperopt import hp, fmin, tpe, Trials
from hyperopt.early_stop import no_progress_loss
from Bio import SeqIO
import os
import matplotlib.pyplot as plt
from functools import partial

from . import DEC
from . import ae_utils
import numpy as np
import pandas as pd
from sklearn import metrics as metric
from . import metrics
from . import eval
from tqdm import trange


def black_box_DBI(param, features):
    kmeans = KMeans(int(param["k_cluster"])).fit(features)
    predict_labels = kmeans.predict(features)
    predict_label = np.array(predict_labels)
    return metric.davies_bouldin_score(features, predict_label)


def param_hyperopt2(param_space, features, max_eval=50):
    trial = Trials()
    early_stop_fn = no_progress_loss(20)
    object_fun = partial(black_box_DBI, features=features)
    params_best = fmin(
        fn=object_fun,
        space=param_space,
        algo=tpe.suggest,
        max_evals=max_eval,
        trials=trial,
        early_stop_fn=early_stop_fn
    )
    print("best params", params_best)
    print(int(params_best["k_cluster"]))
    return params_best, trial


def bin2fasta(input, output, pred_label):
    bin_files = {}
    ext = input.split(".")[-1].lower()
    fmt = "fasta" if ext in ["fasta", "fna", "fa"] else "fastq"
    bin_dir = os.path.join(output, "binned_reads")
    if not os.path.exists(bin_dir):
        os.mkdir(bin_dir)
    for r, record in enumerate(SeqIO.parse(input, fmt)):
        bin_id = pred_label[r]
        bin_path = os.path.join(bin_dir, f"Bin-{bin_id}.fasta")
        if bin_id not in bin_files:
            bin_files[bin_id] = open(bin_path, "w+")
        bin_files[bin_id].write(f">read-{r}\n{record.seq}\n")


def train_dec(read_path, output, latent_dims, hidden_layers, epochs, batch_size, device="cpu", nsp=None):
    com_path = os.path.join(output, "profiles", "com_profs.npy")
    cov_path = os.path.join(output, "profiles", "cov_profs.npy")
    com_profiles = np.load(com_path)
    cov_profiles = np.load(cov_path)
    dataloader = ae_utils.make_data_loader(cov_profiles, com_profiles, batch_size=batch_size, drop_last=False, shuffle=True)

    vae = ae_utils.VAE(
        cov_profiles.shape[1],
        com_profiles.shape[1],
        latent_dims=latent_dims,
        hidden_layers=hidden_layers,
        device=device
    )
    model_path = os.path.join(output, "model.pt")
    vae.load_state_dict(torch.load(model_path)["state"])
    features, index = vae.encode(dataloader)

    n_sample = int(0.1 * len(features))
    sample_idx = np.random.choice(len(features), size=n_sample, replace=False)
    features_sample = features[sample_idx]

    if nsp != None:
        param_space = {"k_cluster": hp.quniform("k_cluster", int(nsp*0.5), int(nsp*1.5), 1)}
    else:
        param_space = {"k_cluster": hp.quniform("k_cluster", 2, 100, 1)}
    params_best, trials = param_hyperopt2(param_space, features_sample, 50)
    n_cluster = int(params_best["k_cluster"])
    print(f"detect {n_cluster} clusters")

    dec = DEC.DEC(n_cluster, vae, hidden=latent_dims).to(device)
    kmeans = KMeans(n_clusters=n_cluster, random_state=0).fit(features)
    cluster_centers = kmeans.cluster_centers_

    if device == "cuda:0":
        cluster_centers = torch.tensor(cluster_centers, dtype=torch.float).cuda()
    #cluster_centers = torch.tensor(cluster_centers, dtype=torch.float)
    dec.clusteringlayer.cluster_centers = torch.nn.Parameter(cluster_centers)

    for param in dec.clusteringlayer.parameters():
        param.requires_grad = False

    loss_function = nn.KLDivLoss(reduction="sum")
    optimizer = torch.optim.SGD(params=dec.parameters(), lr=0.001, momentum=0.9)

    for epoch in trange(epochs):
        for _, data in enumerate(dataloader):
            cov, com, indice = data
            cov = cov.to(device)
            com = com.to(device)
            out_target = dec(cov, com)
            target = dec.target_distribution(out_target).detach()
            loss = loss_function(out_target.log(), target) / out_target.shape[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            

    dec_save = os.path.join(output, "dec_layer.pt")
    vae_save = os.path.join(output, "vae_layer.pt")
    torch.save({"state": dec.state_dict()}, dec_save)
    torch.save({"state": vae.state_dict()}, vae_save)

    total_num = com_profiles.shape[0]
    pred_label = torch.empty(total_num, dtype=torch.int)
    dataloader = ae_utils.make_data_loader(cov_profiles, com_profiles, batch_size=batch_size, drop_last=False, shuffle=False)

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            cov, com, indice = data
            cov = cov.to(device)
            com = com.to(device)
            out_target = dec(cov, com)
            out = out_target.argmax(1).cpu()
            start = i * batch_size
            end = start + batch_size
            if end < total_num:
                pred_label[start:end] = out
            else:
                pred_label[start:] = out

    csv_out = os.path.join(output, "bin.csv")
    pd.DataFrame(pred_label).to_csv(csv_out)
    bin2fasta(read_path, output, list(np.array(pred_label)))


def dec(output, latent_dims, hidden_layers, label_dir, epochs, device="cpu"):
    com_path = os.path.join(output, "profiles", "com_profs.npy")
    cov_path = os.path.join(output, "profiles", "cov_profs.npy")
    com_profiles = np.load(com_path)
    cov_profiles = np.load(cov_path)
    dataloader = ae_utils.make_data_loader(cov_profiles, com_profiles, batch_size=30000, drop_last=False, shuffle=True)
    label = np.array(pd.read_csv(label_dir, index_col=0)).reshape(-1)
    result_file = "./test"

    vae = ae_utils.VAE(
        cov_profiles.shape[1],
        com_profiles.shape[1],
        latent_dims=latent_dims,
        hidden_layers=hidden_layers,
        device=device
    )
    model_path = os.path.join(output, "model.pt")
    vae.load_state_dict(torch.load(model_path)["state"])

    param_space = {"k_cluster": hp.choice("k_cluster", range(2, 100))}
    params_best, trials = param_hyperopt2(param_space, features, 50)
    n_cluster = params_best["k_cluster"]
    dec = DEC.DEC(n_cluster, vae, hidden=latent_dims).to(device)
    features, index = vae.encode(dataloader)

    kmeans = KMeans(n_clusters=n_cluster, random_state=0).fit(features)
    cluster_centers = kmeans.cluster_centers_
    if device == "cuda":
        cluster_centers = torch.tensor(cluster_centers, dtype=torch.float).cuda()
    cluster_centers = torch.tensor(cluster_centers, dtype=torch.float)
    dec.clusteringlayer.cluster_centers = torch.nn.Parameter(cluster_centers)
    dec.clusteringlayer.requires_grad(False)

    y_pred = kmeans.predict(features)
    y_true = label[index]
    accuracy = metrics.acc(y_true, y_pred)
    print(f"Initial Accuracy: {accuracy}")
    eval.clusters_table(y_pred, label, True, os.path.join(result_file, "kmeans.xls"))

    loss_function = nn.KLDivLoss(reduction="sum")
    optimizer = torch.optim.SGD(params=dec.parameters(), lr=0.1, momentum=0.9)
    max_acc = 0.0

    for epoch in trange(epochs):
        for _, data in enumerate(dataloader):
            cov, com, indice = data
            cov = cov.to(device)
            com = com.to(device)
            out_target = dec(cov, com)
            target = dec.target_distribution(out_target).detach()
            loss = loss_function(out_target.log(), target) / out_target.shape[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            cov, com, indice = dataloader.dataset.tensors
            cov = cov.to(device)
            com = com.to(device)
            out_target = dec(cov, com)
            out = out_target.argmax(1).cpu()
            acc = metrics.acc(label, out.numpy())
            print(f"acc is {acc:.5f}")

        print(f"epoch:{epoch + 1}--acc is {acc:.5f}")
        if acc > max_acc:
            dec_save = os.path.join(result_file, "dec_layer.pt")
            vae_save = os.path.join(result_file, "vae_layer.pt")
            torch.save({"acc": acc, "state": dec.state_dict()}, dec_save)
            torch.save({"acc": acc, "state": vae.state_dict()}, vae_save)
            print("model save")
            max_acc = acc
            eval.clusters_table(out.numpy(), label, True, os.path.join(result_file, "result.xls"))


def test_dec(latent_dims, hidden_layers, epochs, device="cpu"):
    sim_dir = "./sim10"
    com_path = os.path.join(sim_dir, "com_profs.npy")
    cov_path = os.path.join(sim_dir, "cov_profs.npy")
    com_profiles = np.load(com_path)
    cov_profiles = np.load(cov_path)
    label_dir = os.path.join(sim_dir, "gut_sim10_4_freq_label.csv")
    dataloader = ae_utils.make_data_loader(cov_profiles, com_profiles, batch_size=30000, drop_last=False, shuffle=True)
    label = np.array(pd.read_csv(label_dir, index_col=0)).reshape(-1)
    result_file = "./test"
    output = "./test"

    vae = ae_utils.VAE(
        cov_profiles.shape[1],
        com_profiles.shape[1],
        latent_dims=latent_dims,
        hidden_layers=hidden_layers,
        device=device
    )
    vae_model_path = os.path.join(sim_dir, "4mervae_layer.pt")
    vae.load_state_dict(torch.load(vae_model_path)["state"])
    features, index = vae.encode(dataloader)
    n_cluster = 10

    dec = DEC.DEC(n_cluster, vae, hidden=latent_dims).to(device)
    kmeans = KMeans(n_clusters=n_cluster, random_state=0).fit(features)
    cluster_centers = kmeans.cluster_centers_
    if device == "cuda":
        cluster_centers = torch.tensor(cluster_centers, dtype=torch.float).cuda()
    cluster_centers = torch.tensor(cluster_centers, dtype=torch.float)
    dec.clusteringlayer.cluster_centers = torch.nn.Parameter(cluster_centers)

    y_pred = kmeans.predict(features)
    y_true = label[index]
    accuracy = metrics.acc(y_true, y_pred)
    print(f"Initial Accuracy: {accuracy}")
    eval.clusters_table(y_pred, label, True, os.path.join(result_file, "kmeans.xls"))

    loss_function = nn.KLDivLoss(reduction="sum")
    trainable_params = filter(lambda p: p.requires_grad, dec.parameters())
    optimizer = torch.optim.SGD(params=trainable_params, lr=0.01, momentum=0.9)
    max_acc = 0.0

    for epoch in range(epochs):
        for _, data in enumerate(dataloader):
            cov, com, indice = data
            cov = cov.to(device)
            com = com.to(device)
            out_target = dec(cov, com)
            target = dec.target_distribution(out_target).detach()
            loss = loss_function(out_target.log(), target) / out_target.shape[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            cov, com, indice = dataloader.dataset.tensors
            cov = cov.to(device)
            com = com.to(device)
            out_target = dec(cov, com)
            out = out_target.argmax(1).cpu()
            acc = metrics.acc(label, out.numpy())
            print(f"acc is {acc:.5f}")

        print(f"epoch:{epoch + 1}--acc is {acc:.5f}")
        if acc > max_acc:
            dec_save = os.path.join(result_file, "dec_layer.pt")
            vae_save = os.path.join(result_file, "vae_layer.pt")
            torch.save({"acc": acc, "state": dec.state_dict()}, dec_save)
            torch.save({"acc": acc, "state": vae.state_dict()}, vae_save)
            print("model save")
            max_acc = acc
            eval.clusters_table(out.numpy(), label, True, os.path.join(result_file, "result.xls"))
            cluster_centers = kmeans.cluster_centers_

        if (epoch + 1) % 5 == 0:
            kmeans = KMeans(n_clusters=n_cluster, random_state=0).fit(features)
            if device == "cuda":
                cluster_centers = torch.tensor(cluster_centers, dtype=torch.float).cuda()
            cluster_centers = torch.tensor(cluster_centers, dtype=torch.float).clone().detach()
            dec.clusteringlayer.cluster_centers = torch.nn.Parameter(cluster_centers)