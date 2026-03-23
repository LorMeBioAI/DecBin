from torch import nn
import torch
from torch.nn import Parameter
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from torch.utils.data import TensorDataset
from torch.utils.data import Dataset
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
import seaborn as sns

def black_box_DBI(param, features):
    kmeans = KMeans(param['k_cluster']).fit(features)
    predict_labels = kmeans.predict(features)

    #DBI
    predict_label = np.array(predict_labels)
    return metric.davies_bouldin_score(features, predict_label)

def param_hyperopt2(param_space,features, max_eval=50):
    trial = Trials()
    early_stop_fn = no_progress_loss(20)
    object_fun = partial(black_box_DBI,features=features)
    params_best = fmin(fn=object_fun, space = param_space, algo=tpe.suggest, max_evals=max_eval,
                       trials=trial, early_stop_fn=early_stop_fn)
    print('best params', params_best)
    print(params_best['k_cluster'])
    return params_best, trial


def bin2fasta(input,output, pred_label):
    """
    :param input: Original fasta file
    :param output: Output folder
    :param pred_label: Binning labels
    :param indice: Slicing order
    :return: None
    """
    bin_files = {}
    fmt ="fasta" if input.split(
        '.')[-1].lower() in ["fasta", "fna", "fa"] else "fastq"
    if not os.path.exists(f"{output}/binned_reads"):
        os.mkdir(f"{output}/binned_reads")
    for r,record in enumerate(SeqIO.parse(input, fmt)):
        if pred_label[r] not in bin_files:
            bin_files[pred_label[r]] = open(f"{output}/binned_reads/Bin-{pred_label[r]}.fasta", "w+")
        bin_files[pred_label[r]].write(f">read-{r}\n")
        bin_files[pred_label[r]].write(f"{record.seq}\n")

def train_dec(read_path,output,latent_dims, hidden_layers, epochs,batch_size,device='cpu'):
    com_profiles = np.load(f"{output}/profiles/com_profs.npy")
    cov_profiles = np.load(f"{output}/profiles/cov_profs.npy")
    dataloader = ae_utils.make_data_loader(cov_profiles, com_profiles, batch_size=batch_size, drop_last=False, shuffle=True)
    # testloader = ae_utils.make_data_loader(cov_profiles, comp_profiles, drop_last=False, shuffle=False)
    vae = ae_utils.VAE(cov_profiles.shape[1], com_profiles.shape[1],
                       latent_dims=latent_dims,
                       hidden_layers=hidden_layers,
                       device=device)
    # dec = DEC.DEC(n_cluster, vae, hidden=latent_dims).to(device)
    vae.load_state_dict(torch.load(f"{output}/model.pt")['state'])
    features, index = vae.encode(dataloader)
    param_space = {"k_cluster": hp.choice('k_cluster', range(2, 100))}
    params_best, trials = param_hyperopt2(param_space,features, 50)
    n_cluster=params_best['k_cluster']
    print(f"dective {n_cluster} clusters")
    dec = DEC.DEC(n_cluster, vae, hidden=latent_dims).to(device)
    # ============K-means=======================================
    kmeans = KMeans(n_clusters=n_cluster, random_state=0).fit(features)
    cluster_centers = kmeans.cluster_centers_
    
    if device=='cuda':
        cluster_centers = torch.tensor(cluster_centers, dtype=torch.float).cuda()
    cluster_centers = torch.tensor(cluster_centers, dtype=torch.float)
    dec.clusteringlayer.cluster_centers = torch.nn.Parameter(cluster_centers)
    
    dec.clusteringlayer.requires_grad(False)
    # =========================================================
    # training
    loss_function = nn.KLDivLoss(reduction='sum')
    optimizer = torch.optim.SGD(params=dec.parameters(), lr=0.1, momentum=0.9)
    for epoch in range(epochs):
        for i, data in enumerate(dataloader):
            cov, com, indice = data
            cov = cov.to(device)
            # com, indice = data
            com = com.to(device)
            out_target = dec(cov, com)
            target = dec.target_distribution(out_target).detach()
            loss = loss_function(out_target.log(), target) / out_target.shape[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    torch.save({'state': dec.state_dict()}, f'{output}/dec_layer.pt')
    torch.save({'state': vae.state_dict()}, f'{output}/vae_layer.pt')
    # predicting
    pred_label = torch.empty((len(dataloader),1),dtype=torch.int)
    dataloader = ae_utils.make_data_loader(cov_profiles, com_profiles, batch_size=batch_size, drop_last=False, shuffle=False)
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            cov, com, indice = data
            cov = cov.to(device)
            # com, indice = data
            com = com.to(device)
            out_target = dec(cov, com)
            out = (out_target.argmax(1)).cpu()
            if (i+1)*batch_size<len(com_profiles):
                pred_label[i * batch_size:(i + 1) * batch_size] = out
            else:
                pred_label[i*batch_size:]=out
    pd.DataFrame(pred_label).to_csv(f"{output}/bin.csv")
    bin2fasta(read_path,output,list(np.array(pred_label)))

def detect_cluster(output,latent_dims, hidden_layers,label_dir,epochs,device='cpu'):
    com_profiles = np.load(f"{output}/profiles/com_profs.npy")
    cov_profiles = np.load(f"{output}/profiles/cov_profs.npy")
    dataloader = ae_utils.make_data_loader(cov_profiles, com_profiles, batch_size=30000, drop_last=False, shuffle=True)
    label = np.array(pd.read_csv(label_dir, index_col=0)).reshape(-1)
    result_file = "./test"
    # testloader = ae_utils.make_data_loader(cov_profiles, comp_profiles, drop_last=False, shuffle=False)
    vae = ae_utils.VAE(cov_profiles.shape[1], com_profiles.shape[1],
                       latent_dims=latent_dims,
                       hidden_layers=hidden_layers,
                       device=device)
    # dec = DEC.DEC(n_cluster, vae, hidden=latent_dims).to(device)
    vae.load_state_dict(torch.load(f"{output}/model.pt")['state'])
    param_space = {"k_cluster": hp.choice('k_cluster', range(2, 100))}
    params_best, trials = param_hyperopt2(param_space, 50)
    print(params_best['k_cluster'])


def dec(output,latent_dims, hidden_layers,label_dir,epochs,device='cpu'):
    com_profiles = np.load(f"{output}/profiles/com_profs.npy")
    cov_profiles = np.load(f"{output}/profiles/cov_profs.npy")
    dataloader = ae_utils.make_data_loader(cov_profiles, com_profiles, batch_size=30000, drop_last=False, shuffle=True)
    label = np.array(pd.read_csv(label_dir,index_col=0)).reshape(-1)
    result_file=r"./test"
    # testloader = ae_utils.make_data_loader(cov_profiles, comp_profiles, drop_last=False, shuffle=False)
    vae = ae_utils.VAE(cov_profiles.shape[1], com_profiles.shape[1],
                       latent_dims=latent_dims,
                       hidden_layers=hidden_layers,
                       device=device)
    # dec = DEC.DEC(n_cluster, vae, hidden=latent_dims).to(device)
    vae.load_state_dict(torch.load(f"{output}/model.pt")['state'])
    param_space = {"k_cluster": hp.choice('k_cluster', range(2, 100))}
    params_best, trials = param_hyperopt2(param_space, 50)
    n_cluster=params_best['k_cluster']
    dec = DEC.DEC(n_cluster, vae, hidden=latent_dims).to(device)
    features, index = vae.encode(dataloader)
    # ============K-means=======================================
    kmeans = KMeans(n_clusters=n_cluster, random_state=0).fit(features)
    cluster_centers = kmeans.cluster_centers_
    
    if device=='cuda':
        cluster_centers = torch.tensor(cluster_centers, dtype=torch.float).cuda()
    cluster_centers = torch.tensor(cluster_centers, dtype=torch.float)
    dec.clusteringlayer.cluster_centers = torch.nn.Parameter(cluster_centers)
    
    dec.clusteringlayer.requires_grad(False)
    # =========================================================
    y_pred = kmeans.predict(features)
    y_true = label[index]
    accuracy = metrics.acc(y_true, y_pred)
    print('Initial Accuracy: {}'.format(accuracy))
    eval.clusters_table(y_pred, label, True, result_file + r'\kmeans.xls')
    # training
    loss_function = nn.KLDivLoss(reduction='sum')
    optimizer = torch.optim.SGD(params=dec.parameters(), lr=0.1, momentum=0.9)
    max_acc = 0.0
    for epoch in range(epochs):
        for i, data in enumerate(dataloader):
            cov, com, indice = data
            cov = cov.to(device)
            # com, indice = data
            com = com.to(device)
            output = dec(cov, com)
            target = dec.target_distribution(output).detach()
            out = (output.argmax(1)).cpu()
            loss = loss_function(output.log(), target) / output.shape[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            cov, com, indice = dataloader.dataset.tensors
            cov = cov.to(device)
            # com, indice = dataloader.dataset.tensors
            com = com.to(device)
            out_target = dec(cov, com)
            target = dec.target_distribution(out_target).detach()
            out = (out_target.argmax(1)).cpu()
            acc = metrics.acc(label, out.numpy())
            print("acc is %5f" % acc)

        print("epoch:%d--acc is %5f" % ((epoch + 1), acc))
        if acc > max_acc:
            torch.save({'acc': acc, 'state': dec.state_dict()}, result_file + r'\dec_layer.pt')
            torch.save({'acc': acc, 'state': vae.state_dict()}, result_file + r'\vae_layer.pt')
            print("model save")
            max_acc = acc
            eval.clusters_table(out.numpy(), label, True, result_file + r'\result.xls')

def test_dec(latent_dims, hidden_layers,epochs,device='cpu'):
    com_profiles = np.load(f"./sim10/com_profs.npy")
    cov_profiles = np.load(f"./sim10/cov_profs.npy")
    label_dir = f"./sim10/gut_sim10_4_freq_label.csv"
    dataloader = ae_utils.make_data_loader(cov_profiles, com_profiles, batch_size=30000, drop_last=False, shuffle=True)
    label = np.array(pd.read_csv(label_dir,index_col=0)).reshape(-1)
    result_file="./test"
    output ='./test'
    # testloader = ae_utils.make_data_loader(cov_profiles, comp_profiles, drop_last=False, shuffle=False)
    vae = ae_utils.VAE(cov_profiles.shape[1], com_profiles.shape[1],
                       latent_dims=latent_dims,
                       hidden_layers=hidden_layers,
                       device=device)
    # dec = DEC.DEC(n_cluster, vae, hidden=latent_dims).to(device)
    vae.load_state_dict(torch.load(f"./sim10/3mervae_layer.pt")['state'])
    features, index = vae.encode(dataloader)
    n_cluster=10
    """
    param_space = {"k_cluster": hp.choice('k_cluster', range(2, 100))}
    params_best, trials = param_hyperopt2(param_space,features,50)
    n_cluster=params_best['k_cluster']
    """

    dec = DEC.DEC(n_cluster, vae, hidden=latent_dims).to(device)
    # ============K-means=======================================
    kmeans = KMeans(n_clusters=n_cluster, random_state=0).fit(features)
    cluster_centers = kmeans.cluster_centers_
    
    if device=='cuda':
        cluster_centers = torch.tensor(cluster_centers, dtype=torch.float).cuda()
    cluster_centers = torch.tensor(cluster_centers, dtype=torch.float)
    dec.clusteringlayer.cluster_centers = torch.nn.Parameter(cluster_centers)
    #dec.clusteringlayer.cluster_centers.requires_grad_(False)
    # =========================================================
    y_pred = kmeans.predict(features)
    y_true = label[index]
    accuracy = metrics.acc(y_true, y_pred)
    print('Initial Accuracy: {}'.format(accuracy))
    eval.clusters_table(y_pred, label, True, result_file + r'\kmeans.xls')
    
    loss_function = nn.KLDivLoss(reduction='sum')
    optimizer = torch.optim.SGD(params=filter(lambda p:p.requires_grad,dec.parameters()), lr=0.01, momentum=0.9)
    max_acc = 0.0
    for epoch in range(epochs):
        for i, data in enumerate(dataloader):
            cov, com, indice = data
            cov = cov.to(device)
            # com, indice = data
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
            # com, indice = dataloader.dataset.tensors
            com = com.to(device)
            out_target = dec(cov, com)
            out = (out_target.argmax(1)).cpu()
            acc = metrics.acc(label, out.numpy())
            print("acc is %5f" % acc)
        print("epoch:%d--acc is %5f" % ((epoch + 1), acc))
        if acc > max_acc:
            torch.save({'acc': acc, 'state': dec.state_dict()}, result_file + r'\dec_layer.pt')
            torch.save({'acc': acc, 'state': vae.state_dict()}, result_file + r'\vae_layer.pt')
            print("model save")
            max_acc = acc
            eval.clusters_table(out.numpy(), label, True, result_file + r'\result.xls')
            cluster_centers = kmeans.cluster_centers_
        if (epoch+1)%5==0:
            kmeans = KMeans(n_clusters=n_cluster, random_state=0).fit(features)
            if device == 'cuda':
                cluster_centers = torch.tensor(cluster_centers, dtype=torch.float).cuda()
            cluster_centers = torch.tensor(cluster_centers, dtype=torch.float).clone().detach()
            dec.clusteringlayer.cluster_centers = torch.nn.Parameter(cluster_centers)
    """
    pred_label = torch.empty((len(com_profiles)), dtype=torch.int)
    pred_probability = torch.empty((len(com_profiles),10))
    print(pred_label.shape)
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            cov, com, indice = data
            cov = cov.to(device)
            # com, indice = data
            com = com.to(device)
            out_target = dec(cov, com)
            out = (out_target.argmax(1)).cpu()
            out_probility = out_target
            if (i+1)*30000<len(com_profiles):
                pred_label[i * 30000:(i + 1) * 30000] = out
                pred_probability[i*30000:(i+1)*30000] = out_probility
            else:
                pred_label[i*30000:]=out
                pred_probability[i * 30000:] = out_probility
    pred_label = np.array(pred_label)
    pred_index0 = np.where(pred_label == 0)
    index_probability = pred_probability[pred_index0]
    """

    #pd.DataFrame(pred_label).to_csv(f"{output}/bin.csv")
    #bin2fasta("E:/human_gut/sim10/gut_sim10.fasta", output, list(np.array(pred_label)))