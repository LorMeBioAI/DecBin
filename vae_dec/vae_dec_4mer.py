from torch import nn
import torch
from torch.nn import Parameter
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from torch.utils.data import TensorDataset
from torch.utils.data import Dataset
from hyperopt import hp, fmin, tpe, Trials
from hyperopt.early_stop import no_progress_loss

import DEC
import ae_utils
import numpy as np
import pandas as pd
from sklearn import metrics as metric
import metrics
import eval



device = 'cuda' if torch.cuda.is_available() else 'cpu'
latent_dims = 4
hidden_layers = [128,128]
constraints = None
epochs = 40
n_cluster = 43

def black_box_function2(param):
    kmeans = KMeans(param['k_cluster']).fit(features)
    predict_labels = kmeans.predict(features)
    
    predict_label = np.array(predict_labels)
    return -metric.silhouette_score(features, predict_label, metric='euclidean')

def black_box_DBI(param):
    kmeans = KMeans(param['k_cluster']).fit(features)
    predict_labels = kmeans.predict(features)

    #DBI
    predict_label = np.array(predict_labels)
    return metric.davies_bouldin_score(features, predict_label)

def param_hyperopt1(max_eval=50):
    trial = Trials()
    early_stop_fn = no_progress_loss(10)
    params_best = fmin(fn=black_box_function2, space = param_space, algo=tpe.suggest, max_evals=max_eval,
                       trials=trial, early_stop_fn=early_stop_fn)
    print('best params', params_best)
    print(params_best['k_cluster'])
    return params_best, trial

def param_hyperopt2(max_eval=50):
    trial = Trials()
    early_stop_fn = no_progress_loss(20)
    params_best = fmin(fn=black_box_DBI, space = param_space, algo=tpe.suggest, max_evals=max_eval,
                       trials=trial, early_stop_fn=early_stop_fn)
    print('best params', params_best)
    print(params_best['k_cluster'])
    return params_best, trial

if __name__ == "__main__":
    cov_file = r".\sim70\3mer\cov_profs.npy"
    com_file = r".\sim70\3mer\com_profs.npy"
    cov_profile = np.load(cov_file)
    com_profile = np.load(com_file)
    result_file = r".\sim70\3mer"
    label_txt = r".\sim70\sim70_label.txt"
    label_txt = list(pd.read_csv(label_txt, index_col=None, header=None).iloc[:,0])
    label_file = r".\sim70\sim70_label.csv"
    #feature = torch.from_numpy(np.load(com_file))
    label = np.array(pd.read_csv(label_file, index_col=0)).reshape(-1)

    dataloader = ae_utils.make_data_loader(cov_profile, com_profile, batch_size=30000, drop_last=False, shuffle=True)
    #testloader = ae_utils.make_data_loader(cov_profiles, comp_profiles, drop_last=False, shuffle=False)
    vae = ae_utils.VAE(cov_profile.shape[1], com_profile.shape[1],
                       latent_dims=latent_dims,
                       hidden_layers=hidden_layers,
                       constraints=constraints,
                       device=device)
    #dec = DEC.DEC(n_cluster, vae, hidden=latent_dims).to(device)
    vae.load_state_dict(torch.load(r".\sim70\model.pt")['state'])
    dec = DEC.DEC(n_cluster, vae, hidden=latent_dims).to(device)
    features, index = vae.encode(dataloader)
    """
    param_space = {"k_cluster":hp.choice('k_cluster',range(2,100))}
    params_best, trials = param_hyperopt2(50)"""
    # ============K-means=======================================
    kmeans = KMeans(n_clusters=n_cluster, random_state=0).fit(features)
    cluster_centers = kmeans.cluster_centers_
    
    cluster_centers = torch.tensor(cluster_centers, dtype=torch.float).cuda()
    dec.clusteringlayer.cluster_centers = torch.nn.Parameter(cluster_centers)
    # =========================================================
    y_pred = kmeans.predict(features)
    y_true = label[index]
    accuracy = metrics.acc(y_true, y_pred)
    print('Initial Accuracy: {}'.format(accuracy))
    eval.clusters_table(y_pred, label_txt, True, result_file+r'\kmeans.xls')
    
    loss_function = nn.KLDivLoss(reduction='sum')
    optimizer = torch.optim.SGD(params=dec.parameters(), lr=0.1, momentum=0.9)
    max_acc = 0.0
    for epoch in range(epochs):
        for i, data in enumerate(dataloader):
            cov, com, indice = data
            cov = cov.to(device)
            #com, indice = data
            com = com.to(device)
            output = dec(cov,com)
            target = dec.target_distribution(output).detach()
            out = (output.argmax(1)).cpu()
            loss = loss_function(output.log(), target) / output.shape[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            cov, com, indice = dataloader.dataset.tensors
            cov = cov.to(device)
            #com, indice = dataloader.dataset.tensors
            com = com.to(device)
            output = dec(cov, com)
            target = dec.target_distribution(output).detach()
            out = (output.argmax(1)).cpu()
            acc = metrics.acc(label, out.numpy())
            print("acc is %5f" % acc)

        print("epoch:%d--acc is %5f" %((epoch+1), acc))
        if acc > max_acc:
            torch.save({'acc': acc, 'state': dec.state_dict()}, result_file+r'\dec_layer.pt')
            torch.save({'acc': acc, 'state': vae.state_dict()}, result_file + r'\vae_layer.pt')
            print("model save")
            max_acc = acc
            eval.clusters_table(out.numpy(), label_txt, True,result_file+r'\result.xls')






