import pickle
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
from torch.utils.data import DataLoader
from graph_constructor import GraphDatasetHFGNN, collate_fn_hfgnn
import os
import pandas as pd
import re
from rdkit import Chem
from prody import *
from dgl import graph
import argparse
from utils import *
from prefetch_generator import BackgroundGenerator
import time
import datetime
from model_v2 import HFGNN
import warnings
warnings.filterwarnings("ignore")





class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__()) #加速数据加载


def run_a_train_epoch(model, loss_fn, train_dataloader, optimizer, device):
    # training model for one epoch
    model.train()
    for i_batch, batch in enumerate(train_dataloader):
        model.zero_grad() #梯度设置为0
        hyper_bg, bg3, Ys, _ = batch
        hyper_bg, bg3, Ys = hyper_bg.to(device), bg3.to(device), Ys.to(device)#交给gpu计算
        outputs, weights = model(hyper_bg, bg3)
        loss = loss_fn(outputs, Ys)
        loss.backward()
        optimizer.step()


def run_a_eval_epoch(model, validation_dataloader, device):
    true = []
    pred = []
    key = []
    model.eval()
    with torch.no_grad():
        for i_batch, batch in enumerate(validation_dataloader):
            # DTIModel.zero_grad()
            hyper_bg, bg3, Ys, keys = batch
            hyper_bg, bg3, Ys = hyper_bg.to(device), bg3.to(device), Ys.to(device)
            #bg, bg3, Ys, Hyper_index_ls, Hyper_index_ps, num_atoms_m1s, num_atoms_m2s = bg.to(device), bg3.to(device), Ys.to(device), Hyper_index_ls.to(device), Hyper_index_ps.to(device), num_atoms_m1s.to(device), num_atoms_m2s.to(device)
            outputs, weights = model(hyper_bg, bg3)
            true.append(Ys.data.cpu().numpy())
            pred.append(outputs.data.cpu().numpy())
            key.append(keys)
    return true, pred, key, weights.data.cpu().numpy()





if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--gpuid', type=str, default='0', help="gpu id for training model")
    argparser.add_argument('--lr', type=float, default=10 ** -3.5, help="Learning rate")
    argparser.add_argument('--epochs', type=int, default=5000, help="Number of epochs in total")
    argparser.add_argument('--batch_size', type=int, default=128, help="Batch size")
    argparser.add_argument('--tolerance', type=float, default=0.0, help="early stopping tolerance")
    argparser.add_argument('--patience', type=int, default=50, help="early stopping patience")
    argparser.add_argument('--l2', type=float, default=0.0005, help="L2 regularization")
    argparser.add_argument('--repetitions', type=int, default=5, help="the number of independent runs")
    argparser.add_argument('--node_feat_size', type=int, default=54 + 40)  # both acsf feature and basic atom feature
    argparser.add_argument('--edge_feat_size_2d', type=int, default=12)
    argparser.add_argument('--edge_feat_size_3d', type=int, default=21)
    argparser.add_argument('--graph_feat_size', type=int, default=256)
    argparser.add_argument('--num_layers', type=int, default=3, help='the number of intra-molecular layers')
    argparser.add_argument('--outdim_g3', type=int, default=200, help='the output dim of inter-molecular layers')
    argparser.add_argument('--d_FC_layer', type=int, default=200, help='the hidden layer size of task networks')
    argparser.add_argument('--n_FC_layer', type=int, default=2, help='the number of hidden layers of task networks')
    argparser.add_argument('--dropout', type=float, default=0.4, help='dropout ratio')
    argparser.add_argument('--n_tasks', type=int, default=1)
    argparser.add_argument('--num_workers', type=int, default=2,
                           help='number of workers for loading data in Dataloader')
    argparser.add_argument('--num_process', type=int, default=4,
                           help='number of process for generating graphs')
    argparser.add_argument('--dic_path_suffix', type=str, default='0')

    # paras acsf setting
    argparser.add_argument('--EtaR', type=float, default=4.00, help='EtaR')
    argparser.add_argument('--ShfR', type=float, default=3.17, help='ShfR')
    argparser.add_argument('--Zeta', type=float, default=8.00, help='Zeta')
    argparser.add_argument('--ShtZ', type=float, default=3.14, help='ShtZ')
    args = argparser.parse_args()
    print(args)
    lr, epochs, batch_size, num_workers = args.lr, args.epochs, args.batch_size, args.num_workers
    tolerance, patience, l2, repetitions = args.tolerance, args.patience, args.l2, args.repetitions

    # paras for model
    node_feat_size, edge_feat_size_2d, edge_feat_size_3d = args.node_feat_size, args.edge_feat_size_2d, args.edge_feat_size_3d
    graph_feat_size, num_layers = args.graph_feat_size, args.num_layers
    outdim_g3, d_FC_layer, n_FC_layer, dropout, n_tasks= args.outdim_g3, args.d_FC_layer, args.n_FC_layer, args.dropout, args.n_tasks
    dic_path_suffix = args.dic_path_suffix
    num_process = args.num_process

    # paras for acsf setting
    EtaR, ShfR, Zeta, ShtZ = args.EtaR, args.ShfR, args.Zeta, args.ShtZ
    test_scrpits = False



    home_path = '../input_data'
    # data_split_file = '/apdcephfs/private_dejunjiang/105/dejunjiang/wspy/hxp/PDB2016All_Splits_new_1.pkl'
    pdb_info_file = '../input_data/PDB2016ALL.csv'
    train_dir = '../input_data/complexes_train'
    valid_dir = '../input_data/complexes_val'
    test_dir = '../input_data/complexes_test'
    path_marker = '/'


    all_data = pd.read_csv(pdb_info_file)
    train_keys_new = os.listdir(train_dir)
    train_labels_new = []
    train_dirs_new = []
    for key in train_keys_new:
        train_labels_new.append(all_data[all_data['PDB'] == key]['label'].values[0])
        train_dirs_new.append(train_dir + path_marker + key)  # ./input_data/complexes_train/key
    valid_keys_new = os.listdir(valid_dir)
    valid_labels_new = []
    valid_dirs_new = []
    for key in valid_keys_new:
        valid_labels_new.append(all_data[all_data['PDB'] == key]['label'].values[0])
        valid_dirs_new.append(valid_dir + path_marker + key)
    test_keys_new = os.listdir(test_dir)
    test_labels_new = []
    test_dirs_new = []
    for key in test_keys_new:
        test_labels_new.append(all_data[all_data['PDB'] == key]['label'].values[0])
        test_dirs_new.append(test_dir + path_marker + key)


    # generating the graph objective using multi process
    train_dataset = GraphDatasetHFGNN(keys=train_keys_new, labels=train_labels_new, data_dirs=train_dirs_new,
                                    graph_ls_file=home_path + path_marker + 'train_data_best_pose.pkl',
                                    graph_dic_path=home_path + path_marker + 'tmpfiles', num_process=num_process,
                                    dis_threshold=8.00, path_marker='/')#g,g3,key,labels
    valid_dataset = GraphDatasetHFGNN(keys=valid_keys_new, labels=valid_labels_new, data_dirs=valid_dirs_new,
                                    graph_ls_file=home_path + path_marker + 'valid_data_best_pose.pkl',
                                    graph_dic_path=home_path + path_marker + 'tmpfiles', num_process=num_process,
                                    dis_threshold=8.00, path_marker='/')
    test_dataset = GraphDatasetHFGNN(keys=test_keys_new, labels=test_labels_new, data_dirs=test_dirs_new,
                                   graph_ls_file=home_path + path_marker + 'test_data_best_pose.pkl',
                                   graph_dic_path=home_path + path_marker + 'tmpfiles', num_process=num_process,
                                   dis_threshold=8.00, path_marker='/')


    stat_res = []

    # print('the number of 4csj data:', len(_4csj_dataset))
    # print('the number of 5g5w data:', len(_5g5w_dataset))
    for repetition_th in range(repetitions):
        dt = datetime.datetime.now()
        filename = home_path + path_marker + 'model_save/{}_{:02d}_{:02d}_{:02d}_{:d}.pth'.format(
            dt.date(), dt.hour, dt.minute, dt.second, dt.microsecond)#模型名称
        print('Independent run %s' % repetition_th)
        print('model file %s' % filename)
        set_random_seed(repetition_th)#固定随机数种子使得每次复现结果相同
        train_dataloader = DataLoaderX(train_dataset, batch_size, shuffle=True, num_workers=num_workers,
                                       collate_fn=collate_fn_hfgnn)

        valid_dataloader = DataLoaderX(valid_dataset, batch_size, shuffle=False, num_workers=num_workers,
                                       collate_fn=collate_fn_hfgnn)

        # model


        DTIModel = HFGNN(node_feat_size=node_feat_size, edge_feat_size=edge_feat_size_3d,
                       num_layers=num_layers,
                       graph_feat_size=graph_feat_size, outdim_g3=outdim_g3,
                       d_FC_layer=d_FC_layer, n_FC_layer=n_FC_layer, dropout=dropout,
                       n_tasks=n_tasks)
        print('number of parameters : ', sum(p.numel() for p in DTIModel.parameters() if p.requires_grad))
        if repetition_th == 0:
            print(DTIModel)
        device = torch.device("cuda:%s" % args.gpuid if torch.cuda.is_available() else "cpu")#设置设备
        DTIModel.to(device)#把模型放到GPU里
        optimizer = torch.optim.Adam(DTIModel.parameters(), lr=lr, weight_decay=l2)#设置优化器,weight_decay 权重衰减（L2惩罚）

        stopper = EarlyStopping(mode='lower', patience=patience, tolerance=tolerance, filename=filename)
        loss_fn = nn.MSELoss()

        for epoch in range(epochs):
            st = time.time()
            # train
            run_a_train_epoch(DTIModel, loss_fn, train_dataloader, optimizer, device)#模型。损失函数，数据，优化器，设备

            # validation
            train_true, train_pred, _, _ = run_a_eval_epoch(DTIModel, train_dataloader, device)
            valid_true, valid_pred, _, _ = run_a_eval_epoch(DTIModel, valid_dataloader, device)

            train_true = np.concatenate(np.array(train_true), 0)
            train_pred = np.concatenate(np.array(train_pred), 0)

            valid_true = np.concatenate(np.array(valid_true), 0)
            valid_pred = np.concatenate(np.array(valid_pred), 0)

            train_rmse = np.sqrt(mean_squared_error(train_true, train_pred))
            valid_rmse = np.sqrt(mean_squared_error(valid_true, valid_pred))
            early_stop = stopper.step(valid_rmse, DTIModel)
            end = time.time()
            if early_stop:
                break
            print(
                "epoch:%s \t train_rmse:%.4f \t valid_rmse:%.4f \t time:%.3f s" % (
                    epoch, train_rmse, valid_rmse, end - st))

        # load the best model
        stopper.load_checkpoint(DTIModel)
        train_dataloader = DataLoaderX(train_dataset, batch_size, shuffle=False, num_workers=num_workers,
                                       collate_fn=collate_fn_hfgnn)
        valid_dataloader = DataLoaderX(valid_dataset, batch_size, shuffle=False, num_workers=num_workers,
                                       collate_fn=collate_fn_hfgnn)
        test_dataloader = DataLoaderX(test_dataset, batch_size, shuffle=False, num_workers=num_workers,
                                      collate_fn=collate_fn_hfgnn)


        train_true, train_pred, tr_keys, _ = run_a_eval_epoch(DTIModel, train_dataloader, device)
        valid_true, valid_pred, val_keys, _ = run_a_eval_epoch(DTIModel, valid_dataloader, device)
        test_true, test_pred, te_keys, _ = run_a_eval_epoch(DTIModel, test_dataloader, device)

        # metrics
        train_true = np.concatenate(np.array(train_true), 0).flatten()
        train_pred = np.concatenate(np.array(train_pred), 0).flatten()
        tr_keys = np.concatenate(np.array(tr_keys), 0).flatten()

        valid_true = np.concatenate(np.array(valid_true), 0).flatten()
        valid_pred = np.concatenate(np.array(valid_pred), 0).flatten()
        val_keys = np.concatenate(np.array(val_keys), 0).flatten()

        test_true = np.concatenate(np.array(test_true), 0).flatten()
        test_pred = np.concatenate(np.array(test_pred), 0).flatten()
        te_keys = np.concatenate(np.array(te_keys), 0).flatten()



        pd_tr = pd.DataFrame({'keys': tr_keys, 'train_true': train_true, 'train_pred': train_pred})
        pd_va = pd.DataFrame({'keys': val_keys, 'valid_true': valid_true, 'valid_pred': valid_pred})
        pd_te = pd.DataFrame({'keys': te_keys, 'test_true': test_true, 'test_pred': test_pred})


        pd_tr.to_csv(home_path + path_marker + 'stats/{}_{:02d}_{:02d}_{:02d}_{:d}_tr.csv'.
                     format(dt.date(), dt.hour, dt.minute, dt.second, dt.microsecond), index=False)
        pd_va.to_csv(home_path + path_marker + 'stats/{}_{:02d}_{:02d}_{:02d}_{:d}_va.csv'.format(
            dt.date(), dt.hour, dt.minute, dt.second, dt.microsecond), index=False)
        pd_te.to_csv(home_path + path_marker + 'stats/{}_{:02d}_{:02d}_{:02d}_{:d}_te.csv'.format(
            dt.date(), dt.hour, dt.minute, dt.second, dt.microsecond), index=False)


        train_rmse, train_r2, train_mae, train_rp = np.sqrt(mean_squared_error(train_true, train_pred)), \
                                                    r2_score(train_true, train_pred), \
                                                    mean_absolute_error(train_true, train_pred), \
                                                    pearsonr(train_true, train_pred)
        valid_rmse, valid_r2, valid_mae, valid_rp = np.sqrt(mean_squared_error(valid_true, valid_pred)), \
                                                    r2_score(valid_true, valid_pred), \
                                                    mean_absolute_error(valid_true, valid_pred), \
                                                    pearsonr(valid_true, valid_pred)
        test_rmse, test_r2, test_mae, test_rp = np.sqrt(mean_squared_error(test_true, test_pred)), \
                                                r2_score(test_true, test_pred), \
                                                mean_absolute_error(test_true, test_pred), \
                                                pearsonr(test_true, test_pred)

        print('***best %s model***' % repetition_th)
        print("train_rmse:%.4f \t train_r2:%.4f \t train_mae:%.4f \t train_rp:%.4f" % (
            train_rmse, train_r2, train_mae, train_rp[0]))
        print("valid_rmse:%.4f \t valid_r2:%.4f \t valid_mae:%.4f \t valid_rp:%.4f" % (
            valid_rmse, valid_r2, valid_mae, valid_rp[0]))
        print("test_rmse:%.4f \t test_r2:%.4f \t test_mae:%.4f \t test_rp:%.4f" % (
            test_rmse, test_r2, test_mae, test_rp[0]))
