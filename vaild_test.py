# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 15:38:06 2022

@author: ChenMingfeng
"""

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score, classification_report
from sklearn.model_selection import train_test_split

from TCAD_Net import ADTransformer
from util.build_criterion import *
from util.data_loader import DataSetAD, DataSetADFew
from util.utils import aucPerformance, dataLoading_vaild
from tqdm import tqdm, trange

import numpy as np
def aucPerformance(mse, labels):
    # 计算AUC-ROC and AUC-PR
    roc_auc = roc_auc_score(labels, mse)
    ap = average_precision_score(labels, mse)
    # print("AUC-ROC: %.4f, AUC-PR: %.4f" % (roc_auc, ap))
    return roc_auc, ap

def extract_top_n_predictions(pre_labels, true_label):
    pred_dict = {}

    for i in range(1,len(pre_labels)):
        pred_dict['top-{}-预测ID'.format(i)] = pre_labels[i-1]
        pred_dict['top-{}-实际ID'.format(i)] = true_label[i - 1]

    return pred_dict



if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    print(device)
    dataset_information = [
                           # ['bank-additional-full-deonehot', 41189, 20, 4, 0.3],
                           # ['celeba_baldvsnonbald_normalised',202600, 39, 3, 0.6],
                           #  ['census-income-full-mixed-binarized', 299286, 100, 25, 0.14],
                           ['creditcardfraud_normalised',284808, 29, 1, 0.5],
                           # ['shuttle_normalization',49098, 9, 1, 0.6],
                           #  ['annthyroid_21feat_normalised', 7201, 21, 3, 0.5],
                           # ['UNSW_NB15_traintest_backdoor-deonehot', 95330, 42, 6, 0.5],
                           # ['mammography_normalization', 11183, 6, 1, 0.4]
                            ] #no
    for numb, information in enumerate(dataset_information):
        block_size = information[2]
        sample_size = information[3]

        model = ADTransformer(block_size, num_layers=8, heads= sample_size , device=device).cuda()
        model.eval()

        state_dict = torch.load('./Model/Transformer_Base_'+ information[0] +'.pt', map_location=device)
        model.load_state_dict(state_dict)
        sum = 0
        seed = 1024
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        x, labels, feature_name = dataLoading_vaild('./dataset/'+ information[0]+'.csv')
        x_train, x_test, y_train, y_test = train_test_split(x, labels, test_size=0.2, random_state=seed, stratify=labels)
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        print("验证集中 0 的数量：", np.sum(y_test==0), "; 验证集中 1 的数量：", np.sum(y_test==1))
        # pca
        if (information[0] == 'census-income-full-mixed-binarized'):
            pca = joblib.load('Model/pca_for_census.joblib')
            x_test = pca.transform(x_test)
            feature_name = [f"att{i}" for i in range(0, 101)]

        vaild_data_few = DataSetADFew(x_test, y_test)
        test_loader = tqdm(torch.utils.data.DataLoader(vaild_data_few, batch_size=4096, shuffle=False))

        # val_metric_sum = 0.0
        val_step = 0
        AD_accuracy_score = 0.0
        df_pred = pd.DataFrame()
        rauc = np.zeros(len(test_loader))
        ap = np.zeros(len(test_loader))

        for val_step, (features,labels) in enumerate(test_loader):
            # 关闭梯度计算
            with torch.no_grad():
                preditData = model(features).cpu().numpy().squeeze()
                test_lables = labels.cpu().numpy().squeeze()
                acdata = model.accuracy_predict(features, information[4])
                # 获取预测标签
                pre_label = acdata.numpy().squeeze()
                AD_accuracy_score += accuracy_score(pre_label, test_lables)
                # 打印基本指标
                print("AD_accuracy_score_mean", AD_accuracy_score / (val_step + 1))
                # 获取pr和roc值
                rauc[val_step], ap[val_step] = aucPerformance(preditData, test_lables)

                # pred = model(features)
                # score = pred.cpu().detach().numpy()
                # a = pred - labels
                # prd = torch.norm(a) / torch.norm(labels)
                # sum += prd.item()
        print(information[0]+"average AUC-ROC: %.4f, average AUC-PR: %.4f" % (np.mean(rauc), np.mean(ap)))
        print(df_pred.shape)
        # avg = sum / 112
        # print(avg)