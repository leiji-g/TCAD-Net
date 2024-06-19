# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 20:30:01 2022

@author: ChenMingfeng
"""
import argparse
import csv

import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from FEM import ADTransformer
from util.build_criterion import *
from util.data_loader import *
from util.utils import *

from sklearn.decomposition import PCA
from vaild_test import aucPerformance


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
def fewNegAuto(args):

    #-----------------------------------
    block_size = args.dimension
    random_seed = args.ramdn_seed
    if random_seed != -1:
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
    names = args.data_set

    x, labels = dataLoading(args.input_path+names+'.csv')

    dir_name = args.output_file

    Roc_max = 0.0
    PR_max = 0.0

    x_train, x_test, y_train, y_test = train_test_split(x, labels, test_size=0.2, random_state=random_seed, stratify=labels)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    outlier_indices = np.where(labels == 1)[0]
    # 获取x中label=1的元素
    outliers = x[outlier_indices]
    n_outliers_org = outliers.shape[0]

    outlier_indices = np.where(y_train == 1)[0]
    inlier_indices = np.where(y_train == 0)[0]
    n_outliers = len(outlier_indices)
    print("Original training size: %d, No. outliers(异常): %d, No. inliers（正常）: %d" % (x_train.shape[0], n_outliers, len(inlier_indices)))

    n_noise = len(np.where(y_train == 0)[0]) * 0.02 / (1. - 0.02)
    n_noise = int(n_noise)
    rng = np.random.RandomState(random_seed)

    if n_outliers > args.known_outliers:
        mn = n_outliers - args.known_outliers
        remove_idx = rng.choice(outlier_indices, mn, replace=False)
        x_train = np.delete(x_train, remove_idx, axis=0)
        y_train = np.delete(y_train, remove_idx, axis=0)
    noises = inject_noise(outliers, n_noise, random_seed)
    x_train = np.append(x_train, noises, axis = 0)
    y_train = np.append(y_train, np.zeros((noises.shape[0], 1)))

    outlier_indices = np.where(y_train == 1)[0]
    inlier_indices = np.where(y_train == 0)[0]
    print("modify label size: %d, No. outliers(异常): %d, No. inliers（正常）: %d, No. noise（污染）: %d" % (y_train.shape[0], outlier_indices.shape[0], inlier_indices.shape[0], noises.shape[0]))

    print('test_dataset:', x_test.shape[0], "test_anomal:", np.where(y_test == 1)[0].shape[0])

    # pca
    if (args.data_set == 'census-income-full-mixed-binarized'):
        pca = PCA(n_components=block_size, random_state=random_seed)
        pca.fit(x_train)
        # 保存 PCA 模型
        joblib.dump(pca, 'Model/pca_for_census.joblib')
        x_train = pca.transform(x_train)
    model = ADTransformer(block_size, num_layers=args.num_layers, heads= args.heads , device=device).to(device)

    criterion = build_criterion("FocalLoss", args.gamma, args.alpha).to(device)

    learning_rate = args.learning_rate
    num_epochs = args.epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5, )

    train_data_few = DataSetADFew(x_train, y_train)
    vaild_data_few = DataSetADFew(x_test, y_test)

    train_loader = torch.utils.data.DataLoader(train_data_few, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(vaild_data_few, batch_size=4096, shuffle=False)
    count = 0
    early_stop_numb = 70
    for epoch in range(num_epochs):
        for index, (x,ylabel) in enumerate(train_loader):
            out = model(x)
            loss = criterion(out, ylabel)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sched.step(epoch)
            if(index%1000==0):
                print("loss:", loss)

        model.eval()
        val_loss_sum = 0.0
        val_metric_sum = 0.0
        val_step = 0
        AD_accuracy_score = 0.0
        rauc = np.zeros(len(test_loader))
        ap = np.zeros(len(test_loader))

        for val_step, (features, labels) in enumerate(test_loader):
            with torch.no_grad():
                pred = model(features)
                score = pred.cpu().detach().numpy()
                val_loss_sum = 0
                test_lables = labels.cpu().numpy()

                AD_accuracy_score += accuracy_score(model.accuracy_predict(features), test_lables)
                rauc[val_step], ap[val_step] = aucPerformance(pred.cpu().numpy(), test_lables)

        print("AD_accuracy_score_mean", AD_accuracy_score/(val_step+1))
        rauc_mean = np.mean(rauc)
        ap_mean = np.mean(ap)

        print("average AUC-ROC: %.4f, average AUC-PR: %.4f" % (rauc_mean, ap_mean))
        print("val_loss_sum:", val_loss_sum)
        # 查看梯度
        lr = 0
        for p in optimizer.param_groups:
            lr = p['lr']
        print('lr:', lr)
        count += 1
        if ap_mean + rauc_mean > PR_max + Roc_max:
            count = 0
            Roc_max = rauc_mean
            PR_max = ap_mean
            torch.save(model.state_dict(),'./Model/Transformer_Base_' + names + ".pt")
            writeData = [names + " , " + str(epoch) + ",max AUC-ROC: %.4f, max AUC-PR: %.4f" % (Roc_max, PR_max)]
            with open(dir_name, 'a+', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(writeData)
        if ap_mean + rauc_mean < PR_max + Roc_max:
            count += 1;
        if count > early_stop_numb:
            print('Early stopping......................')
            break

        print('\n' + '==========' * 8 + '%d' % epoch)

    print("max AUC-ROC: %.4f, max AUC-PR: %.4f" % (Roc_max, PR_max))
    writeData = ["max AUC-ROC: %.4f, max AUC-PR: %.4f" % (Roc_max, PR_max)]
    with open('./SaveScore/score_aeFeature', 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(writeData)
    with open(dir_name, 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(str(args.learning_rate)+"--------------------------------------")
    print('Finishing Training...')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--input_path", type=str, default='./dataset_2022/', help="the path of the data sets")
    parser.add_argument("--output_file", type=str, default='./compare_result/best_score.csv',
                        help="the path of the data scores")
    parser.add_argument("--data_set", type=str, default='annthyroid_21feat_normalised', help="a list of data set names, numb 49097")
    parser.add_argument("--batch_size", type=int, default=512, help="batch size used in SGD")
    parser.add_argument("--ramdn_seed", type=int, default=1024, help="the random seed number")
    parser.add_argument("--gamma", choices=['1','2','3','4','5','6'], default=2, help="the gamma is devnet parament ,used to set the multiplier ")
    parser.add_argument("--dimension", type=int, default=21,help="the dimension of dataset ,Corresponding block size")
    parser.add_argument("--heads", type=int, default=3, help="the heads of self attention")
    parser.add_argument("--epochs", type=int, default=200, help="the number of epochs")
    parser.add_argument("--learning_rate", type=float, default=9e-4, help="the learning rate in the training data")
    parser.add_argument("--known_outliers", type=int, default=30, help="the number of labeled outliers available at hand")
    parser.add_argument("--num_layers", type=float, default=8, help="TFD-Net with different encoder layers")
    parser.add_argument("--alpha", choices=['0.15', '0.1'], default=0.15,
                        help="the alpha is loss parament")

    args = parser.parse_args()
    # names,sizes,dimension,attention_heads,gamma,learn_rate
    dataset_information = [
                           # ['bank-additional-full-deonehot', 41189, 20, 4, 2, 9e-4],
                           # ['celeba_baldvsnonbald_normalised',202600, 39, 3, 3, 2e-5],
                            ['census-income-full-mixed-binarized', 299286, 100, 25, 2, 2e-5],
                           # ['creditcardfraud_normalised',284808, 29, 1, 3, 3e-4],
                           # ['shuttle_normalization', 49098, 9, 1, 3, 48e-5],
                           #  ['annthyroid_21feat_normalised', 7201, 21, 3, 3, 15e-4],
                           # ['UNSW_NB15_traintest_backdoor-deonehot', 95330, 42, 6, 3, 34e-5],
                           #  ['UNSW_NB15_traintest_backdoor', 95330, 84, 12, 3, 48e-5],
                           #  ['mammography_normalization', 11183, 6, 1, 4, 43e-5]
                            ]
    num_layers = [8]
    for nl in num_layers:
        with open(args.output_file, 'a', encoding='utf-8', newline='') as f:
            write = csv.writer(f)
            write.writerow(["num_layers=" + str(nl) + " gamma=default "])
        for numb, information in enumerate(dataset_information):
            parser.set_defaults(data_set=information[0])
            parser.set_defaults(dimension=information[2])
            parser.set_defaults(heads=information[3])
            parser.set_defaults(gamma=information[4])
            parser.set_defaults(learning_rate=information[5])
            parser.set_defaults(num_layers=nl)
            args = parser.parse_args()
            fewNegAuto(args)
            print("********************************"+str(args.learning_rate)+"********************************")