import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn import manifold
from sklearn.metrics import classification_report
from model import FCN, DilatedConvolutionVis, Classifier, NonLinearClassifierVis, NonLinearClassifier
from tsm_utils import load_data, transfer_labels, set_seed
from data import normalize_per_series
import torch
import tqdm
import torch.nn
from sklearn.manifold import MDS, TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")


def t_sne(xs, ys, datasetname='Wine', tsne=False, seed=42):
    if tsne:
        tsne = TSNE(n_components=2, random_state=seed)
    else:
        tsne = MDS(n_components=2, random_state=seed)

    xs_out = tsne.fit_transform(xs, ys)
    plt.subplot(3, 1, 1)
    plt.title('Raw time-series')
    plt.scatter(xs_out[:,0], xs_out[:,1], c=ys)

    xs = torch.from_numpy(xs).to(DEVICE)
    xs = torch.unsqueeze(xs, 1)

    model = FCN(2).to(DEVICE)
    classifier = Classifier(128, 2).to(DEVICE)
    model.load_state_dict(torch.load('./visuals/'+datasetname+'/direct_fcn_linear_encoder_weights.pt', map_location='cuda:0'))
    classifier.load_state_dict(torch.load('./visuals/'+datasetname+'/direct_fcn_linear_classifier_weights.pt', map_location='cuda:0'))#,map_location=torch.device('cpu')))
    model.eval()
    classifier.eval()
    features, _ = model(xs, vis=True)
    feature_map = tsne.fit_transform(features.cpu().detach().numpy())
    plt.subplot(3, 1, 2)
    plt.title('Embedding map (linear classifier)')
    plt.scatter(feature_map[:,0], feature_map[:,1], c=ys)

    model = FCN(2).to(DEVICE)
    classifier = NonLinearClassifierVis(128, 128, 2).to(DEVICE)
    model.load_state_dict(
        torch.load('./visuals/' + datasetname + '/direct_fcn_nonlinear_encoder_weights.pt', map_location='cuda:0'))
    classifier.load_state_dict(torch.load('./visuals/' + datasetname + '/direct_fcn_nonlinear_classifier_weights.pt',
                                          map_location='cuda:0'))  # ,map_location=torch.device('cpu')))
    model.eval()
    classifier.eval()
    features, _ = model(xs, vis=True)
    val_pred, non_features = classifier(features, vis=True)
    feature_map = tsne.fit_transform(non_features.cpu().detach().numpy())
    plt.subplot(3, 1, 3)
    plt.title('Embedding map (nonlinear classifier)')
    plt.scatter(feature_map[:, 0], feature_map[:, 1], c=ys)

    plt.tight_layout()
    plt.savefig('./visuals/tsne_seed_'+str(seed)+"_"+datasetname+'.png')
    plt.savefig('./visuals/tsne_seed_'+str(seed)+"_"+datasetname+'.pdf')

    plt.clf()


def heatmap(xs, ys, dataset_name='MixedShapesSmallTrain', num_class=5, cls=4):
    model = FCN(num_class)
    model.to(DEVICE)

    ts1 = plt.subplot2grid((2, 15), loc=(0, 0), colspan=4, rowspan=1)
    hm1 = plt.subplot2grid((2, 15), loc=(1, 0), colspan=4)
    ts2 = plt.subplot2grid((2, 15), loc=(0, 5), colspan=4, rowspan=1)
    hm2 = plt.subplot2grid((2, 15), loc=(1, 5), colspan=4)
    ts3 = plt.subplot2grid((2, 15), loc=(0, 10), colspan=4, rowspan=1)
    hm3 = plt.subplot2grid((2, 15), loc=(1, 10), colspan=4)

    x0s = xs[np.where(ys == cls)]
    x0_mean = np.mean(x0s, axis=1)
    x0_mean_mean = np.mean(x0_mean, axis=0)
    class0 = x0s[np.where(np.abs(x0_mean - x0_mean_mean) == min(np.abs(x0_mean - x0_mean_mean)))[0][0]]
    x1 = class0
    x_copy = x1
    # direct cls
    model.load_state_dict(torch.load('./visuals/' + dataset_name +'/direct_fcn_nonlinear_encoder_weights.pt', map_location='cuda:0'))
    model.eval()
    ts1.set_title('Direct classification')
    ts1.plot(range(x_copy.shape[0]), x_copy)
    x1 = torch.from_numpy(x1).to(DEVICE)
    x1 = torch.unsqueeze(x1, 0)
    x1 = torch.unsqueeze(x1, 0)
    gaps, feature = model(x1, vis=True)
    gaps = torch.squeeze(gaps)
    feature = torch.squeeze(feature)
    feature = feature[torch.topk((gaps-gaps.mean())**2, k=16).indices,:].cpu()
    hm1.pcolormesh(feature[0:16],shading='nearest')

    # supervised transfer
    # model.load_state_dict(torch.load('./visuals/' + dataset_name + '/fcn_nonlinear_encoder_finetune_weights_UWaveGestureLibraryZ.pt',map_location='cuda:0'))
    model.load_state_dict(
        torch.load('./visuals/' + dataset_name + '/fcn_nonlinear_encoder_finetune_weights_UWaveGestureLibraryZ.pt',
                   map_location='cuda:0'))
    model.eval()
    ts2.set_title('Positive transfer')
    ts2.plot(range(x_copy.shape[0]), x_copy)
    gaps, feature = model(x1, vis=True)
    gaps = torch.squeeze(gaps)
    feature = torch.squeeze(feature)
    feature = feature[torch.topk((gaps-gaps.mean())**2, k=16).indices,:].cpu()
    hm2.pcolormesh(feature[0:16],shading='nearest')

    # model.load_state_dict(torch.load('./visuals/' + dataset_name + '/fcn_nonlinear_encoder_finetune_weights_ElectricDevices.pt',map_location='cuda:0'))
    model.load_state_dict(
        torch.load('./visuals/' + dataset_name + '/fcn_nonlinear_encoder_finetune_weights_Crop.pt',
                   map_location='cuda:0'))
    model.eval()
    ts3.set_title('Negative transfer')
    ts3.plot(range(x_copy.shape[0]), x_copy)
    gaps, feature = model(x1, vis=True)
    gaps = torch.squeeze(gaps)
    feature = torch.squeeze(feature)
    feature = feature[torch.topk((gaps-gaps.mean())**2, k=16).indices,:].cpu()
    hm3.pcolormesh(feature[0:16], shading='nearest')

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.15, hspace=0.30)
    plt.tight_layout()
    plt.savefig('./visuals/' + dataset_name +'_postive_negative.png')
    plt.savefig('./visuals/' + dataset_name +'_postive_negative.pdf')


def multi_cam(xs, ys):
    # sampling
    x0s = xs[np.where(ys==0)]
    x1s = xs[np.where(ys==1)]

    x0_mean = np.mean(x0s, axis=1)
    x0_mean_mean = np.mean(x0_mean, axis=0)
    class0 = x0s[np.where(np.abs(x0_mean-x0_mean_mean) == min(np.abs(x0_mean-x0_mean_mean)))]
    #class0 = np.expand_dims(class0, 0)
    print(class0.shape)

    x1_mean = np.mean(x1s, axis=1)
    x1_mean_mean = np.mean(x1_mean, axis=0)
    class1 = x1s[np.where(np.abs(x1_mean-x1_mean_mean) == min(np.abs(x1_mean-x1_mean_mean)))][0]
    class1 = np.expand_dims(class1, 0)
    print(class1.shape)

    # print(class0.mean())
    # print(class1.mean())
    def cam(x, label):
        x = torch.from_numpy(x).to(DEVICE)
        #x = torch.unsqueeze(x, 0)
        x = torch.unsqueeze(x, 0)
        features, vis_out = model(x, vis=True)
        pred = classifier(features)

        w_k_c = classifier.state_dict()['dense.weight']
        cas = np.zeros(dtype=np.float16, shape=(vis_out.shape[2]))
        for k, w in enumerate(w_k_c[label,:]):
            cas += (w * vis_out[0, k, :]).cpu().numpy()
        
        minimum = np.min(cas)
        # print(cas)
        cas = cas - minimum
        cas = cas / max(cas)
        cas = cas * 100
        
        x = x.cpu().numpy()
        plt_x = np.linspace(0, x.shape[2]-1, 2000, endpoint=True)

        f = interp1d(range(x.shape[2]), x.squeeze())
        y = f(plt_x)

        f = interp1d(range(x.shape[2]), cas)
        cas = f(plt_x).astype(int)
        
        plt.scatter(x=plt_x, y=y, c=cas, cmap='jet', marker='.', s=2, vmin=0, vmax=100, linewidths=1.0)
        
        plt.yticks([-1.0, 0.0, 1.0])

    plt.figure()
    model = FCN(2).to(DEVICE)
    classifier = Classifier(128, 2).to(DEVICE)
    model.load_state_dict(torch.load('./visuals/GunPoint/direct_fcn_encoder.pt',map_location='cuda:0'))
    classifier.load_state_dict(torch.load('./visuals/GunPoint/direct_fcn_classifier.pt',map_location='cuda:0'))
    model.eval()
    classifier.eval()
    x1 = torch.from_numpy(xs).to(DEVICE)
    x1 = torch.unsqueeze(x1, 1)
    features, _ = model(x1, vis=True)
    val_pred = features
    val_pred = classifier(val_pred)
    ys1 = torch.from_numpy(ys).to(DEVICE)
    val_accu = torch.sum(torch.argmax(val_pred.data, axis=1) == ys1)
    val_accu = val_accu / len(ys)
    print("val accuracy direct = ", val_accu)

    ax1 = plt.subplot(4, 1, 1)
    plt.title('Direct classification via FCN (100%)')
    cam(class0, 0)
    cam(class1, 1)

    model = DilatedConvolutionVis(in_channels=1, embedding_channels=40, out_channels=320, depth=3,
                                  reduced_size=320, kernel_size=3, num_classes=2).to(DEVICE)
    classifier = Classifier(320, 2).to(DEVICE)
    model.load_state_dict(
        torch.load('./visuals/GunPoint/direct_dilated_encoder.pt', map_location='cuda:0'))
    classifier.load_state_dict(
        torch.load('./visuals/GunPoint/direct_dilated_classifier.pt', map_location='cuda:0'))
    model.eval()
    classifier.eval()
    features, _ = model(x1, vis=True)
    val_pred = features
    val_pred = classifier(val_pred)
    ys1 = torch.from_numpy(ys).to(DEVICE)
    val_accu = torch.sum(torch.argmax(val_pred.data, axis=1) == ys1)
    val_accu = val_accu / len(ys)
    print("val accuracy dilated = ", val_accu)

    ax2 = plt.subplot(4, 1, 2)
    plt.title('Direct classification via TCN (50%)')
    cam(class0, 0)
    cam(class1, 1)

    model = FCN(2).to(DEVICE)
    classifier = Classifier(128, 2).to(DEVICE)
    model.load_state_dict(torch.load('./visuals/GunPoint/supervised_encoder_UWaveGestureLibraryX_linear.pt',map_location='cuda:0'))
    classifier.load_state_dict(torch.load('./visuals/GunPoint/supervised_classifier_UWaveGestureLibraryX_linear.pt',map_location='cuda:0'))
    model.eval()
    classifier.eval()
    features, _ = model(x1, vis=True)
    val_pred = features
    val_pred = classifier(val_pred)
    ys1 = torch.from_numpy(ys).to(DEVICE)
    val_accu = torch.sum(torch.argmax(val_pred.data, axis=1) == ys1)
    val_accu = val_accu / len(ys)
    print("val accuracy supervised = ", val_accu)

    ax3 = plt.subplot(4, 1, 3)
    plt.title('Supervised transfer via FCN (100%)')
    cam(class0, 0)
    cam(class1, 1)

    model.load_state_dict(torch.load('./visuals/GunPoint/unsupervised_encoder_UWaveGestureLibraryX_linear.pt',map_location='cuda:0'))
    classifier.load_state_dict(torch.load('./visuals/GunPoint/unsupervised_classifier_UWaveGestureLibraryX_linear.pt',map_location='cuda:0'))
    model.eval()
    classifier.eval()
    features, _ = model(x1, vis=True)
    val_pred = features
    val_pred = classifier(val_pred)
    ys1 = torch.from_numpy(ys).to(DEVICE)
    val_accu = torch.sum(torch.argmax(val_pred.data, axis=1) == ys1)
    val_accu = val_accu / len(ys)
    print("val accuracy unsupervised = ", val_accu)

    ax4 = plt.subplot(4, 1, 4)
    plt.title('Unsupervised transfer via FCN decoder (98.5%)')
    cam(class0, 0)
    cam(class1, 1)

    plt.colorbar(ax=[ax1, ax2, ax3, ax4])  # Add a color scale bar on the right side
    plt.subplots_adjust(left=None, bottom=None, right=0.75, top=None, wspace=0.00, hspace=0.9)
    # plt.tight_layout()
    plt.savefig('./visuals/fcn_dilated_supervised_unsupervised_transfer.png', bbox_inches='tight')
    plt.savefig('./visuals/fcn_dilated_supervised_unsupervised_transfer.pdf', bbox_inches='tight')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default='/SSD/lz/UCRArchive_2018', help='data root') ## /dev_data/zzj/hzy/datasets/UCR
    parser.add_argument('--dataset', type=str, default='GunPoint', help='dataset name')  ## Wine GunPoint FreezerSmallTrain MixedShapesSmallTrain
    parser.add_argument('--backbone', type=str, choices=['dilated', 'fcn'], default='fcn', help='encoder backbone')
    parser.add_argument('--graph', type=str, choices=['cam', 'heatmap', 'tsne'], default='cam')
    parser.add_argument('--random_seed', type=int, default=42, help='shuffle seed')

    args = parser.parse_args()
    set_seed(args)
    
    xs, ys, num_classes = load_data(args.dataroot, args.dataset)
    xs = normalize_per_series(xs)
    ys = transfer_labels(ys)

    if args.graph == 'cam':
        multi_cam(xs, ys)
    elif args.graph == 'heatmap':
        heatmap(xs, ys, dataset_name='Wine', num_class=2,  cls=0)
    elif args.graph == 'tsne':
        # t_sne(xs, ys)
        t_sne(xs, ys, datasetname=args.dataset, tsne=True)