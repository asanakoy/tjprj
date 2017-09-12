from k_shot_classifier import ZeroShotClassifier
import numpy as np
import tfeval.image_getter
import tfeval.features
import scipy.stats as stats


def nn_acc(net, category, layer_names, mat_path, mean_path, batch_size):

    pathtolabels = '/export/home/mbautist/Desktop/workspace/cnn_similarities/datasets/Caltech101/features/labels.npy'
    labels = np.load(pathtolabels)
    feats_params = {
        'category': category,
        'layer_names': layer_names,
        'mean': np.load(mean_path),
        'batch_size': batch_size,
        'im_shape': (227, 227),
        'image_getter': tfeval.image_getter.ImageGetterFromMat(mat_path)
    }

    features = tfeval.features.extract_features(False, net=net, **feats_params)
    features = stats.zscore(features['fc7'])
    zclass = ZeroShotClassifier(features, labels)
    nnn = 10
    acc_zero_shot = zclass.ten_fold_cv_train(nnn)
    return np.asarray(acc_zero_shot)[0]

