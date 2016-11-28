from k_shot_classifier import KshotClassifier
from k_shot_classifier import ZeroShotClassifier
import matplotlib.pylab as pylab
import numpy as np
import scipy.io as sio

# Load features and labels
pathtohogfeatures = '/export/home/mbautist/Desktop/workspace/cnn_similarities/datasets/Caltech101/sim/simMatrix_INIT.npy'


pathtolabels = '/export/home/mbautist/Desktop/workspace/cnn_similarities/datasets/Caltech101/features/labels.npy'


features = np.load(pathtohogfeatures)

data = sio.loadmat('/export/home/mbautist/Desktop/Caltech101/simMatrix_Caltech101_2_rounds_iter1_fc7_zscores.mat')

features = data['sim_matrix']
labels = np.load(pathtolabels)

# Initialize classifiers
# kclass = KshotClassifier(features, labels)
zclass = ZeroShotClassifier(features, labels)

nns = [1, 5, 10, 15, 20]
acc_zero_shot = []
for n in nns:
    print "Training nearest neighbour classifier with {} neighbours".format(n)
    acc_zero_shot.append(zclass.ten_fold_cv_train(n))

pylab.errorbar(np.asarray(nns), np.asarray(acc_zero_shot)[:, 0], yerr=np.asarray(acc_zero_shot)[:, 1])
pylab.show()

# kshot = [1, 10, 20, 30]
# acc_k_shot = []
# for k in kshot:
#     print "Training {} shot ovr multiclass-svm".format(k)
#     acc_k_shot.append(kclass.ten_fold_cv_train(k))
#
# pylab.plot(np.asarray(kshot), np.asarray(acc_k_shot))
# pylab.show()
