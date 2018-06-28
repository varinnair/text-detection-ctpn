from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn import svm
from sklearn.mixture import GaussianMixture
from sklearn.isotonic import IsotonicRegression
from sklearn.mixture import GaussianMixture
from sklearn.isotonic import IsotonicRegression
from PIL import Image
import os, os.path

imgs = []
imgs1 = []
path = "/home/aditya/aditya/text-detection-ctpn/ctpn/aditya/training/all-bank-cheques"
path1 = "/home/aditya/aditya/text-detection-ctpn/ctpn/aditya/training/random-images"
valid_images = [".jpg",".gif",".png",".tga"]

for f in os.listdir(path):
    ext = os.path.splitext(f)[1]
    if ext.lower() not in valid_images:
        continue
    imgs.append(Image.open(os.path.join(path,f)))

for f in os.listdir(path1):
    ext = os.path.splitext(f)[1]
    if ext.lower() not in valid_images:
        continue
    imgs1.append(Image.open(os.path.join(path,f)))


def extract_resnet(img_path):
    # X : images numpy array
    #resnet_model = ResNet50(input_shape=(image_h, image_w, 3), weights='imagenet', include_top=False)  # Since top layer is the fc layer used for predictions
    resnet_model = ResNet50(weights='imagenet')
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features_array = resnet_model.predict(x)
    return features_array


for i in imgs:
    y_val =  1
    X_train = extract_resnet(i)
    # Apply standard scaler to output from resnet50
    ss = StandardScaler()
    ss.fit(X_train)
    X_train = ss.transform(X_train)
    #X_test = ss.transform(X_test)

    # Take PCA to reduce feature space dimensionality
    pca = PCA(n_components=512, whiten=True)
    pca = pca.fit(X_train)
    print('Explained variance percentage = %0.2f' % sum(pca.explained_variance_ratio_))
    X_train = pca.transform(X_train)
    #X_test = pca.transform(X_test)

    # Train classifier and obtain predictions for OC-SVM
    oc_svm_clf = svm.OneClassSVM(gamma=0.001, kernel='rbf', nu=0.08)  # Obtained using grid search
    if_clf = IsolationForest(contamination=0.08, max_features=1.0, max_samples=1.0, n_estimators=40)  # Obtained using grid search

    oc_svm_clf.fit(X_train)
    if_clf.fit(X_train)

    #oc_svm_preds = oc_svm_clf.predict(X_test)
    #if_preds = if_clf.predict(X_test)

    gmm_clf = GaussianMixture(covariance_type='spherical', n_components=18, max_iter=int(1e7))  # Obtained via grid search
    gmm_clf.fit(X_train)
    log_probs_val = gmm_clf.score_samples(X_train)
    isotonic_regressor = IsotonicRegression(out_of_bounds='clip')
    isotonic_regressor.fit(log_probs_val, y_val)  # y_val is for labels 0 - not food 1 - food (validation set)

for i in imgs1:
    X_test = extract_resnet(i)

    pca = PCA(n_components=512, whiten=True)
    ss = StandardScaler()
    oc_svm_clf = svm.OneClassSVM(gamma=0.001, kernel='rbf', nu=0.08)
    gmm_clf = GaussianMixture(covariance_type='spherical', n_components=18, max_iter=int(1e7))

    X_test = ss.transform(X_test)
    X_test = pca.transform(X_test)
    oc_svm_preds = oc_svm_clf.predict(X_test)
    if_clf = IsolationForest(contamination=0.08, max_features=1.0, max_samples=1.0, n_estimators=40)
    isotonic_regressor = IsotonicRegression(out_of_bounds='clip')
    if_clf.fit(X_test)
    if_preds = if_clf.predict(X_test)
    log_probs_test = gmm_clf.score_samples(X_test)
    test_probabilities = isotonic_regressor.predict(log_probs_test)
    isotonic_regressor = IsotonicRegression(out_of_bounds='clip')
    test_predictions = [1 if prob >= 0.5 else 0 for prob in test_probabilities]
    print(test_predictions[i])

    # Obtaining results on the test set

