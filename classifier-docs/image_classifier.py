# using keras (ResNet50) and SVM

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn import svm
import numpy as np
import PIL
import glob
from sklearn.externals import joblib

def extract_resnet(X,i):  
    print(str(i))
    # X : images numpy array
    #resnet_model = ResNet50(input_shape=(image_h, image_w, 3), weights='imagenet', include_top=False)  # Since top layer is the fc layer used for predictions
    resnet_model = ResNet50(weights='imagenet', include_top=False) # Since top layer is the fc layer used for predictions
    features_array = resnet_model.predict(X)
    return features_array

if __name__ == "__main__":

    # training data path
    training_bank_cheques_path = "C:/Users/varin/Documents/GitHub/text-detection-ctpn/classifier-docs/training/all-bank-cheques/*.jpg"

    # testing data paths
    testing_bank_cheques_path = "C:/Users/varin/Documents/GitHub/text-detection-ctpn/classifier-docs/testing/all-bank-cheques/*.jpg"
    testing_random_images_path = "C:/Users/varin/Documents/GitHub/text-detection-ctpn/classifier-docs/testing/random-images/*.jpg"

    # getting all training data imgs as numpy arrays and resizing (720, 720)
    training_bank_cheque_imgs = [image.img_to_array(image.load_img(file, target_size=(720,720))) for file in glob.glob(training_bank_cheques_path)]

    # getting all testing data imgs as numpy arrays and resizing (720, 720)
    testing_bank_cheque_imgs = [image.img_to_array(image.load_img(file, target_size=(720,720))) for file in glob.glob(testing_bank_cheques_path)]
    testing_random_imgs = [image.img_to_array(image.load_img(file, target_size=(720,720))) for file in glob.glob(testing_random_images_path)]
    
    # combining training data
    training_arr = training_bank_cheque_imgs

    # combining testing data
    testing_arr = (testing_bank_cheque_imgs + testing_random_imgs)

    # computing resnet features for training data
    print("computing training resnets")
    i = 1
    X_train = []
    for t in training_arr:
        t = np.array([t])
        features_array = np.array(extract_resnet(t, i))
        X_train.append(features_array)
        i += 1
    
    X_train = np.array(X_train)

    # computing resnet features for testing data
    print()
    print("computing testing resnets")
    i = 1 
    X_test = []
    for t in testing_arr:
        t = np.array([t])
        features_array = np.array(extract_resnet(t, i))
        X_test.append(features_array)
        i += 1

    X_test = np.array(X_test)

    # Apply standard scaler to output from resnet50
    print()
    print("applying standard scaler")

    ss = StandardScaler()
    ss.fit(X_train)
    X_train = ss.transform(X_train)
    X_test = ss.transform(X_test)

    # Take PCA to reduce feature space dimensionality
    print()
    print("using PCA to reduce feature space dimensionality")
    pca = PCA(n_components=512, whiten=True)
    pca = pca.fit(X_train)
    print('Explained variance percentage = %0.2f' % sum(pca.explained_variance_ratio_))
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)

    # Train classifier and obtain predictions for OC-SVM
    print()
    print("training classifier and obtaining predictions for OC-SVM")
    oc_svm_clf = svm.OneClassSVM(gamma=0.001, kernel='rbf', nu=0.08)  # Obtained using grid search
    if_clf = IsolationForest(contamination=0.08, max_features=1.0, max_samples=1.0, n_estimators=40)  # Obtained using grid search

    # fitting models to training data
    oc_svm_clf.fit(X_train)
    if_clf.fit(X_train)

    # predicting data on testing data
    oc_svm_preds = oc_svm_clf.predict(X_test)
    if_preds = if_clf.predict(X_test)

    # printing predictions
    print()
    print("OC-SVM predictions")
    print(oc_svm_preds)
    print()
    print("Isolated forest predictions")
    print(if_preds)

    # saving svm and if models
    joblib.dump(oc_svm_clf, 'oc-svm.pkl')
    joblib.dump(if_clf, 'isolated-forest.pkl')
