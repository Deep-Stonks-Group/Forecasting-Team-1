import sklearn.metrics as skm

def calculate_classification(test_x,test_y,model,label_scaler):
    # Getting predictions for the test set.
    pred_y = model(test_x)

    # Transforming the labels back to original space
    orig_y = label_scaler.inverse_transform(test_y.numpy()).tolist()
    pred_y = label_scaler.inverse_transform(pred_y.detach().numpy()).tolist()

    # Transforming labels to be returns instead of price
    rtrn_y = [orig_y[i][0] - orig_y[i - 1][0] for i in range(1, len(orig_y))]
    rtrn_pred = [pred_y[i][0] - pred_y[i - 1][0] for i in range(1, len(pred_y))]

    # Getting Binary Classification
    bin_y = [1 if i > 0 else 0 for i in rtrn_y]
    bin_pred = [1 if i > 0 else 0 for i in rtrn_pred]
    return bin_y, bin_pred

def get_acc(bin_y,bin_pred):
    # Getting accuracy
    acc = skm.accuracy_score(bin_y,bin_pred)

    # Accuracy if you randomly guessed
    rndm_acc = sum(bin_y) / len(bin_y)
    rndm_acc = max(1 - rndm_acc, rndm_acc)

    return rndm_acc, acc

def get_confusion_matrix(bin_y,bin_pred):
    cfn_mtrx = skm.confusion_matrix(bin_y, bin_pred)
    return cfn_mtrx

def print_metrics(test_x,test_y,model,label_scaler):
    # Getting binary classification
    bin_y, bin_pred = calculate_classification(test_x, test_y, model, label_scaler)

    rndm_acc, acc = get_acc(bin_y, bin_pred)

    cfn_mtrx = get_confusion_matrix(bin_y,bin_pred)

    # Printing Accuracy
    print("Random: " + str(rndm_acc))
    print("Accuracy: " + str(acc))

    # Printing Confusion Matrix
    print("Confusion Matrix: ")
    print(cfn_mtrx)
