from convolutional_nn_tut import ConvNet
import numpy as np
from utils.dataloader import DataLoader
import matplotlib.pyplot as plt

LEARNING_RATE = 3e-4 #mnist 0.0001  # 0.00025    #finetune = 0.000075
BATCH_SIZE = 256
EPOCHS = 20

def new_model(model_name):
    conv_net = ConvNet(
        input_shape=(28, 28, 1),
        conv_filters=(32,   64, 128, 256, ), 
        conv_kernels=(7,    5,   5,    5, ),
        conv_strides=(1,    2,   2,    1, ), 
    )
    conv_net._name = model_name
    conv_net.summary()
    conv_net.compile(LEARNING_RATE)
    return conv_net 

def existing_model(model_name):
    conv_net = ConvNet.load(model_name)
    conv_net._name = model_name # new_name
    # conv_net.summary()
    conv_net.compile(LEARNING_RATE)

    try:
        conv_net.best_epoch    =  np.load(f"MIR_trained_models/{conv_net._name}/best_epoch.npy")
        conv_net.train_loss    =  np.load(f"MIR_trained_models/{conv_net._name}/train_loss.npy")[:conv_net.best_epoch]
        conv_net.val_loss      =  np.load(f"MIR_trained_models/{conv_net._name}/val_loss.npy")[:conv_net.best_epoch]
        conv_net.val_auc       =  np.load(f"MIR_trained_models/{conv_net._name}/val_auc.npy")[:conv_net.best_epoch]
        conv_net.epoch_count   =  np.load(f"MIR_trained_models/{conv_net._name}/best_epoch.npy") +1 # cus we start at new epoch   
        conv_net.m_auc         =  conv_net.val_auc[conv_net.best_epoch-1]
        assert conv_net.best_epoch-1 > 10, "existing model was not trained for at least 10 epoch"
        print("best index of auc so far: ",np.argmax(conv_net.val_auc), " actual best epoch: ", conv_net.best_epoch-1) # cus epoch starts at count 1
    except:
            print("MIR_train.py: model did not use new param saving")
    return conv_net

if __name__ == "__main__":
    model_name = "first_model" # 100 epoch
    conv_net = new_model(model_name)
    # conv_net = existing_model('first_model-18-0.07248-1.79095')
    conv_net.train_on_batch(batch_size=BATCH_SIZE,num_epoch=EPOCHS)
    
    # Step 5: Make predictions on the X_test data
    X_test = DataLoader(BATCH_SIZE,EPOCHS).x_test
    test_predictions = conv_net.model.predict(X_test)
    test_labels = np.argmax(test_predictions, axis=1)  # Convert probabilities to class labels

    # Visualize the predictions on X_test
    num_samples_to_visualize = 10
    sample_indices = np.random.choice(X_test.shape[0], num_samples_to_visualize, replace=False)

    plt.figure(figsize=(12, 6))
    for i, index in enumerate(sample_indices):
        plt.subplot(2, 5, i + 1)
        plt.imshow(X_test[index].reshape(28, 28), cmap='gray')
        predicted_label = test_labels[index]
        plt.title(f"Predicted Label: {predicted_label}")
        plt.axis('off')

    plt.show()
