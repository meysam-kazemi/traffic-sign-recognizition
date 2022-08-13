# =============================================================================
# Importing libraries
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from keras.layers import Dense,Dropout,Conv2D,MaxPooling2D
import tensorflow as tf
import PIL

# =============================================================================
# Loading data
# =============================================================================
path="./German-Traffic-Signs-Dataset-GTSRB-master/"
   
cs = pd.read_csv(path+"Test.csv")


# Create a function that load the data
def load_images(csv_file_name,path=path):
    """csv_file_name --> The name of The csv file that we want to open it\n
    path --> The address of the folder where the file is locateds\n\n\n
    
    example:
        X_train,Y_train=load_images(csv_file="Train.csv")"""
    
    # Read the csv file that contains inside it images address and label
    csv_file = pd.read_csv(path+csv_file_name)

    X = np.zeros((len(csv_file),30,30,3))
    Y = np.zeros((len(csv_file),1))
    
    # Adding images in DataFrame
    for i,image_path in enumerate(csv_file.Path):
        im = Image.open(path+image_path).resize((30,30)) # Opening images and Resize them
        im = np.array(im) # Convert image to array
        # Change data range into range of (0,1)
        im = im /255
        X[i] = im
        Y[i] = np.asanyarray(csv_file["ClassId"][i])
    return (X,Y)
        

Xtrain,Ytrain=load_images("Train.csv")  
Xtest,Ytest=load_images("Test.csv") 

# Shape of train and test datasets
print(f"Shape of Xtrain-->{Xtrain.shape},Shape of Ytrain{Ytrain.shape}")
print("=============================")
print(f"Shape of Xtest-->{Xtest.shape},Shape of Ytest-->{Ytest.shape}")
    

# The name of each sign and its class number
sign_names = pd.read_csv(path+"signnames.csv")
print(f"sign names for classes 0 until 10-->{sign_names[0:10]}")


# =============================================================================
# View samples of the train data
# =============================================================================
plt.figure(figsize=(7,7))
rand = np.random.randint(len(Xtrain),size=(6,))
for i,image in enumerate(Xtrain[rand]):
    plt.subplot(3,2,i+1)
    plt.imshow(image)
    plt.title(sign_names.SignName[Ytrain[rand[i]][0]],color="green")
    plt.axis("off")
plt.show()


# =============================================================================
# Catego
# =============================================================================
Ytrain = tf.keras.utils.to_categorical(Ytrain,43)
Ytest = tf.keras.utils.to_categorical(Ytest,43)


# validation data
from sklearn.model_selection import train_test_split
Xtrain,Xval,Ytrain,Yval=train_test_split(Xtrain,Ytrain,train_size=0.8,random_state=24)


# =============================================================================
# Build a Model
# =============================================================================

from tensorflow.keras import regularizers

# Create a Sequential model
model = tf.keras.models.Sequential()
# Add a convolutional layer with 4 filters
model.add(Conv2D(4, (3,3),activation="relu",input_shape=(30,30,3),
    kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
    bias_regularizer=regularizers.L2(1e-4),
    activity_regularizer=regularizers.L2(1e-5)))
model.add(MaxPooling2D((2,2)))
# model.add(Dropout(0.4))

# Add a convolutional layer with 4 filters
model.add(Conv2D(4,(3,3),activation="relu",
    kernel_regularizer=regularizers.L1L2(l1=1e-3, l2=1e-3)))
model.add(MaxPooling2D((2,2)))
# model.add(Dropout(0.4))

model.add(tf.keras.layers.Flatten())
# Add a linear layer with 128 neurons and relu activation function
model.add(Dense(128,activation="relu",
    kernel_regularizer=regularizers.L1L2(l1=1e-3, l2=1e-3)))
# model.add(Dropout(0.4))
# Add a linear layer with 128 neurons and relu activation function
model.add(Dense(64,activation="relu",
    kernel_regularizer=regularizers.L1L2(l1=1e-3, l2=1e-3)))
model.add(Dropout(0.4))

# Add Output layer with softmax activation function
model.add(Dense(len(sign_names),activation="softmax"))


model.summary()

recall = tf.keras.metrics.Recall()
model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=[recall,"acc"]
             )
# hist = model.fit(Xtrain,Ytrain,epochs=15,validation_data=(Xval,Yval))


"""I trained the model and saved it.
I also saved the history in CSV format because it takes time to run the model"""
# =============================================================================
# Load the saved mdoel
# =============================================================================
model = tf.keras.models.load_model("./mymodel.h5")
history = pd.read_csv("history.csv")

print("evaluate: ",model.evaluate(Xtest,Ytest))

# =============================================================================
# Validation curve
# =============================================================================

# history = hist.history

plt.figure(figsize=(17,4))
plt.title("Validation Curve")

# The diagram of accuracy-epochs 
plt.subplot(1,3,1)
plt.plot(history["acc"],c="blue",label="accuracy")
plt.plot(history["val_acc"],c="green",label="val_accuracy")
# plt.ylim(0,1)
plt.legend(loc="upper right")
plt.xlabel("epochs")
plt.ylabel("accuracy")

# The diagram of recall-epochs
plt.subplot(1,3,2)
plt.plot(history["recall"],c="yellow",label="recall")
plt.plot(history["val_recall"],c="orange",label="val_recall")
plt.legend(loc="upper right")
plt.xlabel("epochs")
plt.ylabel("recall")

# The diagram of loss-epochs
plt.subplot(1,3,3)
plt.plot(history["loss"],c="red",label="loss")
plt.plot(history["val_loss"],c="orange",label="val_loss")
plt.legend(loc="upper right")
plt.xlabel("epochs")
plt.ylabel("loss")

plt.savefig("validation curve.png")
plt.show()



# =============================================================================
# Predict Xtest's classes
# =============================================================================
pred = model.predict(Xtest)
pred = np.argmax(pred,axis=1)
pred 

Ytest = np.argmax(Ytest,axis=1) 

print(f"ture classes for Xtest:{Ytest[:10]}")
print("==========================")
print(f"Predicted classes for Xtest:{pred[:10]}")

print(f"accuracy of model: {round(np.sum(Ytest==pred)/len(pred)*100,2)}%")
# =============================================================================
# View samples of the test data
# =============================================================================
plt.figure(figsize=(7,7))
rand = np.random.randint(len(Xtest),size=(6,))
for i,image in enumerate(Xtest[rand]):
    plt.subplot(3,2,i+1)
    plt.imshow(image)
    pred_class = sign_names.SignName[pred[rand[i]]]
    true_class = sign_names.SignName[Ytest[rand[i]]]
    plt.title(pred_class if pred_class==true_class else f"{pred_class}\ntrue :{true_class}",
              color="green" if pred_class==true_class else "red")
    plt.axis("off")
plt.show()






    