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
# Categoricalizing
# =============================================================================
Ytrain = tf.keras.utils.to_categorical(Ytrain,43)
Ytest = tf.keras.utils.to_categorical(Ytest,43)

# =============================================================================
# Creating a Model
# =============================================================================

# Create a Sequential model
model = tf.keras.models.Sequential()
# Add a convolutional layer with 16 filters
model.add(Conv2D(16, (3,3),activation="relu",input_shape=(30,30,3)))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.2))

# Add a convolutional layer with 32 filters
model.add(Conv2D(32,(3,3),activation="relu"))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.2))

model.add(tf.keras.layers.Flatten())
# Add a linear layer with 128 neurons and relu activation function
model.add(Dense(128,activation="relu"))
model.add(Dropout(0.2))
# Add a linear layer with 128 neurons and relu activation function
model.add(Dense(64,activation="relu"))
model.add(Dropout(0.2))
# Add Output layer with softmax activation function
model.add(Dense(len(sign_names),activation="softmax"))


model.summary()

recall = tf.keras.metrics.Recall()
model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=[recall,"acc"]
             )
model.fit(Xtrain,Ytrain,epochs=15,validation_split=0.2)

model.evaluate(Xtest,Ytest)
    
