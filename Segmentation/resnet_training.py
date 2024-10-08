import numpy as np
import tensorflow as tf

#Load preprocessed images and labels
X_train = np.load('../cs588-capstone/Data/Processed/train_images.npy') # Shape should be (num_samples, height, width, channels)
y_train = np.load('../cs588-capstone/Data/Processed/train_labels.npy') # Shape should be (num_samples,)

#Convert the labels to a TensorFlow tensor
y_train = tf.convert_to_tensor(y_train, dtype=tf.int32)

#Create a TensorFlow dataset for training
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))

#Shuffle and batch the dataset
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)

#Print out the shape of the dataset
for images, labels in train_dataset.take(1):
    print("Images shape:", images.shape)
    print("Labels shape:", labels.shape)

#Use the dataset for training
resnet_model.fit(train_dataset, epochs=10)
