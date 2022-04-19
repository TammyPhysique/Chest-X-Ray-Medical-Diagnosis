Chest X-Ray Medical Diagnosis with Deep Learning
Model Interpretation Methods


# Import necessary packages
import keras
from keras import backend as K
import matplotlib.pyplot as plt
%matplotlib inline
import pandas as pd
import numpy as np
import cv2
import sklearn
import shap
import os
import seaborn as sns
import time
import pickle

sns.set()

# This sets a common size for all the figures we will draw.
plt.rcParams['figure.figsize'] = [10, 7]
Data Exploration
we will work with chest x-ray images taken from the public ChestX-ray8 dataset. In this notebook, you'll get a chance to explore this dataset and familiarize yourself with .

# Read csv file containing training datadata
train_df = pd.read_csv("nih_new/train-small.csv")
valid_df = pd.read_csv("nih_new/valid-small.csv")
test_df = pd.read_csv("nih_new/test.csv")
print(f'There are {train_df.shape[0]} rows and {train_df.shape[1]} columns in the train data frame')
train_df.head()
There are 1000 rows and 16 columns in the train data frame
Image	Atelectasis	Cardiomegaly	Consolidation	Edema	Effusion	Emphysema	Fibrosis	Hernia	Infiltration	Mass	Nodule	PatientId	Pleural_Thickening	Pneumonia	Pneumothorax
0	00008270_015.png	0	0	0	0	0	0	0	0	0	0	0	8270	0	0	0
1	00029855_001.png	1	0	0	0	1	0	0	0	1	0	0	29855	0	0	0
2	00001297_000.png	0	0	0	0	0	0	0	0	0	0	0	1297	1	0	0
3	00012359_002.png	0	0	0	0	0	0	0	0	0	0	0	12359	0	0	0
4	00017951_001.png	0	0	0	0	0	0	0	0	1	0	0	17951	0	0	0
print(f"Train set: The total patient ids are {train_df['PatientId'].count()}, from those the unique ids are {train_df['PatientId'].value_counts().shape[0]} ")
print(f"Validation set: The total patient ids are {valid_df['PatientId'].count()}")
print(f"Test set: The total patient ids are {test_df['PatientId'].count()}")
Train set: The total patient ids are 1000, from those the unique ids are 928 
Validation set: The total patient ids are 200
Test set: The total patient ids are 420
Preventing Data Leakage
It is worth noting that our dataset contains multiple images for each patient. This could be the case, for example, when a patient has taken multiple X-ray images at different times during their hospital visits. In our data splitting, we have ensured that the split is done on the patient level so that there is no data "leakage" between the train, validation, and test datasets.

def check_for_leakage(df1, df2, patient_col):
    """
    Return True if there any patients are in both df1 and df2.

    Args:
        df1 (dataframe): dataframe describing first dataset
        df2 (dataframe): dataframe describing second dataset
        patient_col (str): string name of column with patient IDs
    
    Returns:
        leakage (bool): True if there is leakage, otherwise False
    """

    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
    
    df1_patients_unique = set(df1[patient_col])
    df2_patients_unique = set(df2[patient_col])
    
    patients_in_both_groups = list(df1_patients_unique.intersection(df2_patients_unique))

    # leakage contains true if there is patient overlap, otherwise false.
    leakage = len(patients_in_both_groups) > 0 
    
    ### END CODE HERE ###
    
    return leakage
print("leakage between train and test: {}".format(check_for_leakage(train_df, test_df, 'PatientId')))
print("leakage between valid and test: {}".format(check_for_leakage(valid_df, test_df, 'PatientId')))
leakage between train and test: False
leakage between valid and test: False
Explore data labels
Create a list of the names of each patient condition or disease.

columns = train_df.keys()
columns = list(columns)
print(columns)
['Image', 'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 'PatientId', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']
# Remove unnecesary elements
columns.remove('Image')
columns.remove('PatientId')
# Get the total classes
print(f"There are {len(columns)} columns of labels for these conditions: {columns}")
# Print out the number of positive labels for each class
for column in columns:
    print(f"The class {column} has {train_df[column].sum()} samples")
There are 14 columns of labels for these conditions: ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']
The class Atelectasis has 106 samples
The class Cardiomegaly has 20 samples
The class Consolidation has 33 samples
The class Edema has 16 samples
The class Effusion has 128 samples
The class Emphysema has 13 samples
The class Fibrosis has 14 samples
The class Hernia has 2 samples
The class Infiltration has 175 samples
The class Mass has 45 samples
The class Nodule has 54 samples
The class Pleural_Thickening has 21 samples
The class Pneumonia has 10 samples
The class Pneumothorax has 38 samples
Data Visualization
Using the image names listed in the csv file, retrieve the image associated with each row of data in your dataframe.

# Extract numpy values from Image column in data frame
images = train_df['Image'].values

# Extract 9 random images from it
random_images = [np.random.choice(images) for i in range(9)]

# Location of the image dir
img_dir = 'nih_new/images-small/'

print('Display Random Images')

# Adjust the size of your images
plt.figure(figsize=(20,10))

# Iterate and plot random images
for i in range(9):
    plt.subplot(3, 3, i + 1)
    img = plt.imread(os.path.join(img_dir, random_images[i]))
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    
# Adjust subplot parameters to give specified padding
plt.tight_layout()    
Display Random Images

Investigate a single image
Look at the first image in the dataset and print out some details of the image contents.

# Get the first image that was listed in the train_df dataframe
sample_img = train_df.Image[0]
raw_image = plt.imread(os.path.join(img_dir, sample_img))
plt.imshow(raw_image, cmap='gray')
plt.grid(color='w', linestyle='-', linewidth=1)
plt.colorbar()
plt.title('Raw Chest X Ray Image')
print(f"The dimensions of the image are {raw_image.shape[0]} pixels width and {raw_image.shape[1]} pixels height, one single color channel")
print(f"The maximum pixel value is {raw_image.max():.4f} and the minimum is {raw_image.min():.4f}")
print(f"The mean value of the pixels is {raw_image.mean():.4f} and the standard deviation is {raw_image.std():.4f}")
The dimensions of the image are 1024 pixels width and 1024 pixels height, one single color channel
The maximum pixel value is 0.9804 and the minimum is 0.0000
The mean value of the pixels is 0.4796 and the standard deviation is 0.2757

Investigate pixel value distribution
Plot up the distribution of pixel values in the image shown above.

pixels = np.reshape(raw_image,raw_image.shape[0]*raw_image.shape[1])
plt.hist(pixels, bins=50, label=f'Pixel Mean {np.mean(raw_image):.4f} & Standard Deviation {np.std(raw_image):.4f}')
plt.legend(loc='upper center')
plt.title('Distribution of Pixel Intensities in the Image')
plt.xlabel('Pixel Intensity')
plt.ylabel('# Pixels in Image')
plt.show()

Image Preprocessing in Keras
Standardization
Normalizing images is better suited for training a convolutional neural network. For this task we use the Keras ImageDataGenerator function to perform data preprocessing and data augmentation. The image_generator will adjust the image data such that the new mean of the data will be zero, and the standard deviation of the data will be 1.

In other words, the generator will replace each pixel value in the image with a new value calculated by subtracting the mean and dividing by the standard deviation.

 
Create an image generator for preprocessing. Pre-process the data using the image_generatoras well as reduce the image size down to 320x320 pixels.

# Import data generator from keras https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator
def get_train_generator(df, image_dir, x_col, y_cols, shuffle=True, batch_size=8, seed=1, target_w = 320, target_h = 320):
    """
    Return generator for training set, normalizing using batch
    statistics.

    Args:
      train_df (dataframe): dataframe specifying training data.
      image_dir (str): directory where image files are held.
      x_col (str): name of column in df that holds filenames.
      y_cols (list): list of strings that hold y labels for images.
      batch_size (int): images per batch to be fed into model during training.
      seed (int): random seed.
      target_w (int): final width of input images.
      target_h (int): final height of input images.
    
    Returns:
        train_generator (DataFrameIterator): iterator over training set
    """        
    print("getting train generator...") 
    # Normalize images  --- Generate batches of tensor image data with real-time data augmentation
    image_generator = ImageDataGenerator(
        samplewise_center=True,              #Set each sample mean to 0
        samplewise_std_normalization= True)  # Divide each input by its standard deviation
    
    # flow from directory with specified batch size and target image size
    # flow_from_dataframe ==> https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator
    # RETURNS a DataFrameIterator yielding tuples of (x, y) where x is a numpy array containing a batch of images with 
    # shape (batch_size, *target_size, channels) and y is a numpy array of corresponding labels
    # default data format of ImageGenerator is channels_last
    generator = image_generator.flow_from_dataframe(
            dataframe=df,
            directory=image_dir,
            x_col=x_col,
            y_col=y_cols,
            class_mode="raw",       #  Mode for yielding the targets, one of "binary", "categorical", "input", "multi_output", "raw", sparse" or None. Default: "categorical".
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            target_size=(target_w,target_h))
    
    return generator
def get_test_and_valid_generator(valid_df, test_df, train_df, image_dir, x_col, y_cols, sample_size=100, batch_size=8, 
                                 seed=1, target_w = 320, target_h = 320):
    """
    Return generator for validation set and test test set using 
    normalization statistics from training set.

    Args:
      valid_df (dataframe): dataframe specifying validation data.
      test_df (dataframe): dataframe specifying test data.
      train_df (dataframe): dataframe specifying training data.
      image_dir (str): directory where image files are held.
      x_col (str): name of column in df that holds filenames.
      y_cols (list): list of strings that hold y labels for images.
      sample_size (int): size of sample to use for normalization statistics.
      batch_size (int): images per batch to be fed into model during training.
      seed (int): random seed.
      target_w (int): final width of input images.
      target_h (int): final height of input images.
    
    Returns:
        test_generator (DataFrameIterator) and valid_generator: iterators over test set and validation set respectively
    """
    # get generator to sample dataset
    print(f"\nextracting {sample_size} train images to normalize validation and test datasets...")

    raw_train_generator = ImageDataGenerator().flow_from_dataframe(
        dataframe=train_df, 
        directory=IMAGE_DIR, 
        x_col="Image", 
        y_col=labels, 
        class_mode="raw", 
        batch_size=sample_size, 
        shuffle=True, 
        target_size=(target_w, target_h))
    
    # get data sample
    batch = raw_train_generator.next() # generate a batch of samples and associated labels 
    data_sample = batch[0]             # => we need only the sample imgs ie batch[0]

    # use sample to fit mean and std for test set generator
    image_generator = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization= True)
    
    # fit generator to sample from training data - we use this generator normalizing mean and std using the train sample of 100
    image_generator.fit(data_sample)
    
    print("\ngetting valid generator...")

    # get test generator
    valid_generator = image_generator.flow_from_dataframe(
            dataframe=valid_df,
            directory=image_dir,
            x_col=x_col,
            y_col=y_cols,
            class_mode="raw",
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            target_size=(target_w,target_h))
    
    print("\ngetting test generator...")
    test_generator = image_generator.flow_from_dataframe(
            dataframe=test_df,
            directory=image_dir,
            x_col=x_col,
            y_col=y_cols,
            class_mode="raw",
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            target_size=(target_w,target_h))
    return valid_generator, test_generator
labels = ['Cardiomegaly', 'Emphysema', 'Effusion', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 'Atelectasis',
              'Pneumothorax', 'Pleural_Thickening', 'Pneumonia', 'Fibrosis', 'Edema', 'Consolidation']
IMAGE_DIR = "nih_new/images-small/"
train_generator = get_train_generator(train_df, IMAGE_DIR, "Image", labels)
valid_generator, test_generator= get_test_and_valid_generator(valid_df, test_df, train_df, IMAGE_DIR, "Image", labels)
getting train generator...
Found 1000 validated image filenames.

extracting 100 train images to normalize validation and test datasets...
Found 1000 validated image filenames.

getting valid generator...
Found 200 validated image filenames.

getting test generator...
Found 420 validated image filenames.
# Plot a processed image
sns.set_style("white")
generated_image, label = train_generator.__getitem__(0)
plt.imshow(generated_image[0], cmap='gray')
plt.colorbar()
plt.title('Raw Chest X Ray Image')
print(f"The dimensions of the image are {generated_image.shape[1]} pixels width and {generated_image.shape[2]} pixels height")
print(f"The maximum pixel value is {generated_image.max():.4f} and the minimum is {generated_image.min():.4f}")
print(f"The mean value of the pixels is {generated_image.mean():.4f} and the standard deviation is {generated_image.std():.4f}")
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
The dimensions of the image are 320 pixels width and 320 pixels height
The maximum pixel value is 2.8194 and the minimum is -3.4863
The mean value of the pixels is 0.0000 and the standard deviation is 1.0000

print(generated_image.shape, generated_image[0].shape)
print(raw_image.shape)
(8, 320, 320, 3) (320, 320, 3)
(1024, 1024)
# Include a histogram of the distribution of the pixels
sns.set()
plt.figure(figsize=(10, 7))

# Plot histogram for original iamge
sns.distplot(raw_image.ravel(), 
             label=f'Original Image: mean {np.mean(raw_image):.4f} - Standard Deviation {np.std(raw_image):.4f} \n '
             f'Min pixel value {np.min(raw_image):.4} - Max pixel value {np.max(raw_image):.4}',
             color='blue', 
             kde=False)

# Plot histogram for generated image
sns.distplot(generated_image[0].ravel(), 
             label=f'Generated Image: mean {np.mean(generated_image[0]):.4f} - Standard Deviation {np.std(generated_image[0]):.4f} \n'
             f'Min pixel value {np.min(generated_image[0]):.4} - Max pixel value {np.max(generated_image[0]):.4}', 
             color='red', 
             kde=False)

# Place legends
plt.legend()
plt.title('Distribution of Pixel Intensities in the Image')
plt.xlabel('Pixel Intensity')
plt.ylabel('# Pixel')
plt.show()

Addressing Class Imbalance - Weighted Loss
One of the challenges with working with medical diagnostic datasets is the large class imbalance present in such datasets. Let's plot the frequency of each of the labels in our dataset:

plt.xticks(rotation=90)
plt.bar(x=labels, height=np.mean(train_generator.labels, axis=0))
plt.title("Frequency of Each Class")
plt.show()

We can see from this plot that the prevalance of positive cases varies significantly across the different pathologies. (These trends mirror the ones in the full dataset as well.)

The Hernia pathology has the greatest imbalance with the proportion of positive training cases being about 0.2%.
But even the Infiltration pathology, which has the least amount of imbalance, has only 17.5% of the training cases labelled positive.
Ideally, we would train our model using an evenly balanced dataset so that the positive and negative training cases would contribute equally to the loss.

Impact of class imbalance on loss function
Let's take a closer look at this. Assume we would have used a normal cross-entropy loss for each pathology. We recall that the cross-entropy loss contribution from the 
 training data case is:

where 
 and 
 are the input features and the label, and 
 is the output of the model, i.e. the probability that it is positive.

Note that for any training case, either 
 or else 
, so only one of these terms contributes to the loss (the other term is multiplied by zero, and becomes zero).

We can rewrite the overall average cross-entropy loss over the entire training set  of size  as follows:

 
 
 
Using this formulation, we can see that if there is a large imbalance with very few positive training cases, for example, then the loss will be dominated by the negative class. Summing the contribution over all the training cases for each class (i.e. pathological condition), we see that the contribution of each class (i.e. positive or negative) is:

 
 
Computing Class Frequencies
def compute_class_freqs(labels):
    """
    Compute positive and negative frequences for each class.

    Args:
        labels (np.array): matrix of labels, size (num_examples, num_classes)
    Returns:
        positive_frequencies (np.array): array of positive frequences for each
                                         class, size (num_classes)
        negative_frequencies (np.array): array of negative frequences for each
                                         class, size (num_classes)
    """
    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
    
    # total number of patients (rows)
    N = labels.shape[0]
    
    positive_frequencies = np.sum(labels, axis=0)/N
    negative_frequencies = (N - np.sum(labels, axis=0))/N  # broadcasting of N to a line vector of dim num_classes

    ### END CODE HERE ###
    return positive_frequencies, negative_frequencies
freq_pos, freq_neg = compute_class_freqs(train_generator.labels)
freq_pos
array([0.02 , 0.013, 0.128, 0.002, 0.175, 0.045, 0.054, 0.106, 0.038,
       0.021, 0.01 , 0.014, 0.016, 0.033])
data = pd.DataFrame({"Class": labels, "Label": "Positive", "Value": freq_pos})
data = data.append([{"Class": labels[l], "Label": "Negative", "Value": v} for l,v in enumerate(freq_neg)], ignore_index=True)
plt.xticks(rotation=90)
f = sns.barplot(x="Class", y="Value", hue="Label" ,data=data)

The contributions of positive cases is significantly lower than that of the negative ones. However, we want the contributions to be equal. One way of doing this is by multiplying each example from each class by a class-specific weight factor, 
 and 
, so that the overall contribution of each class is the same.

To have this, we want

which we can do simply by taking

This way, we will be balancing the contribution of positive and negative labels.

pos_weights = freq_neg
neg_weights = freq_pos
pos_contribution = freq_pos * pos_weights 
neg_contribution = freq_neg * neg_weights
Let's verify this by graphing the two contributions next to each other :

data = pd.DataFrame({"Class": labels, "Label": "Positive", "Value": pos_contribution})
data = data.append([{"Class": labels[l], "Label": "Negative", "Value": v} 
                        for l,v in enumerate(neg_contribution)], ignore_index=True)
plt.xticks(rotation=90)
sns.barplot(x="Class", y="Value", hue="Label" ,data=data);

After computing the weights, our final weighted loss for each training case will be

Weighted Loss
def get_weighted_loss(pos_weights, neg_weights, epsilon=1e-7):
    """
    Return weighted loss function given negative weights and positive weights.

    Args:
      pos_weights (np.array): array of positive weights for each class, size (num_classes)
      neg_weights (np.array): array of negative weights for each class, size (num_classes)
    
    Returns:
      weighted_loss (function): weighted loss function
    """
    def weighted_loss(y_true, y_pred):
        """
        Return weighted loss value. 

        Args:
            y_true (Tensor): Tensor of true labels, size is (num_examples, num_classes)
            y_pred (Tensor): Tensor of predicted labels, size is (num_examples, num_classes)
        Returns:
            loss (Float): overall scalar loss summed across all classes
        """
        # initialize loss to zero
        loss = 0.0
        
        ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###

        for i in range(len(pos_weights)):
            # for each class, add average weighted loss for that class 
            loss += - pos_weights[i] * K.mean(y_true[:,i] * K.log(y_pred[:,i] + epsilon)) \
            - neg_weights[i] * K.mean((1-y_true[:,i]) * K.log(1-y_pred[:,i] + epsilon)) #complete this line
        return loss
    
        ### END CODE HERE ###
    return weighted_loss             # this is a function taking 2 arguments y_true and y_pred
DenseNet121
Use a pre-trained DenseNet121 model.Densenet is a convolutional network where each layer is connected to all other layers that are deeper in the network

The first layer is connected to the 2nd, 3rd, 4th etc.
The second layer is connected to the 3rd, 4th, 5th etc.
For a detailed explanation of Densenet, check out the source of the image above, a paper by Gao Huang et al. 2018 called Densely Connected Convolutional Networks. U-net Image
we can load directly from Keras and then add two layers on top of it:

A GlobalAveragePooling2D layer to get the average of the last convolution layers from DenseNet121.
A Dense layer with sigmoid activation to get the prediction logits for each of our classes.
import keras
from keras.applications.densenet import DenseNet121
from keras.models import Model
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping, TensorBoard
from keras.preprocessing import image

def load_C3M3_model():
   
    class_pos = train_df.loc[:, labels].sum(axis=0)
    class_neg = len(train_df) - class_pos
    class_total = class_pos + class_neg

    pos_weights = class_pos / class_total
    neg_weights = class_neg / class_total
    print("Got loss weights")
    # create the base pre-trained model
    base_model = DenseNet121(weights='densenet.hdf5', include_top=False)
    print("Loaded DenseNet")
    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # and a logistic layer
    predictions = Dense(len(labels), activation="sigmoid")(x)
    print("Added layers")

    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(optimizer='adam', loss=get_weighted_loss(neg_weights, pos_weights))
    print("Compiled Model")

    model.load_weights("nih_new/pretrained_model.h5")
    print("Loaded Weights")
    return model
model = load_C3M3_model()
Got loss weights
Loaded DenseNet
Added layers
Compiled Model
Loaded Weights
let's see the layers that our model is composed of.

model.summary()
Model: "model_2"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_2 (InputLayer)            (None, None, None, 3 0                                            
__________________________________________________________________________________________________
zero_padding2d_3 (ZeroPadding2D (None, None, None, 3 0           input_2[0][0]                    
__________________________________________________________________________________________________
conv1/conv (Conv2D)             (None, None, None, 6 9408        zero_padding2d_3[0][0]           
__________________________________________________________________________________________________
conv1/bn (BatchNormalization)   (None, None, None, 6 256         conv1/conv[0][0]                 
__________________________________________________________________________________________________
conv1/relu (Activation)         (None, None, None, 6 0           conv1/bn[0][0]                   
__________________________________________________________________________________________________
zero_padding2d_4 (ZeroPadding2D (None, None, None, 6 0           conv1/relu[0][0]                 
__________________________________________________________________________________________________
pool1 (MaxPooling2D)            (None, None, None, 6 0           zero_padding2d_4[0][0]           
__________________________________________________________________________________________________
conv2_block1_0_bn (BatchNormali (None, None, None, 6 256         pool1[0][0]                      
__________________________________________________________________________________________________
conv2_block1_0_relu (Activation (None, None, None, 6 0           conv2_block1_0_bn[0][0]          
__________________________________________________________________________________________________
conv2_block1_1_conv (Conv2D)    (None, None, None, 1 8192        conv2_block1_0_relu[0][0]        
__________________________________________________________________________________________________
conv2_block1_1_bn (BatchNormali (None, None, None, 1 512         conv2_block1_1_conv[0][0]        
__________________________________________________________________________________________________
conv2_block1_1_relu (Activation (None, None, None, 1 0           conv2_block1_1_bn[0][0]          
__________________________________________________________________________________________________
conv2_block1_2_conv (Conv2D)    (None, None, None, 3 36864       conv2_block1_1_relu[0][0]        
__________________________________________________________________________________________________
conv2_block1_concat (Concatenat (None, None, None, 9 0           pool1[0][0]                      
                                                                 conv2_block1_2_conv[0][0]        
__________________________________________________________________________________________________
conv2_block2_0_bn (BatchNormali (None, None, None, 9 384         conv2_block1_concat[0][0]        
__________________________________________________________________________________________________
conv2_block2_0_relu (Activation (None, None, None, 9 0           conv2_block2_0_bn[0][0]          
__________________________________________________________________________________________________
conv2_block2_1_conv (Conv2D)    (None, None, None, 1 12288       conv2_block2_0_relu[0][0]        
__________________________________________________________________________________________________
conv2_block2_1_bn (BatchNormali (None, None, None, 1 512         conv2_block2_1_conv[0][0]        
__________________________________________________________________________________________________
conv2_block2_1_relu (Activation (None, None, None, 1 0           conv2_block2_1_bn[0][0]          
__________________________________________________________________________________________________
conv2_block2_2_conv (Conv2D)    (None, None, None, 3 36864       conv2_block2_1_relu[0][0]        
__________________________________________________________________________________________________
conv2_block2_concat (Concatenat (None, None, None, 1 0           conv2_block1_concat[0][0]        
                                                                 conv2_block2_2_conv[0][0]        
__________________________________________________________________________________________________
conv2_block3_0_bn (BatchNormali (None, None, None, 1 512         conv2_block2_concat[0][0]        
__________________________________________________________________________________________________
conv2_block3_0_relu (Activation (None, None, None, 1 0           conv2_block3_0_bn[0][0]          
__________________________________________________________________________________________________
conv2_block3_1_conv (Conv2D)    (None, None, None, 1 16384       conv2_block3_0_relu[0][0]        
__________________________________________________________________________________________________
conv2_block3_1_bn (BatchNormali (None, None, None, 1 512         conv2_block3_1_conv[0][0]        
__________________________________________________________________________________________________
conv2_block3_1_relu (Activation (None, None, None, 1 0           conv2_block3_1_bn[0][0]          
__________________________________________________________________________________________________
conv2_block3_2_conv (Conv2D)    (None, None, None, 3 36864       conv2_block3_1_relu[0][0]        
__________________________________________________________________________________________________
conv2_block3_concat (Concatenat (None, None, None, 1 0           conv2_block2_concat[0][0]        
                                                                 conv2_block3_2_conv[0][0]        
__________________________________________________________________________________________________
conv2_block4_0_bn (BatchNormali (None, None, None, 1 640         conv2_block3_concat[0][0]        
__________________________________________________________________________________________________
conv2_block4_0_relu (Activation (None, None, None, 1 0           conv2_block4_0_bn[0][0]          
__________________________________________________________________________________________________
conv2_block4_1_conv (Conv2D)    (None, None, None, 1 20480       conv2_block4_0_relu[0][0]        
__________________________________________________________________________________________________
conv2_block4_1_bn (BatchNormali (None, None, None, 1 512         conv2_block4_1_conv[0][0]        
__________________________________________________________________________________________________
conv2_block4_1_relu (Activation (None, None, None, 1 0           conv2_block4_1_bn[0][0]          
__________________________________________________________________________________________________
conv2_block4_2_conv (Conv2D)    (None, None, None, 3 36864       conv2_block4_1_relu[0][0]        
__________________________________________________________________________________________________
conv2_block4_concat (Concatenat (None, None, None, 1 0           conv2_block3_concat[0][0]        
                                                                 conv2_block4_2_conv[0][0]        
__________________________________________________________________________________________________
conv2_block5_0_bn (BatchNormali (None, None, None, 1 768         conv2_block4_concat[0][0]        
__________________________________________________________________________________________________
conv2_block5_0_relu (Activation (None, None, None, 1 0           conv2_block5_0_bn[0][0]          
__________________________________________________________________________________________________
conv2_block5_1_conv (Conv2D)    (None, None, None, 1 24576       conv2_block5_0_relu[0][0]        
__________________________________________________________________________________________________
conv2_block5_1_bn (BatchNormali (None, None, None, 1 512         conv2_block5_1_conv[0][0]        
__________________________________________________________________________________________________
conv2_block5_1_relu (Activation (None, None, None, 1 0           conv2_block5_1_bn[0][0]          
__________________________________________________________________________________________________
conv2_block5_2_conv (Conv2D)    (None, None, None, 3 36864       conv2_block5_1_relu[0][0]        
__________________________________________________________________________________________________
conv2_block5_concat (Concatenat (None, None, None, 2 0           conv2_block4_concat[0][0]        
                                                                 conv2_block5_2_conv[0][0]        
__________________________________________________________________________________________________
conv2_block6_0_bn (BatchNormali (None, None, None, 2 896         conv2_block5_concat[0][0]        
__________________________________________________________________________________________________
conv2_block6_0_relu (Activation (None, None, None, 2 0           conv2_block6_0_bn[0][0]          
__________________________________________________________________________________________________
conv2_block6_1_conv (Conv2D)    (None, None, None, 1 28672       conv2_block6_0_relu[0][0]        
__________________________________________________________________________________________________
conv2_block6_1_bn (BatchNormali (None, None, None, 1 512         conv2_block6_1_conv[0][0]        
__________________________________________________________________________________________________
conv2_block6_1_relu (Activation (None, None, None, 1 0           conv2_block6_1_bn[0][0]          
__________________________________________________________________________________________________
conv2_block6_2_conv (Conv2D)    (None, None, None, 3 36864       conv2_block6_1_relu[0][0]        
__________________________________________________________________________________________________
conv2_block6_concat (Concatenat (None, None, None, 2 0           conv2_block5_concat[0][0]        
                                                                 conv2_block6_2_conv[0][0]        
__________________________________________________________________________________________________
pool2_bn (BatchNormalization)   (None, None, None, 2 1024        conv2_block6_concat[0][0]        
__________________________________________________________________________________________________
pool2_relu (Activation)         (None, None, None, 2 0           pool2_bn[0][0]                   
__________________________________________________________________________________________________
pool2_conv (Conv2D)             (None, None, None, 1 32768       pool2_relu[0][0]                 
__________________________________________________________________________________________________
pool2_pool (AveragePooling2D)   (None, None, None, 1 0           pool2_conv[0][0]                 
__________________________________________________________________________________________________
conv3_block1_0_bn (BatchNormali (None, None, None, 1 512         pool2_pool[0][0]                 
__________________________________________________________________________________________________
conv3_block1_0_relu (Activation (None, None, None, 1 0           conv3_block1_0_bn[0][0]          
__________________________________________________________________________________________________
conv3_block1_1_conv (Conv2D)    (None, None, None, 1 16384       conv3_block1_0_relu[0][0]        
__________________________________________________________________________________________________
conv3_block1_1_bn (BatchNormali (None, None, None, 1 512         conv3_block1_1_conv[0][0]        
__________________________________________________________________________________________________
conv3_block1_1_relu (Activation (None, None, None, 1 0           conv3_block1_1_bn[0][0]          
__________________________________________________________________________________________________
conv3_block1_2_conv (Conv2D)    (None, None, None, 3 36864       conv3_block1_1_relu[0][0]        
__________________________________________________________________________________________________
conv3_block1_concat (Concatenat (None, None, None, 1 0           pool2_pool[0][0]                 
                                                                 conv3_block1_2_conv[0][0]        
__________________________________________________________________________________________________
conv3_block2_0_bn (BatchNormali (None, None, None, 1 640         conv3_block1_concat[0][0]        
__________________________________________________________________________________________________
conv3_block2_0_relu (Activation (None, None, None, 1 0           conv3_block2_0_bn[0][0]          
__________________________________________________________________________________________________
conv3_block2_1_conv (Conv2D)    (None, None, None, 1 20480       conv3_block2_0_relu[0][0]        
__________________________________________________________________________________________________
conv3_block2_1_bn (BatchNormali (None, None, None, 1 512         conv3_block2_1_conv[0][0]        
__________________________________________________________________________________________________
conv3_block2_1_relu (Activation (None, None, None, 1 0           conv3_block2_1_bn[0][0]          
__________________________________________________________________________________________________
conv3_block2_2_conv (Conv2D)    (None, None, None, 3 36864       conv3_block2_1_relu[0][0]        
__________________________________________________________________________________________________
conv3_block2_concat (Concatenat (None, None, None, 1 0           conv3_block1_concat[0][0]        
                                                                 conv3_block2_2_conv[0][0]        
__________________________________________________________________________________________________
conv3_block3_0_bn (BatchNormali (None, None, None, 1 768         conv3_block2_concat[0][0]        
__________________________________________________________________________________________________
conv3_block3_0_relu (Activation (None, None, None, 1 0           conv3_block3_0_bn[0][0]          
__________________________________________________________________________________________________
conv3_block3_1_conv (Conv2D)    (None, None, None, 1 24576       conv3_block3_0_relu[0][0]        
__________________________________________________________________________________________________
conv3_block3_1_bn (BatchNormali (None, None, None, 1 512         conv3_block3_1_conv[0][0]        
__________________________________________________________________________________________________
conv3_block3_1_relu (Activation (None, None, None, 1 0           conv3_block3_1_bn[0][0]          
__________________________________________________________________________________________________
conv3_block3_2_conv (Conv2D)    (None, None, None, 3 36864       conv3_block3_1_relu[0][0]        
__________________________________________________________________________________________________
conv3_block3_concat (Concatenat (None, None, None, 2 0           conv3_block2_concat[0][0]        
                                                                 conv3_block3_2_conv[0][0]        
__________________________________________________________________________________________________
conv3_block4_0_bn (BatchNormali (None, None, None, 2 896         conv3_block3_concat[0][0]        
__________________________________________________________________________________________________
conv3_block4_0_relu (Activation (None, None, None, 2 0           conv3_block4_0_bn[0][0]          
__________________________________________________________________________________________________
conv3_block4_1_conv (Conv2D)    (None, None, None, 1 28672       conv3_block4_0_relu[0][0]        
__________________________________________________________________________________________________
conv3_block4_1_bn (BatchNormali (None, None, None, 1 512         conv3_block4_1_conv[0][0]        
__________________________________________________________________________________________________
conv3_block4_1_relu (Activation (None, None, None, 1 0           conv3_block4_1_bn[0][0]          
__________________________________________________________________________________________________
conv3_block4_2_conv (Conv2D)    (None, None, None, 3 36864       conv3_block4_1_relu[0][0]        
__________________________________________________________________________________________________
conv3_block4_concat (Concatenat (None, None, None, 2 0           conv3_block3_concat[0][0]        
                                                                 conv3_block4_2_conv[0][0]        
__________________________________________________________________________________________________
conv3_block5_0_bn (BatchNormali (None, None, None, 2 1024        conv3_block4_concat[0][0]        
__________________________________________________________________________________________________
conv3_block5_0_relu (Activation (None, None, None, 2 0           conv3_block5_0_bn[0][0]          
__________________________________________________________________________________________________
conv3_block5_1_conv (Conv2D)    (None, None, None, 1 32768       conv3_block5_0_relu[0][0]        
__________________________________________________________________________________________________
conv3_block5_1_bn (BatchNormali (None, None, None, 1 512         conv3_block5_1_conv[0][0]        
__________________________________________________________________________________________________
conv3_block5_1_relu (Activation (None, None, None, 1 0           conv3_block5_1_bn[0][0]          
__________________________________________________________________________________________________
conv3_block5_2_conv (Conv2D)    (None, None, None, 3 36864       conv3_block5_1_relu[0][0]        
__________________________________________________________________________________________________
conv3_block5_concat (Concatenat (None, None, None, 2 0           conv3_block4_concat[0][0]        
                                                                 conv3_block5_2_conv[0][0]        
__________________________________________________________________________________________________
conv3_block6_0_bn (BatchNormali (None, None, None, 2 1152        conv3_block5_concat[0][0]        
__________________________________________________________________________________________________
conv3_block6_0_relu (Activation (None, None, None, 2 0           conv3_block6_0_bn[0][0]          
__________________________________________________________________________________________________
conv3_block6_1_conv (Conv2D)    (None, None, None, 1 36864       conv3_block6_0_relu[0][0]        
__________________________________________________________________________________________________
conv3_block6_1_bn (BatchNormali (None, None, None, 1 512         conv3_block6_1_conv[0][0]        
__________________________________________________________________________________________________
conv3_block6_1_relu (Activation (None, None, None, 1 0           conv3_block6_1_bn[0][0]          
__________________________________________________________________________________________________
conv3_block6_2_conv (Conv2D)    (None, None, None, 3 36864       conv3_block6_1_relu[0][0]        
__________________________________________________________________________________________________
conv3_block6_concat (Concatenat (None, None, None, 3 0           conv3_block5_concat[0][0]        
                                                                 conv3_block6_2_conv[0][0]        
__________________________________________________________________________________________________
conv3_block7_0_bn (BatchNormali (None, None, None, 3 1280        conv3_block6_concat[0][0]        
__________________________________________________________________________________________________
conv3_block7_0_relu (Activation (None, None, None, 3 0           conv3_block7_0_bn[0][0]          
__________________________________________________________________________________________________
conv3_block7_1_conv (Conv2D)    (None, None, None, 1 40960       conv3_block7_0_relu[0][0]        
__________________________________________________________________________________________________
conv3_block7_1_bn (BatchNormali (None, None, None, 1 512         conv3_block7_1_conv[0][0]        
__________________________________________________________________________________________________
conv3_block7_1_relu (Activation (None, None, None, 1 0           conv3_block7_1_bn[0][0]          
__________________________________________________________________________________________________
conv3_block7_2_conv (Conv2D)    (None, None, None, 3 36864       conv3_block7_1_relu[0][0]        
__________________________________________________________________________________________________
conv3_block7_concat (Concatenat (None, None, None, 3 0           conv3_block6_concat[0][0]        
                                                                 conv3_block7_2_conv[0][0]        
__________________________________________________________________________________________________
conv3_block8_0_bn (BatchNormali (None, None, None, 3 1408        conv3_block7_concat[0][0]        
__________________________________________________________________________________________________
conv3_block8_0_relu (Activation (None, None, None, 3 0           conv3_block8_0_bn[0][0]          
__________________________________________________________________________________________________
conv3_block8_1_conv (Conv2D)    (None, None, None, 1 45056       conv3_block8_0_relu[0][0]        
__________________________________________________________________________________________________
conv3_block8_1_bn (BatchNormali (None, None, None, 1 512         conv3_block8_1_conv[0][0]        
__________________________________________________________________________________________________
conv3_block8_1_relu (Activation (None, None, None, 1 0           conv3_block8_1_bn[0][0]          
__________________________________________________________________________________________________
conv3_block8_2_conv (Conv2D)    (None, None, None, 3 36864       conv3_block8_1_relu[0][0]        
__________________________________________________________________________________________________
conv3_block8_concat (Concatenat (None, None, None, 3 0           conv3_block7_concat[0][0]        
                                                                 conv3_block8_2_conv[0][0]        
__________________________________________________________________________________________________
conv3_block9_0_bn (BatchNormali (None, None, None, 3 1536        conv3_block8_concat[0][0]        
__________________________________________________________________________________________________
conv3_block9_0_relu (Activation (None, None, None, 3 0           conv3_block9_0_bn[0][0]          
__________________________________________________________________________________________________
conv3_block9_1_conv (Conv2D)    (None, None, None, 1 49152       conv3_block9_0_relu[0][0]        
__________________________________________________________________________________________________
conv3_block9_1_bn (BatchNormali (None, None, None, 1 512         conv3_block9_1_conv[0][0]        
__________________________________________________________________________________________________
conv3_block9_1_relu (Activation (None, None, None, 1 0           conv3_block9_1_bn[0][0]          
__________________________________________________________________________________________________
conv3_block9_2_conv (Conv2D)    (None, None, None, 3 36864       conv3_block9_1_relu[0][0]        
__________________________________________________________________________________________________
conv3_block9_concat (Concatenat (None, None, None, 4 0           conv3_block8_concat[0][0]        
                                                                 conv3_block9_2_conv[0][0]        
__________________________________________________________________________________________________
conv3_block10_0_bn (BatchNormal (None, None, None, 4 1664        conv3_block9_concat[0][0]        
__________________________________________________________________________________________________
conv3_block10_0_relu (Activatio (None, None, None, 4 0           conv3_block10_0_bn[0][0]         
__________________________________________________________________________________________________
conv3_block10_1_conv (Conv2D)   (None, None, None, 1 53248       conv3_block10_0_relu[0][0]       
__________________________________________________________________________________________________
conv3_block10_1_bn (BatchNormal (None, None, None, 1 512         conv3_block10_1_conv[0][0]       
__________________________________________________________________________________________________
conv3_block10_1_relu (Activatio (None, None, None, 1 0           conv3_block10_1_bn[0][0]         
__________________________________________________________________________________________________
conv3_block10_2_conv (Conv2D)   (None, None, None, 3 36864       conv3_block10_1_relu[0][0]       
__________________________________________________________________________________________________
conv3_block10_concat (Concatena (None, None, None, 4 0           conv3_block9_concat[0][0]        
                                                                 conv3_block10_2_conv[0][0]       
__________________________________________________________________________________________________
conv3_block11_0_bn (BatchNormal (None, None, None, 4 1792        conv3_block10_concat[0][0]       
__________________________________________________________________________________________________
conv3_block11_0_relu (Activatio (None, None, None, 4 0           conv3_block11_0_bn[0][0]         
__________________________________________________________________________________________________
conv3_block11_1_conv (Conv2D)   (None, None, None, 1 57344       conv3_block11_0_relu[0][0]       
__________________________________________________________________________________________________
conv3_block11_1_bn (BatchNormal (None, None, None, 1 512         conv3_block11_1_conv[0][0]       
__________________________________________________________________________________________________
conv3_block11_1_relu (Activatio (None, None, None, 1 0           conv3_block11_1_bn[0][0]         
__________________________________________________________________________________________________
conv3_block11_2_conv (Conv2D)   (None, None, None, 3 36864       conv3_block11_1_relu[0][0]       
__________________________________________________________________________________________________
conv3_block11_concat (Concatena (None, None, None, 4 0           conv3_block10_concat[0][0]       
                                                                 conv3_block11_2_conv[0][0]       
__________________________________________________________________________________________________
conv3_block12_0_bn (BatchNormal (None, None, None, 4 1920        conv3_block11_concat[0][0]       
__________________________________________________________________________________________________
conv3_block12_0_relu (Activatio (None, None, None, 4 0           conv3_block12_0_bn[0][0]         
__________________________________________________________________________________________________
conv3_block12_1_conv (Conv2D)   (None, None, None, 1 61440       conv3_block12_0_relu[0][0]       
__________________________________________________________________________________________________
conv3_block12_1_bn (BatchNormal (None, None, None, 1 512         conv3_block12_1_conv[0][0]       
__________________________________________________________________________________________________
conv3_block12_1_relu (Activatio (None, None, None, 1 0           conv3_block12_1_bn[0][0]         
__________________________________________________________________________________________________
conv3_block12_2_conv (Conv2D)   (None, None, None, 3 36864       conv3_block12_1_relu[0][0]       
__________________________________________________________________________________________________
conv3_block12_concat (Concatena (None, None, None, 5 0           conv3_block11_concat[0][0]       
                                                                 conv3_block12_2_conv[0][0]       
__________________________________________________________________________________________________
pool3_bn (BatchNormalization)   (None, None, None, 5 2048        conv3_block12_concat[0][0]       
__________________________________________________________________________________________________
pool3_relu (Activation)         (None, None, None, 5 0           pool3_bn[0][0]                   
__________________________________________________________________________________________________
pool3_conv (Conv2D)             (None, None, None, 2 131072      pool3_relu[0][0]                 
__________________________________________________________________________________________________
pool3_pool (AveragePooling2D)   (None, None, None, 2 0           pool3_conv[0][0]                 
__________________________________________________________________________________________________
conv4_block1_0_bn (BatchNormali (None, None, None, 2 1024        pool3_pool[0][0]                 
__________________________________________________________________________________________________
conv4_block1_0_relu (Activation (None, None, None, 2 0           conv4_block1_0_bn[0][0]          
__________________________________________________________________________________________________
conv4_block1_1_conv (Conv2D)    (None, None, None, 1 32768       conv4_block1_0_relu[0][0]        
__________________________________________________________________________________________________
conv4_block1_1_bn (BatchNormali (None, None, None, 1 512         conv4_block1_1_conv[0][0]        
__________________________________________________________________________________________________
conv4_block1_1_relu (Activation (None, None, None, 1 0           conv4_block1_1_bn[0][0]          
__________________________________________________________________________________________________
conv4_block1_2_conv (Conv2D)    (None, None, None, 3 36864       conv4_block1_1_relu[0][0]        
__________________________________________________________________________________________________
conv4_block1_concat (Concatenat (None, None, None, 2 0           pool3_pool[0][0]                 
                                                                 conv4_block1_2_conv[0][0]        
__________________________________________________________________________________________________
conv4_block2_0_bn (BatchNormali (None, None, None, 2 1152        conv4_block1_concat[0][0]        
__________________________________________________________________________________________________
conv4_block2_0_relu (Activation (None, None, None, 2 0           conv4_block2_0_bn[0][0]          
__________________________________________________________________________________________________
conv4_block2_1_conv (Conv2D)    (None, None, None, 1 36864       conv4_block2_0_relu[0][0]        
__________________________________________________________________________________________________
conv4_block2_1_bn (BatchNormali (None, None, None, 1 512         conv4_block2_1_conv[0][0]        
__________________________________________________________________________________________________
conv4_block2_1_relu (Activation (None, None, None, 1 0           conv4_block2_1_bn[0][0]          
__________________________________________________________________________________________________
conv4_block2_2_conv (Conv2D)    (None, None, None, 3 36864       conv4_block2_1_relu[0][0]        
__________________________________________________________________________________________________
conv4_block2_concat (Concatenat (None, None, None, 3 0           conv4_block1_concat[0][0]        
                                                                 conv4_block2_2_conv[0][0]        
__________________________________________________________________________________________________
conv4_block3_0_bn (BatchNormali (None, None, None, 3 1280        conv4_block2_concat[0][0]        
__________________________________________________________________________________________________
conv4_block3_0_relu (Activation (None, None, None, 3 0           conv4_block3_0_bn[0][0]          
__________________________________________________________________________________________________
conv4_block3_1_conv (Conv2D)    (None, None, None, 1 40960       conv4_block3_0_relu[0][0]        
__________________________________________________________________________________________________
conv4_block3_1_bn (BatchNormali (None, None, None, 1 512         conv4_block3_1_conv[0][0]        
__________________________________________________________________________________________________
conv4_block3_1_relu (Activation (None, None, None, 1 0           conv4_block3_1_bn[0][0]          
__________________________________________________________________________________________________
conv4_block3_2_conv (Conv2D)    (None, None, None, 3 36864       conv4_block3_1_relu[0][0]        
__________________________________________________________________________________________________
conv4_block3_concat (Concatenat (None, None, None, 3 0           conv4_block2_concat[0][0]        
                                                                 conv4_block3_2_conv[0][0]        
__________________________________________________________________________________________________
conv4_block4_0_bn (BatchNormali (None, None, None, 3 1408        conv4_block3_concat[0][0]        
__________________________________________________________________________________________________
conv4_block4_0_relu (Activation (None, None, None, 3 0           conv4_block4_0_bn[0][0]          
__________________________________________________________________________________________________
conv4_block4_1_conv (Conv2D)    (None, None, None, 1 45056       conv4_block4_0_relu[0][0]        
__________________________________________________________________________________________________
conv4_block4_1_bn (BatchNormali (None, None, None, 1 512         conv4_block4_1_conv[0][0]        
__________________________________________________________________________________________________
conv4_block4_1_relu (Activation (None, None, None, 1 0           conv4_block4_1_bn[0][0]          
__________________________________________________________________________________________________
conv4_block4_2_conv (Conv2D)    (None, None, None, 3 36864       conv4_block4_1_relu[0][0]        
__________________________________________________________________________________________________
conv4_block4_concat (Concatenat (None, None, None, 3 0           conv4_block3_concat[0][0]        
                                                                 conv4_block4_2_conv[0][0]        
__________________________________________________________________________________________________
conv4_block5_0_bn (BatchNormali (None, None, None, 3 1536        conv4_block4_concat[0][0]        
__________________________________________________________________________________________________
conv4_block5_0_relu (Activation (None, None, None, 3 0           conv4_block5_0_bn[0][0]          
__________________________________________________________________________________________________
conv4_block5_1_conv (Conv2D)    (None, None, None, 1 49152       conv4_block5_0_relu[0][0]        
__________________________________________________________________________________________________
conv4_block5_1_bn (BatchNormali (None, None, None, 1 512         conv4_block5_1_conv[0][0]        
__________________________________________________________________________________________________
conv4_block5_1_relu (Activation (None, None, None, 1 0           conv4_block5_1_bn[0][0]          
__________________________________________________________________________________________________
conv4_block5_2_conv (Conv2D)    (None, None, None, 3 36864       conv4_block5_1_relu[0][0]        
__________________________________________________________________________________________________
conv4_block5_concat (Concatenat (None, None, None, 4 0           conv4_block4_concat[0][0]        
                                                                 conv4_block5_2_conv[0][0]        
__________________________________________________________________________________________________
conv4_block6_0_bn (BatchNormali (None, None, None, 4 1664        conv4_block5_concat[0][0]        
__________________________________________________________________________________________________
conv4_block6_0_relu (Activation (None, None, None, 4 0           conv4_block6_0_bn[0][0]          
__________________________________________________________________________________________________
conv4_block6_1_conv (Conv2D)    (None, None, None, 1 53248       conv4_block6_0_relu[0][0]        
__________________________________________________________________________________________________
conv4_block6_1_bn (BatchNormali (None, None, None, 1 512         conv4_block6_1_conv[0][0]        
__________________________________________________________________________________________________
conv4_block6_1_relu (Activation (None, None, None, 1 0           conv4_block6_1_bn[0][0]          
__________________________________________________________________________________________________
conv4_block6_2_conv (Conv2D)    (None, None, None, 3 36864       conv4_block6_1_relu[0][0]        
__________________________________________________________________________________________________
conv4_block6_concat (Concatenat (None, None, None, 4 0           conv4_block5_concat[0][0]        
                                                                 conv4_block6_2_conv[0][0]        
__________________________________________________________________________________________________
conv4_block7_0_bn (BatchNormali (None, None, None, 4 1792        conv4_block6_concat[0][0]        
__________________________________________________________________________________________________
conv4_block7_0_relu (Activation (None, None, None, 4 0           conv4_block7_0_bn[0][0]          
__________________________________________________________________________________________________
conv4_block7_1_conv (Conv2D)    (None, None, None, 1 57344       conv4_block7_0_relu[0][0]        
__________________________________________________________________________________________________
conv4_block7_1_bn (BatchNormali (None, None, None, 1 512         conv4_block7_1_conv[0][0]        
__________________________________________________________________________________________________
conv4_block7_1_relu (Activation (None, None, None, 1 0           conv4_block7_1_bn[0][0]          
__________________________________________________________________________________________________
conv4_block7_2_conv (Conv2D)    (None, None, None, 3 36864       conv4_block7_1_relu[0][0]        
__________________________________________________________________________________________________
conv4_block7_concat (Concatenat (None, None, None, 4 0           conv4_block6_concat[0][0]        
                                                                 conv4_block7_2_conv[0][0]        
__________________________________________________________________________________________________
conv4_block8_0_bn (BatchNormali (None, None, None, 4 1920        conv4_block7_concat[0][0]        
__________________________________________________________________________________________________
conv4_block8_0_relu (Activation (None, None, None, 4 0           conv4_block8_0_bn[0][0]          
__________________________________________________________________________________________________
conv4_block8_1_conv (Conv2D)    (None, None, None, 1 61440       conv4_block8_0_relu[0][0]        
__________________________________________________________________________________________________
conv4_block8_1_bn (BatchNormali (None, None, None, 1 512         conv4_block8_1_conv[0][0]        
__________________________________________________________________________________________________
conv4_block8_1_relu (Activation (None, None, None, 1 0           conv4_block8_1_bn[0][0]          
__________________________________________________________________________________________________
conv4_block8_2_conv (Conv2D)    (None, None, None, 3 36864       conv4_block8_1_relu[0][0]        
__________________________________________________________________________________________________
conv4_block8_concat (Concatenat (None, None, None, 5 0           conv4_block7_concat[0][0]        
                                                                 conv4_block8_2_conv[0][0]        
__________________________________________________________________________________________________
conv4_block9_0_bn (BatchNormali (None, None, None, 5 2048        conv4_block8_concat[0][0]        
__________________________________________________________________________________________________
conv4_block9_0_relu (Activation (None, None, None, 5 0           conv4_block9_0_bn[0][0]          
__________________________________________________________________________________________________
conv4_block9_1_conv (Conv2D)    (None, None, None, 1 65536       conv4_block9_0_relu[0][0]        
__________________________________________________________________________________________________
conv4_block9_1_bn (BatchNormali (None, None, None, 1 512         conv4_block9_1_conv[0][0]        
__________________________________________________________________________________________________
conv4_block9_1_relu (Activation (None, None, None, 1 0           conv4_block9_1_bn[0][0]          
__________________________________________________________________________________________________
conv4_block9_2_conv (Conv2D)    (None, None, None, 3 36864       conv4_block9_1_relu[0][0]        
__________________________________________________________________________________________________
conv4_block9_concat (Concatenat (None, None, None, 5 0           conv4_block8_concat[0][0]        
                                                                 conv4_block9_2_conv[0][0]        
__________________________________________________________________________________________________
conv4_block10_0_bn (BatchNormal (None, None, None, 5 2176        conv4_block9_concat[0][0]        
__________________________________________________________________________________________________
conv4_block10_0_relu (Activatio (None, None, None, 5 0           conv4_block10_0_bn[0][0]         
__________________________________________________________________________________________________
conv4_block10_1_conv (Conv2D)   (None, None, None, 1 69632       conv4_block10_0_relu[0][0]       
__________________________________________________________________________________________________
conv4_block10_1_bn (BatchNormal (None, None, None, 1 512         conv4_block10_1_conv[0][0]       
__________________________________________________________________________________________________
conv4_block10_1_relu (Activatio (None, None, None, 1 0           conv4_block10_1_bn[0][0]         
__________________________________________________________________________________________________
conv4_block10_2_conv (Conv2D)   (None, None, None, 3 36864       conv4_block10_1_relu[0][0]       
__________________________________________________________________________________________________
conv4_block10_concat (Concatena (None, None, None, 5 0           conv4_block9_concat[0][0]        
                                                                 conv4_block10_2_conv[0][0]       
__________________________________________________________________________________________________
conv4_block11_0_bn (BatchNormal (None, None, None, 5 2304        conv4_block10_concat[0][0]       
__________________________________________________________________________________________________
conv4_block11_0_relu (Activatio (None, None, None, 5 0           conv4_block11_0_bn[0][0]         
__________________________________________________________________________________________________
conv4_block11_1_conv (Conv2D)   (None, None, None, 1 73728       conv4_block11_0_relu[0][0]       
__________________________________________________________________________________________________
conv4_block11_1_bn (BatchNormal (None, None, None, 1 512         conv4_block11_1_conv[0][0]       
__________________________________________________________________________________________________
conv4_block11_1_relu (Activatio (None, None, None, 1 0           conv4_block11_1_bn[0][0]         
__________________________________________________________________________________________________
conv4_block11_2_conv (Conv2D)   (None, None, None, 3 36864       conv4_block11_1_relu[0][0]       
__________________________________________________________________________________________________
conv4_block11_concat (Concatena (None, None, None, 6 0           conv4_block10_concat[0][0]       
                                                                 conv4_block11_2_conv[0][0]       
__________________________________________________________________________________________________
conv4_block12_0_bn (BatchNormal (None, None, None, 6 2432        conv4_block11_concat[0][0]       
__________________________________________________________________________________________________
conv4_block12_0_relu (Activatio (None, None, None, 6 0           conv4_block12_0_bn[0][0]         
__________________________________________________________________________________________________
conv4_block12_1_conv (Conv2D)   (None, None, None, 1 77824       conv4_block12_0_relu[0][0]       
__________________________________________________________________________________________________
conv4_block12_1_bn (BatchNormal (None, None, None, 1 512         conv4_block12_1_conv[0][0]       
__________________________________________________________________________________________________
conv4_block12_1_relu (Activatio (None, None, None, 1 0           conv4_block12_1_bn[0][0]         
__________________________________________________________________________________________________
conv4_block12_2_conv (Conv2D)   (None, None, None, 3 36864       conv4_block12_1_relu[0][0]       
__________________________________________________________________________________________________
conv4_block12_concat (Concatena (None, None, None, 6 0           conv4_block11_concat[0][0]       
                                                                 conv4_block12_2_conv[0][0]       
__________________________________________________________________________________________________
conv4_block13_0_bn (BatchNormal (None, None, None, 6 2560        conv4_block12_concat[0][0]       
__________________________________________________________________________________________________
conv4_block13_0_relu (Activatio (None, None, None, 6 0           conv4_block13_0_bn[0][0]         
__________________________________________________________________________________________________
conv4_block13_1_conv (Conv2D)   (None, None, None, 1 81920       conv4_block13_0_relu[0][0]       
__________________________________________________________________________________________________
conv4_block13_1_bn (BatchNormal (None, None, None, 1 512         conv4_block13_1_conv[0][0]       
__________________________________________________________________________________________________
conv4_block13_1_relu (Activatio (None, None, None, 1 0           conv4_block13_1_bn[0][0]         
__________________________________________________________________________________________________
conv4_block13_2_conv (Conv2D)   (None, None, None, 3 36864       conv4_block13_1_relu[0][0]       
__________________________________________________________________________________________________
conv4_block13_concat (Concatena (None, None, None, 6 0           conv4_block12_concat[0][0]       
                                                                 conv4_block13_2_conv[0][0]       
__________________________________________________________________________________________________
conv4_block14_0_bn (BatchNormal (None, None, None, 6 2688        conv4_block13_concat[0][0]       
__________________________________________________________________________________________________
conv4_block14_0_relu (Activatio (None, None, None, 6 0           conv4_block14_0_bn[0][0]         
__________________________________________________________________________________________________
conv4_block14_1_conv (Conv2D)   (None, None, None, 1 86016       conv4_block14_0_relu[0][0]       
__________________________________________________________________________________________________
conv4_block14_1_bn (BatchNormal (None, None, None, 1 512         conv4_block14_1_conv[0][0]       
__________________________________________________________________________________________________
conv4_block14_1_relu (Activatio (None, None, None, 1 0           conv4_block14_1_bn[0][0]         
__________________________________________________________________________________________________
conv4_block14_2_conv (Conv2D)   (None, None, None, 3 36864       conv4_block14_1_relu[0][0]       
__________________________________________________________________________________________________
conv4_block14_concat (Concatena (None, None, None, 7 0           conv4_block13_concat[0][0]       
                                                                 conv4_block14_2_conv[0][0]       
__________________________________________________________________________________________________
conv4_block15_0_bn (BatchNormal (None, None, None, 7 2816        conv4_block14_concat[0][0]       
__________________________________________________________________________________________________
conv4_block15_0_relu (Activatio (None, None, None, 7 0           conv4_block15_0_bn[0][0]         
__________________________________________________________________________________________________
conv4_block15_1_conv (Conv2D)   (None, None, None, 1 90112       conv4_block15_0_relu[0][0]       
__________________________________________________________________________________________________
conv4_block15_1_bn (BatchNormal (None, None, None, 1 512         conv4_block15_1_conv[0][0]       
__________________________________________________________________________________________________
conv4_block15_1_relu (Activatio (None, None, None, 1 0           conv4_block15_1_bn[0][0]         
__________________________________________________________________________________________________
conv4_block15_2_conv (Conv2D)   (None, None, None, 3 36864       conv4_block15_1_relu[0][0]       
__________________________________________________________________________________________________
conv4_block15_concat (Concatena (None, None, None, 7 0           conv4_block14_concat[0][0]       
                                                                 conv4_block15_2_conv[0][0]       
__________________________________________________________________________________________________
conv4_block16_0_bn (BatchNormal (None, None, None, 7 2944        conv4_block15_concat[0][0]       
__________________________________________________________________________________________________
conv4_block16_0_relu (Activatio (None, None, None, 7 0           conv4_block16_0_bn[0][0]         
__________________________________________________________________________________________________
conv4_block16_1_conv (Conv2D)   (None, None, None, 1 94208       conv4_block16_0_relu[0][0]       
__________________________________________________________________________________________________
conv4_block16_1_bn (BatchNormal (None, None, None, 1 512         conv4_block16_1_conv[0][0]       
__________________________________________________________________________________________________
conv4_block16_1_relu (Activatio (None, None, None, 1 0           conv4_block16_1_bn[0][0]         
__________________________________________________________________________________________________
conv4_block16_2_conv (Conv2D)   (None, None, None, 3 36864       conv4_block16_1_relu[0][0]       
__________________________________________________________________________________________________
conv4_block16_concat (Concatena (None, None, None, 7 0           conv4_block15_concat[0][0]       
                                                                 conv4_block16_2_conv[0][0]       
__________________________________________________________________________________________________
conv4_block17_0_bn (BatchNormal (None, None, None, 7 3072        conv4_block16_concat[0][0]       
__________________________________________________________________________________________________
conv4_block17_0_relu (Activatio (None, None, None, 7 0           conv4_block17_0_bn[0][0]         
__________________________________________________________________________________________________
conv4_block17_1_conv (Conv2D)   (None, None, None, 1 98304       conv4_block17_0_relu[0][0]       
__________________________________________________________________________________________________
conv4_block17_1_bn (BatchNormal (None, None, None, 1 512         conv4_block17_1_conv[0][0]       
__________________________________________________________________________________________________
conv4_block17_1_relu (Activatio (None, None, None, 1 0           conv4_block17_1_bn[0][0]         
__________________________________________________________________________________________________
conv4_block17_2_conv (Conv2D)   (None, None, None, 3 36864       conv4_block17_1_relu[0][0]       
__________________________________________________________________________________________________
conv4_block17_concat (Concatena (None, None, None, 8 0           conv4_block16_concat[0][0]       
                                                                 conv4_block17_2_conv[0][0]       
__________________________________________________________________________________________________
conv4_block18_0_bn (BatchNormal (None, None, None, 8 3200        conv4_block17_concat[0][0]       
__________________________________________________________________________________________________
conv4_block18_0_relu (Activatio (None, None, None, 8 0           conv4_block18_0_bn[0][0]         
__________________________________________________________________________________________________
conv4_block18_1_conv (Conv2D)   (None, None, None, 1 102400      conv4_block18_0_relu[0][0]       
__________________________________________________________________________________________________
conv4_block18_1_bn (BatchNormal (None, None, None, 1 512         conv4_block18_1_conv[0][0]       
__________________________________________________________________________________________________
conv4_block18_1_relu (Activatio (None, None, None, 1 0           conv4_block18_1_bn[0][0]         
__________________________________________________________________________________________________
conv4_block18_2_conv (Conv2D)   (None, None, None, 3 36864       conv4_block18_1_relu[0][0]       
__________________________________________________________________________________________________
conv4_block18_concat (Concatena (None, None, None, 8 0           conv4_block17_concat[0][0]       
                                                                 conv4_block18_2_conv[0][0]       
__________________________________________________________________________________________________
conv4_block19_0_bn (BatchNormal (None, None, None, 8 3328        conv4_block18_concat[0][0]       
__________________________________________________________________________________________________
conv4_block19_0_relu (Activatio (None, None, None, 8 0           conv4_block19_0_bn[0][0]         
__________________________________________________________________________________________________
conv4_block19_1_conv (Conv2D)   (None, None, None, 1 106496      conv4_block19_0_relu[0][0]       
__________________________________________________________________________________________________
conv4_block19_1_bn (BatchNormal (None, None, None, 1 512         conv4_block19_1_conv[0][0]       
__________________________________________________________________________________________________
conv4_block19_1_relu (Activatio (None, None, None, 1 0           conv4_block19_1_bn[0][0]         
__________________________________________________________________________________________________
conv4_block19_2_conv (Conv2D)   (None, None, None, 3 36864       conv4_block19_1_relu[0][0]       
__________________________________________________________________________________________________
conv4_block19_concat (Concatena (None, None, None, 8 0           conv4_block18_concat[0][0]       
                                                                 conv4_block19_2_conv[0][0]       
__________________________________________________________________________________________________
conv4_block20_0_bn (BatchNormal (None, None, None, 8 3456        conv4_block19_concat[0][0]       
__________________________________________________________________________________________________
conv4_block20_0_relu (Activatio (None, None, None, 8 0           conv4_block20_0_bn[0][0]         
__________________________________________________________________________________________________
conv4_block20_1_conv (Conv2D)   (None, None, None, 1 110592      conv4_block20_0_relu[0][0]       
__________________________________________________________________________________________________
conv4_block20_1_bn (BatchNormal (None, None, None, 1 512         conv4_block20_1_conv[0][0]       
__________________________________________________________________________________________________
conv4_block20_1_relu (Activatio (None, None, None, 1 0           conv4_block20_1_bn[0][0]         
__________________________________________________________________________________________________
conv4_block20_2_conv (Conv2D)   (None, None, None, 3 36864       conv4_block20_1_relu[0][0]       
__________________________________________________________________________________________________
conv4_block20_concat (Concatena (None, None, None, 8 0           conv4_block19_concat[0][0]       
                                                                 conv4_block20_2_conv[0][0]       
__________________________________________________________________________________________________
conv4_block21_0_bn (BatchNormal (None, None, None, 8 3584        conv4_block20_concat[0][0]       
__________________________________________________________________________________________________
conv4_block21_0_relu (Activatio (None, None, None, 8 0           conv4_block21_0_bn[0][0]         
__________________________________________________________________________________________________
conv4_block21_1_conv (Conv2D)   (None, None, None, 1 114688      conv4_block21_0_relu[0][0]       
__________________________________________________________________________________________________
conv4_block21_1_bn (BatchNormal (None, None, None, 1 512         conv4_block21_1_conv[0][0]       
__________________________________________________________________________________________________
conv4_block21_1_relu (Activatio (None, None, None, 1 0           conv4_block21_1_bn[0][0]         
__________________________________________________________________________________________________
conv4_block21_2_conv (Conv2D)   (None, None, None, 3 36864       conv4_block21_1_relu[0][0]       
__________________________________________________________________________________________________
conv4_block21_concat (Concatena (None, None, None, 9 0           conv4_block20_concat[0][0]       
                                                                 conv4_block21_2_conv[0][0]       
__________________________________________________________________________________________________
conv4_block22_0_bn (BatchNormal (None, None, None, 9 3712        conv4_block21_concat[0][0]       
__________________________________________________________________________________________________
conv4_block22_0_relu (Activatio (None, None, None, 9 0           conv4_block22_0_bn[0][0]         
__________________________________________________________________________________________________
conv4_block22_1_conv (Conv2D)   (None, None, None, 1 118784      conv4_block22_0_relu[0][0]       
__________________________________________________________________________________________________
conv4_block22_1_bn (BatchNormal (None, None, None, 1 512         conv4_block22_1_conv[0][0]       
__________________________________________________________________________________________________
conv4_block22_1_relu (Activatio (None, None, None, 1 0           conv4_block22_1_bn[0][0]         
__________________________________________________________________________________________________
conv4_block22_2_conv (Conv2D)   (None, None, None, 3 36864       conv4_block22_1_relu[0][0]       
__________________________________________________________________________________________________
conv4_block22_concat (Concatena (None, None, None, 9 0           conv4_block21_concat[0][0]       
                                                                 conv4_block22_2_conv[0][0]       
__________________________________________________________________________________________________
conv4_block23_0_bn (BatchNormal (None, None, None, 9 3840        conv4_block22_concat[0][0]       
__________________________________________________________________________________________________
conv4_block23_0_relu (Activatio (None, None, None, 9 0           conv4_block23_0_bn[0][0]         
__________________________________________________________________________________________________
conv4_block23_1_conv (Conv2D)   (None, None, None, 1 122880      conv4_block23_0_relu[0][0]       
__________________________________________________________________________________________________
conv4_block23_1_bn (BatchNormal (None, None, None, 1 512         conv4_block23_1_conv[0][0]       
__________________________________________________________________________________________________
conv4_block23_1_relu (Activatio (None, None, None, 1 0           conv4_block23_1_bn[0][0]         
__________________________________________________________________________________________________
conv4_block23_2_conv (Conv2D)   (None, None, None, 3 36864       conv4_block23_1_relu[0][0]       
__________________________________________________________________________________________________
conv4_block23_concat (Concatena (None, None, None, 9 0           conv4_block22_concat[0][0]       
                                                                 conv4_block23_2_conv[0][0]       
__________________________________________________________________________________________________
conv4_block24_0_bn (BatchNormal (None, None, None, 9 3968        conv4_block23_concat[0][0]       
__________________________________________________________________________________________________
conv4_block24_0_relu (Activatio (None, None, None, 9 0           conv4_block24_0_bn[0][0]         
__________________________________________________________________________________________________
conv4_block24_1_conv (Conv2D)   (None, None, None, 1 126976      conv4_block24_0_relu[0][0]       
__________________________________________________________________________________________________
conv4_block24_1_bn (BatchNormal (None, None, None, 1 512         conv4_block24_1_conv[0][0]       
__________________________________________________________________________________________________
conv4_block24_1_relu (Activatio (None, None, None, 1 0           conv4_block24_1_bn[0][0]         
__________________________________________________________________________________________________
conv4_block24_2_conv (Conv2D)   (None, None, None, 3 36864       conv4_block24_1_relu[0][0]       
__________________________________________________________________________________________________
conv4_block24_concat (Concatena (None, None, None, 1 0           conv4_block23_concat[0][0]       
                                                                 conv4_block24_2_conv[0][0]       
__________________________________________________________________________________________________
pool4_bn (BatchNormalization)   (None, None, None, 1 4096        conv4_block24_concat[0][0]       
__________________________________________________________________________________________________
pool4_relu (Activation)         (None, None, None, 1 0           pool4_bn[0][0]                   
__________________________________________________________________________________________________
pool4_conv (Conv2D)             (None, None, None, 5 524288      pool4_relu[0][0]                 
__________________________________________________________________________________________________
pool4_pool (AveragePooling2D)   (None, None, None, 5 0           pool4_conv[0][0]                 
__________________________________________________________________________________________________
conv5_block1_0_bn (BatchNormali (None, None, None, 5 2048        pool4_pool[0][0]                 
__________________________________________________________________________________________________
conv5_block1_0_relu (Activation (None, None, None, 5 0           conv5_block1_0_bn[0][0]          
__________________________________________________________________________________________________
conv5_block1_1_conv (Conv2D)    (None, None, None, 1 65536       conv5_block1_0_relu[0][0]        
__________________________________________________________________________________________________
conv5_block1_1_bn (BatchNormali (None, None, None, 1 512         conv5_block1_1_conv[0][0]        
__________________________________________________________________________________________________
conv5_block1_1_relu (Activation (None, None, None, 1 0           conv5_block1_1_bn[0][0]          
__________________________________________________________________________________________________
conv5_block1_2_conv (Conv2D)    (None, None, None, 3 36864       conv5_block1_1_relu[0][0]        
__________________________________________________________________________________________________
conv5_block1_concat (Concatenat (None, None, None, 5 0           pool4_pool[0][0]                 
                                                                 conv5_block1_2_conv[0][0]        
__________________________________________________________________________________________________
conv5_block2_0_bn (BatchNormali (None, None, None, 5 2176        conv5_block1_concat[0][0]        
__________________________________________________________________________________________________
conv5_block2_0_relu (Activation (None, None, None, 5 0           conv5_block2_0_bn[0][0]          
__________________________________________________________________________________________________
conv5_block2_1_conv (Conv2D)    (None, None, None, 1 69632       conv5_block2_0_relu[0][0]        
__________________________________________________________________________________________________
conv5_block2_1_bn (BatchNormali (None, None, None, 1 512         conv5_block2_1_conv[0][0]        
__________________________________________________________________________________________________
conv5_block2_1_relu (Activation (None, None, None, 1 0           conv5_block2_1_bn[0][0]          
__________________________________________________________________________________________________
conv5_block2_2_conv (Conv2D)    (None, None, None, 3 36864       conv5_block2_1_relu[0][0]        
__________________________________________________________________________________________________
conv5_block2_concat (Concatenat (None, None, None, 5 0           conv5_block1_concat[0][0]        
                                                                 conv5_block2_2_conv[0][0]        
__________________________________________________________________________________________________
conv5_block3_0_bn (BatchNormali (None, None, None, 5 2304        conv5_block2_concat[0][0]        
__________________________________________________________________________________________________
conv5_block3_0_relu (Activation (None, None, None, 5 0           conv5_block3_0_bn[0][0]          
__________________________________________________________________________________________________
conv5_block3_1_conv (Conv2D)    (None, None, None, 1 73728       conv5_block3_0_relu[0][0]        
__________________________________________________________________________________________________
conv5_block3_1_bn (BatchNormali (None, None, None, 1 512         conv5_block3_1_conv[0][0]        
__________________________________________________________________________________________________
conv5_block3_1_relu (Activation (None, None, None, 1 0           conv5_block3_1_bn[0][0]          
__________________________________________________________________________________________________
conv5_block3_2_conv (Conv2D)    (None, None, None, 3 36864       conv5_block3_1_relu[0][0]        
__________________________________________________________________________________________________
conv5_block3_concat (Concatenat (None, None, None, 6 0           conv5_block2_concat[0][0]        
                                                                 conv5_block3_2_conv[0][0]        
__________________________________________________________________________________________________
conv5_block4_0_bn (BatchNormali (None, None, None, 6 2432        conv5_block3_concat[0][0]        
__________________________________________________________________________________________________
conv5_block4_0_relu (Activation (None, None, None, 6 0           conv5_block4_0_bn[0][0]          
__________________________________________________________________________________________________
conv5_block4_1_conv (Conv2D)    (None, None, None, 1 77824       conv5_block4_0_relu[0][0]        
__________________________________________________________________________________________________
conv5_block4_1_bn (BatchNormali (None, None, None, 1 512         conv5_block4_1_conv[0][0]        
__________________________________________________________________________________________________
conv5_block4_1_relu (Activation (None, None, None, 1 0           conv5_block4_1_bn[0][0]          
__________________________________________________________________________________________________
conv5_block4_2_conv (Conv2D)    (None, None, None, 3 36864       conv5_block4_1_relu[0][0]        
__________________________________________________________________________________________________
conv5_block4_concat (Concatenat (None, None, None, 6 0           conv5_block3_concat[0][0]        
                                                                 conv5_block4_2_conv[0][0]        
__________________________________________________________________________________________________
conv5_block5_0_bn (BatchNormali (None, None, None, 6 2560        conv5_block4_concat[0][0]        
__________________________________________________________________________________________________
conv5_block5_0_relu (Activation (None, None, None, 6 0           conv5_block5_0_bn[0][0]          
__________________________________________________________________________________________________
conv5_block5_1_conv (Conv2D)    (None, None, None, 1 81920       conv5_block5_0_relu[0][0]        
__________________________________________________________________________________________________
conv5_block5_1_bn (BatchNormali (None, None, None, 1 512         conv5_block5_1_conv[0][0]        
__________________________________________________________________________________________________
conv5_block5_1_relu (Activation (None, None, None, 1 0           conv5_block5_1_bn[0][0]          
__________________________________________________________________________________________________
conv5_block5_2_conv (Conv2D)    (None, None, None, 3 36864       conv5_block5_1_relu[0][0]        
__________________________________________________________________________________________________
conv5_block5_concat (Concatenat (None, None, None, 6 0           conv5_block4_concat[0][0]        
                                                                 conv5_block5_2_conv[0][0]        
__________________________________________________________________________________________________
conv5_block6_0_bn (BatchNormali (None, None, None, 6 2688        conv5_block5_concat[0][0]        
__________________________________________________________________________________________________
conv5_block6_0_relu (Activation (None, None, None, 6 0           conv5_block6_0_bn[0][0]          
__________________________________________________________________________________________________
conv5_block6_1_conv (Conv2D)    (None, None, None, 1 86016       conv5_block6_0_relu[0][0]        
__________________________________________________________________________________________________
conv5_block6_1_bn (BatchNormali (None, None, None, 1 512         conv5_block6_1_conv[0][0]        
__________________________________________________________________________________________________
conv5_block6_1_relu (Activation (None, None, None, 1 0           conv5_block6_1_bn[0][0]          
__________________________________________________________________________________________________
conv5_block6_2_conv (Conv2D)    (None, None, None, 3 36864       conv5_block6_1_relu[0][0]        
__________________________________________________________________________________________________
conv5_block6_concat (Concatenat (None, None, None, 7 0           conv5_block5_concat[0][0]        
                                                                 conv5_block6_2_conv[0][0]        
__________________________________________________________________________________________________
conv5_block7_0_bn (BatchNormali (None, None, None, 7 2816        conv5_block6_concat[0][0]        
__________________________________________________________________________________________________
conv5_block7_0_relu (Activation (None, None, None, 7 0           conv5_block7_0_bn[0][0]          
__________________________________________________________________________________________________
conv5_block7_1_conv (Conv2D)    (None, None, None, 1 90112       conv5_block7_0_relu[0][0]        
__________________________________________________________________________________________________
conv5_block7_1_bn (BatchNormali (None, None, None, 1 512         conv5_block7_1_conv[0][0]        
__________________________________________________________________________________________________
conv5_block7_1_relu (Activation (None, None, None, 1 0           conv5_block7_1_bn[0][0]          
__________________________________________________________________________________________________
conv5_block7_2_conv (Conv2D)    (None, None, None, 3 36864       conv5_block7_1_relu[0][0]        
__________________________________________________________________________________________________
conv5_block7_concat (Concatenat (None, None, None, 7 0           conv5_block6_concat[0][0]        
                                                                 conv5_block7_2_conv[0][0]        
__________________________________________________________________________________________________
conv5_block8_0_bn (BatchNormali (None, None, None, 7 2944        conv5_block7_concat[0][0]        
__________________________________________________________________________________________________
conv5_block8_0_relu (Activation (None, None, None, 7 0           conv5_block8_0_bn[0][0]          
__________________________________________________________________________________________________
conv5_block8_1_conv (Conv2D)    (None, None, None, 1 94208       conv5_block8_0_relu[0][0]        
__________________________________________________________________________________________________
conv5_block8_1_bn (BatchNormali (None, None, None, 1 512         conv5_block8_1_conv[0][0]        
__________________________________________________________________________________________________
conv5_block8_1_relu (Activation (None, None, None, 1 0           conv5_block8_1_bn[0][0]          
__________________________________________________________________________________________________
conv5_block8_2_conv (Conv2D)    (None, None, None, 3 36864       conv5_block8_1_relu[0][0]        
__________________________________________________________________________________________________
conv5_block8_concat (Concatenat (None, None, None, 7 0           conv5_block7_concat[0][0]        
                                                                 conv5_block8_2_conv[0][0]        
__________________________________________________________________________________________________
conv5_block9_0_bn (BatchNormali (None, None, None, 7 3072        conv5_block8_concat[0][0]        
__________________________________________________________________________________________________
conv5_block9_0_relu (Activation (None, None, None, 7 0           conv5_block9_0_bn[0][0]          
__________________________________________________________________________________________________
conv5_block9_1_conv (Conv2D)    (None, None, None, 1 98304       conv5_block9_0_relu[0][0]        
__________________________________________________________________________________________________
conv5_block9_1_bn (BatchNormali (None, None, None, 1 512         conv5_block9_1_conv[0][0]        
__________________________________________________________________________________________________
conv5_block9_1_relu (Activation (None, None, None, 1 0           conv5_block9_1_bn[0][0]          
__________________________________________________________________________________________________
conv5_block9_2_conv (Conv2D)    (None, None, None, 3 36864       conv5_block9_1_relu[0][0]        
__________________________________________________________________________________________________
conv5_block9_concat (Concatenat (None, None, None, 8 0           conv5_block8_concat[0][0]        
                                                                 conv5_block9_2_conv[0][0]        
__________________________________________________________________________________________________
conv5_block10_0_bn (BatchNormal (None, None, None, 8 3200        conv5_block9_concat[0][0]        
__________________________________________________________________________________________________
conv5_block10_0_relu (Activatio (None, None, None, 8 0           conv5_block10_0_bn[0][0]         
__________________________________________________________________________________________________
conv5_block10_1_conv (Conv2D)   (None, None, None, 1 102400      conv5_block10_0_relu[0][0]       
__________________________________________________________________________________________________
conv5_block10_1_bn (BatchNormal (None, None, None, 1 512         conv5_block10_1_conv[0][0]       
__________________________________________________________________________________________________
conv5_block10_1_relu (Activatio (None, None, None, 1 0           conv5_block10_1_bn[0][0]         
__________________________________________________________________________________________________
conv5_block10_2_conv (Conv2D)   (None, None, None, 3 36864       conv5_block10_1_relu[0][0]       
__________________________________________________________________________________________________
conv5_block10_concat (Concatena (None, None, None, 8 0           conv5_block9_concat[0][0]        
                                                                 conv5_block10_2_conv[0][0]       
__________________________________________________________________________________________________
conv5_block11_0_bn (BatchNormal (None, None, None, 8 3328        conv5_block10_concat[0][0]       
__________________________________________________________________________________________________
conv5_block11_0_relu (Activatio (None, None, None, 8 0           conv5_block11_0_bn[0][0]         
__________________________________________________________________________________________________
conv5_block11_1_conv (Conv2D)   (None, None, None, 1 106496      conv5_block11_0_relu[0][0]       
__________________________________________________________________________________________________
conv5_block11_1_bn (BatchNormal (None, None, None, 1 512         conv5_block11_1_conv[0][0]       
__________________________________________________________________________________________________
conv5_block11_1_relu (Activatio (None, None, None, 1 0           conv5_block11_1_bn[0][0]         
__________________________________________________________________________________________________
conv5_block11_2_conv (Conv2D)   (None, None, None, 3 36864       conv5_block11_1_relu[0][0]       
__________________________________________________________________________________________________
conv5_block11_concat (Concatena (None, None, None, 8 0           conv5_block10_concat[0][0]       
                                                                 conv5_block11_2_conv[0][0]       
__________________________________________________________________________________________________
conv5_block12_0_bn (BatchNormal (None, None, None, 8 3456        conv5_block11_concat[0][0]       
__________________________________________________________________________________________________
conv5_block12_0_relu (Activatio (None, None, None, 8 0           conv5_block12_0_bn[0][0]         
__________________________________________________________________________________________________
conv5_block12_1_conv (Conv2D)   (None, None, None, 1 110592      conv5_block12_0_relu[0][0]       
__________________________________________________________________________________________________
conv5_block12_1_bn (BatchNormal (None, None, None, 1 512         conv5_block12_1_conv[0][0]       
__________________________________________________________________________________________________
conv5_block12_1_relu (Activatio (None, None, None, 1 0           conv5_block12_1_bn[0][0]         
__________________________________________________________________________________________________
conv5_block12_2_conv (Conv2D)   (None, None, None, 3 36864       conv5_block12_1_relu[0][0]       
__________________________________________________________________________________________________
conv5_block12_concat (Concatena (None, None, None, 8 0           conv5_block11_concat[0][0]       
                                                                 conv5_block12_2_conv[0][0]       
__________________________________________________________________________________________________
conv5_block13_0_bn (BatchNormal (None, None, None, 8 3584        conv5_block12_concat[0][0]       
__________________________________________________________________________________________________
conv5_block13_0_relu (Activatio (None, None, None, 8 0           conv5_block13_0_bn[0][0]         
__________________________________________________________________________________________________
conv5_block13_1_conv (Conv2D)   (None, None, None, 1 114688      conv5_block13_0_relu[0][0]       
__________________________________________________________________________________________________
conv5_block13_1_bn (BatchNormal (None, None, None, 1 512         conv5_block13_1_conv[0][0]       
__________________________________________________________________________________________________
conv5_block13_1_relu (Activatio (None, None, None, 1 0           conv5_block13_1_bn[0][0]         
__________________________________________________________________________________________________
conv5_block13_2_conv (Conv2D)   (None, None, None, 3 36864       conv5_block13_1_relu[0][0]       
__________________________________________________________________________________________________
conv5_block13_concat (Concatena (None, None, None, 9 0           conv5_block12_concat[0][0]       
                                                                 conv5_block13_2_conv[0][0]       
__________________________________________________________________________________________________
conv5_block14_0_bn (BatchNormal (None, None, None, 9 3712        conv5_block13_concat[0][0]       
__________________________________________________________________________________________________
conv5_block14_0_relu (Activatio (None, None, None, 9 0           conv5_block14_0_bn[0][0]         
__________________________________________________________________________________________________
conv5_block14_1_conv (Conv2D)   (None, None, None, 1 118784      conv5_block14_0_relu[0][0]       
__________________________________________________________________________________________________
conv5_block14_1_bn (BatchNormal (None, None, None, 1 512         conv5_block14_1_conv[0][0]       
__________________________________________________________________________________________________
conv5_block14_1_relu (Activatio (None, None, None, 1 0           conv5_block14_1_bn[0][0]         
__________________________________________________________________________________________________
conv5_block14_2_conv (Conv2D)   (None, None, None, 3 36864       conv5_block14_1_relu[0][0]       
__________________________________________________________________________________________________
conv5_block14_concat (Concatena (None, None, None, 9 0           conv5_block13_concat[0][0]       
                                                                 conv5_block14_2_conv[0][0]       
__________________________________________________________________________________________________
conv5_block15_0_bn (BatchNormal (None, None, None, 9 3840        conv5_block14_concat[0][0]       
__________________________________________________________________________________________________
conv5_block15_0_relu (Activatio (None, None, None, 9 0           conv5_block15_0_bn[0][0]         
__________________________________________________________________________________________________
conv5_block15_1_conv (Conv2D)   (None, None, None, 1 122880      conv5_block15_0_relu[0][0]       
__________________________________________________________________________________________________
conv5_block15_1_bn (BatchNormal (None, None, None, 1 512         conv5_block15_1_conv[0][0]       
__________________________________________________________________________________________________
conv5_block15_1_relu (Activatio (None, None, None, 1 0           conv5_block15_1_bn[0][0]         
__________________________________________________________________________________________________
conv5_block15_2_conv (Conv2D)   (None, None, None, 3 36864       conv5_block15_1_relu[0][0]       
__________________________________________________________________________________________________
conv5_block15_concat (Concatena (None, None, None, 9 0           conv5_block14_concat[0][0]       
                                                                 conv5_block15_2_conv[0][0]       
__________________________________________________________________________________________________
conv5_block16_0_bn (BatchNormal (None, None, None, 9 3968        conv5_block15_concat[0][0]       
__________________________________________________________________________________________________
conv5_block16_0_relu (Activatio (None, None, None, 9 0           conv5_block16_0_bn[0][0]         
__________________________________________________________________________________________________
conv5_block16_1_conv (Conv2D)   (None, None, None, 1 126976      conv5_block16_0_relu[0][0]       
__________________________________________________________________________________________________
conv5_block16_1_bn (BatchNormal (None, None, None, 1 512         conv5_block16_1_conv[0][0]       
__________________________________________________________________________________________________
conv5_block16_1_relu (Activatio (None, None, None, 1 0           conv5_block16_1_bn[0][0]         
__________________________________________________________________________________________________
conv5_block16_2_conv (Conv2D)   (None, None, None, 3 36864       conv5_block16_1_relu[0][0]       
__________________________________________________________________________________________________
conv5_block16_concat (Concatena (None, None, None, 1 0           conv5_block15_concat[0][0]       
                                                                 conv5_block16_2_conv[0][0]       
__________________________________________________________________________________________________
bn (BatchNormalization)         (None, None, None, 1 4096        conv5_block16_concat[0][0]       
__________________________________________________________________________________________________
relu (Activation)               (None, None, None, 1 0           bn[0][0]                         
__________________________________________________________________________________________________
global_average_pooling2d_2 (Glo (None, 1024)         0           relu[0][0]                       
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 14)           14350       global_average_pooling2d_2[0][0] 
==================================================================================================
Total params: 7,051,854
Trainable params: 6,968,206
Non-trainable params: 83,648
__________________________________________________________________________________________________
# Print out the total number of layers
layers_ = model.layers
print('total number of layers =',len(layers_))
total number of layers = 429
# The find() method returns an integer value:
# If substring doesn't exist inside the string, it returns -1, otherwise returns first occurence index
conv2D_layers = [layer for layer in model.layers 
                if str(type(layer)).find('Conv2D') > -1]
print('Model input -------------->', model.input)
print('Feature extractor output ->', model.get_layer('conv5_block16_concat').output)
print('Model output ------------->', model.output)
Model input --------------> Tensor("input_2:0", shape=(?, ?, ?, 3), dtype=float32)
Feature extractor output -> Tensor("conv5_block16_concat_1/concat:0", shape=(?, ?, ?, 1024), dtype=float32)
Model output -------------> Tensor("dense_2/Sigmoid:0", shape=(?, 14), dtype=float32)
Training
history = model.fit_generator(train_generator, 
                              validation_data=valid_generator,
                              steps_per_epoch=100, 
                              validation_steps=25, 
                              epochs = 1)

plt.plot(history.history['loss'])
plt.ylabel("loss")
plt.xlabel("epoch")
plt.title("Training Loss Curve")
plt.show()
WARNING:tensorflow:From C:\Users\lveys\anaconda3\envs\tensorflow1.15\lib\site-packages\keras\backend\tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/3
  8/100 [=>............................] - ETA: 14:27 - loss: 0.6813
---------------------------------------------------------------------------
KeyboardInterrupt                         Traceback (most recent call last)
<ipython-input-86-38ed1388e22c> in <module>()
      3                               steps_per_epoch=100,
      4                               validation_steps=25,
----> 5                               epochs = 3)
      6 
      7 plt.plot(history.history['loss'])

~\anaconda3\envs\tensorflow1.15\lib\site-packages\keras\legacy\interfaces.py in wrapper(*args, **kwargs)
     89                 warnings.warn('Update your `' + object_name + '` call to the ' +
     90                               'Keras 2 API: ' + signature, stacklevel=2)
---> 91             return func(*args, **kwargs)
     92         wrapper._original_function = func
     93         return wrapper

~\anaconda3\envs\tensorflow1.15\lib\site-packages\keras\engine\training.py in fit_generator(self, generator, steps_per_epoch, epochs, verbose, callbacks, validation_data, validation_steps, validation_freq, class_weight, max_queue_size, workers, use_multiprocessing, shuffle, initial_epoch)
   1730             use_multiprocessing=use_multiprocessing,
   1731             shuffle=shuffle,
-> 1732             initial_epoch=initial_epoch)
   1733 
   1734     @interfaces.legacy_generator_methods_support

~\anaconda3\envs\tensorflow1.15\lib\site-packages\keras\engine\training_generator.py in fit_generator(model, generator, steps_per_epoch, epochs, verbose, callbacks, validation_data, validation_steps, validation_freq, class_weight, max_queue_size, workers, use_multiprocessing, shuffle, initial_epoch)
    218                                             sample_weight=sample_weight,
    219                                             class_weight=class_weight,
--> 220                                             reset_metrics=False)
    221 
    222                 outs = to_list(outs)

~\anaconda3\envs\tensorflow1.15\lib\site-packages\keras\engine\training.py in train_on_batch(self, x, y, sample_weight, class_weight, reset_metrics)
   1512             ins = x + y + sample_weights
   1513         self._make_train_function()
-> 1514         outputs = self.train_function(ins)
   1515 
   1516         if reset_metrics:

~\anaconda3\envs\tensorflow1.15\lib\site-packages\tensorflow_core\python\keras\backend.py in __call__(self, inputs)
   3474 
   3475     fetched = self._callable_fn(*array_vals,
-> 3476                                 run_metadata=self.run_metadata)
   3477     self._call_fetch_callbacks(fetched[-len(self._fetches):])
   3478     output_structure = nest.pack_sequence_as(

~\anaconda3\envs\tensorflow1.15\lib\site-packages\tensorflow_core\python\client\session.py in __call__(self, *args, **kwargs)
   1470         ret = tf_session.TF_SessionRunCallable(self._session._session,
   1471                                                self._handle, args,
-> 1472                                                run_metadata_ptr)
   1473         if run_metadata:
   1474           proto_data = tf_session.TF_GetBuffer(run_metadata_ptr)

KeyboardInterrupt: 
Prediction and Evaluation
Now that we have a model, let's evaluate it using our test set. We can conveniently use the predict_generator function to generate the predictions for the images in our test set.

predicted_vals = model.predict_generator(test_generator, steps = len(test_generator))
predicted_vals.shape  # number of test samples x number of classes to predict
(420, 14)
ROC Curve and AUROC
Compute metric called the AUC (Area Under the Curve) from the ROC (Receiver Operating Characteristic) curve. ideally we want a curve that is more to the left so that the top has more "area" under it, which indicates that the model is performing better.

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

def get_roc_curve(labels, predicted_vals, generator):
    auc_roc_vals = []
    for i in range(len(labels)):
        try:
            gt = generator.labels[:, i]
            pred = predicted_vals[:, i]
            auc_roc = roc_auc_score(gt, pred)
            auc_roc_vals.append(auc_roc)
            fpr_rf, tpr_rf, _ = roc_curve(gt, pred)
            plt.figure(1, figsize=(10, 10))
            plt.plot([0, 1], [0, 1], 'k--')
            plt.plot(fpr_rf, tpr_rf,
                     label=labels[i] + " (" + str(round(auc_roc, 3)) + ")")
            plt.xlabel('False positive rate')
            plt.ylabel('True positive rate')
            plt.title('ROC curve')
            plt.legend(loc='best')
        except:
            print(
                f"Error in generating ROC curve for {labels[i]}. "
                f"Dataset lacks enough examples."
            )
    plt.savefig('ROC.png')
    plt.show()
    return auc_roc_vals
auc_rocs = get_roc_curve(labels, predicted_vals, test_generator)

print("areas under the curve : {} \n for all {} classes".format(auc_rocs,len(auc_rocs)))
areas under the curve : [0.8960540540540539, 0.7963598901098901, 0.7626343118605727, 0.7731351351351351, 0.650828677402695, 0.8193055555555555, 0.6556871078729002, 0.785787037037037, 0.802640099626401, 0.7123737854829492, 0.678, 0.7279327823188274, 0.8563243243243243, 0.7516837180607682] 
 for all 14 classes
Interpreting Deep Learning Models
Let's load in an X-ray image.

sns.reset_defaults()

def get_mean_std_per_batch(df, H=320, W=320):
    sample_data = []
    for idx, img in enumerate(df.sample(100)["Image"].values):
        path = IMAGE_DIR + img
        sample_data.append(np.array(image.load_img(path, target_size=(H, W))))

    mean = np.mean(sample_data[0])
    std = np.std(sample_data[0])
    return mean, std    

def load_image_normalize(path, mean, std, H=320, W=320):
    x = image.load_img(path, target_size=(H, W))
    x -= mean
    x /= std
    x = np.expand_dims(x, axis=0)
    return x

def load_image(path, df, preprocess=True, H = 320, W = 320):
    """Load and preprocess image."""
    x = image.load_img(path, target_size=(H, W))
    if preprocess:
        mean, std = get_mean_std_per_batch(df, H=H, W=W)
        x -= mean
        x /= std
        x = np.expand_dims(x, axis=0)
    return x

im_path = IMAGE_DIR + '00025288_001.png' 
x = load_image(im_path, train_df, preprocess=False)
plt.imshow(x, cmap = 'gray')
plt.show()

Next, let's get our predictions. Before we plug the image into our model, we have to normalize it. Run the next cell to compute the mean and standard deviation of the images in our training set.

mean, std = get_mean_std_per_batch(train_df)
Now we are ready to normalize and run the image through our model to get predictions.

labels = ['Cardiomegaly', 'Emphysema', 'Effusion', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 'Atelectasis',
              'Pneumothorax', 'Pleural_Thickening', 'Pneumonia', 'Fibrosis', 'Edema', 'Consolidation']

processed_image = load_image_normalize(im_path, mean, std)
preds = model.predict(processed_image)
pred_df = pd.DataFrame(preds, columns = labels)
pred_df.loc[0, :].plot.bar()
plt.title("Predictions")
plt.savefig('predictions.png')
plt.show()

pred_df
Cardiomegaly	Emphysema	Effusion	Hernia	Infiltration	Mass	Nodule	Atelectasis	Pneumothorax	Pleural_Thickening	Pneumonia	Fibrosis	Edema	Consolidation
0	0.049361	0.025269	0.113772	0.280401	0.475629	0.96631	0.582365	0.256831	0.216697	0.433484	0.494812	0.37037	0.279044	0.690861
We see, for example, that the model predicts Mass (abnormal spot or area in the lungs that are more than 3 centimeters) with high probability. Indeed, this patient was diagnosed with mass. However, we don't know where the model is looking when it's making its own diagnosis. To gain more insight into what the model is looking at, we can use GradCAMs.

GradCAM
GradCAM is a technique to visualize the impact of each region of an image on a specific output for a Convolutional Neural Network model. Through GradCAM, we can generate a heatmap by computing gradients of the specific class scores we are interested in visualizing.

Getting Intermediate Layers
Perhaps the most complicated part of computing GradCAM is accessing intermediate activations in our deep learning model and computing gradients with respect to the class output. Typically we'll only be extracting one of the last few. The last few layers usually have more abstract information. To access a layer, we can use model.get_layer(layer).output, which takes in the name of the layer in question. Here we are going to extract the raw output of the last convolutional layer conv5_block16_concat.

spatial_maps =  model.get_layer('conv5_block16_concat').output
print(spatial_maps)
Tensor("conv5_block16_concat_1/concat:0", shape=(?, ?, ?, 1024), dtype=float32)
Now, this tensor is just a placeholder, it doesn't contain the actual activations for a particular image. To get this we will use Keras.backend.function to return intermediate computations while the model is processing a particular input. This method takes in an input and output placeholders and returns a function. This function will compute the intermediate output (until it reaches the given placeholder) evaluated given the input. For example, if you want the layer that you just retrieved (conv5_block16_concat), you could write the following:

get_spatial_maps = K.function([model.input], [spatial_maps])
print(get_spatial_maps)
<tensorflow.python.keras.backend.GraphExecutionFunction object at 0x00000259800090F0>
We see that we now have a Function object. Now, to get the actual intermediate output evaluated with a particular input, we just plug in an image to this function:

# get an image
x = load_image_normalize(im_path, mean, std)
print(f"x is of type {type(x)}")
print(f"x is of shape {x.shape}")
x is of type <class 'numpy.ndarray'>
x is of shape (1, 320, 320, 3)
# get the 0th item in the list
spatial_maps_x = get_spatial_maps([x])[0]
print(f"spatial_maps_x is of type {type(spatial_maps_x)}")
print(f"spatial_maps_x is of shape {spatial_maps_x.shape}")
print(f"spatial_maps_x without the batch dimension has shape {spatial_maps_x[0].shape}")
spatial_maps_x is of type <class 'numpy.ndarray'>
spatial_maps_x is of shape (1, 10, 10, 1024)
spatial_maps_x without the batch dimension has shape (10, 10, 1024)
We now have the activations for that particular image, and we can use it for interpretation. The function that is returned by calling K.function([model.input], [spatial_maps]) (saved here in the variable get_spatial_maps) is sometimes referred to as a "hook", letting you peek into the intermediate computations in the model.

Getting Gradients
The other major step in computing GradCAMs is getting gradients with respect to the output for a particular class.

# get the output of the model
output_with_batch_dim = model.output
print(f"Model output includes batch dimension, has shape {output_with_batch_dim.shape}")
print(f"excluding the batch dimension, the output for all 14 categories of disease has shape {output_with_batch_dim[0].shape}")
Model output includes batch dimension, has shape (?, 14)
excluding the batch dimension, the output for all 14 categories of disease has shape (14,)
The output has 14 categories, one for each disease category, indexed from 0 to 13. Cardiomegaly is the disease category at index 0.

# Get the first category's output (Cardiomegaly) at index 0
y_category_0 = output_with_batch_dim[0][0]
print(f"The Cardiomegaly output is at index 0, and has shape {y_category_0.shape}")
The Cardiomegaly output is at index 0, and has shape ()
# Get gradient of y_category_0 with respect to spatial_maps
# The first parameter is the value you are taking the gradient of, and the second is the parameter you are taking that gradient with respect to.
gradient_l = K.gradients(y_category_0, spatial_maps)
print(f"gradient_l is of type {type(gradient_l)} and has length {len(gradient_l)}")

# gradient_l is a list of size 1.  Get the gradient at index 0
gradient = gradient_l[0]
print(gradient)
gradient_l is of type <class 'list'> and has length 1
Tensor("gradients/AddN:0", shape=(?, ?, ?, 1024), dtype=float32)
Again, this is just a placeholder. Just like for intermediate layers, we can use K.function to compute the value of the gradient for a particular input.

The K.function() takes in

a list of inputs: in this case, one input, 'model.input'
a list of tensors: in this case, one output tensor 'gradient'
It returns a function that calculates the activations of the list of tensors.

This returned function returns a list of the activations, one for each tensor that was passed into K.function().
# Create the function that gets the gradient
get_gradient = K.function([model.input], [gradient])
type(get_gradient)
tensorflow.python.keras.backend.GraphExecutionFunction
# get an input x-ray image
x = load_image_normalize(im_path, mean, std)
print(f"X-ray image has shape {x.shape}")
X-ray image has shape (1, 320, 320, 3)
The get_gradient function takes in a list of inputs, and returns a list of the gradients, one for each image.

# use the get_gradient function to get the gradient (pass in the input image inside a list)
grad_x_l = get_gradient([x])
print(f"grad_x_l is of type {type(grad_x_l)} and length {len(grad_x_l)}")

# get the gradient at index 0 of the list.
grad_x_with_batch_dim = grad_x_l[0]
print(f"grad_x_with_batch_dim is type {type(grad_x_with_batch_dim)} and shape {grad_x_with_batch_dim.shape}")

# To remove the batch dimension, take the value at index 0 of the batch dimension
grad_x = grad_x_with_batch_dim[0]
print(f"grad_x is type {type(grad_x)} and shape {grad_x.shape}")
grad_x_l is of type <class 'list'> and length 1
grad_x_with_batch_dim is type <class 'numpy.ndarray'> and shape (1, 10, 10, 1024)
grad_x is type <class 'numpy.ndarray'> and shape (10, 10, 1024)
Implementing GradCAM
def grad_cam(input_model, image, category_index, layer_name):
    """
    GradCAM method for visualizing input saliency.
    
    Args:
        input_model (Keras.model): model to compute cam for
        image (tensor): input to model, shape (1, H, W, 3)
        cls (int): class to compute cam with respect to
        layer_name (str): relevant layer in model
        H (int): input height
        W (int): input width
    Return:
        cam ()
    """
    cam = None
    
    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###

    # 1. Get placeholders for class output and last layer
    # Get the model's output
    output_with_batch_dim = input_model.output
    
    # Remove the batch dimension
    output_all_categories = output_with_batch_dim[0]
    
    # Retrieve only the disease category at the given category index
    y_c = output_all_categories[category_index]
    
    # Get the input model's layer specified by layer_name, and retrive the layer's output tensor
    spatial_map_layer = input_model.get_layer(layer_name).output

    # 2. Get gradients of last layer with respect to output

    # get the gradients of y_c with respect to the spatial map layer (it's a list of length 1)
    grads_l = K.gradients(y_c, spatial_map_layer)
    
    # Get the gradient at index 0 of the list
    grads = grads_l[0]
        
    # 3. Get hook for the selected layer and its gradient, based on given model's input
    # Hint: Use the variables produced by the previous two lines of code
    spatial_map_and_gradient_function = K.function([input_model.input], [spatial_map_layer, grads])
    
    # Put in the image to calculate the values of the spatial_maps (selected layer) and values of the gradients
    spatial_map_all_dims, grads_val_all_dims = spatial_map_and_gradient_function([image])

    # Reshape activations and gradient to remove the batch dimension
    # Shape goes from (B, H, W, C) to (H, W, C)
    # B: Batch. H: Height. W: Width. C: Channel    
    # Reshape spatial map output to remove the batch dimension
    spatial_map_val = spatial_map_all_dims[0]
    
    # Reshape gradients to remove the batch dimension
    grads_val = grads_val_all_dims[0]
    
    # 4. Compute weights using global average pooling on gradient 
    # grads_val has shape (Height, Width, Channels) (H,W,C)
    # Take the mean across the height and also width, for each channel
    # Make sure weights have shape (C)
    weights = np.mean(grads_val, axis=(0,1))
    
    # 5. Compute dot product of spatial map values with the weights
    cam = np.dot(spatial_map_val, weights)    # shape (10,10,1024) x shape(1024,) resulting into shape(10,10)

    ### END CODE HERE ###
    
    # We'll take care of the postprocessing.
    H, W = image.shape[1], image.shape[2]
    cam = np.maximum(cam, 0) # ReLU so we only get positive importance
    cam = cv2.resize(cam, (W, H), cv2.INTER_NEAREST)
    cam = cam / cam.max()

    return cam
Generate the CAM for the image
im = load_image_normalize(im_path, mean, std)
cam = grad_cam(model, im, 5, 'conv5_block16_concat') # Mass is class 5
Visualize the CAM and the original image.
plt.rcParams['figure.figsize'] = [10, 7]
plt.subplot(121)
plt.imshow(load_image(im_path, train_df, preprocess=False), cmap='gray')
plt.title("Original")
plt.axis('off')

plt.subplot(122)
plt.imshow(load_image(im_path, train_df, preprocess=False), cmap='gray')
plt.imshow(cam, cmap='magma', alpha=0.5)
plt.title("GradCAM")
plt.axis('off')
plt.show()

We can see that it focuses on the large (white) empty area on the right lung. Indeed this is a clear case of Mass.

Using GradCAM to Visualize Multiple Labels
We can use GradCAMs for multiple labels on the same image. Let's do it for the labels with best AUC for our model, Cardiomegaly, Mass, and Edema.

def compute_gradcam(model, img, mean, std, data_dir, df, 
                    labels, selected_labels, layer_name='conv5_block16_concat'):
    """
    Compute GradCAM for many specified labels for an image. 
    This method will use the `grad_cam` function.
    
    Args:
        model (Keras.model): Model to compute GradCAM for
        img (string): Image name we want to compute GradCAM for.
        mean (float): Mean to normalize to image.
        std (float): Standard deviation to normalize the image.
        data_dir (str): Path of the directory to load the images from.
        df(pd.Dataframe): Dataframe with the image features.
        labels ([str]): All output labels for the model.
        selected_labels ([str]): All output labels we want to compute the GradCAM for.
        layer_name: Intermediate layer from the model we want to compute the GradCAM for.
    """
    img_path = data_dir + img
    preprocessed_input = load_image_normalize(img_path, mean, std)
    predictions = model.predict(preprocessed_input)
    print("Ground Truth: ", ", ".join(np.take(labels, np.nonzero(df[df["Image"] == img][labels].values[0]))[0]))

    plt.figure(figsize=(20, 15))
    plt.subplot(151)
    plt.title("Original")
    plt.axis('off')
    plt.imshow(load_image(img_path, df, preprocess=False), cmap='gray')
    
    j = 1
    
    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###    
    # Loop through all labels
    for i in range(len(labels)): # complete this line
        # Compute CAM and show plots for each selected label.
        
        # Check if the label is one of the selected labels
        if labels[i] in selected_labels: # complete this line
            
            # Use the grad_cam function to calculate gradcam
            gradcam = grad_cam(model, preprocessed_input, i, layer_name)
            
            ### END CODE HERE ###
            
            print("Generating gradcam for class %s (p=%2.2f)" % (labels[i], round(predictions[0][i], 3)))
            plt.subplot(151 + j)
            plt.title(labels[i] + ": " + str(round(predictions[0][i], 3)))
            plt.axis('off')
            plt.imshow(load_image(img_path, df, preprocess=False), cmap='gray')
            plt.imshow(gradcam, cmap='magma', alpha=min(0.5, predictions[0][i]))
            j +=1
Run the following cells to print the ground truth diagnosis for a given case and show the original x-ray as well as GradCAMs for Cardiomegaly, Mass, and Edema.

image_filename = '00016650_000.png'
labels_to_show = ['Cardiomegaly', 'Mass', 'Edema']
compute_gradcam(model, image_filename, mean, std, IMAGE_DIR, train_df, labels, labels_to_show)
Ground Truth:  Cardiomegaly
Generating gradcam for class Cardiomegaly (p=0.88)
Generating gradcam for class Mass (p=0.22)
Generating gradcam for class Edema (p=0.06)

The model correctly predicts absence of mass or edema. The probability for mass is higher, and we can see that it may be influenced by the shapes in the middle of the chest cavity, as well as around the shoulder.

image_filename = '00005410_000.png'
compute_gradcam(model, image_filename, mean, std, IMAGE_DIR, train_df, labels, labels_to_show)
Ground Truth:  Mass
Generating gradcam for class Cardiomegaly (p=0.01)
Generating gradcam for class Mass (p=0.95)
Generating gradcam for class Edema (p=0.15)

In the example above, the model correctly focuses on the mass near the center of the chest cavity.

image_name = '00004090_002.png'
compute_gradcam(model, image_name, mean, std, IMAGE_DIR, train_df, labels, labels_to_show)
Ground Truth:  Edema
Generating gradcam for class Cardiomegaly (p=0.26)
Generating gradcam for class Mass (p=0.24)
Generating gradcam for class Edema (p=0.86)

Here the model correctly picks up the signs of edema near the bottom of the chest cavity. We can also notice that Cardiomegaly has a high score for this image, though the ground truth doesn't include it. This visualization might be helpful for error analysis; for example, we can notice that the model is indeed looking at the expected area to make the prediction.

image_name = '00025288_001.png'
compute_gradcam(model, image_name, mean, std, IMAGE_DIR, train_df, labels, labels_to_show)
Ground Truth:  Mass
Generating gradcam for class Cardiomegaly (p=0.04)
Generating gradcam for class Mass (p=0.97)
Generating gradcam for class Edema (p=0.30)

Here the model picks up signs of the mass near the center of the chest cavity on the right. Edema has a high score for this image, though the ground truth doesn't include it.

Interpretation tools like this one can be helpful for discovery of markers, error analysis, and even in deployment.
 