# ATLAS - Geolocation project
In this file we would explain our code briefly
## dataset visualization
This folder has our code to visualize where our images from one of the datasets are from.
It requires the images.csv of the photospheres dataset
coverage.py - creates a heatmap of where our images are from. The script uses pandas, folium and was generated by chatGPT.
## feature_extraction
This folder contain the code for the first stage of our pipeline
### data augmentation
This is the code we used to convert the panoramic images to planar images.
It requires the images.csv of the photospheres dataset
convert_panorama.py - This file has the function `panorama_to_plane`, which receive a path to a panoramic image, the FOV of the image, the output size of the image, the yaw and pitch angles of the planar image
This file was taken from https://blogs.codingballad.com/unwrapping-the-view-transforming-360-panoramas-into-intuitive-videos-with-python-6009bd5bca94
augment.py - This file convert the photosphere dataset to planar images. It requires the original dataset to be in a directory called images and saves the output to directory called big_dataset
It also creates a csv augmented_data.csv to find the data on each image.
### Hafner the rest of the folder is alll you buddy.
## filter_refinement
### filtering
This folder contains the code we used to filter the probability vector we received from the classifier based on certain filters.
it requires the big_dataset_labeled.csv and model_lr0.0005_predictions_with_prob.csv
get_clusters.py- Creates the clusters.csv which holds the center of all the clusters.
filter_test.py - This script implement our 2 filter methods and compare them by creating a quantile graph of them.
The main functions are `cluster_filter` and `deviation_filter`.
### Rest is you buddy
## moco
require the big_dataset_labeled.csv, city_images_dataset.csv, train.csv.
This folder implements the last stage of our pipeline which uses the moco model to generate the final prediction based on a set of reference images we called baseset.
get_baseset.py - This script randomly select 600 samples from each cluster and writes them into `sample_df.csv`
extract_baseset.py - Take 'sample_df.csv' and for each image it extracts its query from the `Embedder` and saves it into `baseset.csv`
test_custom_dataset.py - This holds the `CustomImageDataset` class which we used to access images in our dataset, it inherits from torch Dataset.
embedder.py - This script has both the training loop for the moco model and the `Embedder` class which is the class that applies the model on the images.
utils.py - This class has several utilities functions like the cluster filter, the prediction function based on the `baseset` and an image query, function for reading csv files and some general purpose functions.
compare_nn.py - In this function we compared the performance of the pipeline with just the classifier, classifier and the cluster filter, all the pipeline with several k values for the prediction.
We then graphed the results in 2 graphs, one on all the sample and the second on the best 70% of our sample, because we found it to be more informative.
The sample is a random 10% of the validation set.