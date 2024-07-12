# Kaggle competition - Humpback Whale Identification

Kaggle competition [link](https://www.kaggle.com/c/humpback-whale-identification/overview)

# Humpback Whale Identification

## Competition Description
In this competition, participants are challenged to build an algorithm to identify individual whales in images. You’ll analyze Happywhale’s database of over 25,000 images, gathered from research institutions and public contributors. By contributing, you’ll help to open rich fields of understanding for marine mammal population dynamics around the globe.

## Evaluation
Submissions are evaluated according to the Mean Average Precision @ 5 (MAP@5).

## My Work

### Dataset Characteristics
The dataset presents several challenges:
1. **Labeling**: The dataset includes images with labels (whale IDs) for only 62% of the images. The remaining 38% are unlabeled, marked as "new_whale." This means 9,664 images have labels, while 25,361 are labeled as "new_whale."
   
2. **Image Quality**: The images are primarily grayscale, with some in color. The standard size is 1050 x 750 pixels.

3. **Image Clarity**: The dataset contains blurry images.

4. **Single Image Whales**: There are 2,073 whales that have only one image, accounting for 41% of the dataset.

### Steps to Classify the "new_whale" Dataset
To increase the number of annotated images, I utilized the following approach:

- I leveraged a previous competition's dataset, which included almost identical images, to identify and classify known or new whale images based on similarity.
- Identification of very similar images was accomplished by creating a hash function for each image and checking the distance. The hash function employed was similar to industry standards for copyright violation prevention, with a similarity threshold of at least 90% and a Hamming distance smaller than six.

### Creation of Augmented Dataset
To focus on the whale's tail while minimizing background noise, I employed bounding boxes:
1. Each image was converted to HSV, and saturation and grayscale were analyzed.
2. In the tail region, two patterns were identified:
   - A threshold was established in the image histograms, and regions below this threshold were marked.
   - A union between two images was created, taking pixel coordinates from the union.
   - For images with high saturation and low grayscale, a threshold was determined from the external background and combined with the grayscale from the lower threshold, creating another union for pixel coordinates.

Bounding boxes smaller than 10% of the image were considered failures.

### Image Augmentation Techniques
Each image underwent three types of augmentations:
1. Flipping the image left to right and adding Gaussian blur.
2. Randomly shifting the image within a confined range (left/right and up/down).
3. Combining the two augmentation techniques mentioned above.

## The Model
The model used for this task is Inception ResNet V2. The architecture was enhanced by substituting the final layer with additional average pooling, dropout layers, and a fully connected layer. The model was built using Keras and validated on a dedicated dataset, with various hyperparameters tested. To further improve performance, Test Time Augmentation (TTA) techniques were employed.

## Files
- **project.ipynb**  - Jupyter notebook containing all the code.
- **Run_model.py**   - Script that contains the code from the notebook for easy execution.
- **Submission.csv**  - The submission file of my model.
- **Matlab folder**   - Contains bounding boxes created for test/train, along with the code and Excel files.
- **project.pdf**     - Project description document.
