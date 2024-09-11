# GenuineEye: RealFake Classifier

## Group Members
- Atharva Jaiswal (229309131)
- Khushi Bansal (229309142)
- Lakshya Mehta (229309175)
- Yashashvi Sarran (229309119)

## Project Overview
GenuineEye is a machine learning project aimed at developing a classifier to distinguish between real and fake images. The project involves several steps including data gathering, preprocessing, model development, hypertuning, and performance evaluation.

## Project Flow
1. Gathering Data
2. Preprocessing Data
3. Model Development & Hypertuning
4. Performance Evaluation

![Project Flow](images/project\ flow.png)

## Roles and Responsibilities
- **Yashashvi Sarran (229309119)**: Gathering Data
- **Khushi Bansal (229309142)**: Preprocessing Data
- **Atharva Jaiswal (229309131)**: Model Development & Hypertuning
- **Lakshay Mehta (229309175)**: Performance Evaluation

## Data Collection
### Choosing from Potential Datasets
- **High Resolution Images**:
  - Advantages: High resolution
  - Disadvantages: Not enough images
- **Abundant Image Dataset**:
  - Advantages: Abundant images, formatted
  - Disadvantages: Low resolution

### Final Choice
We selected a dataset with a substantial volume of data (120,000 images) to optimize the performance of our neural networks. The large dataset helps in learning diverse patterns, mitigating overfitting, and improving the model's robustness.

## Setting Up the Environment
### Loading Libraries
- **TensorFlow**: For neural network tasks
- **os**: For directory navigation
- **OpenCV (cv2)**: For advanced image processing
- **Imghdr**: For reading various image file extensions
- **Numpy**: For numeric computation and matrix algebra
- **Matplotlib**: For plotting images

### Memory Constraints and GPU Configuration
Implemented to avoid `OutOfMemoryError`.

## Preprocessing
- Upscaling images to enhance resolution.
- Splitting data into training (70%), validation (20%), and test (10%) sets.

## Model Development
### Sequential CNN
- **Vanishing Gradient Problem**: Mitigated using ReLU (Rectified Linear Unit) activation.
- **ReLU Activation**: Introduces non-linearity, helps deal with the vanishing gradient problem.
- **Sigmoid Activation**: Ensures output is between 0 and 1.

### Training and Hyperparameter Tuning
- Logging the training process using TensorBoard.
- Using binary cross-entropy loss for binary classification.

## Performance Evaluation
### Metrics
- **Accuracy**: Proportion of correctly classified examples.
- **Precision**: Proportion of true positives out of all predicted positives.
- **Recall**: Proportion of true positives out of all actual positives.

### Results
- **Precision**: 85%
- **Recall**: 86.73%
- **Accuracy**: 85.41%

## Visualization
- Accuracy and validation accuracy curves during training.
- Loss and validation loss curves.

## References
- [Accuracy, Precision, Recall](https://www.evidentlyai.com/classification-metrics/accuracy-precision-recall)
- [Neural Network Basics: Loss and Cost Functions](https://medium.com/artificialis/neural-network-basics-loss-and-cost-functions-9d089e9de5f8)
- [Understanding Loss Function in Deep Learning](https://www.analyticsvidhya.com/blog/2022/06/understanding-loss-function-in-deep-learning/)
