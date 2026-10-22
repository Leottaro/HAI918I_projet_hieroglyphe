# 2. A Deep Learning Approach to Ancient Egyptian Hieroglyphs Classification (Glyphnet)

## Abstract

**Three well-known CNN architectures**

| Network | Main feature |
|---|---|
| **ResNet-50** | Uses residual connections to help train deeper networks. |
| **Inception-v3** | Uses different branches to process visual features at different scales. |
| **Xception** | Uses depthwise separable convolutions to handle spatial features and channel mixing separately. |

**Transfer learning**

* Training from scratch: Start with initialized weights and train the model using hieroglyph images.
* Transfer learning: Use pretrained weights, then train a classification head suited to the current classes.

**Core contribution**

Design Glyphnet specifically for hieroglyph classification: Modify an existing network architecture to create a CNN tailored to this task, named Glyphnet.

The idea behind it is:
General-purpose networks need to handle complex natural image classification tasks. For this hieroglyph classification task, could a simpler network still achieve good recognition results?

Glyphnet borrows the separable convolution design from Xception and reduces the network's complexity.

## Introduction

### Single hieroglyph classification

1. Use two labeled datasets: one public and one collected by the authors.
2. Test three existing CNNs.
3. Compare transfer learning with training from scratch.
4. Take inspiration from existing networks to design Glyphnet for the current task.
5. Compare the networks' classification performance through experiments.


## The problem of hieroglyph recognition and the datasets used


**The ancient Egyptian hieroglyphic writing system**

* Signifier: The outward form of a sign, which here means its visible shape.
* Signified: The meaning expressed by that form.

Recognizing a symbol's shape and explaining its role in a sentence are two different tasks. The classifier in this paper identifies the symbol's class; at this stage, it does not determine the symbol's linguistic function in a particular word.

### Original datasets

The authors use two datasets, called $D_1$ and $D_2$.
Each image contains a single hieroglyph, labeled with a Gardiner code.

| Item | $D_1$ | $D_2$ |
|---|---|---|
| Source | Public dataset, from photos of the interior walls of the Pyramid of Unas | Collected and labeled by the authors |
| Number of images | 4,310 | 1,310 |
| Number of classes | 172 | 48 |
| Original color format | Grayscale | RGB color |
| Original dimensions | $75\times50$ | Variable |
| Image characteristics | Fairly consistent source and physical medium | Different media, periods, and writers, including carved and painted signs |

Images in $D_1$ have a fairly similar overall style, while those in $D_2$ vary more in appearance.

The authors introduce $D_2$ to expose the model to more variation and improve generalization.

**Preprocessing of images belonging to the D2 dataset**

These images are cropped from scanned photos, and some have irregular outlines. Placing them in rectangular images leaves artificially added white areas.

Color crop → Convert to grayscale → Fit into a square while keeping the aspect ratio → Fill the artificial white areas with texture → Resize to $100\times100$

**Pseudorandom Gaussian texture** means random texture generated using a Gaussian distribution. The authors use it to fill the artificial white background so that these areas look closer to the original background around the symbol.

### Merged datasets

The authors first identify the classes shared by both datasets, then collect images of those classes from both sources.

$Y_1$ and $Y_2$ are the sets of class labels in the two datasets. The classes kept are:
$$
Y_1\cap Y_2
$$

$$
D=\{(x,y)\in D_1\cup D_2\mid y\in Y_1\cap Y_2\}
$$

The final dataset contains 4,309 images across 40 classes.

#### Class imbalance

![Class imbalance](./note_imgs/class_imbalance.png)

The number of images varies a lot between classes. Some have hundreds of images, while others have very few. This is called class imbalance, and it may make the model learn more from the classes with more samples.

### Data augmentation and dataset balancing

First, split the dataset.

| Subset | Proportion | Purpose |
|---|---:|---|
| Training set | 70% | Update the model's parameters. |
| Validation set | 15% | Monitor generalization and select a model checkpoint. |
| Test set | 15% | Evaluate the final model. |

Then apply data augmentation only to the training set, so that an original image and its transformed versions do not end up in both the training and test sets.

The authors use the following transformations:

| Transformation | Setting in this paper |
|---|---|
| Horizontal and vertical translation | Maximum 10 pixels |
| Small rotations | $\pm10^\circ$ |
| Random scaling | 0.95–1.05 |
| Horizontal flipping | Left-right mirroring |

These transformations help the model handle changes in position, angle, and size, reducing its dependence on one fixed way of presenting a symbol.


The authors do not add the same number of images to every class. Instead, they:

* Add augmented samples to classes with fewer images than a certain threshold.
* Remove some images from classes with too many samples.
* Set the maximum number of training images per class to twice the average number per class.

*Add samples to minority classes and downsample classes with too many images.*

The original training set contains 3,014 images.
After augmentation and sample count adjustments, it contains 3,670 images.
Adding horizontally flipped versions increases this to 7,340 images.

## Architectures for hieroglyph classification

### Architectures for image classification

#### ResNet-50

[ResNet-50 link](https://arxiv.org/abs/1512.03385)

Residual Network: Add a shortcut alongside a convolutional block. Intuitively, adding more convolutional layers should allow the network to learn more complex features.
But deeper networks are also harder to train. During backpropagation, gradients pass through many layers and may become very small or very large. Even without obvious vanishing gradients, simply stacking more layers can make optimization harder. ResNet's main contribution is to make deep networks easier to optimize.

The residual network architecture addresses the vanishing/exploding gradient problem.

#### Inception-v3

Send the same input through several branches.

The question it addresses can be understood as: How large an area should a convolutional block look at to extract features?

Different ranges can focus on different structures.
For hieroglyph images:

* A smaller range may focus on stroke edges and endpoints.
* A larger range may focus on combinations of strokes and local outlines.

A simple CNN often uses just one kernel size in a given layer, such as $3\times3$. Inception sends the same input through several branches at once.

Each branch extracts different features, and their outputs are combined, usually by concatenating them along the channel dimension.

What does a $1\times1$ convolution do?

After the outputs of the previous layer's four branches are concatenated, they enter the next layer's $1\times1$ branch. It reads all the input channels at each spatial position and learns how to combine them.

#### Xception

The name comes from Extreme Inception and can be understood as taking Inception's idea further.

The key concept is depthwise separable convolution.

What two things does a regular convolution do at the same time?
Suppose the input has 32 channels. To produce one output channel, a regular $3\times3$ convolution kernel:

1. Looks at a nearby $3\times3$ spatial region in each channel.
2. Combines information from the different input channels.

In other words, a regular convolution handles both:

- Spatial relationships: How neighboring pixels or features combine.
- Channel relationships: How different feature channels combine.

The separable convolution used in Xception splits these tasks into two steps.

1. Depthwise convolution: Apply a spatial convolution independently to each input channel.
2. Pointwise convolution: Then use a $1\times1$ convolution to combine information across channels and set the number of output channels.

There is no activation function between these two operations. This does not mean the entire Xception network has no ReLU.

### Proposed network: Glyphnet

For this hieroglyph classification task, the number of layers and convolutional filters can be reduced to lower computation and the risk of overfitting while maintaining classification performance.

#### The network architecture

![Glyphnet architecture](./note_imgs/Glyphnet_architecture.png)

---

Input: 100×100×1 grayscale image

$$\downarrow$$

Block 1: Two regular convolutions, outputting 64 channels

$$\downarrow$$

Block 2: Two separable convolutions, outputting 128 channels

$$\downarrow$$

Block 3: Two separable convolutions, outputting 128 channels

$$\downarrow$$

Block 4: Two separable convolutions, outputting 256 channels

$$\downarrow$$

Block 5: Two separable convolutions, outputting 256 channels

$$\downarrow$$

Block 6: A separable convolution, outputting 512 channels

$$\downarrow$$

Global Average Pooling → Dropout → Fully Connected → Softmax

---

The spatial dimensions gradually shrink while the number of channels increases. Finally, the features are converted into class probabilities.

| Parameter count | Number |
|---|---:|
| Total parameters | 498,856 |
| Trainable parameters | 494,504 |

The model has about 499,000 parameters in total, far fewer than the more than 20 million in the general-purpose networks discussed earlier.
The main reasons are:

* Fewer layers and channels.
* Extensive use of separable convolutions.
* Global average pooling to keep the classification head small.


#### Network hyperparameters

| Item | Setting |
|---|---|
| Implementation framework | Keras, TensorFlow |
| Classification loss | Categorical cross-entropy |
| Optimizer | Adam |
| Batch size | 32 |
| Initial learning rate | 0.001 |
| Learning rate schedule | Halved every 15 epochs |
| Regularization | BN, L2 regularization on the fully connected layer, Dropout |
| Dropout rate | 0.15 |
| Weight initialization | Glorot Uniform |


#### Transfer learning

Use knowledge learned from one task for another related task.

In a CNN, this knowledge mainly lies in the learned convolution kernels and other parameters.
For example, after training on many images, a model has already learned to extract visual features such as edges, textures, and shapes. For hieroglyph images, these features can be reused rather than learned all over again.

Earlier layers usually extract more basic visual features, while later layers combine them to perform the specific classification task.

When the task changes, some basic features may still be useful, but the final classification rules usually need to change.

For example, if the original model's output classes differ from the hieroglyph classes, its original output layer cannot be used directly and needs to be replaced with a classification head suited to the new task.

The approach used in this paper: Freeze most of the lower layers and replace the top classification layers.
The authors apply the following steps to the pretrained networks introduced earlier:

1. Keep most of the original network's lower layers.
2. Freeze their weights.
3. Replace the original final output layers.
4. Train the newly added classification layers using hieroglyph data.

| Part | Used in the forward pass? | Parameters updated? |
|---|---|---|
| Frozen feature extraction layers | Yes | No |
| Newly added trainable classification layers | Yes | Yes |

## Experimental results

### Network performance using transfer learning

| Network | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| ResNet-50 | **0.906** | **0.882** | **0.825** | **0.840** |
| Inception-v3 | 0.864 | 0.730 | 0.737 | 0.717 |
| Xception | 0.834 | 0.715 | 0.720 | 0.703 |

### Network performance when trained from scratch

| Network | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| ResNet-50 | 0.945 | 0.919 | 0.905 | 0.903 |
| Inception-v3 | 0.948 | 0.917 | 0.904 | 0.900 |
| Xception | 0.956 | 0.919 | 0.930 | 0.919 |
| **Glyphnet** | **0.976** | **0.975** | **0.965** | **0.968** |

Glyphnet scores highest on all four metrics.
Its accuracy is 97.6%, higher than Xception, the best general-purpose network in this group. Its precision, recall, and F1 are also the highest.

The results support the authors' view that, under this paper's experimental conditions, the smaller task-specific network outperforms the comparison models.

### Stratified 5-fold cross-validation

Stratified:
Keep the class proportions as similar as possible across the splits.

Five-fold:
Usually, divide the data into five parts. Take turns using one part for held-out evaluation and the remaining parts for training.

Later in training, the curves stabilize and the shaded areas narrow, showing that the results across these runs become more similar. Glyphnet performs fairly consistently across these data splits.

### Effect of data augmentation on the training set

The blue curve stays above the red curve later in training. The authors use this to highlight the positive effect of data augmentation.