<div id="top"></div>

<!-- TABLE OF CONTENTS -->
<summary>Table of Contents</summary>
<ol>
    <li>
        <a href="#about-the-project">About The Project</a>
        <ul>
            <li><a href="#folder-structure">Folder Structure</a></li>
            <li><a href="#understanding-datasets">Understanding Datasets</a></li>
            <li><a href="#getting-started">Getting Started</a></li>
        </ul>
    </li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
</ol>


<!-- ABOUT THE PROJECT -->
# About The Project

This project is for the Udacity's AI Programming with Python Nanodegree program. 

The assignment is broken down to 2 parts
1. Jupyter Notebook Image Classifier
2. Command Line Image Classifier

<br/>

**Part 1**
> You'll work through a Jupyter notebook to implement an image classifier with PyTorch

<br/>

**Part 2**
> Now that you've built and trained a deep neural network on the flower data set, it's time to convert it into an application that others can use.
<br/><br/>
The first file, train.py, will train a new network on a dataset and save the model as a checkpoint. The second file, predict.py, uses a trained network to predict the class for an input image. Feel free to create as many other files as you need. 

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- Folder Structure -->
## Folder Structure

The following are the files found in the workspace

```
.
├── cat_to_name.json                   
├── checkpoint_constants.py             
├── flowers -> ../../../data/flowers   
├── Image Classifier Project.ipynb       
├── image_file.py                       
├── image_folder.py                     
├── load_checkpoint.py                  
├── model_architectures.py              
├── model_classifier.py                 
├── model_criterion.py                  
├── model_optimizer.py                  
├── model_prediction.py                 
├── model_training.py                   
├── predict.py                          
├── README.md                           
├── save_checkpoint.py                   
├── screenshots                         
├── time_helper.py                      
├── train.py                            
└── workspace_utils.py                  
```

|#|File/Folder|Description|
|---|---|---|
|1|cat_to_name.json|File containing the folder name to flower name mapping|
|2|checkpoint_constants.py|Python script containing the fields to be saved in checkpoint|
|3|flowers|Folder containing the images to use for training, testing and validating of the model|
|4|Image Classifier Project.ipynb|Jupyter Notebook of the image classifier|
|5|image_file.py|Python script to facilitate the loading and transforming images for prediction|
|6|image_folder.py|Python script to facilitate the loading and transforming images from folders for training|
|7|load_checkpoint.py|Python script to load a model from a checkpoint|
|8|model_architectures.py|Python script containing the architectures that are currently supported|
|9|model_classifier.py|Python script to create the classifier network required by the model|
|10|model_criterion.py|Python script to generate the criterion required for training the model|
|11|model_optimizer.py|Python script to generate the optimizer required for training the model|
|12|model_prediction.py|Python script used to run a prediction on an image|
|13|model_training.py|Python script used to train a model for prediction the images|
|14|predict.py|The main Python script for prediction of an image|
|15|README.md|Contains the information about the project|
|16|save_checkpoint.py|Python script to save a model to a checkpoint|
|17|screenshots|Folder containing the images required by the README.md|
|18|time_helper.py|Python script which provide helper methods for time-related functions|
|19|train.py|The main Python script for training the model|
|20|workspace_utils.py|Python script which provide helper methods to keep the session alive|

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- understanding-datasets -->
## Understanding Datasets

The datasets is broken down into 3 parts
1. train - use for training of the model
2. test  - use for testing of the model during training
3. valid - use for testing of the model after training

Example:
```
flowers/train/1
flowers/test/1
flowers/valid/1
```

<br/>

There are a total of `102` categories for the flower datasets and each folder name represent a flower type. 

Example:
```
flowers/train/1
flowers/train/2
flowers/train/3
```

The folder name to flower mapping is stored in the `cat_to_name.json` file

Example:
```json
{
    "1": "pink primrose",
    "2": "hard-leaved pocket orchid", 
    "3": "canterbury bells"
}
```

The following show an example of an image:
/flowers/valid/1/image_06755.jpg

![Flower Image Sample][flower-sample-resource]
<br/>

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

First we have to run `train.py` to train the models on the training dataset

```python
python3 train.py data_directory
```

The following is some of the available options for the `train.py`
|parameters|description|example
|---|---|---|
|save_dir|Model checkpoint|--save_dir save_directory|
|--arch|Model architecture|--arch "vgg13"|
|--learning_rate|Learning Rate of the model|--learning_rate 0.01|
|--hidden_units|Hidden Units for the model|--hidden_units 512|
|--epochs|Training cycles for the model|--epochs 20|
|--gpu|Enable GPU for the training|--gpu|
<br/>

The following is a sample of the execution of the `train.py`
![Training Progress][training-model-resource]
<br/>

After the model is created, we can then proceed to execute `predict.py` to run the prediction on the test dataset

```python
python3 predict.py /path/to/image checkpoint
```

The following is some of the available options for the `predict.py`
|parameters|description|example
|---|---|---|
|--top_k|Return top K most likely classes|--top_k 3|
|--category_names|Map category name from provided json|--category_names cat_to_name.json|
|--gpu|Enable GPU for the inference|--gpu|
<br/>

The following is a sample of the execution of the `predict.py`
![Prediction Result][prediction-model-resource]
<br/>

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

Here are some resources that have helped in completing this project!

* [Readme Template][readme-template-url]
* [Pytorch Transformation][pytorch-transformation-url]
* [Pytorch Models][pytorch-models-url]
* [PyTorch Load And Save Models][pytorch-save-load-models-url]
* [PIL Resize Image][pil-rezie-image-url]
* [PIL Cropped Image][pil-crop-image-url]
* [PIL Image to Tensor][pil-convert-image-to-tensor-url]
* [Matplotlib Subplots][matplotlib-subplots-url]
* [Matplotlib Horizontal Bar][matplotlib-horizontal-bar-url]
* [Matplotlib Centre Image][matplotlib-centre-images-url]

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[flower-sample-resource]: ./screenshots/image_06755.jpg
[training-model-resource]: ./screenshots/train_model_%231.png
[prediction-model-resource]: ./screenshots/predict_%231.png
[readme-template-url]: https://github.com/othneildrew/Best-README-Template/blob/master/README.md
[pytorch-transformation-url]: https://pytorch.org/vision/0.13/auto_examples/plot_transforms.html#sphx-glr-auto-examples-plot-transforms-py
[pytorch-models-url]: https://pytorch.org/docs/0.2.0/torchvision/models.html
[pytorch-save-load-models-url]: https://pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html
[pil-rezie-image-url]: https://pythonprogramming.altervista.org/resize-images-with-pil-2/
[pil-crop-image-url]: https://stackoverflow.com/questions/16646183/crop-an-image-in-the-centre-using-pil
[pil-convert-image-to-tensor-url]: https://discuss.pytorch.org/t/pytorch-pil-to-tensor-and-vice-versa/6312/9
[matplotlib-subplots-url]: https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_demo.html
[matplotlib-horizontal-bar-url]: https://matplotlib.org/stable/gallery/lines_bars_and_markers/barh.html
[matplotlib-centre-images-url]: https://stackoverflow.com/questions/18380168/center-output-plots-in-the-notebook 