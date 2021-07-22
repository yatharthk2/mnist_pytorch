
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
<!--[![Stargazers][stars-shield]][stars-url]-->
[![Issues][issues-shield]][issues-url]




<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/yatharthk2/Inpainting">
    <img src="https://github.com/yatharthk2/mnist_pytorch/blob/master/readme_images/mnist.png" alt="Logo" width="800" height="500">
  </a>

  <p align="center">
    <h3 align="center">A project for efficiently merging concepts of NN and containerizing it into working container , which is deployment ready </h3>
    <br />
    <a href="https://github.com/yatharthk2/mnist_pytorch/blob/master/readme.md"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/yatharthk2/mnist_pytorch/issues">Report Bug</a>
    ·
    <a href="https://github.com/yatharthk2/mnist_pytorch/issues">Request Feature</a>
  </p>
</p>

<!-- ABOUT THE PROJECT -->
## About The Project
This project is based upon very basic building block of Neural Networks which is implementation of mnist . The implementation has 4 layer (1-input layer , 2-hidden layer , 1-softmax layer) , but what's peculiar about this project that I was able to make ubuntu based image of this project which was capable of doing following -
* Run seperate miniconda environment named 'pytorch' on the Docker based  Ubuntu image.
* The env 'pytorch' had all dependencies required for running the model (numpy ,torchaudio ,torchvision, tqdm ,argparse)
* A volume piont was also mounted to the Docker image for the ease of running of source file within the container.
* Upon running the train.py , the checkpoints would be saved in this defined volume which can be accessed without running the container through the local machine(windows in my case)
*   The model is also configuerd to load the last checkpoint and continue training in the segments (the checkpoints can be accessed from the volume). which is quite a handy feature in itself .

### Built With
1) Pytorch
2) tqdm
3) numpy
4) docker
5) argparser



<!-- GETTING STARTED -->
## Getting Started
I made this project with intention of integrating concepts of Neuralnetwork and Docker container , but since i am not a Pro user of docker environment , i wont be able to provide the Image through the DockerHub(due to file being too large) , but below are the simple steps for creating your own Docker image .
* I am assuming docker is already installed on your pc 

Step 1. Clone the repository.

Step 2. Install the dependencies inyour environment using
```sh
  pip install -r requirements.txt
  ```
Step 3. Run model.py .(It will download mnist dataset from torch ) (once in root dir)
 ```sh
  python run model.py   
  ```
Step 4. Compile docker file (once in root dir)
```sh
  docker build -t <the name you want to give to your image> .
  ```
Step 5. Run container based upon your image ()
```sh
  docker run -v <add absolute path to add volume from local host>:/root/mnist_source -ti <name of your image>
  ```
  * The volume path should be absolute  . If want to rename the volume folder name in the image then make changes in model.py before building docker image and then change image path in -v<>
  * This should make a container of image
  
### Running model from image(once inside the container - /bin/bash)
Step 1. Activate the env 
```sh
  source activate pytorch 
  ```
  * all the dependecies are alredy installed while building the image
 
Step 2. go to working dir 
```sh
  cd src
  ```
Step 3. run model file within src file of container 
* The training will start and the check points will be saved in the volume mount .

### Configuring Model.py hyperparameters
* all the HP are listed inside the file itself 
* To continue training on prrevious checkpoint , make sure the mnist_source has the previously trained weights , then run 
```sh
  python run model.py --load_model = True
  ```
<!-- CONTRIBUTING -->
## References
1. <a href="https://pytorch.org/docs/stable/index.html"><strong>Torch documentation</strong></a>
2. <a href="https://docs.docker.com/"><strong>Docker Documentation</strong></a> 
3. <a href="https://www.youtube.com/watch?v=0qG_0CPQhpg&t=2713s"><strong>Docker tutorial</strong></a> 

<!-- CONTACT -->
## Contact
* Contact Yatharth Kapadia @yatharthk2.nn@gmail.com





<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/yatharthk2/mnist_pytorch?color=red&logo=github&logoColor=green&style=flat-square
[contributors-url]: https://github.com/yatharthk2/mnist_pytorch/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/yatharthk2/mnist_pytorch?color=red&logo=github&logoColor=green&style=flat-square
[forks-url]: https://github.com/yatharthk2/Inpainting/network/members
<!--[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge-->
<!--[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers-->
[issues-shield]: https://img.shields.io/bitbucket/issues/yatharthk2/mnist_pytorch?color=red&logo=github&logoColor=green&style=flat-square
[issues-url]:https://github.com/yatharthk2/mnist_pytorch/issues

[product-screenshot]: C:\Users\yatha\OneDrive\Desktop\projects\Inpainting_project\Inpainting\train_video.gif
