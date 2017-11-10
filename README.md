# VQ-VAE (Neural Discrete Representation Learning) Tensorflow

## Intro

This repository implements the paper, [Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937) (VQ-VAE) in Tensorflow.

:warning: This is not an official implementation, and might have some glitch (,or a major defect).

## Requirements

- Python 3.5
- Tensorflow (v1.3 or higher)
- numpy, better_exceptions, tqdm, etc.

## Results

  All training is done with Quadro M4000 GPU. Training MNIST only takes less than 10 minutes.

- [x] MNIST

  | Original Images | Reconstructed Images |
  | ------------- |:-------------:|
  |![MNIST original images](/assets/mnist_test_original.png) | ![MNIST Reconstructed Images](/assets/mnist_test_recon.png) |

  The result on MNIST test dataset. (K=20, D=64, latent space=3 by 3)

  I also observed its latent space by changing single value for each latent space from one of the observed latent code. The result is shown below.
  ![MNIST Latent Observation](/assets/mnist_diff_codes.png)

  It seems that spatial location of latent code is improtant. By changing latent code on a specific location, the pixel matches with the location is disturbed.

  ![MNIST Latent Observation - Random Walk](/assets/mnist_randomwalk.gif)

  This results shows the 1000 generated images starting from knwon latent codes and changing aa single latent code at radnom location by +1 or -1.
  Most of the images are redundant (unrealistic), so it indicates that there are much room for compression.

  If you want to further explore the latent space, then try to play with notebook files I provided.

- [x] CIFAR 10

  | Original Images | Reconstructed Images |
  | ------------- |:-------------:|
  |![MNIST original images](/assets/cifar10_test_original.png) | ![MNIST Reconstructed Images](/assets/cifar10_test_recon.png) |

  I was able to get 4.65 bits/dims. (K=10, D=256, latent space=8 by 8)


## Training

It will download required datasets on the directory `./datasets/{mnist,cifar10}` by itself.
Hence, just run the code will do the trick.

### Run train

- Run mnist: `python mnist.py`
- Run cifar10: `python cifar10.py`

Change the hyperparameters accordingly as you want. Please check at the bottom of each script.

## Evaluation

I provide the model and the code for generating (,or reconstructing) images in the form of Jupyter notebook.
Run jupyter notebook server, then run it to see more results with provided models.

If you want to test NLL, then run `test()` function on `cifar.py` by uncomment the line. You can find it at the bottom of the file.

## TODO

- [ ] PixelCNN?
- [ ] WaveNet?

Contributions are welcome!

## Thoughts and Help request

- The results seems correct, but there is a chance that the implmentation is not perfectly correct (especially, gradient copying...). If you find any glitches (or, a major defect) then, please let me know!
- I am currently not sure how exactly NLL should be computed. Anyone who wants me a proper explantion on this?
- It is unclear from the paper how they leverage pertrained PixelCNN and WaveNet for prior(z). What does it mean?

