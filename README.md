# Defense-GAN_Pytorch

This repository containts the Pytorch implementation for [Defense-GAN: Protecting Classifiers Against Adversarial Attacks Using Generative Models](https://arxiv.org/pdf/1805.06605.pdf), by Samangouei, P., Kabkab, M., & Chellappa, R., at ICLR 2018. 


We use [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset to test models. Also, we use [Foolbox](https://foolbox.readthedocs.io/en/latest/) to generate three different type of adversarial examples.

Adversarial Attacks

- Fast Gradient Sign Method(FGSM) : [Explaining and Harnessing Adversarial Examples
](https://arxiv.org/pdf/1412.6572.pdf)
- DeepFool(DF) : [DeepFool: A Simple and Accurate
Method to Fool Deep Neural Networks](https://arxiv.org/pdf/1511.04599.pdf)
- Saliency Map Attacks(SM) : [The Limitations of Deep Learning in Adversarial Settings](https://arxiv.org/pdf/1511.07528.pdf)


## Code Descriptions

`cifar10_train.ipynb` : train CNN model to classify CIFAR10 dataset

`cifar10_test.ipynb` : test trained CNN model into clean images and adversarial examples

`generate_adversarial_examples.ipynb` : generate adversarial examples - FGSM, DF, and SM

`train_wgan_cifar10.py` : train WGAN model

`cifar10_Defense-GAN.ipynb` : test defense-GAN algorithm against adversarial examples 

## Usage for train_wgan_cifar10.py

Examples

```
python train_wgan_cifar10.py
```

```
python defense.py --data_path data/ --iterations 20000 --deviceD 0 --deviceG 1
```

You can see more detailed arguments.

```
python train_wgan_cifar10.py -h
```

