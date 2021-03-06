# MNIST-Drawer & Generator

![MNIST-Drawer](https://robertfoerster.com/images/MNIST-Drawer.png)

# GANs
<img src="https://raw.githubusercontent.com/foersterrobert/MNIST-Drawer/master/GANs/assets/gan.gif" width="80%" style="display: block; margin: auto;"/>

## Fundamentals of GANs
#### When building a Generative Adversarial Network we have two models competing with each other. One is called a Generator and the other a Discriminator. In our case, the Generator creates digits while the Discriminator compares them with real MNIST-digits and tries to decide whether they are fake or not. Both models are trained in parallel and continuously play a min-max game.

### Discriminator loss
![DISC](GANs/assets/discriminatorloss.JPG)

#### The loss function of the Discriminator which it tries to maximize looks like this, where z is a random noise created with PyTorch. D(x) can be seen as lossReal and D(G(z)) as lossFake.
#### The output of our discriminator goes through a sigmoid function. Thus its results will be clamped between 0 and 1.
![sigmoid](GANs/assets/sigmoid.png)
![sigmoid-graph](GANs/assets/sigmoidGraph.png)

#### Based on our context, the outcome of our Discriminator should be the probability of a Digit actually being part of the MNIST-Dataset. 

### Generator loss

![GEN](GANs/assets/generatorloss.jpg)

#### The loss function of the discriminator basically is the latter part of the first equation we saw earlier. But now it should be minimized.

![GEN1](GANs/assets/generatorloss1.jpeg)

#### Instead, we can also try maximizing this function right here, in order to remove the problem of a saturating loss when converging to zero.

### DCGANs
#### DeepConvolutional-GANs make our algorithm more robust for generating images by turning our models into Convolutional Neural Networks.

### CGANs
#### Conditional-GANs allow us to generate Images with a certain value or in our case a certain digit.

### WGANs
#### Wasserstein-GANs change our Discriminator to a Critic meaning the output will be a score compared to a probability. This creates further stabilisation in training and makes it easier for us to estimate progress.
