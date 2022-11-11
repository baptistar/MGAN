# Monotone GANs (MGAN)

This repository contains the code for the monotone GANs generative model proposed in [Kovachki et al., 2021](https://arxiv.org/abs/2006.06755) for conditional sampling. The approach learns a generative adversarial network (GAN) on the joint space of parameters and observations of a statistical model. The block-triangular structure of the GAN permits sampling directly the conditionals of the joint distribution, in particular the posterior distribution for parameters given a specific realization of the data. The code is written in Python and uses for PyTorch to define the neural networks and perform automatic differentiation. Some of the scripts for generating training and test data are written in MATLAB.

## Authors

Ricardo Baptista (Caltech), Nikola Kovachki (NVIDIA), Bamdad Hosseini (Washington), Youssef Marzouk (MIT)

E-mails: rsb@caltech.edu, nkovachki@caltech.edu, bamdadh@uw.edu, ymarz@mit.edu
