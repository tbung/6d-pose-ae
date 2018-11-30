# 6D Pose Estimation with Autoencoders

[![CodeFactor](https://www.codefactor.io/repository/github/tbung/6d-pose-ae/badge)](https://www.codefactor.io/repository/github/tbung/6d-pose-ae)

## Key Idea
The idea of this mini research project is to implictly learn the 6D Pose of an object through implicit learning by the means of an Augmented Autoencoder.
The Augmented Autoencoder will have to seperate latent spaces (z1 & z2) where the first (z1) will be used to determine the 3D orientation 
and the second (z2) will be used to determine the 3D translation. 

The concept of the Augmented Autoencoder is to 



where the 3D orientation and the 3D translation are mapped in differ 
This Project is inspired by the ECCV oral Paper: Implicit 3D Orientation Learning by Martin Sundermeyer et AL.

The Augmented Autoencoder which gets introduced in this paper shows that it is possible to encoder orientation information of a body in a compressed latent space.
Our Key Idea is it to have 2 latent spaces. The first latent space will encode the 3D orientation while the second latent space encodes the x, y and z axis of the object.
This allows for more precise 6D Pose orientation. 


So ofcourse the first goal is to reimplement the experiment with the 
