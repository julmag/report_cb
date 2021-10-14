---
title: Forward models in the cerebellum
author: Julian Thukral, Julien Vitay and Fred Hamker
---

The cerebellum is thought to be able to learn forward models, which allow to predict the sensory consequences of planned movements and adapt behavior accordingly. Although classically considered as a feedforward structure learning in a  supervised manner, recent proposals highlighted the importance of the internal recurrent connectivity of the cerebellum to produce rich dynamics, as well as the importance of reinforcement-like mechanisms for its plasticity. Based on these models, we propose a neuro-computational model of the cerebellum using an inhibitory reservoir architecture and biologically plausible learning mechanisms based on perturbation learning. The model is trained to predict the position of a simple robotic arm after ballistic movements. Understanding how the cerebellum is able to learn forward models might allow elucidating the biological basis of model-based reinforcement learning.

## Model

Equations:

$$
\begin{aligned}
p_i(t) = \Bigg[\sum_j^{N_z} & D_{ij}\, m_{ij}\, \sum_{s=1}^t \exp\bigg({-\frac{t-s}{\tau_m}}\bigg)\, z_j(s-1) \\
+ & \sum_j^{N_i} H_{ij}\, w_{IO}\, \sum_{s=1}^t \exp\bigg({-\frac{t-s}{\tau_\text{IO}}}\bigg) \, l_j(s-1)\Bigg]^+ \\
\end{aligned}
$$

Figures:

![**Figure 1:** Structure of the model.](img/model.svg){width=80%}

Videos from Youtube:

<div class='embed-container'><iframe src='https://www.youtube.com/embed/qhUvQiKec2U' frameborder='0' allowfullscreen></iframe></div>

Local videos:

<div class='embed-container'><iframe src='./videos/cover.mp4' frameborder='0' allowfullscreen></iframe></div>

## Tasks

## Results

