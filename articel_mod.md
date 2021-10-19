---
title: Forward models in the cerebellum
author: Julian Thukral, Julien Vitay and Fred Hamker
---


# Model



The current network is a reservoir computing model structurally inspired by the human cerebellum. Input (mossy fibers) is fed through a intermediary layer into a reservoir and the output cells (projection neurons). A general hebbian algorithm layer is used as the intermediary to decorrelate the input information before it is redirected to the reservoir. The reservoir is read out by a feedforward layer the purkinje cells which in turn inhibit the projection neurons (dentate nucleus neurons). Synaptic weights are only adjusted between the reservoir and the purkinje cell layer. The projection neurons fire rate functions as the model output on which basis the error is calculated. The layers of the purkinje cells, inferior olive cells and the projection neurons consist of each one neuron for the x and y coordinate. 






![**Figure 1:** Structure of the model.](img/cb_model.png){width=80%}



The model is trained to predict the next position of the hand of a 2d arm ((x<sub>t+1</sub>, y<sub>t+1</sub>)) based on the current position ((x<sub>t</sub>, y<sub>t</sub>)) and a movement command in   form of the Δ of the joint angles (ΔӨ<sub>elbow</sub> and ΔӨ<sub>shoulder</sub> ) . The base of the arm is situated at the coordinate origin. Additionally the input contains the information about the Δ of the last step to the current step i.e.  (Δx = x<sub>t</sub> - x<sub>t-1</sub> ; Δy = y<sub>t</sub> - y<sub>t-1</sub>). 

Furthermore the input was based on the target circle and not on the predictions of the model i.e. the current position of of the hand at timestep t+1 is represented by the target of timestep t and not of the model prediciton of timestep t. Thus ΔӨ<sub>elbow</sub> and ΔӨ<sub>shoulder</sub> was calculated as the movement command from the last position on the target circle to the current position on the target circle. The same principle applies to the information about the last step i.e. Δx = x<sub>t</sub> - x<sub>t-1</sub> = x<sub>t</sub> - x<sub>target_t</sub> . 

The error was calculated as normalized MSE based on the difference/delta of the predicted (x<sub>t+1</sub>, y<sub>t+1</sub>) and (x<sub>target</sub>, y<sub>target</sub>).

Training is done with 100.000 Circles, with 8 predictions/steps each. Each circle differentiated in the center of the circle, the radius, and the starting position of the hand in the circle. The degrees determining the movement per timestep was constant at 43 degrees each step. Thus each circle needed 8 steps for one complete circumnavigation. 







### Equations:



Learning only happened in the synapses between the reservoir and the purkinje cell layer. Weights are adjusted with a modified delta learning algorithm:
$$
\begin{aligned}Δw_{ij} =  𝜼 * (r_{i} * error_{j} - c*w_{ij)}) \\\end{aligned}
$$

The learning rate was set to 𝜼 = 0.005 . A cost parameter was added and set to c=0.001. 

Fire rate of the the purkinje cell layer is not dynamic and described by
$$
\begin{aligned}r_{j} = \sum^i w_{ij}* r_i \\\end{aligned}
$$


Reservoir Neurons follow first-order ODEs:
$$
\tau + \frac {dx(t)}{dt} + x(t)= \sum w^{in} * r^{in}(t)+g
$$
with *τ* = 10 and g = 1 .

Fire rate of the reservoir neurons was defined as :
$$
r(t) = tanh(x(t))
$$

### Task 

In the random dot task, the movement of the dot displayed on the monitor consisted of the movement of the test subject and added noise. The ratio was specified by the control level,which defined the percentage of control a test subject had ofter the movement of the dot.  To simulate a similar task for the model a visual display circle was calculated. It was based on the a target circle with added noise. The noise added was take from the same noise array as in the dot task. Each step a random movement of the array was taken, normalized and added to the movement of the target circle. The ratio was specified by the control level.  

This visual display was fed as input into the model. i.e.:

* the current position of the effector (x<sub>t</sub>, y<sub>t</sub>)<sub>vd</sub> was taken from the visual display and not from the target circle or the prediction of them model of the 	 previous step.
* ΔӨ<sub>elbow</sub> and ΔӨ<sub>shoulder</sub> were calculated for the movement from (x<sub>t</sub>, y<sub>t</sub>)<sub>vd</sub> to the target on the target circle for current step
* $\Delta x$ and Δy were based on the last movement by the visual display (Δx = (x<sub>t</sub> - x<sub>t-1</sub>)<sub>vd</sub> ; Δy = (y<sub>t</sub> - y<sub>t-1</sub>)<sub>vd</sub> )








$\tau$









$$
\begin{aligned}
p_i(t) = \Bigg[\sum_j^{N_z} & D_{ij}\, m_{ij}\, \sum_{s=1}^t \exp\bigg({-\frac{t-s}{\tau_m}}\bigg)\, z_j(s-1) \\
+ & \sum_j^{N_i} H_{ij}\, w_{IO}\, \sum_{s=1}^t \exp\bigg({-\frac{t-s}{\tau_\text{IO}}}\bigg) \, l_j(s-1)\Bigg]^+ \\
\end{aligned}
$$

## Figures:

![**Figure 1:** Structure of the model.](img/model.svg){width=80%}

Videos from Youtube:

<div class='embed-container'><iframe src='https://www.youtube.com/embed/qhUvQiKec2U' frameborder='0' allowfullscreen></iframe></div>

Local videos:

<div class='embed-container'><iframe src='./videos/cover.mp4' frameborder='0' allowfullscreen></iframe></div>

## Tasks

## Results
 Videos

<div class='embed-container'><iframe src='./videos/training_circle_20-99999.mp4' frameborder='0' allowfullscreen></iframe></div>

![**Figure 1:** Structure of the model.](img/model.svg){width=80%}