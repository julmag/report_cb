---
title: Forward models in the cerebellum
author: Julian Thukral, Julien Vitay and Fred Hamker
---

**TODO:** small abstract presenting the main idea (why forward models, cerebellum, etc) before jumping into the model.

For a human body to move accuratly, movement control solely based on sensory feedback would be too slow. The body would control the movement based on where it was and not where it is, resulting in movements that for example overshooting the target. It is believed that movement control is based on internal models predicting the future state of the body (or rather the future sensory feedback). The two internal models of motor control are the inverse model, which issues a motor command given the current state and the desired state of the body and the forward model which predicts the future state of the body given the current state and an efference copy of said motor command. The comparison between the predicted and the obtained state results in the prediction error, which again is used in motor control, motor learning and as a key component in generating a Sense of Agency. 
The Sense of Agency describes the experience of controlling our own actions and through them events in the outside world. Usually we don’t question if we are the agent of our own actions, it only comes into focus by the element of surprise when there is an incongruence of intention and action outcome. 
The cerebellum is thought to be the locus of the forward model. Acting not only as predictor of movements but also as the comparator between obtained and desired/predicted state. In this work we have built a neural forward model based on the architecture of the cerebellum. The model is trained to predict the future position of the hand of a 2D planar arm going in a circular motion.  
Furthermore we tested the capabilities of the model with an experiment based on the Random Dot Task.
 
\
\
\



# Methods

## Description of the model

The proposed network is a reservoir computing model structurally inspired by the human cerebellum. Inputs from the cerebral cortex are fed through a intermediary layer into the reservoir via mossy fibers and to the output cells (projection neurons). A general hebbian algorithm (gha) layer is used as an intermediary step to decorrelate the input information before it is redirected to the reservoir. 

The reservoir consists of randomly and recurrently connected neurons. It emulates the recurrent connectivity between granule cells (excitatory) and golgi cells (inhibitory), wich is able to exhibit strong non-linear dynamics.

The activity of the reservoir is read out by a layer of purkinje cells, which in turn inhibit the projection neurons (dentate nucleus neurons). The projection neurons fire rate functions as the model output on which basis the error is calculated. Based on the error feedback from the inferior olive cells synaptic weights are adjusted between the reservoir and the purkinje cell layer. The layers of the purkinje cells, inferior olive cells and the projection neurons consist of each one neuron for the x and y coordinate.  

![**Figure 1:** Structure of the model.](img/cb_model.png){width=40%}


The model is trained to predict the next position of the hand of a 2d arm (($x_{t+1}$, $y_{t+1}$)) based on the current position (($x_{t}$, $y_{t}$)) and a movement command in form of the $\Delta$ of the joint angles ($\Delta\Theta_{elbow}$ and $\Delta\Theta_{shoulder}$ ). The base of the arm is situated at the coordinate origin. Additionally, the input contains the information about the visual displacement from the last step to the current step i.e.  ($\Delta x = x_{t} - x_{t-1}$; $\Delta y = y_{t} - y_{t-1}$). 


![**Figure 2:** 2D arm model. Source: doi:10.1109/IRIS.2017.8250090](img/arm.png){width=60%}



The visual inputs correspond to positions on the target circle and not to predictions of the model. The current position of the hand at $t+1$ is represented by the target of timestep $t$ and not by the model prediction at $t$. $\Delta\Theta_{elbow}$ and $\Delta\Theta_{shoulder}$ are calculated as the movement command from the last position on the target circle to the current position on the target circle. The same principle applies to the information about the last step i.e. $\Delta x = x_t - x_{t-1} = x_{target_t-1}-x_{t-1}$. Keep in mind that $x_t = x_{target_t-1}$. 

The error is calculated as a normalized mean-square error (MSE) based on the difference between the predicted position ($x_{t+1}$, $y_{t+1}$) and the target ($x_{target}$, $y_{target}$). 

Training is done using 100.000 circles, with 8 predictions/steps each. Each circle differentiates in the center of the circle, the radius, and the starting position of the hand in the circle. Each movement per timestep step was set to a movement of the hand of a constant 43 degrees. Thus each circle needed 8 steps for one complete circumnavigation. 

![**Figure 3:** The target moves 43 degrees in reference to the circle center. The target position of the hand at $x_{t+0} is used as the input of the current position of the hand at $x_{t+1}.](img/training_plots/input_explanation.png){width=40%}



## Equations

### Neurons

**TODO:** Make sentences and include all important equations.


The firerate of the input neurons was set to the desired input. There were six input neurons representing each of the six different input (x,y, $\Delta\Theta_{elbow}$, $\Delta\Theta_{shoulder}$, $\Delta x$, $\Delta y$).

\
\

The neurons of the gha layer can be cescribed with the following equation:

$$
\begin{aligned}r_{j} = \sum^i w_{ij}\, r_i \\\end{aligned}
$$

where $w_{ij}$ represents the general hebbian algorithm pre-trained weights between input layer and gha layer and $r_i$ is given the connected input neuron.  


\
\


The reservoir neurons follow a first-order ODEs:

$$
    \tau + \frac {dx(t)}{dt} + x(t)= \sum w^{in} \, r^{in}(t)+g
$$

$\tau = 10$ meant the dynamics in the reservoir are relativly quick and the scaling factor $g = 1$ characterizes the strentgh of the recurrent connections in the reservoir at the edge of chaos, a base prequesite for an Echo State Reservoir. The weights $w^{in}$ are set using a random uniform distribution between the $min=-0.5$ and the $max=0.5$, while $r^{in}$ is given by the firerrate of the gha pre-neuron. \
The activation function of the reservoir neurons was given as: 

$$ 
r = tanh(x(t))
$$



\


The firing rate of the the Purkinje cell layer is not dynamic and described by:

$$
\begin{aligned}r_{j} = \sum^i w_{ij}\, r_i \\\end{aligned}
$$
The weights were initialized according to a normal distribution ($m = 0$ and $\sigma = 0.1$) and updated each step using the learning rule. 

The firerate of the Inferior olive neurons wich feed the error feedback to the Purkinje Cells is calculated and set in python each step. 

\
\
The projection neurons are defined by the similar equation as the prukinje cells:
$$
\begin{aligned}r_{j} = \sum^i w^{in}_{ij}\, I_i  - \sum^i w^{purk}_{ij}\, r_i \\\end{aligned}
$$

The projection neurons recieve a copy of the input from the mossy fibres and input from the purkinjie cells. $w^{in}_{ij}$ and $w^{purk}_{ij}$ are set to $1$. The firerate of the projection neurons equals the copy of the mossy fibre input minus the purkinjie firerate. Hence the purkinjie cells don't learn to predict the new coordinates of the hand of the arm but the movement between the old coordinates and the new. 


### Synapses

**TODO:** GHA


Sanger's rule aka Generalized Hebbian Algorithm
$$
\begin{aligned}\Delta w_{ij} =  \eta \, \bigg(y_i \, x_j - y_i \sum_{k=1}^i w_{kj} \, y_k\bigg)\\\end{aligned}
$$

Same rule in Matrix form. 

$$
\begin{aligned}\Delta w_{(t)} =  \eta_{(t)} \, \bigg(y_{(t)} \, x_{(t)}^T - LT[y_{(t)} \, y_{(t)}^T] \, w_{(t)}\bigg) \\\end{aligned}
$$


Learning only happened in the synapses between the reservoir and the Purkinje cell layer and at the gha layer. 

The gha layer 

Weights are adjusted with a modified delta learning algorithm:

$$
\begin{aligned}\Delta w_{ij} =  \eta \, (r_{i} \, e_{j} - c \, w_{ij)}) \\\end{aligned}
$$


The learning rate was set to $\eta = 0.005$ . A cost parameter was added and set to $c = 0.001$. 
Each step in the circle the error, $e_j$,  was calculated in python as target - model_prediction. And fed into the prukinje cell layer via the inferior olive cells. 




## Random dot task 

In the random dot task, the movement of the dot displayed on the monitor consists of the movement of the test subject and added noise. The ratio is specified by the control level, which defines the percentage of control a test subject has after the movement of the dot. To simulate a similar task for the model, a visual display circle is first calculated, based on the target circle with added noise. The added noise is taken from the same noise array as in the dot task, an array with prerecorded pseudorandom movements. At each step, a random movement in the array is taken, normalized and added to the movement of the target circle. The control level specifies the percentage of the movement done by noise or true movement.  

**TODO** make a figure to better explain.

This visual display is fed as input into the model. i.e.:

* the current position of the effector (($x_{t}$, $y_{t})_{vd}$) is taken from the visual display and not from the target circle or the prediction of them model of the previous step.
*  $\Delta\Theta_{elbow}$ and $\Delta\Theta_{shoulder}$ are calculated for the movement from ($x_{t}$, $y_{t})_{vd}$ to the target on the target circle for current step
* $\Delta x$ and $\Delta y$ are based on the last movement by the visual display ($\Delta x = (x_t - x_{t-1})_{vd}$ ; $\Delta y = (y_t - y_{t-1})_{vd}$ )

The condition in which the model is fed with the visual display will be called model agency condition. 
The error for each circle was calculated as mean square error.  

Testing was conducted with 1000 trials without learning, starting with control level 0 and incrementing it by +0.001 each run. Each trial consists of 20 circles. Circles were created as in the training phase, differentiating in radius, circle center and starting position on the circle. 


# Results 

## Training

As can be seen in Fig. 2, the training MSE decreases rapidly during training, showing that learning the forward model of the arm is successful. 

![**Figure 2:** Training MSE of the first 100 Training Circles. MSE Circle 100 = 0.3177.](img/training_plots/training_Circle_0-100.png){ style="width: 40%; margin: auto;" }


The following video illustrates how the model performance progresses through training. **TODO: give more explanations on what should be observed.**

<video controls width=60%>
    <source src="./videos/training/training_circles_0-99999.mp4"
            type="video/mp4">
</video>


## Test trials with different control levels

Below are examplary videos of a test run. For each control level, 5 circles were run. The videos show the visual display and the movement of the model agency condition at the control levels 0.2, 0.5 and 0.7. The control level and the mse for the circle are displayed in the title. 

**TODO:** explain more what should be observed.


**Control level 0.2**

<video controls width=60%>
    <source src="./videos/test_only_ac_to_vd/test_video_cl_0.2.mp4"
            type="video/mp4">
</video>



**Control level 0.5**

<video controls width=60%>
    <source src="./videos/test_only_ac_to_vd/test_video_cl_0.5.mp4"
            type="video/mp4">
</video>


**Control level 0.7**

<video controls width=60%>
    <source src="./videos/test_only_ac_to_vd/test_video_cl_0.7.mp4"
            type="video/mp4">
</video>

## Relationship between the control level and the prediction error

Fig. 3 shows the influence of the control level on the prediction error (i.e. the MSE of the test circles raw values above, moving average below). As expected, the MSE increases as the control level decreases, signalling the discrepancy between the forward model predicting position and the noisy visual feedback. The sigmoidal shape of that relationship can be considered as a prediction of the model and confirmed experimentally.


![**Figure 3:** Test MSE of Model Agency condition to Visual Display](img/test_plots/ED_from_model_ac_to_vd_cl_0-1.png){ style="width: 40%; margin: auto;" }









































