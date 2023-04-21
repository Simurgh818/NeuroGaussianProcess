# Adaptive Bayesian Optimization for State-Dependent Brain Stimulation

Brain stimulation has become an important treatment option for a variety of neurological and psychiatric diseases. A key challenge in improving brain stimulation is selecting the optimal set of stimulation parameters for each patient, as parameter spaces are too large for brute-force search and their induced effects can exhibit complex subject-specific behavior. To achieve greatest effectiveness, stimulation parameters may additionally need to be adjusted based on an underlying neural state, which may be unknown, unmeasurable, or challenging to quantify a priori. In this study, we first develop a simulation of a state-dependent brain stimulation experiment using rodent optogenetic stimulation data. We then use this simulation to demonstrate and evaluate two implementations of an adaptive Bayesian optimization algorithm that can model a dynamically changing response to stimulation parameters without requiring knowledge of the underlying neural state. 

[DEAP dataset](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/index.html)
["DEAP: A Database for Emotion Analysis using Physiological Signals (PDF)", S. Koelstra, C. Muehl, M. Soleymani, J.-S. Lee, A. Yazdani, T. Ebrahimi, T. Pun, A. Nijholt, I. Patras, EEE Transactions on Affective Computing, vol. 3, no. 1, pp. 18-31, 2012] (https://www.eecs.qmul.ac.uk/mmv/datasets/deap/doc/tac_special_issue_2011.pdf).

 
## Installation instructions:
```install
git clone https://github.com/Simurgh818/NeuroGaussianProcess.git
```

### Requirements
This script was developed on a Windows machine with 7 core processor and 16 GB RAM. 

## Description of scripts:
Once installed, the user can open the jupyter notebook called StateIndependent_BaO.ipynb 
 

## Results

Below plot shows ...:

![Results]()

## Contributor
[Sina Dabiri](https://github.com/Simurgh818) and Eric Cole
