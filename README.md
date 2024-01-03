# VMBQC

This repository contains the code related to the manuscript [_"Variational measurement-based quantum computation for generative modeling"_](https://arxiv.org/pdf/2310.13524.pdf). The article mainly talks about the applications of quantum circuits based on [Measurement-Based Quantum Computation](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.86.5188)(MBQC) in the realm of generative modeling. All the simulations are implemented using [Pennylane](https://pennylane.ai/) and [pytorch](https://pytorch.org/).

## Overview of the Code:
1. The mathematica file named "**mbqcvqc_example_publish.nb**" contains the necessary code to prove **Theorem1** in the manuscript.
2. The folder generative VMBQC contains two files:


   **$(a)$** The **VMBQC_functions.py** script contains the main model of our manuscript i.e. the quantum circuit built using the theory of [Quantum Cellular Automata](https://arxiv.org/abs/2312.13185) inspired from MBQC. We again refer to our [article](https://arxiv.org/pdf/2310.13524.pdf) for details. 
