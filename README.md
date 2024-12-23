# VMBQC

This repository contains the code related to the manuscript [_"Variational measurement-based quantum computation for generative modeling"_](https://arxiv.org/pdf/2310.13524.pdf). The article mainly talks about the applications of quantum circuits based on [Measurement-Based Quantum Computation](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.86.5188)(MBQC) in the realm of generative modeling. All the simulations are implemented using [Pennylane](https://pennylane.ai/) and [pytorch](https://pytorch.org/).

## Overview of the Code:
1. The mathematica file named "**mbqcvqc_example_publish.nb**" contains the necessary code to prove **Theorem1** in the manuscript.

2. Packages required to compile the python script: `pennylane`, `pytorch`, `jax`, `sympy`, `numba`
3. The folder generative VMBQC contains two files:


   **$(a)$** The **VMBQC_functions.py** script contains the main model of our manuscript i.e. the quantum circuit built using the theory of [Quantum Cellular Automata](https://arxiv.org/abs/2312.13185) inspired from MBQC. We refer to our [article](https://arxiv.org/pdf/2310.13524.pdf) for details.


   **$(b)$** The **8 qubits Double Gaussian.ipynb** file contains the steps of the algorithm of our manuscript starting from initializing the model, sampling bitstrings, and calculating MMD loss and gradients manually to optimize the entire model. We, again, refer to our [article](https://arxiv.org/pdf/2310.13524.pdf) for details.

   To run and reproduce similar results to our paper, one can run the above Jupyter Notebook for different settings. For instance, in **Fig.4** in our manuscript, the samples are generated from the VMBQC circuit itself. In that case, the samples (from channels) are generated using the following functions (after setting the number of qubits and depth of the circuits):
   ```

    model=VMBQC(qubits,layers,N)
   
    def get_samples_jl(runs, params):
        arr=[]
        p=params[:int(len(params)/2)] 
        t=params[len(p):] 
        
        arr.append(model.corrected_machine_f2(p,t))
        return np.concatenate(arr)
    
    

    def sample_circ(par):
        
        # usinf joblib to parallalize the process of generating sufficient samples
        results = Parallel(n_jobs = 20)(delayed(get_samples_jl)(r, par) for r in range(runs))
        s_a=list(itertools.chain.from_iterable(results))
        binary_array = np.array(s_a)
        #print(binary_array)
        powers_of_two = 2 ** np.arange(binary_array.shape[1])[::-1]
        decimal_array = np.sum(binary_array * powers_of_two, axis=1)
        decimal_list = decimal_array.tolist()
        
        return decimal_list
   ```

   Before executing the above functions make sure that you have the necessary packages installed in your machine.

   Whereas if you want to collect samples from the unitary model or the model where we do not correct byproduct at the end of the circuit then simply running the below function will do the job
   ```
    
    def sample_circ(par):
    
       binary_array = np.array(eqv_circ_st(params))
       powers_of_two = 2 ** np.arange(binary_array.shape[1])[::-1]
       decimal_array = np.sum(binary_array * powers_of_two, axis=1)
       decimal_list = decimal_array.tolist()
    
       return decimal_list
   ```

## Reproducting results
The results from the paper can be found in the folder `/Results`. Each data file contains multiple datasets which need to be averaged over in order to reproduce the plots in the paper.

To reproduce the exact results one simply needs to load the data file using ` np.load('VMBQC_learnt_by_channel_8_qubits_fig_4.npy')`, for example, and average it over the number of models. Each model is trained with either 200 (in some cases 199) epochs.

## Cite our work
If you wish to cite our git repository please use the following metadata:

```
@software{Majumder_VMBQC,
author = {Majumder, Arunava},
title = {{VMBQC}},
url = {https://github.com/ArunM10/VMBQC}
}
```

