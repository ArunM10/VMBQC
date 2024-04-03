#!/usr/bin/env python
# coding: utf-8

# In[1]:

# installing relevant packages (some of them are not used in the current code but can be used in future)

import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
import time
import torch
from jax.config import config
config.update("jax_enable_x64", True)
import jax
import jax.numpy as jnp
import pennylane as qml
from joblib import Parallel, delayed
from jax import jit
from sympy import symbols, sqrt, exp, log, sin, pi, Matrix, expand, eye, trace, collect, Mul, Add
import numba as nb
from numba import jit2
from collections import defaultdict
import itertools
from scipy.special import rel_entr


# In[2]:
# Defining the sigmoid and inverse sigmoid functions 

def sigmoid(x):
        return 1/(1+np.exp(-x))
    
def inv_sigmoid(x):
    return np.log(x/(1-x))


# In[3]:

# Function that can be used to plot the heatmap of the correction probabilities between [0,1] 

def plot_prob(p,qubits,depth):
    # p: the set of correction probabilities

    prob_weights=[]
    for j in range(qubits):
        a=[]
        for i in range(depth):
            a.append((sigmoid((p[:qubits*depth]))).reshape(depth, qubits)[i][j])
        prob_weights.append(a)

    
    x_depth = -0.5 + np.linspace(1, depth+1, depth+1)  ########
    y_depth = -0.5 + np.linspace(1, qubits+1, qubits+1) ########

    # Generate random weights for each point
    np.random.seed(42)  # Set a seed for reproducibility
    weights = prob_weights
    # Create a meshgrid of the coordinates
    X, Y = np.meshgrid(x_depth, y_depth)

    fig = plt.figure(figsize=(8, 4))

    # Plot the temperature plot
    plt.imshow(weights, cmap='hot', origin='lower', extent=[min(x_depth), max(x_depth), min(y_depth), max(y_depth)], vmin=0, vmax=1)
    plt.colorbar(label='Probabilities')
    plt.ylabel('Qubit')
    plt.xlabel('Depth')
    plt.title('Initial probabilities')
    plt.gca().invert_yaxis()
    plt.show()



##### Rules for propagating byproducts ############

u = symbols('u') # This is a global notation which will be used later 

# This is the polynomial representation of the CQCA, the eq.7 in the article: https://quantum-journal.org/papers/q-2019-05-20-142/pdf/
def t_poly():
    return Matrix([[u**(-1)+u,1],[1,0]])

# It can be used to represent any "nth" power of "t" i.e. t_poly()
def t_poly_n(n):
    t=t_poly()
    t=t**n
    return Matrix([[expand(term) for term in row] for row in (t).tolist()])

# This function will represent byproducts in each layer but one thing to must remember** is that the byproducts in the
# last layer should propagate first at the end then the 2nd last and at the end the 1st layer and after all that 
# we have to correct the byproducts at the end
def bp_rep_t(q_idx):
    # q_idx: qubit index in each layer, e.g. [0,1,4]
    arr=[]
    for i in q_idx:
        arr.append(u**i)
    return Matrix([[sum(arr)],[0]])


# How the transition operator acts on the input bitstring
def transition_act(n,q_idx):
    # q_idx: position matrix of the byproducts interms of polynomials
    # n: number of transition operators (as layers) acting on the input byproduct layer
    
    r=t_poly_n(n)*bp_rep_t(q_idx)
    return Matrix([[expand(term) for term in row] for row in (r).tolist()])



#### 

def count_distinct_elements(matrix):
    distinct_elements = set()
    for element in matrix:
        if isinstance(element, Add):
            distinct_elements.update(element.args)
        elif element != 0:
            distinct_elements.add(element)
    return distinct_elements


# In[12]:


# Here I want to use the property that for any "n" 2*(u**(n))=0 as the two byprods at same place will be cancelled

def simplify_terms(matrix):
    u = symbols('u')
    
    
    if matrix[0]==1:
        a1=[1]
    elif matrix[0]==0:
        a1=[0]
    else:
        dist_ele_r1=count_distinct_elements(Matrix([matrix[0]]))
    
        if len(dist_ele_r1)==1:
            a1=[]
            t1=matrix[0]
            
            coeff=0
            power=0
            coeff=t1.as_coeff_exponent(u)[0]%2
            power=t1.as_coeff_exponent(u)[1]
            a1.append(coeff*u**(power))
        

        else:
            a1=[]

                # first we do for the 1st row
            
            for t1 in dist_ele_r1:   #matrix[0].args:
                coeff=0
                power=0
                coeff=t1.as_coeff_exponent(u)[0]%2
                power=t1.as_coeff_exponent(u)[1]
                a1.append(coeff*u**(power))
      
    ##############################################
    
    if matrix[1]==1:
        a2=[1]
    elif matrix[1]==0:
        a2=[0]
    else:
        dist_ele_r2=count_distinct_elements(Matrix([matrix[1]]))
        if len(dist_ele_r2)==1:
            a2=[]
            t1=matrix[1]
            
            coeff=0
            power=0
            coeff=t1.as_coeff_exponent(u)[0]%2
            power=t1.as_coeff_exponent(u)[1]
            a2.append(coeff*u**(power))
        

        else:
            a2=[]

                # first we do for the 1st row
            
            for t1 in dist_ele_r2:    #matrix[1].args:
                coeff=0
                power=0
                
                coeff=t1.as_coeff_exponent(u)[0]%2
                power=t1.as_coeff_exponent(u)[1]
                a2.append(coeff*u**(power))
            

    return Matrix([[sum(a1)],[sum(a2)]])


# In[13]:


# def next_layer_bp_idx(out_pos):
    
    
#     max_idx=qubits
#     min_idx=0
    
#     # out_pos: position of the byprod after the transition function is applied i.e. transition_act()

#     # first we will extract the indices from the out_pos
#     x_pos=out_pos[0]
    
#     if x_pos==0: #### So if any row has only zero then the default index would be -1 and for that there will be no
#                  #### operations in the quantum circuits     
#         Xs=[]
        
#     elif x_pos==1:
#         Xs=[0]
#     else:
#         dist_ele_r1=count_distinct_elements(Matrix([out_pos[0]]))
#         if len(dist_ele_r1)==1:
#             x_elements=x_pos
#             x_idxs=[x_elements.as_coeff_exponent(u)[1]]
#             Xs=[(i % max_idx + max_idx) % max_idx for i in x_idxs]
#         else:
#             x_elements=x_pos.args
#             x_idxs=[i.as_coeff_exponent(u)[1] for i in x_elements]
#             Xs=[(i % max_idx + max_idx) % max_idx for i in x_idxs]
            
    
    
#     z_pos=out_pos[1]
#     if z_pos==0:
#         #print('y')
#         Zs=[]
        
#     elif z_pos==1:
#         Zs=[0]
#     else:
#         dist_ele_r2=count_distinct_elements(Matrix([out_pos[1]]))
#         if len(dist_ele_r2)==1:
#             z_elements=z_pos# .as_coeff_exponent(u)[1] is used to get the power of 'u'
#             z_idxs=[z_elements.as_coeff_exponent(u)[1]]
#             Zs=[(i % max_idx + max_idx) % max_idx for i in z_idxs]
#         else:
#             z_elements=z_pos.args# .as_coeff_exponent(u)[1] is used to get the power of 'u'
#             z_idxs=[i.as_coeff_exponent(u)[1] for i in z_elements]
#             Zs=[(i % max_idx + max_idx) % max_idx for i in z_idxs]
    
#     return Xs,Zs


# In[14]:


def next_layer_bp_idx(out_pos,qubits):
    
    
    max_idx=qubits
    min_idx=0
    
    # out_pos: position of the byprod after the transition function is applied i.e. transition_act()

    # first we will extract the indices from the out_pos
    x_pos=out_pos[0]
    
    if x_pos==0: #### So if any row has only zero then the default index would be -1 and for that there will be no
                 #### operations in the quantum circuits     
        Xs=[]
        
    elif x_pos==1:
        Xs=[0]
    else:
        dist_ele_r1=count_distinct_elements(Matrix([out_pos[0]]))
        if len(dist_ele_r1)==1:
            x_elements=x_pos
            x_idxs=[x_elements.as_coeff_exponent(u)[1]]
            Xs=[(i % max_idx + max_idx) % max_idx for i in x_idxs]
        else:
            x_elements=x_pos.args
            x_idxs=[i.as_coeff_exponent(u)[1] for i in x_elements]
            Xs=[(i % max_idx + max_idx) % max_idx for i in x_idxs]
            
    
    
    z_pos=out_pos[1]
    if z_pos==0:
        #print('y')
        Zs=[]
        
    elif z_pos==1:
        Zs=[0]
    else:
        dist_ele_r2=count_distinct_elements(Matrix([out_pos[1]]))
        if len(dist_ele_r2)==1:
            z_elements=z_pos# .as_coeff_exponent(u)[1] is used to get the power of 'u'
            z_idxs=[z_elements.as_coeff_exponent(u)[1]]
            Zs=[(i % max_idx + max_idx) % max_idx for i in z_idxs]
        else:
            z_elements=z_pos.args# .as_coeff_exponent(u)[1] is used to get the power of 'u'
            z_idxs=[i.as_coeff_exponent(u)[1] for i in z_elements]
            Zs=[(i % max_idx + max_idx) % max_idx for i in z_idxs]
    
    return Xs,Zs


# In[15]:

# rule for correction of the byproducts(bp) at the end
def cal_bp_only(q_idx,layer,layers,qubits):
    n=layers-layer-1
    # we only require the index of the bp at a specific layer and then we can add them add the end.
    matrix = transition_act(n,q_idx)

    modified_matrix = simplify_terms(matrix)
    
    return (simplify_terms(bp_from_idx(next_layer_bp_idx((modified_matrix),qubits)))) 


# In[16]:


def bp_from_idx(idxs):
    rows=idxs[0]
    cols=idxs[1]
    
    arr1=[]
    for i in rows:
        arr1.append(u**i)
        
    arr2=[]
    for i in cols:
        arr2.append(u**i)
    return Matrix([[sum(arr1)],[sum(arr2)]])



# In[17]:


# This function will calculate the indices for the qubits in each layer which will later be used to caculate 
# the byproducts at the end that needs to be corrected and the indices of layers should be used in reverse order
# i.e.

def layer_qubit_idx(dic,layers,qubits):
        
        indices = {}
        for l in range(layers):
            a = []
            for q in range(qubits):
                if dic[str(l)+' '+str(q)] == 1:
                    a.append(q)
            if a:
                indices[str(l)] = a
        return indices


# In[19]:

# Function for random sampling from a 2-D grid (qubits,depth) structure  
def random_choice_2d(probabilities,depth,qubits):
    assert probabilities.shape == (depth, qubits), 

    result = np.zeros_like(probabilities, dtype=int)
    for index, p in np.ndenumerate(probabilities):
        result[index] = np.random.choice([0, 1], p=[1-sigmoid(p), sigmoid(p)])

    return result

# Function to sample the "s" values in the 2-D grid
def replace_elements(matrix,depth,qubits):
    assert matrix.shape == (depth, qubits), 

    replacements = np.where(matrix == 0, np.random.choice([0, 1], size=(depth, qubits), p=[0.5, 0.5]), 0)

    return replacements


#@jit(nopython=True)
def calculate_positions(array):
    positions = []
    for row in array:
        row_positions = np.where(row == 1)[0].tolist()
        positions.append(row_positions)

    return positions


# In[20]:

# Model we are using with Pauli corrections at the end of the circuit
class VMBQC:
    
    def __init__(self,qubits,layers,N):
        # 
        
        self.qubits=qubits
        self.layers=layers
        self.N=N
     
    def sigmoid(self,x):
        return 1/(1+np.exp(-x))
    
    
    # Here we define a function that can output the indices where we need to have the byproducts for better training purposes
    
    def machine_f1(self,P):
        # P = corrections probs that will be used to sample whether we will correct or not 
        
        ### Note that the probs to sample C's is same for all the byproducts, later we can change that and make it
        ### different for different byproducts
    
        qubit_index=np.arange(0,self.qubits)
        layer_index=np.arange(0,self.layers)
        
        x=np.array([0,1]) # 0: We don't correct ; 1: We always correct
        
        
        
        S = np.empty((self.layers * self.qubits,), dtype=int)#[]#np.empty((self.layers , self.qubits))
        C_arr = np.empty((self.layers * self.qubits,), dtype=int)
        #print(len(C_arr))

        # Randomly sample correction values for all qubits and layers in one shot
        
        #s=time.time()
        S=replace_elements(random_choice_2d(P.reshape(self.layers,self.qubits),self.layers,self.qubits),self.layers,self.qubits)

       # e=time.time()
       # print('t2->',e-s)
        return S,C_arr
    
    
    
    
    
    
    
#     # rule for correction of the byproducts at the end
#     def cal_bp(self,q_idx,layer): # we only require the index of the bp at a specific layer and then we can add them add the end.
#         n=self.layers-layer-1

#         matrix = transition_act(n,q_idx)

#         modified_matrix = simplify_terms(matrix)

#         return next_layer_bp_idx(simplify_terms(bp_from_idx(next_layer_bp_idx((modified_matrix)))))
    
    
    
  
    
    
    # Here we define a function "corrected_machine_f2" which takes the output from "machine_f1" as the input and generate the circuit with byproducts
    
    def corrected_machine_f2(self,p,t):
        # t = Thetas in the quantum circuits
        
        
        
        b_p=self.machine_f1(p) # These are indices where the byproducts will appear 
        # the first index in b_p states the qubit index and the second one states the layer index
        
        
        
        # Defining each Periodic QCA layer
        def qca_layer(t,l):
            # Layers of CZs
            for q in range(self.qubits-1):
                qml.CZ(wires=[q,q+1])
            qml.CZ(wires=[0,self.qubits-1])
            
            # Layers of Rz and H
            for q in range(self.qubits):
                qml.RZ(t[self.qubits*l+q],wires=q)
                qml.Hadamard(wires=q)
                
        
              
        
        
        dev=qml.device("lightning.qubit", wires=self.qubits,shots=self.N)
        #@qml.qnode(dev)
        
        def qc(t):
            
            
            [qml.Hadamard(wires=q) for q in range(self.qubits)]
            #s=time.time()
            for l in range(self.layers):
                qca_layer(t,l)
                
                # Introducing byproducts based on previous samples
                for q in range(self.qubits):
                    if b_p[0][l][q]==1:
                        qml.PauliX(wires=q)
            
            
            
            #### Correcting the byproducts
            qubit_idx=calculate_positions(b_p[0])
            
            
            
            index=Matrix([[0],[0]])
            
            
            # The rest of the part below calculates the propagation of the byproducts to the end of the circuit and the correcting them at the end
            
            for l in range(self.layers-1,-1,-1):
                
                if len(qubit_idx[l])!=0:
                    
                    index+=(cal_bp_only(qubit_idx[l],l,self.layers,self.qubits))
                    
                else:
                    continue
                    
             
            
            
            
            index=next_layer_bp_idx(simplify_terms(index),self.qubits)
            
            
            
            x_idx=index[0]
                    
            if len(x_idx)!=0:
                #print('y')
                [qml.PauliX(wires=q) for q in x_idx]
            z_idx=index[1]

            if len(z_idx)!=0:
                #print('y')
                [qml.PauliZ(wires=q) for q in z_idx]
            #e=time.time()
            #print('t1->',e-s,'sec')
            
            
            return qml.sample()
        
        
        
        qnode1 = qml.QNode(qc, dev)
        merged_circuit = qml.transforms.cancel_inverses(qc)
        qml.transforms.commutation_dag
        qnode2 = qml.QNode(merged_circuit, dev)
        
        
        return qnode2(t)#,qml.draw_mpl(qnode2)(t),b_p#,b_p#,qml.draw_mpl(qnode1)(t)#,b_p#,qnode2(t),qml.draw_mpl(qnode2)(t),b_p
        


