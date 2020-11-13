#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import required packages
import tensorflow as tf
from tensorflow import keras
import numpy as np
import scipy.io


# In[2]:


# read Power Amplifier input data
In_file = 'Input_NN.mat'
In_x = scipy.io.loadmat(In_file)

# X_Input has "n_examples" rows, and each row is a vector of "memory_complex_PAs+current_input" length
x_input = np.asarray(In_x['X_Input'])
n_examples = len(x_input)
memory_complex_PAs_plus_current_input = x_input[0].shape

# it is assumed the PA has memory_complex_PAs memory, 
    #thus the output of PA at time t is a function of inputs from time [t:t-memory_complex_PAs]
    #memory_complex_PAs = 2 * memory_real_PAs, it means we need previous " memory_complex_PAs/2" complex inputs of PA

# normalize input for a better training of NN
mean_np=np.mean(x_input,axis=None)
max_np=np.max(np.abs(x_input),axis=None)
x_input=(x_input-mean_np)/(max_np)
X_Train = x_input


# In[3]:


# read Outputs of PA at time t
Out_file = 'Output_NN.mat'
Out_y = scipy.io.loadmat(Out_file)

# Y_Output has n_examples rows, and each row is the current output of the PA, which has real and imaginary parts
y_output = np.asarray(Out_y['Y_Output'])

mean_np_y=np.mean(y_output,axis=None)
std_np_y=np.max(np.abs(y_output),axis=None)
y_output=(y_output-mean_np_y)/(std_np_y)

Y_Train = y_output


# In[4]:


# the FC NN has the following layers
# input  layer
input_shape  = memory_complex_PAs_plus_current_input
# hidden layers
layer1_nodes = 64
layer2_nodes = 16
layer3_nodes = 8
# output layer
N_out_layer  = 2


# In[5]:


# Defining a model for traning
def create_model():
# input layer/one hidden layer/output layer/
    model = keras.Sequential([
    keras.layers.Flatten(input_shape=input_shape),
    keras.layers.Dense(layer1_nodes, activation='tanh'),                                                    
    keras.layers.Dense(layer2_nodes, activation='tanh'),                                                                                                       
    keras.layers.Dense(layer3_nodes, activation='tanh'), 
    keras.layers.Dense(N_out_layer,  activation='tanh'),
    ])    

    # setup the optimizer
    adm = tf.keras.optimizers.Adam(lr=0.001)
    # compile the model
    model.compile(optimizer=adm,loss = 'mse',)
    return model


# In[6]:


# Create a basic model instance
model = create_model()

# summary of the model
model.summary()

# Model training
epo=20
b_size=25
model.fit(X_Train, Y_Train, validation_split = 0.05, epochs=epo, batch_size=b_size)


# In[7]:


# read test input data
In_file = 'Input_NN_Test.mat'
In_x = scipy.io.loadmat(In_file)

# X_Input has n_examples rows, and each row is a vector of "memory_complex_PAs+current_input" length
x_input_Test = np.asarray(In_x['X_Input'])
n_examples_Test = len(x_input_Test)

# normalize input
x_input_Test=(x_input_Test-mean_np)/(max_np)

X_Test = x_input_Test

# find the estimated output from the model
Y_hat = model.predict(X_Test)


# In[8]:


# read test Outputs of PA
Out_file = 'Output_NN_Test.mat'
Out_y = scipy.io.loadmat(Out_file)

# Y_Output has n_examples rows, and each row is the current output of the PA
y_output_Test = np.asarray(Out_y['Y_Output'])

# normalize measured output data
y_output_Test=(y_output_Test-mean_np_y)/(std_np_y)

Y_Test = y_output_Test


# In[9]:


# find the MSE of NN and the measured data
MSE_Model = np.square(Y_hat-Y_Test).mean(axis = None)


# In[10]:


# find the absolute value of the NN output
Y_real_hat = Y_hat[:,0]
Y_imag_hat = Y_hat[:,1]
Y_abs_hat = np.sqrt(np.square(Y_imag_hat) + np.square(Y_real_hat))

# find the absolute value of the measured data
Y_real = Y_Test[:,0]
Y_imag = Y_Test[:,1]
Y_abs_Test = np.sqrt(np.square(Y_imag) + np.square(Y_real))


# In[12]:


# plot the measured absolute value of the output over the estimated abs
import matplotlib.pyplot as plt
plt.plot(np.abs(Y_abs_hat),np.abs(Y_abs_Test),'.')
plt.ylabel('Measurements')
plt.xlabel('Estimation')
plt.title('Abs output: measurements vs. estimated')
plt.show()


# In[14]:


# plot the spectrum of the estimated NN output
Y_hat_complex = Y_real_hat + 1j * Y_imag_hat
Y_hat_complex = np.array(Y_hat_complex)
fft_y_hat = np.fft.fftshift(np.fft.fft(Y_hat_complex))
fft_y_hat_log10 = 10*np.log10(fft_y_hat)
plt.plot(np.abs(fft_y_hat_log10))

# find the ACPR of the NN
fft_abs_channel = np.sum(np.square(np.abs(fft_y_hat[4500:5500])))
a = fft_y_hat[1:4499]
a = np.append(a,fft_y_hat[5501:])
fft_abs_adj     = np.sum(np.square(np.abs(a)))
ACLR = fft_abs_adj/fft_abs_channel
print(ACLR)


# In[15]:


# plot the spectrum of the PA
Y_complex_meas = Y_real + 1j * Y_imag
Y_complex_meas = np.array(Y_complex_meas)
fft_y_meas = np.fft.fftshift(np.fft.fft(Y_complex_meas))
fft_y_meas_log10 = 10*np.log10(fft_y_meas)
plt.plot(np.abs(fft_y_meas_log10))

# find the ACPR of the PA
fft_abs_channel = np.sum(np.square(np.abs(fft_y_meas[4500:5500])))
a = fft_y_meas[1:4499]
a = np.append(a,fft_y_meas[5501:])
fft_abs_adj     = np.sum(np.square(np.abs(a)))
ACLR = fft_abs_adj/fft_abs_channel
print(ACLR)


# In[ ]:




