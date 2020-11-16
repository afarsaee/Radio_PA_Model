clc
clear all
close all
%% This script generates random constellation "x" to be fed to Power Amplifier
% it sends "x" to "weblab" to measure the power amplifier outputs "y"
%% The goal is to fit the real part of data
Test_flag = 1;
if Test_flag == 1
    n_tot = 100;
else
    n_tot = 10000;
end
%% Generating "x" and measuring "y"
over_sampling = 10;
n_example = n_tot*over_sampling;
x=0.02*randconst(n_tot,1);
x=resample(x,over_sampling,1);
y=weblab(x);
%% plot the output vs input of Power Amplifier
plot(abs(x),abs(y),'.')
%% Concatenate the real and imaginary parts of "x" and "y" in a single vector
x_concatenate = zeros(size(n_tot*over_sampling*2,1));
y_concatenate = zeros(size(n_tot*over_sampling*2,1));
for i = 1:n_tot*10
   x_concatenate(1+(i-1)*2) = real(x(i));
   x_concatenate(2+(i-1)*2) = imag(x(i));
   y_concatenate(1+(i-1)*2,1) = real(y(i));
   y_concatenate(2+(i-1)*2,1) = imag(y(i));
end
%% Assume that Power Amplifier has a memory of "memory_complex_PA"
% Thus, the output at time 't' is a function of inputs at time 't, t-1,
% ..., t-10'
memory_complex_PA = 10;
x_concatenate = [zeros(1,10),x_concatenate];

%% Generate mat files for python script
% X_Input are the block of inputs  of "length memory_complex_PA+2" that is needed to
% find "y" at time 't' 
% Y_Output is the output at time 't' (real and imaginary)
X_Input  = zeros(n_example,memory_complex_PA+2);
Y_Output = zeros(n_example,2);
for i = 1:n_example
    X_Input(i,:)  = x_concatenate(1+(i-1)*2:memory_complex_PA + i*2);
    Y_Output(i,:) = y_concatenate(1+(i-1)*2:i*2)';
end
%% Write the final data
if Test_flag == 1
    save('Output_NN_Test.mat', 'Y_Output');
    save('Input_NN_Test.mat', 'X_Input');
else
    save('Output_NN.mat', 'Y_Output');
    save('Input_NN.mat', 'X_Input');
end