import numpy as np #Python library

#Simple Feed Forward Neural Network

def sigmoid(x, deriv=False):
 if(deriv==True): #If we pass in true we calcultate if not no.
        return (x*(1-x))

 return 1/(1+np.exp(-x))

##Hardcoded input data
x = np.array([[0,0,1],
 [0,1,1],
 [1,0,1],
 [1,1,1]])

##Each of these is associated with on the lines
y = np.array([[0],
 [1],
 [1],
 [0]])

#Seed
np.random.seed(1)

#Creating the synapses in between the neurons
syn0 = 2*np.random.random((3,4)) - 1 #3 by 4 matrix of weights, the 1 is for bias
syn1 = 2*np.random.random((4,1)) - 1 #4 by 1 matrix of weights, the 1 is for bias

#Training the nerual network
for j in range(60000): #Train for 60000 steps
    #Layer Creation(Input, Hidden and Output)
    l0 = x
    l1 = sigmoid(np.dot(l0, syn0)) #Multiplying the synapse matrix by layer 0
    l2 = sigmoid(np.dot(l1, syn1)) #Previous Layer multiplied by the synapse

    #Back Propagation, trying to reduce error everytime we do the loop
    l2_error = y - l2 #Arbitrary starting number
    if (j % 10000) == 0: #Only print every 10,000 steps
        print ('Error:' + str(np.mean(np.abs(l2_error))))
    #Calculting Deltas
    l2_delta = l2_error * sigmoid(l2, deriv=True) #First Delta
    l1_error = l2_delta.dot(syn1.T) #matrix multiplying l2_delta by the 1 synapse transposed(flipped)
    l1_delta = l1_error * sigmoid(l1, deriv=True)

    #Updating the Synapses
    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)

print ('Ouput after training:')
print (l2)
