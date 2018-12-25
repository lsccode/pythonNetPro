#!/usr/bin/env python
# coding: utf-8

# In[1]:


# python notebook for Make Your Own Neural Network
# working with the MNIST data set
# this code demonstrates rotating the training images to create more examples
#
# (c) Tariq Rashid, 2016
# license is GPLv2


# In[2]:


import numpy
import matplotlib.pyplot
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


# scipy.ndimage for rotating image arrays
import scipy.ndimage


# In[4]:


# open the CSV file and read its contents into a list
data_file = open("mnist_dataset/mnist_train_100.csv", 'r')
data_list = data_file.readlines()
data_file.close()


# In[5]:


# which record will be use
record = 6


# In[6]:


# scale input to range 0.01 to 1.00
all_values = data_list[record].split(',')
scaled_input = ((numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01).reshape(28,28)


# In[7]:


print(numpy.min(scaled_input))
print(numpy.max(scaled_input))


# In[8]:


# plot the original image
matplotlib.pyplot.imshow(scaled_input, cmap='Greys', interpolation='None')


# In[9]:


# create rotated variations
# rotated anticlockwise by 10 degrees
inputs_plus10_img = scipy.ndimage.rotate(scaled_input, 10.0, cval=0.01, order=1, reshape=False)
# rotated clockwise by 10 degrees
inputs_minus10_img = scipy.ndimage.rotate(scaled_input, -10.0, cval=0.01, order=1, reshape=False)


# In[10]:


print(numpy.min(inputs_plus10_img))
print(numpy.max(inputs_plus10_img))


# In[11]:


# plot the +10 degree rotated variation
matplotlib.pyplot.imshow(inputs_plus10_img, cmap='Greys', interpolation='None')


# In[12]:


# plot the +10 degree rotated variation
matplotlib.pyplot.imshow(inputs_minus10_img, cmap='Greys', interpolation='None')


# In[ ]:




