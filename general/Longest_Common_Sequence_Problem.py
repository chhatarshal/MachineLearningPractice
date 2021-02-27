#!/usr/bin/env python
# coding: utf-8

# In[1]:


input = {
    'seq1' :'abcd',
    'seq2' : 'ab'
}
output = 2


# In[2]:


def get_LongestCommonSequence(seq1, seq2, inx1 = 0, inx2 = 0):
    if len(seq1) == inx1:
        return 0
    if len(seq2) == inx2:
        return 0
    if seq1[inx1] == seq2[inx2]:
        return 1 + get_LongestCommonSequence(seq1, seq2, inx1+1, inx2+1)
    return max(get_LongestCommonSequence(seq1, seq2, inx1+1, inx2), get_LongestCommonSequence(seq1, seq2, inx1, inx2+1))
    


# In[3]:


get_LongestCommonSequence(input['seq1'], input['seq2'])


# In[4]:


input = {
    'seq1' :'32342342',
    'seq2' : '343243433'
}


# In[5]:


get_LongestCommonSequence(input['seq1'], input['seq2'])


# In[6]:


input = {
    'seq1' :'india',
    'seq2' : 'iia'
}


# In[7]:


get_LongestCommonSequence(input['seq1'], input['seq2'])


# In[ ]:




