import pickle
import paddorch
import numpy as np


a = paddorch.FloatTensor(np.array([ 0.9, 0.1]))
pickle.dump(a,open("tensor.pkl",'wb'))


b=pickle.load(open("tensor.pkl",'rb'))
print(b)



a = paddorch.LongTensor(np.array([ 9, 1]))
pickle.dump(a,open("tensor.pkl",'wb'))


b=pickle.load(open("tensor.pkl",'rb'))
print(b)