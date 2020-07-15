from numpy import dot
from numpy.linalg import norm

a = [1,1,1,1,0,1,1]
b = [1,1,1,1,1,0,0]
print(dot(a,b))
print(norm(a))
print(norm(b))
cos_sim = dot(a, b)/(norm(a)*norm(b))
print(cos_sim)