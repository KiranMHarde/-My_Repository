#1. Given are 2 similar dimensional numpy arrays, how to get a numpy array output
#in which every element is an element-wise sum of the 2 numpy arrays?
import numpy as np
a = np.array([[1,2,3],
              [4,5,6]])

b = np.array([[10,11,12],
              [13,14,15]])

c = a + b

print(c)


#2.Given a numpy array (matrix), how to get a numpy array output which is equal to
#the original matrix multiplied by a scalar?

a = np.array([[4,5,6],
              [6,3,2]])

b = 2*a 

print(b)


#3. Create an identity matrix of dimension 4-by-4.

i = np.eye(4)

print(i)

#4. Convert a 1-D array to a 3-D array

a = np.array([x for x in range(27)])

o = a.reshape((3,3,3))

print(o)


#5. Convert a binary numpy array (containing only 0s and 1s) to a boolean numpy array

a = np.array([[1, 0, 0],
              [1, 1, 1],
              [0, 0, 0]])

o = a.astype('bool')

print(o)



#6. Convert all the elements of a numpy array from float to integer datatype


a = np.array([[1.5, 3.4, 2.5],
              [2.5,2.5, 2.5]])

o = a.astype('int')

print(o)


#7. Stack 2 numpy arrays horizontally i.e., 2 arrays having the same 1st dimension(number of rows in 2D arrays)



a1 = np.array([[7,8,9],
               [4,5,6]])

a2 = np.array([[4,5,8],
               [10,11,12]])

a3 = np.hstack((a1, a2))

print(a3)


#8. Output a sequence of equally gapped 5 numbers in the range 0 to 100 (both inclusive)

list_of_numbers = [x for x in range(0, 101, 5)]

o = np.array(list_of_numbers)

print(o)


#9. Output a matrix (numpy array) of dimension 2-by-3 with each and every value equal to 5

o = np.full((2, 3), 5)

print(o)


#10.Given 2 numpy arrays as matrices, output the result of multiplying the 2 matrices(as a numpy array)

a = np.array([[1,2,3],
              [4,5,6],
              [3,2,1]])

b = np.array([[2,3,4],
              [5,6,7],
              [4,5,3]])

o = a@b

print(o)



#11. Output the array element indexes such that the array elements appear in the ascending order

array = np.array([10,1,5,2])

indexes = np.argsort(array)

print(indexes)


#12.Multiply a 5x3 matrix by a 3x2 matrix (real matrix product).

x = np.random.random((5,3))
print("First array:")
print(x)
y = np.random.random((3,2))
print("Second array:")
print(y)
z = np.dot(x, y)
print("Dot product of two arrays:")
print(z)










































