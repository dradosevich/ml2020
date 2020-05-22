#Danny Radosevich
import numpy as np

students = {}
students[1] = [1,1,2]
students[2] = [1,1,1]
students[3] = [1,0,2]
students[4] = [0,0,2]
students[5] = [1,1,1]
students[6] = [1,0,.5]
students[7] = [0,0,.5]
students[8] = [0,1,1]
students[9] = [1,0,1]
students[10] = [1,1,.5]

sele = [0,1,2]
stwe = [0,1,.5]

for stu in students:
    print("sele",stu)
    a = np.asarray(sele)
    b = np.asarray(students[stu])
    print(np.linalg.norm(a-b))
print("-------------------------------------")
for stu in students:
    print("stwe",stu)
    a = np.asarray(stwe)
    b = np.asarray(students[stu])
    print(np.linalg.norm(a-b))

#problem two
#f = mu*C^(-1)*x^t -(1/2)mu*C^(-1)mu^T +ln(p)
print("-------------------------------------")
sele = np.asarray(sele)
stwe = np.asarray(stwe)
muS = np.array([4/5,3/5,8/5])
muU = np.array([3/5,2/5,7/10])
cov = np.identity(3)
cov = np.linalg.inv(cov)
x   = np.array([[1,1,2],
               [1,1,1],
               [1,0,2],
               [0,0,2],
               [1,1,1],
               [1,0,1/2],
               [0,0,1/2],
               [0,1,1],
               [1,0,1],
               [1,1,1/2]])
daMus =[muS,muU]
for mu in daMus:
    partOne = np.dot(mu,cov)
    partOne = np.dot(partOne,np.transpose(sele))
    partTwo = np.dot(mu,cov)
    partTwo = np.dot(partTwo, np.transpose(mu))
    partTwo = np.dot((1/2),partTwo)
    print(partOne-partTwo+np.log(1/2))
for mu in daMus:
    partOne = np.dot(mu,cov)
    partOne = np.dot(partOne,np.transpose(stwe))
    partTwo = np.dot(mu,cov)
    partTwo = np.dot(partTwo, np.transpose(mu))
    partTwo = np.dot((1/2),partTwo)
    print(partOne-partTwo+np.log(1/2))
