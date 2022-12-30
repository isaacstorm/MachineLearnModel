#!/opt/anaconda3/bin/python
#KRR

import numpy as np
import math as ma

raw = open('data0.txt','r')
data = raw.read()
spd = data.split()
arr =np.array(spd).reshape((560000,3))
arr = arr.astype(float)
b1 = np.vsplit(arr,8000)
#vx =np.array([9.36879,0,0])
#vy =np.array([3.64335,17.20651,0])
#vz =np.array([4.06528,-0.78588,26.08687])
vect  =open('ce0','r')
vect = vect.read()
vect = vect.split()
vect = np.array(vect).reshape((3,3))
vect = vect.astype(float)
vx =np.array([vect[0,0],0,0])
vy =np.array([vect[1,2],vect[0,1],0])
vz =np.array([vect[2,1],vect[2,2],vect[0,2]])
print(vx,vy,vz)
dx= ma.sqrt(sum(vx**2))
dy= ma.sqrt(sum(vy**2))
dz= ma.sqrt(sum(vz**2))
vx = vx/dx
vy = vy/dy
vz = vz/dz
m = 24000
id =open('id0','r')
id = id.read()
id = id.split()
id = np.array(id).reshape((-1,2))
id = id.astype(int)
def inte(ci,cj,r2):
    rz = ma.exp(r2)
    rlist = np.empty([0,0])
    for i in range(0,3):
        for j in range(0,3):
            inte = ma.exp(r2/(1/ci[i]+1/cj[j]))
            rlist= np.append(inte,rlist)
    return rlist
def vec(x,y):
    dvec = min(abs(x),y-abs(x))*np.sign(x)
    if abs(dvec) == abs(x):
        dvec = dvec
    else:
        dvec = -dvec
    return dvec
def decrmake(i,j):
    ilist = b1[i]
    jlist = b1[j]
    iC1 = ilist[0:5]
    iC2 = ilist[17:22]
    iC3 = ilist[34:39]
    iC4 = ilist[52:56]
    iC4 = np.append(iC4,ilist[69])
    jC1 = jlist[0:5]
    jC2 = jlist[17:22]
    jC3 = jlist[34:39]
    jC4 = jlist[52:56]
    jC4 = np.append(jC4,jlist[69])
    iC1 = np.append(iC1,iC2)
    iC2 = np.append(iC3,iC4)
    jC1 = np.append(jC1,jC2)
    jC2 = np.append(jC3,jC4)
    iC = np.append(iC1,iC2).reshape(20,-1)
    jC = np.append(jC1,jC2).reshape(20,-1)
    cminter = np.empty([0,0])
    for a in range(0,20):
        for b in range(0,20):
            xij = iC[a,0]-jC[b,0]
            yij = iC[a,1]-jC[b,1]
            zij = iC[a,2]-jC[b,2]
            zij = zij/vz[2]
            yij = (yij - zij*vz[1])/vy[1]
            xij = (xij-zij*vz[0]-yij*vy[0])/vx[0]
            dxij = vec(xij,dx)
            dyij = vec(yij,dy)
            dzij = vec(zij,dz)
            zij = dzij*vz[2]
            yij = dyij*vy[1]+dzij*vz[1]
            xij = dxij*vx[0]+dyij*vy[0]+dzij*vz[0]

            sp1 = 0.3664980000E+02
            sp2 = 0.7705450000E+01
            sp3 = 0.1958570000E+01
            ci = np.array([sp1,sp2,sp3])
            cj = np.array([sp1,sp2,sp3])
            r2 = -(xij**2+yij**2+zij**2)
            rlist = inte(ci,cj,r2)
            cminter = np.append(cminter,rlist)

    return cminter.reshape(1,3600)
inp = np.empty([0,0])
for k in range(0,m):
   
    inp = np.append(inp,decrmake(id[k,0]-1,id[k,1]-1))

inp = inp.reshape(-1,3600)
np.savetxt('A321exx0.txt',inp, fmt='%s',delimiter=' ' )
