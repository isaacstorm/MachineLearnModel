#!/opt/anaconda3/bin/python

import numpy as np
cm =open('A321exx0.txt','r')
cm = cm.read()
cm = cm.split()
cm = np.array(cm).reshape((-1,3600))
cm = cm.astype(float)
impo =open('import.txt','r')
impo = impo.read()
impo = impo.split()
impo = np.array(impo).reshape((-1,3600))
impo = impo.astype(float)
impo_ser = impo.copy()
cm_edit = cm.copy()
cm_edit = np.append(impo_ser,cm_edit,axis=0)
cm_edit = cm_edit.T
print(cm_edit[:,0].argsort())
cm_edit = cm_edit[cm_edit[:,0].argsort()]
cm_edit = cm_edit.T
cm_edit = cm_edit[1:,:3200]
np.savetxt('A3210_edit1.txt',cm_edit, fmt='%s',delimiter=' ' )







