from pyevtk.hl import gridToVTK
import numpy as np
from multiprocessing import Pool
# p=Pool(10)
N = 256

master_path = '/scratch/tpatel28/topo_mag/EW_sim'

X = range(N)
Y = range(N)
Z = range(N)

X,Y,Z=np.meshgrid(X,Y,Z)

for i in range(0,2025,25):
# def transform(i):
	file = master_path + '/raw_' + str(i)
	data = np.load(file)
	gridToVTK("./structured_%s"%i,X,Y,Z,pointData={"|\Phi|" : data})
	return

# p.map(transform, range(0,2800,25))