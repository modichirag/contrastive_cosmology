import numpy as np
import nbodykit
from nbodykit.lab import BigFileCatalog
from pmesh.pm import ParticleMesh, RealField
from nbodykit.lab import FFTPower, ProjectedFFTPower, ArrayMesh

nc = 256
pm = ParticleMesh(Nmesh=[nc, nc, nc], BoxSize=1024, dtype='f8')

splits = 4
idx = [[i*nc//splits, (i+1)*nc//splits] for i in range(splits)]
print("slice boundary : ", idx)

def getslice(fpath):
    f = BigFileCatalog(fpath)
    mesh = pm.paint(f['Position'].compute())[...]
    slices = [mesh[i*nc//splits:(i+1)*nc//splits].sum(axis=0) for i in range(splits)]
    slices = slices + [mesh[:, i*nc//splits:(i+1)*nc//splits, :].sum(axis=1) for i in range(splits)]
    slices = slices + [mesh[:, :, i*nc//splits:(i+1)*nc//splits].sum(axis=2) for i in range(splits)]
    slices = np.array(slices)
    return slices
    


for isim in range(2000):
    if isim%10 == 0: print(isim)
    for j in range(3):
        try:
            fpath = '/mnt/ceph/users/cmodi/simbig/catalogs/hod.quijote_LH%d.z0p0.zheng07.%d.bf'%(isim, j)
            slices = getslice(fpath)
            np.save('/mnt/ceph/users/cmodi/simbig/mesh/LH%d.z0p0.zheng07.%d'%(isim, j), slices)
        except Exception as e: print(e)
        
        try:
            fpath = '/mnt/ceph/users/cmodi/simbig/catalogs/hod.quijote_LH%d.z0p0.zheng07_ab.%d.bf'%(isim, j)
            slices = getslice(fpath)
            np.save('/mnt/ceph/users/cmodi/simbig/mesh/LH%d.z0p0.zheng07_ab.%d'%(isim, j), slices)
        except Exception as e:
            print(e)


