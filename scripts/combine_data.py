import numpy as np
import pickle, json

#path = '/mnt/ceph/users/cmodi/contrastive/data/z10-N0001/zheng07-fid/'
#path = '/mnt/ceph/users/cmodi/contrastive/data/z10-broad/zheng07/'
#path = '/mnt/ceph/users/cmodi/contrastive/data/z10-N0001/zheng07_ab/'
#path = '/mnt/ceph/users/cmodi/contrastive/data/z10/zheng07/'
#path = '/mnt/ceph/users/cmodi/contrastive/data/z05-chang/zheng07/'
#path = '/mnt/ceph/users/cmodi/contrastive/data/z10-N0001/zheng07_ab-rock/'


# for path in ['/mnt/ceph/users/cmodi/contrastive/data/quijote/Rockstar/z10-N0004/zheng07_velab/',
#              ]:
#     keys =  ['power', 'power_rsd', 'gals', 'hodp', 'power_ell', 'hfracs']
#     #keys =  ['power', 'power_rsd',  'power_ell']
#     #keys =  [ 'hfracs']

#     print(path)
#     for key in keys:
#         print("\nCombine %s\n"%key)
#         ar = []
#         for i_lhc in range(0, 2000):
#             if i_lhc %50 ==0:print(i_lhc)
#             f = np.load(path + '%04d/%s.npy'%(i_lhc, key), allow_pickle=True).astype(np.float64).real
#             ar.append(f)

#         ar = np.array(ar)
#         print(ar.shape)
#         np.save(path + key, ar)


for path in [
             '/mnt/ceph/users/cmodi/contrastive/data/quijote/FoF/z10-N0004/zheng07_velab/',
             ]:
    #keys =  ['power', 'power_rsd', 'gals', 'hodp', 'power_ell', 'hfracs', 'qspec', 'bspec']
    keys =  ['qspec', 'bspec']
    #keys =  ['scat_0', 'scat_1', 'scat_2']

    print(path)
    for key in keys:
        print("\nCombine %s\n"%key)
        ar = []
        for i_lhc in range(0, 2000):
            if i_lhc %50 ==0:print(i_lhc)
            f = np.load(path + '%04d/%s.npy'%(i_lhc, key), allow_pickle=True).astype(np.float64).real
            ar.append(f)

        ar = np.array(ar)
        print(ar.shape)
        np.save(path + key, ar)



################
## ps = []
## for i_lhc in range(0, 2000):
##     if i_lhc %50 ==0:print(i_lhc)
##     f = np.load(path + '%04d/power_h.npy'%(i_lhc))[:, 1]
##     ps.append(f)
#
## ps = np.array(ps)
## print(ps.shape)
## np.save(path + 'power_h', ps)
#
####
## ps = []
## for i_lhc in range(0, 2000):
##     if i_lhc %50 ==0:print(i_lhc)
##     f = np.load(path + '%04d/power_h1e-4.npy'%(i_lhc))[:, 1]
##     ps.append(f)
#
## ps = np.array(ps)
## print(ps.shape)
## np.save(path + 'power_h1e-4', ps)


# ps = []
# for i_lhc in range(0, 2000):
#     if i_lhc %50 ==0:print(i_lhc)
#     f = np.array([np.load(path + '%04d/compensated_power_%d.npy'%(i_lhc, i))[:, 1] for i in range(10)])
#     ps.append(f)

# ps = np.array(ps)
# print(ps.shape)
# np.save(path + 'compensated_power', ps)


# ps = []
# for i_lhc in range(0, 2000):
#     if i_lhc %50 ==0:print(i_lhc)
#     f = np.array([np.load(path + '%04d/power_%d.npy'%(i_lhc, i))[:, 1] for i in range(10)])
#     ps.append(f)

# ps = np.array(ps)
# print(ps.shape)
# np.save(path + 'power', ps)


# ps = []
# for i_lhc in range(0, 2000):
#     if i_lhc %50 ==0:print(i_lhc)
#     tmp = []
#     for i in range(10):
#         with open(path + '%04d/gals_%d.json'%(i_lhc, i)) as ff:
#             g = json.load(ff)
#             #print(g)
#         tmp.append([g['total'], g['centrals'], g['satellites']])
#     ps.append(tmp)
# ps = np.array(ps)
# print(ps.shape)
# #print(ps)
# np.save(path + 'gals', ps)


# ###
# ps = []
# for i_lhc in range(0, 2000):
#     if i_lhc %50 ==0:print(i_lhc)
#     tmp = []
#     for i in range(10):
#         g = np.load(path + '%04d/hodp_%d.npy'%(i_lhc, i), allow_pickle=True)
#         tmp.append([float(i[:-1]) for i in str(g).split(" ")[1::2]])
#     ps.append(tmp)
# ps = np.array(ps)
# print(ps.shape)
# np.save(path + 'hodp', ps)
