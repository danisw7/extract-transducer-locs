import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

from point_cloud import plot_PCs, affinePC

def naive_dist(a, b, p):
    """
    ===========================================================================+
    Find matching points from b to a using a minium distance assignment.

    INPUT
    a                           (np.array)(float) point cloud of reference
                                (NUM_PTS_A, 3)
    b                           (np.array)(float) point cloud of reference
                                (NUM_PTS_B, 3)
    p                           (int) order of the p-norm used to calculate 
                                distances between points in a and b.
                       
    OUTPUT
    assignments                 (np.array)(int) mapping of points B to A
                                (NUM_PTS_A,)
    ===========================================================================+
    """

    length_a = a.shape[0]
    length_b = b.shape[0]

    # calculate the benefit matrix between sets ai and bi
    B = np.repeat(b[:,:,np.newaxis],   length_a, axis=2)
    A = np.repeat(a.T[np.newaxis,:,:], length_b, axis=0)
    cost_matrix = (np.linalg.norm(A-B, ord=p, axis=1)**p)

    # find min dist
    d = np.min(cost_matrix, axis=1)
    
    return d

def refit_transducer(label):

    i = label

    path_file = f"fittedTransducers/{i}/transCyl_{i}.pk"
    with open(path_file, 'rb') as f:
        x = pickle.load(f)
    loc = x['location']
    A_PC = x['Objects']
    loss = A_PC.loss_l2(A_PC.params, A_PC.pc_ct, A_PC.pc_us, A_PC.center_ct, method='naive')
    print(f"{i}:  {loss/len(A_PC.pc_us)}")

    # get normal
    normal_z = np.zeros((10,3))
    normal_z.T[2] = np.linspace(0,1,10) 
    x = A_PC.pc_ct.copy()
    A_PC.pc_ct = normal_z
    normal_ = A_PC.apply_aff().copy()
    A_PC.pc_ct = x
    _ = A_PC.apply_aff()

    # plot distances
    temp = A_PC.pc_us
    # orject
    temp_xy = temp - normal_[-1]*temp
    distance = naive_dist(A_PC.pc_ct_r[1200:], temp_xy, 2)
    plt.plot(np.sort(distance))
    plt.grid()
    plt.xlabel('point #')
    plt.ylabel('L2 distance')
    path_file = f"fittedTransducers/{i}/error_curve.png"
    plt.savefig(path_file)
    plt.close()

    # cure
    x = A_PC.pc_us[:,0]
    y = A_PC.pc_us[:,1]
    B_PC = affinePC(A_PC.pc_us[distance < 0.2], A_PC.pc_ct)
    #B_PC = affinePC(A_PC.pc_us[distance < 0.1], A_PC.pc_ct)
    B_PC.center_ct = A_PC.center_ct
    dx = 20
    dphi = 0.05
    start=A_PC.params
    bounds_t = [(start[3]-dx, start[3]+dx), 
                (start[4]-dx, start[4]+dx),
                (start[5]-dx, start[5]+dx)]
    bounds_r = [(start[0]-dphi, start[0]+dphi), 
                (start[1]-dphi, start[1]+dphi),
                (start[2]-dphi, start[2]+dphi)]
    bounds = bounds_r + bounds_t
    B_PC.fit(start=start, bounds=bounds, method='naive')
    print(f"Affine parametes [rot_x, rot_y, rot_z, t_x, t_y, t_z]: \n{B_PC.params}")
    regrefined_pc = B_PC.apply_aff()
    loss_new = B_PC.loss_l2(B_PC.params, B_PC.pc_ct, B_PC.pc_us, B_PC.center_ct, method='naive')
    print(f"{i} {loss/len(A_PC.pc_us)} vs {loss_new/len(B_PC.pc_us)}")

    # save
    bucket = {"location" : loc,
            "Objects"  : B_PC}
    with open(f"fittedTransducers/{i}/transCyl_{i}_refitCutoff.pk", 'wb') as handle:
        pickle.dump(bucket, handle)
    
    fig = plot_PCs([regrefined_pc, A_PC.pc_us, B_PC.pc_us])
    fig.write_html(f"fittedTransducers/{i}/transCyl_{i}_refitCutoff.html")

    return loc, loss_new/len(B_PC.pc_us)

if __name__=="__main__":
     
    label_points = np.fromfile('ExtractedTransducerClustersPCs-jul24.npy').reshape((-1,4))
    refined_points = label_points[:,:3]
    labels_rerun = label_points[:,-1]
    cap = np.load('cap.npy')
    cyl = np.load('cyl.npy')
    count = 0
    errors = []
    centers = []
    loss_ori = []
    labels = []
    refit = True

    labels_x = np.load('refit_labels.npy').T[0]

    #labels_x = np.array([ 442,  546,  685,  962,  993, 1093, 1094, 1124, 1194, 1197, 1199,
    #   1222, 1235, 1248, 1251, 1256, 1280, 1300, 1302, 1303, 1304, 1305,
    #   1322, 1323, 1325, 1326, 1351, 1371, 1375, 1405, 1406, 1431, 1447,
    #   1473, 1481, 1483, 1507])
    #labels_x = np.arange(0,1696)

    if refit:
        with open(f"fittedTransducers/locs-jul24-fit.pk", 'rb') as f:
            x = pickle.load(f)
        l = np.array(x['losses'])
        loss_ori.append(l)

    for label in labels_x:
        print(label)
        test = refined_points[labels_rerun==label]
        center_loc, los = refit_transducer(label)
        errors.append(los)
        centers.append(center_loc)
        labels.append(int(label))
        print(f"{l} vs {los}")

    print(np.sort(np.array(-errors)))
    bucket = {"locations" : centers,
            "labels"  : labels,
            "losses" : errors,
            "original losses": loss_ori} 

    with open(f"fittedTransducers/locs-jul24-refit.pk", 'wb') as handle:
        pickle.dump(bucket, handle)