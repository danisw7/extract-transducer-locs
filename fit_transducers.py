import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

from point_cloud import plot_PCs, affinePC


def fit_transducer(test, cap, cyl, label):

    # slow, use script on server!

    test = refined_points[labels_rerun==label] # select a decent one, I have not found a way
                                    # to QC this clustering

    try: 
        os.mkdir(f"fittedTransducers/{label}/") # create dir
    except:
        pass

    # remove excess points?
    print(f"Number of points in observed cluster {len(test)}")

    center_test = np.mean(test, axis=0)
    test -= center_test
    start = [np.pi/3,0,np.pi/3,0,0,0] # it will be important to have a rough starting point for 
                        # each! the original geomtery will be very helpful

    print('start_fitting...')
    print(f'center cap: {cap.mean(axis=0)}')
    Cap_PC = affinePC(test, cap)
    Cap_PC.fit(start=start, bounds=[(-2*np.pi,2*np.pi)]*3+[(None,None)]*3, method='naive')
    print(f"Affine parametes [rot_x, rot_y, rot_z, t_x, t_y, t_z]: \n{Cap_PC.params}")
    reg_pc = Cap_PC.apply_aff()
    params_transd = Cap_PC.params.copy()

    bucket = {"location" : center_test,
              "Objects"  : Cap_PC}
    with open(f"fittedTransducers/{label}/transCap_{label}.pk", 'wb') as handle:
        pickle.dump(bucket, handle)

    fig = plot_PCs([reg_pc, test])
    fig.write_html(f"fittedTransducers/{label}/transCap_{label}.html")

    # refit with extended ref
    start = params_transd # it will be important to have a rough starting point for 
                        # each! the original geomtery will be very helpful
    dx = 2
    dphi = 0.2
    bounds_t = [(start[3]-dx, start[3]+dx), 
                (start[4]-dx, start[4]+dx),
                (start[5]-dx, start[5]+dx)]
    bounds_r = [(start[0]-dphi, start[0]+dphi), 
                (start[1]-dphi, start[1]+dphi),
                (start[2]-dphi, start[2]+dphi)]

    # determine polarity
    ref = np.zeros((len(cyl)+len(cap),3))
    ref[:len(cyl)] = cyl
    ref[len(cyl):] = cap

    normal_cap = np.cross(reg_pc[0], reg_pc[1]) # take any 2 cap points
    normal_cap /= np.linalg.norm(normal_cap)
    trans = params_transd[3:]/np.linalg.norm(params_transd[3:])

    if np.dot(normal_cap, trans) < 0:
        print("flip cylinder geom...")
        ref[:,2] = -ref[:,2]

    print('start_fitting...')
    Cyl_PC = affinePC(test, ref)
    Cyl_PC.center_ct = np.array([0,0,0])
    Cyl_PC.fit(start=start, bounds=bounds_r+bounds_t, method='naive')
    print(f"Affine parametes [rot_x, rot_y, rot_z, t_x, t_y, t_z]: \n{Cyl_PC.params}")
    regrefined_pc = Cyl_PC.apply_aff()
    loss = Cyl_PC.loss_l2(Cyl_PC.params, Cyl_PC.pc_ct, Cyl_PC.pc_us, Cyl_PC.center_ct, method='naive')
    if loss/len(Cyl_PC.pc_us) > 0.0003:
        error = label
        print(f"{label}:  {loss/len(Cyl_PC.pc_us)}")
    else:
        error = -1

    bucket = {"location" : center_test,
              "Objects"  : Cyl_PC}
    with open(f"fittedTransducers/{label}/transCyl_{label}.pk", 'wb') as handle:
        pickle.dump(bucket, handle)

    fig = plot_PCs([regrefined_pc, test])
    fig.write_html(f"fittedTransducers/{label}/transCyl_{label}.html")

    return error, center_test

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

if __name__=="__main__":
     
    label_points = np.fromfile('ExtractedTransducerClustersPCs-jul24.npy').reshape((-1,4))
    refined_points = label_points[:,:3]
    labels_rerun = label_points[:,-1]
    cap = np.load('cap.npy')
    cyl = np.load('cyl.npy')
    count = 0
    errors = []
    centers = []
    labels = []
    refit = False

    #labels_x = np.array([ 442,  546,  685,  962,  993, 1093, 1094, 1124, 1194, 1197, 1199,
    #   1222, 1235, 1248, 1251, 1256, 1280, 1300, 1302, 1303, 1304, 1305,
    #   1322, 1323, 1325, 1326, 1351, 1371, 1375, 1405, 1406, 1431, 1447,
    #   1473, 1481, 1483, 1507])
    labels_x = np.arange(0,1696)

    if refit:
        with open(f"fittedTransducers/locs_jul24-refit.pk", 'rb') as f:
            x = pickle.load(f)
        l = np.array(x['losses'])

    for label in labels_x:
        print(label)
        test = refined_points[labels_rerun==label]
        center_loc, los = fit_transducer(test, cap, cyl, label)
        errors.append(los)
        centers.append(center_loc)
        labels.append(int(label))

    print(np.sort(-np.array(errors)))
    bucket = {"locations" : centers,
            "labels"  : labels,
            "losses" : errors} 

    with open(f"fittedTransducers/locs-jul24-fit.pk", 'wb') as handle:
        pickle.dump(bucket, handle)