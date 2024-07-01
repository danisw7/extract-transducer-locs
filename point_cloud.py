import numpy as np
from stride import *
from optimparallel import minimize_parallel
import pandas as pd
import plotly.express as px
from scipy.optimize import linear_sum_assignment

class affinePC():

    def __init__(self, pc_us, pc_ct):
        """
        +======================================================================+
        This class takes in two point clouds pc_us and pc_ct and can 
        find the affine transformation that registers the pc_ct cloud onto 
        pc_us, optimising a simplified closest neighbor L2 loss function.

        The best-fit parameters are stored in params. With axis convention x,y,z
        params = [rot-x, rot-y, rot-z, transl-x, transl-y, transl-z]. After
        applying the affine transformation, the result is returned and stored
        in pc_ct_r.

        For convenience, upon initialisation, the pc_ct is centered around it 
        center-of-mass, and stored in center_ct to ensure the fitted rotation 
        happens around the center-of-mass and minimizes additional translations 
        rotation around other points might induce.

        Note that a custom affine transform can be applied to a different 
        point cloud, but that care must be taken that the same center-point 
        is also supplied.

        TODO:
        - attach full affine matrix
        - upgrade to pytorch

        Example:
        A_PC = affinePC(A, B)       # instanciate
        A_PC.fit(start, bounds)      # fit
        B_onto_A = A_PC.apply_aff() # apply

        INPUTS:
        pc_us                   np.array(float) PC to register to 
                                shape (N1_points,3)
        pc_ct                   np.array(float) PC to register  
                                shape (N2_points,3)
        +======================================================================+
        """
        self.pc_ct = pc_ct
        self.pc_us = pc_us
        self.center_ct = np.mean(pc_ct, axis=0)
        self.params = np.zeros(6)
        self.pc_ct_r = pc_ct
        self.method = 'naive'

    def apply_aff(self):
        self.pc_ct_r = apply_affine(self.pc_ct, self.params, self.center_ct)
        return self.pc_ct_r

    def loss_l2(self, params, pc_ct, pc_us, center_ct, method):
        return loss(params, pc_ct, pc_us, center_ct, method)
    
    def fit(self, start, bounds, method):
        """
        +======================================================================+
        Fit affine transform using L-BFGS-B from optimparallel.

        INPUT:
        start                   list(float) starting points affine transform
                                length (6)
        bounds                  list(tuple(float)) boundary locations of 
                                each parameter length (6, length (2))
        method                  (str) "optimal" : lin_sum assignment
                                      "naive"   : min dist assignment
        +======================================================================+
        """
        pc_us_ = self.pc_us
        pc_ct_ = self.pc_ct
        center_ct_ = self.center_ct
        result = minimize_parallel(fun=self.loss_l2,
                                   x0=start,
                                   args=(pc_ct_, pc_us_, center_ct_, method),
                                   bounds=bounds,
                                   tol=1e-6,
                                   options={'disp':True})
        self.params = result.x
        return self.params
    
def plot_PCs(PCLocsArray_list):
    """    
    ===========================================================================+
    Plotting function for point cloud arrays. Send in as a list of arrays

    INPUT
    PCLocsArray_list            (list)(np.array)(float) list of arrays of 
                                coordinates for each point in real space per
                                cloud (m,N_m,3)

    OUTPUT
    fig                         (px.fig) plotly figure handle
    ===========================================================================+
    """

    df = pd.DataFrame(columns=['x','y','z', 'type'])
    for i, PCLocsArray in enumerate(PCLocsArray_list):
        df_PC=pd.DataFrame(PCLocsArray,columns=['x','y','z'])
        df_PC['type'] = i+1
        df = pd.concat([df, df_PC])

    fig = px.scatter_3d(df,x='x',y='y',z='z',color='type')
    return fig


def rot_mat(x):
    """
    ===========================================================================+
    2D Rotation matrix

    INPUT
    x                           (float) angle to rotate [radians]

    OUTPUT
    R                           (np.array)(float) Rotation matrix with angle x
                                (2,2)
    ===========================================================================+
    """
    R = np.zeros((2,2))

    # load matrix 
    R[0,0] = np.cos(x)
    R[0,1] = -np.sin(x)
    R[1,0] = np.sin(x)
    R[1,1] = np.cos(x)

    return R

def apply_affine(coordinates, affine_transform_params, center):
    """
    ===========================================================================+
    Apply affine transformation for coordinates with affine_transform_params, 
    with its rotation around center. First the object is translated to center 
    with TR, then the rotation R is applied (x, y, z), translate back (TR-1) 
    and the translation T is applied: 

    transformed_coords = T @ TR-1 @ Rz @ Ry @ Rz @ Rx @ TR @ coords

    with all operations in 4x4 matrix format.

    The affine transform is constrained to only allow rotations and translations

    affine_transform_params contain:
    0: rotation around x axis [rad]
    1: rotation around y axis [rad]
    2: rotation around z axis [rad]
    3: translation along x axis [voxels or m]
    4: translation along y axis [voxels or m]
    5: translation along z axis [voxels or m]


    INPUT
    coordinates                 (np.array)(float) coordinate in voxel space 
                                or real space to transform (N,3)
    affine_transform_params     (list) affine transform parameters (6)
    center                      (center) point of rotation in voxel or real
                                space (3,)

    OUTPUT
    transf_coordinates          (np.array)(float) transformed coordinates in 
                                voxel or real space (N,3)
    ===========================================================================+
    """
    # stacks up the parameters of an affine transform and applies to a list of 
    # coordinates
    
    # Reset to center for rotations
    TR = np.array([[1,0,0,-center[0]],
                   [0,1,0,-center[1]],
                   [0,0,1,-center[2]],
                   [0,0,0,1]])
    # push back to original pos for rotations
    TRinv = np.array([[1,0,0,center[0]],
                      [0,1,0,center[1]],
                      [0,0,1,center[2]],
                      [0,0,0,1]])

    # rotations
    theta_x=affine_transform_params[0]
    theta_y=affine_transform_params[1]
    theta_z=affine_transform_params[2]

    # feed parameters into matrix
    R_x = np.eye(4)
    R_y = np.eye(4)
    R_z = np.eye(4)
    R_x[(1,1,2,2),(1,2,1,2)] = rot_mat(theta_x).reshape(-1)
    R_y[(0,0,2,2),(0,2,0,2)] = rot_mat(theta_y).reshape(-1)
    R_z[(0,0,1,1),(0,1,0,1)] = rot_mat(theta_z).reshape(-1)

    # translations
    T = np.eye(4)
    T[(0,1,2),(3,3,3)] = affine_transform_params[3:]

    # scaling
    # s_x=affine_transform_params[6]
    # s_y=affine_transform_params[7]
    # s_z=affine_transform_params[8]
    # S=np.array([[s_x,0,0,0],[0,s_y,0,0],[0,0,s_z,0],[0,0,0,1]])

    transform = T @ TRinv @ R_x @ R_y @ R_z @ TR 

    return np.array([(transform @ np.hstack((coord,1)))[:3] 
                        for coord in coordinates])

def extract_ct_anchor_points(image_vol, 
                             NUM_PTS=5000, 
                             mode='outer', 
                             threshold_max=1500,
                             threshold_min=1000, 
                             start=0,
                             offset=np.array([0,0,0]),
                             Csphere=1.6):
    """
    ===========================================================================+
    Use image thresholding to find skull outline in ct_image
    for this we need the center of the array to fall nicely within the head 
    volume, piercing skull surface cleanly in most directions

    It can be used in two modes:
    'inner':
    Starting from the exterior, trace a ray, radiating towards the center 
    through the volume and note the voxel coordinates where first time the 
    threshold_max is crossed.

    'outer':
    Starting from the center, trace a ray, starting at radius 'start' radiating 
    outwards through the volume and note the voxel coordinates where first time 
    a threshold_max is crossed after threshold_min has been is crossed during
    prior steps along that ray.

    The points are sampled uniformly across the top hemisphere using the 
    golden spiral method.

    This function is used once to generate determine the outline of the skull in
    a scan volume.

    This function places the origin of the coordinate system at the center of 
    the volume, offset with 'center'.

    The extracted coordinates are reported with the center of the array 
    as the origin.

    INPUT
    image_vol                   (np.array)(float) voxel space with vp or CT scan
                                (Nx,Ny,Nz)
    NUM_PTS                     (int) number of anchor points to extract
    mode                        (str) direction from which to determine surface
                                inner: start from center, move out
                                outer: start from edge, move to center
    threshold_max               (float) threshold to detect skull
    threshold_min               (float) first minimum has to be reached before 
                                skull can be detected (only with mode='inner')
    start                       (int) detection radius to start at (only with 
                                mode='inner')
    offset                      (np.array)(float) center of skull to start from
                                relative to center of array (3,)
    Csphere                     (float) coverage of sphere to detect in 
                                (1: full sphere, 2: hemisphere)

    OUTPUT
    coords                      (np.array)(int) coordinates of anchor points in 
                                voxel space (NUM_PTS, 3)
    ===========================================================================+
    """
    anchor_points=[]
    THRESHOLD_MAX=threshold_max
    if mode == 'outer':
        THRESHOLD_MIN=threshold_max+1
    THRESHOLD_MIN=threshold_min # only used with mode='inner'

    # create uniform sampling of a sphere
    indices = np.arange(0, NUM_PTS, dtype=float) + 0.5
    angles1 = np.arccos(1 - 2*indices/(NUM_PTS*Csphere)) 
    angles2 = np.pi * (1 + 5**0.5) * indices

    # determine center and max radius in image_vol voxel cube
    max_r = np.linalg.norm(np.divide(image_vol.shape,2))
    d_r = max_r/max(image_vol.shape)
    center_a = np.divide(image_vol.shape,2)
    center_d = center_a + offset

    # for every ray along (theta_i, phi_i) determine the anchorpoint of skull
    for theta, phi in zip(angles1, angles2):
        if mode == 'outer':
            r=max_r
            outside_head=True
            while outside_head and r>0:
                xyz=np.floor(center_d+np.array([r*np.sin(theta)*np.cos(phi),
                                                r*np.sin(theta)*np.sin(phi),
                                                r*np.cos(theta)])).astype(int)
                if np.all((xyz<image_vol.shape) & (xyz>=0)) \
                    and image_vol[xyz[0],xyz[1],xyz[2]] > THRESHOLD_MAX:
                    outside_head=False
                    anchor_points.append(xyz)
                else:
                    r-=d_r
        if mode =='inner':
            r=start
            inside_head=True
            minFlag=False
            while inside_head and r<max_r:
                xyz=np.floor(center_d+np.array([r*np.sin(theta)*np.cos(phi),
                                                r*np.sin(theta)*np.sin(phi),
                                                r*np.cos(theta)])).astype(int)
                if np.all((xyz<image_vol.shape) & (xyz>=0)) \
                    and (image_vol[xyz[0],xyz[1],xyz[2]] < THRESHOLD_MIN):
                    minFlag = True
                if np.all((xyz<image_vol.shape) & (xyz>=0)) \
                    and (image_vol[xyz[0],xyz[1],xyz[2]] > THRESHOLD_MAX) \
                    and minFlag:
                    inside_head=False
                    minFlag=False
                    anchor_points.append(xyz)
                else:
                    r+=d_r
           
    # remove duplicate anchorpoints (if any)
    anchor_pts_clean = np.unique(np.array(anchor_points), axis=0)
    return anchor_pts_clean - \
           np.ones_like(anchor_pts_clean)*center_a.reshape(1,-1)

def match_anchor_points(anchor_pts_ct, anchor_pts_us):
    """
    ===========================================================================+
    Find the set of anchor points extracted from CT_scan that are closest to 
    reference achor points. Find the best match using minimum dist.

    INPUT
    anchor_pts_ct               (np.array)(int) CT anchor points
                                (NUM_PTS, 3)
    anchor_pts_us               (np.array)(int) Reference anchor points
                                (N, 3)

    OUTPUT
    match_anchor_pts            (np.array)(int) coordinates of matched anchor  
                                points (NUM_PTS, 3)
    ===========================================================================+
    """
    A = np.repeat(anchor_pts_ct[:,:,np.newaxis], anchor_pts_us.shape[0], axis=2)
    B = np.repeat(anchor_pts_us.T[np.newaxis,:,:],anchor_pts_ct.shape[0],axis=0)

    distances = np.sum((A-B)**2, axis=1)

    # return minimum distances to each ref pnt
    return anchor_pts_ct[np.argmin(distances,axis=0)]

def lin_sum(a, b, p):
    """
    ===========================================================================+
    Apply scipy version of the Jonker-Volgenant algorithm to solve the linear 
    assignment problem.

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

    assignment_pairs = linear_sum_assignment(cost_matrix)

    # sort assignments
    assignments = assignment_pairs[1][np.argsort(assignment_pairs[0])]
    
    return assignments

def optimal_anchor_points(anchor_pts_ct, anchor_pts_us):
    """
    ===========================================================================+
    Find the set of anchor points extracted from CT_scan that are closest to 
    reference anchor points. Find the best match using scipy lin_sum_assignment.

    INPUT
    anchor_pts_ct               (np.array)(int) CT anchor points
                                (NUM_PTS, 3)
    anchor_pts_us               (np.array)(int) Reference anchor points
                                (N, 3)

    OUTPUT
    match_anchor_pts            (np.array)(int) coordinates of matched anchor  
                                points (NUM_PTS, 3)
    ===========================================================================+
    """
    assignments = lin_sum(anchor_pts_ct, anchor_pts_us, 2)
    return anchor_pts_ct[assignments]

# define loss functions
def loss(affine_transform_params, anchor_pts_ct, anchor_pts_us, center, 
         method='optimal'):
    """
    ===========================================================================+
    L2 loss for the affine transform (rotation, translation)

    For affine_transform_parameters see apply_affine()

    INPUT
    affine_transform_params     (list)(float) affine transform parameters (6)
    anchor_pts_ct               (np.array)(int) CT anchor points
                                (NUM_PTS, 3)
    anchor_pts_us               (np.array)(int) Reference anchor points
                                (N, 3)
    center                      (np.array)(int) center of rotation (3,)
    method                      (str) method of point matching:
                                'optimal' : using lin_sum assignment
                                'naive'   : using min distance assignment

    OUTPUT
    L2_loss                     (float) L2 distance loss between anchor points
    ===========================================================================+
    """
    if method == 'optimal':
        anchor_pts_ct_matched=optimal_anchor_points(
                                    apply_affine(anchor_pts_ct,
                                                affine_transform_params, 
                                                center),
                                                anchor_pts_us)
    if method == 'naive':
        anchor_pts_ct_matched=match_anchor_points(
                                    apply_affine(anchor_pts_ct,
                                                affine_transform_params, 
                                                center),
                                                anchor_pts_us)
    return np.mean(np.linalg.norm(anchor_pts_ct_matched-anchor_pts_us,axis=1))

def fit_affine_anchor_points(anchor_pts_ct,
                             anchor_pts_us, 
                             center, 
                             start=[0,0,0,0,0,0],
                             bounds=[(-0.3,0.3)]*3+[(None,None)]*3):
    """
    ===========================================================================+
    Fit the affine parameters for the affine transformation (rotation, 
    translation) of the CT anchor points to the reference anchor points.

    Uses parallel version of scipy minimize (L-BFGS)

    INPUT
    anchor_pts_ct               (np.array)(int) CT anchor points
                                (NUM_PTS, 3)
    anchor_pts_us               (np.array)(int) Reference anchor points
                                (N, 3)
    center                      (np.array)(float) Center of rotation
    start                       (list)(float) Starting point for optim (6)
    bounds                      (list)(tuple) Bounds on parameters

    OUTPUT
    result                      (list)(float) optimised parameters (6)
    ===========================================================================+
    """
    result = minimize_parallel(fun=loss,
                               x0=start,
                               args=(anchor_pts_ct, anchor_pts_us, center),
                               bounds=bounds,
                               tol=1e-6,
                               options={'disp':True})
    print(result.x)
    return result.x

def removeOutlier(reflections, cut_off=0.2):
    """
    ===========================================================================+
    Function to remove outliers that do not lie sufficiently close to a fitted
    ellipsoid around the bulk of the points

    INPUT
    reflections                 (np.array)(float) array of coordinates of ToF
                                reflections [m] (N,3)
    cut_off                     (float) outlier cut off threshold

    OUTPUT
    reflCleaned                 (np.array)(float) array of coordinates of ToF
                                reflections with outliers removed (N*,3)
    ===========================================================================+
    """
    meanx = np.mean(reflections[:,0])
    meany = np.mean(reflections[:,1])
    meanz = np.mean(reflections[:,2]) 
    #center = np.array([meanx, meany, meanz])
    X = reflections[:,0] - meanx
    Y = reflections[:,1] - meany
    Z = reflections[:,2] - meanz

    A = np.vstack([X**2, Y**2, Z**2, X*Z, Y*Z, X*Y, X, Y, Z]).T
    b = np.ones_like(X)
    coeffs = np.linalg.lstsq(A, b)[0].squeeze()

    reflCleaned = []

    for i, (x,y,z) in enumerate(zip(X,Y,Z)):
        a = np.hstack([x**2, y**2, z**2, x*z, y*z, x*y, x, y, z])
        error = 1 - np.sum(a*coeffs)
        #normal = np.array([2*x*coeffs[0] + coeffs[3]*z + coeffs[5]*y + coeffs[6],
        #                   2*y*coeffs[1] + coeffs[4]*z + coeffs[5]*x + coeffs[7],
        #                   2*z*coeffs[2] + coeffs[3]*x + coeffs[4]*y + coeffs[8]])
        #normal /= np.linalg.norm(normal)
        if abs(error) > cut_off:
            #reflections[i,:] = reflections[i,:] - error*normal
            #reflections[i,:] = np.array([0,0,0])
            pass
        else:
            reflCleaned.append(reflections[i])
    
    print('Removed: {}/{}'.format(i-len(reflCleaned),i))
    
    return np.array(reflCleaned)

if __name__=="__main__":
    
    # example
    # load data
    true_model = np.load('data/phan_ss_skull.npy') # grid is not rescaled 

    # trial transform parameters
    theta_x = 0.1
    move_x = 30
    """
    # get anchorpoints 
    anchor_pts_us = extract_ct_anchor_points(np.pad(true_model.data,40), 
                                             NUM_PTS=1024) 
                    # observed anchor points, very few but correctly positioned
    anchor_pts_ct = extract_ct_anchor_points(np.pad(true_model.data,40), 
                                             NUM_PTS=15000)

    # save starting points (for reloading)
    np.save('output/test_us_anchor_pts.npy', anchor_pts_us)
    np.save('output/test_ct_anchor_pts.npy', anchor_pts_ct)
    """
    anchor_pts_us = np.load('output/test_us_anchor_pts.npy')
    anchor_pts_ct = np.load('output/test_ct_anchor_pts.npy')[::3]

    # apply affine
    center = np.mean(anchor_pts_ct, axis=0)
    anchor_pts_ct_rottrans = apply_affine(anchor_pts_ct, 
                                          [theta_x, 0, 0, move_x, 0, 0], 
                                          center)
    
    # plot pointclouds
    fig = plot_PCs([anchor_pts_us, anchor_pts_ct_rottrans])
    fig.show()

    # find optimal transform
    affPCUS_CT = affinePC(anchor_pts_us, anchor_pts_ct_rottrans)
    params = affPCUS_CT.fit(start=[0,0,0,0,0,0], 
                            bounds=[(-0.3,0.3)]*3+[(None,None)]*3,
                            method='optimal')
    affPCUS_CT.apply_aff()
    print("Best parameters: {}".format(params))

    # plot pointclouds
    fig = plot_PCs([anchor_pts_us, affPCUS_CT.pc_ct_r])
    fig.show()  
