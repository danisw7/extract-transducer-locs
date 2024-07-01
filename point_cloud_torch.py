import numpy as np
import torch
from tqdm import tqdm
from stride import *
import pandas as pd
import plotly.express as px
from numba import jit
import time
from scipy.optimize import linear_sum_assignment

disp = False

class OptimalTransportLoss():
    """
    This class represents the optimal transport or p-Wasserstein distance between 
    the observed and modelled data, calculated using the graph space transform method.

    Metivier, L., Brossier, R., Merigot, Q., & Oudet, E. (2019). A graph space optimal 
    transport distance as a generalization of lp distances: application to a seismic
    imaging inverse problem. Inverse Problems, 35(8), 085001.

    Parameters
    ----------
    delta_t : float
        Control parameter between the relative importance of time vs amplitude shifts.
    p : int, optional
        The order of the p-Wasserstein metric.
    e_start : float, optional
        Starting minimum positive bidding increment.
        Defaults to 1e4.
    e_end : float, optional
        Final minimum positive bidding increment.
        Defaults to 1e-5.
    e_fac : float, optional
        Factor by which the minimum positive bidding increment is reduced each e-scaling loop.
        Defaults to 10.
    d_samples : int, optional
        Factor by which the time dimension of the traces are downsampled prior to calculating 
        optimal assignments. Defaults to 1.
        
    """

    def __init__(self, **kwargs):
        
        self.t_modelled = None
        self.t_observed = None
        self.t_loss = None
        self.p = kwargs.pop('p', 2)
        self.e_start = kwargs.pop('e_start', 1e4)
        self.e_end = kwargs.pop('e_end', 1e-1)
        self.e_fac = kwargs.pop('e_fac', 30)
        self.d_sample = kwargs.pop('d_sample', 1)
        self.method = kwargs.pop('method', 'naive')
        self.assignments = None


    def optimal_assignments(self, modelled, observed):
        """
        =======================================================================+
        Solves the optimal assignment problem using the auction algorithm

        Parameters
        ----------
        modelled : Traces
            Traces object containing the modelled data.
        observed : Traces
            Traces object containing the observed data.
        
        Returns
        -------
        ndarray (num_receivers, num_samples)
            Optimal assignments between modelled and observed per trace. 
        =======================================================================+
        """

        m = modelled 
        o = observed

        start = time.time()
        if self.method == 'auction':
            print('                                     Running auction algorithm')
            self.assignments = auction_algorithm(o, m, self.p, self.e_start, self.e_end, self.e_fac)
            print(f'completed in: {(time.time() - start):.2f}s')
            print('                                     Obtained optimal assignments')
        elif self.method == 'lin_sum':
            self.assignments = lin_sum(o, m, self.p)
        elif self.method == 'naive':
            self.assignments = naive(o, m, self.p)
        else:
            raise Exception('No valid assignment method chosen...')

    def forward(self, params, reference, observed, center):

        # set some grad params
        params.requires_grad_(requires_grad=True)
        reference.requires_grad_(requires_grad=False)
        observed.requires_grad_(requires_grad=False)
        center.requires_grad_(requires_grad=False)

        # get correct arrays
        modelled = apply_affine_torch(reference, params, center)
        self.t_observed = observed
        n_modelled = modelled.detach().cpu().numpy()
        n_observed = observed.detach().cpu().numpy()

        # optimal assigmenet
        self.optimal_assignments(n_observed, n_modelled)

        # turn on grad
        modelled.requires_grad_(requires_grad=True)
        self.t_modelled = modelled

        # do this yolo
        proc_t_modelled = self.t_modelled
        proc_t_observed = self.t_observed

        self.t_loss = 0
        self.t_loss = torch.sum((proc_t_modelled[self.assignments] - proc_t_observed) ** self.p)

        return self.t_loss.item()
    
    def backward(self):
        self.t_loss.backward()

class affinePC_torch():

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
        - set up two optimisers
        - implement bounds
        - attach full affine matrix
        - clean up code

        Example:
        A_PC = affinePC(A, B)       # instanciate
        A_PC.fit(start, bounds)     # fit
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
        self.center_ct = torch.mean(pc_ct, axis=0)
        self.transform = torch.eye(4)
        self.params = torch.zeros(6)
        self.pc_ct_r = pc_ct
        self.pc_us_r = pc_us
        self.method = 'naive'
        self.loss = None

    def apply_aff(self):
        self.pc_ct_r = apply_affine_torch(self.pc_ct, 
                                          self.params, 
                                          self.center_ct)
        return self.pc_ct_r
    
    def apply_affinv(self):
        self.pc_us_r = apply_affine_inverse_torch(self.pc_us,
                                                  self.params,
                                                  self.center_ct)
        return self.pc_us_r
    
    def get_affine_transform(self):
        self.transform = affine_transform(self.params,
                                          self.center_ct)
        return self.transform
    
    def fit(self, start, method, 
            bounds=[(None, None)]*6, max_oper=1000, lr=0.5, tol=1e-7):
        """
        +======================================================================+
        Fit affine transform using Adam from torch.

        INPUT:
        start                   list(float) starting points affine transform
                                length (6)
        bounds                  list(tuple(float)) boundary locations of 
                                each parameter length (6, length (2))
        method                  (str) "lin_sum" : lin_sum assignment
                                      "naive"   : min dist assignment
                                      "auction" : action algo assignment
        max_oper                (int)
        lr                      (float) learning rate
        bounds                  (dummy)

        OUTPUT:
        params                  (np.array) optimised parameters for transform
        +======================================================================+
        """
        pc_us_ = self.pc_us
        pc_ct_ = self.pc_ct
        center_ct_ = self.center_ct
        result, loss = fit_affine_torch(pc_ct_, 
                                  pc_us_, 
                                  center_ct_, 
                                  start,
                                  bounds=bounds,
                                  method=method,
                                  max_oper=max_oper,
                                  lr=lr, 
                                  tol=tol)
        self.loss = loss
        self.params = result
        return self.params

@jit(nopython=True, nogil=True, parallel=True)
def auction_algorithm(a, b, p, e_start, e_end, e_fac):
    """
    Author: George Strong 04/2023
    Numba-based JIT-compiled and parallelised implementation of the auction algorithm

    Parameters
    ----------
    a : ndarray (num_samples, 3)
        Graph-space transformed trace(s) data with point cloud coordinate 
        in final dimension for set 1.
    b : ndarray (num_samples, 3)
        Graph-space transformed trace(s) data with point cloud coordinate 
        in final dimension for set 3.
    p : int
        The order of the p-norm used to calculate distances between points in a and b.
    e_start : float
        Starting minimum positive bidding increment.
    e_end : float
        Final minimum positive bidding increment.
    e_fac : float
        Factor by which the minimum positive bidding increment is reduced each e-scaling loop.
    
    Returns
    -------
    ndarray (num_receivers, num_samples)
        Optimal assignments between a and b. 

    """

    # store length of the sets
    length = a.shape[0]

    # save dimensions of complete unassigned array
    unassigned = np.arange(0, length)

    # calculate the benefit matrix between sets ai and bi
    benefit_matrix = np.empty((length, length), dtype=np.float32)
    for k in range(length):
        for l in range(length):
            benefit_matrix[k, l] = -(np.linalg.norm(a[k] - b[l], ord=p) ** p)

    # define object prices outside of e-scaling loops
    object_prices = np.zeros(length)

    # set the minimum positive bidding increment to the starting value
    e = e_start
    assignments = np.empty(length, dtype=np.int32) 

    while e > e_end:

        # create assignments array for current sets ai and bi
        assignments = np.empty(length, dtype=np.int32)  # object (represented by indices) to bidder assignment
        assignments[:] = -1  # if object is unassigned store -1

        # all bidders are unassigned to begin with and stored as indexes in unassigned
        u = unassigned.copy()
        # store the corresponding number of unassigned bidders
        num_u = len(u)

        while num_u > 0:

            # Bidding phase
            # ---------------------------------------------------------------------------------

            # randomly select a single unassigned bidder
            bidder = u[u >= 0][np.random.randint(0, num_u)]

            # calculate the total value of each object for the selected bidder, taking price into account
            values = benefit_matrix[bidder] - object_prices

            # calculate the best and second best values as well as the index of the best object
            # set initial best index as well as initial best and second best values

            Ji = 0  # initial index of best value
            Vi = values[Ji]  # initial best value
            Wi = values[Ji]  # initial second best value

            # loop over all objects
            for obj in range(len(values)):

                # calculate the value of the current object
                obj_value = values[obj]

                # check if the value of the current object is better than previous best value
                if obj_value > Vi:
                    # downgrade previous best value to second best value
                    Wi = Vi
                    # set the value of the current object as the new best value
                    Vi = obj_value
                    # update the index of the best
                    Ji = obj

                # if not better than previous best value, check if better than previous second best value
                elif obj_value > Wi:
                    # set the value of the current object as the new second best value
                    Wi = obj_value

            # calculate bidding increment for the best object
            Bi = Vi - Wi + e

            # Assignment phase
            # ---------------------------------------------------------------------------------

            # raise price of best object by bidding increment
            object_prices[Ji] += Bi

            # check if object Ji is already assigned to a bidder
            if assignments[Ji] >= 0:
                # return previously assigned bidder to unassigned array
                u[assignments[Ji]] = assignments[Ji]
                # increase the number of unassigned bidders by 1
                num_u += 1

            # assign bidder to best object
            assignments[Ji] = bidder
            # remove bidder from unassigned array
            u[bidder] = -1
            # deduct 1 from the number of unassigned bidders
            num_u -= 1

        # reduce minimum positive bidding increment by a factor of e_fac
        e /= e_fac

    
    return assignments

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

def naive_dist(a, b, p):
    """
    ===========================================================================+
    Find matching points from b to a using a minium distance assignment.

    INPUT
    a                           (np.array)(float) point cloud of to be moved obj
                                (NUM_PTS_A, 3)
    b                           (np.array)(float) point cloud of reference
                                (NUM_PTS_B, 3)
    p                           (int) order of the p-norm used to calculate 
                                distances between points in a and b.
                       
    OUTPUT
    d                           (np.array)(int) distances
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

def naive(a, b, p):
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

    # find assignments
    assignments = np.argmin(cost_matrix, axis=1)
    
    return assignments

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

def rot_mat_torch(x):
    """
    ===========================================================================+
    ===========================================================================+
    """
    R = torch.zeros((2,2))

    R[0,0] = torch.cos(x)
    R[0,1] = -torch.sin(x)
    R[1,0] = torch.sin(x)
    R[1,1] = torch.cos(x)

    return R

def apply_affine_torch(coordinates, affine_transform_params, center):
    """
    ===========================================================================+
    Apply affine transformation for coordinates with affine_transform_params, 
    with its rotation around center. First the object is translated to center 
    with TR, then the rotation R is applied (x, y, z), translate back (TR-1) 
    and the translation T is applied: 

    transformed_coords = T @ TR-1 @ Rz @ Ry @ Rz @ Rx @ TR @ coords

    with all operations in 4x4 matrix format.

    affine_transform_params contain:
    0: rotation around x axis [0.1 degree]
    1: rotation around y axis [0.1 degree]
    2: rotation around z axis [0.1 degree]
    3: translation along x axis [voxels or m]
    4: translation along y axis [voxels or m]
    5: translation along z axis [voxels or m]

    NOTE: rotation params are scaled to better match translational space!

    INPUT
    coordinates                 (torch.Tensor)(float) coordinate in voxel space 
                                or real space to transform (N,3)
    affine_transform_params     (list) affine transform parameters (6)
    center                      (center) point of rotation in voxel or real
                                space (3,)

    OUTPUT
    transf_coordinates          (torch.Tensor)(float) transformed coordinates in 
                                voxel or real space (N,3)
    ===========================================================================+
    """
    one = torch.Tensor([1])
    transform = affine_transform(affine_transform_params, center)

    return torch.stack([(transform @ torch.cat((coord,one)))[:3] for coord in coordinates], dim=0)

def apply_affine_inverse_torch(coordinates, affine_transform_params, center):
    """
    ===========================================================================+
    Apply affine transformation for coordinates with affine_transform_params, 
    with its rotation around center. First the object is translated to center 
    with TR, then the rotation R is applied (x, y, z), translate back (TR-1) 
    and the translation T is applied: 

    transformed_coords = T @ TR-1 @ Rz @ Ry @ Rz @ Rx @ TR @ coords

    with all operations in 4x4 matrix format.

    affine_transform_params contain:
    0: rotation around x axis [0.1 degree]
    1: rotation around y axis [0.1 degree]
    2: rotation around z axis [0.1 degree]
    3: translation along x axis [voxels or m]
    4: translation along y axis [voxels or m]
    5: translation along z axis [voxels or m]

    NOTE: rotation params are scaled to better match translational space!

    INPUT
    coordinates                 (torch.Tensor)(float) coordinate in voxel space 
                                or real space to transform (N,3)
    affine_transform_params     (list) affine transform parameters (6)
    center                      (center) point of rotation in voxel or real
                                space (3,)

    OUTPUT
    transf_coordinates          (torch.Tensor)(float) transformed coordinates in 
                                voxel or real space (N,3)
    ===========================================================================+
    """
    one = torch.Tensor([1])
    transform = affine_transform(affine_transform_params, center)
    inverse = torch.inverse(transform)

    return torch.stack([(inverse @ torch.cat((coord,one)))[:3] for coord in coordinates], dim=0)

def affine_transform(affine_transform_params, center):

    # stacks up the parameters of an affine transform and applies to a list of coordinates
    scale_rotparams = 1/1800*np.pi

    # Reset to center for rotations
    TR = torch.Tensor([[1,0,0,-center[0]],[0,1,0,-center[1]],[0,0,1,-center[2]],[0,0,0,1]])
    # push back to original pos for rotations
    TRinv =torch.Tensor([[1,0,0,center[0]],[0,1,0,center[1]],[0,0,1,center[2]],[0,0,0,1]])

    # rotations
    theta_x=affine_transform_params[0]*scale_rotparams # rescale these to allow updates
    theta_y=affine_transform_params[1]*scale_rotparams
    theta_z=affine_transform_params[2]*scale_rotparams

    # feed parameters into matrix such that grad is preserved
    R_x = torch.eye(4)
    R_y = torch.eye(4)
    R_z = torch.eye(4)
    R_x[(1,1,2,2),(1,2,1,2)] = rot_mat_torch(theta_x).reshape(-1)
    R_y[(0,0,2,2),(0,2,0,2)] = rot_mat_torch(theta_y).reshape(-1)
    R_z[(0,0,1,1),(0,1,0,1)] = rot_mat_torch(theta_z).reshape(-1)

    # translations
    T = torch.eye(4)
    T[(0,1,2),(3,3,3)] = affine_transform_params[3:]

    # scaling
    # s_x=affine_transform_params[6]
    # s_y=affine_transform_params[7]
    # s_z=affine_transform_params[8]
    # S=torch.Tensor([[s_x,0,0,0],[0,s_y,0,0],[0,0,s_z,0],[0,0,0,1]])

    transform = T @ TRinv @ R_x @ R_y @ R_z @ TR 

    return transform

def extract_ct_anchor_points(image_vol, 
                             NUM_PTS=5000, 
                             mode='outer', 
                             threshold_max=1500,
                             threshold_min=1000, 
                             start=0,
                             offset=torch.Tensor([0,0,0]),
                             Csphere=1.6):
    """
    ===========================================================================+
    Use image thresholding to find skull outline in image_vol
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
    the CT scan.

    This function places the origin of the coordinate system at the center of 
    the volume, offset with 'center'.

    The extracted coordinates are reported with the center of the array 
    as the origin.

    INPUT
    ct_image                    (torch.Tensor)(float) voxel space with vp
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
    offset                      (torch.Tensor)(float) center of skull to start 
                                from relative to center of array (3,)
    Csphere                     (float) coverage of sphere to detect in 
                                (1: full sphere, 2: hemisphere)

    OUTPUT
    coords                      (torch.Tensor)(int) coordinates of anchor points 
                                in voxel space (NUM_PTS, 3)
    ===========================================================================+
    """
    anchor_points=[]
    THRESHOLD_MAX=threshold_max
    if mode == 'outer':
        THRESHOLD_MIN=threshold_max+1
    THRESHOLD_MIN=threshold_min # only used with mode='inner'

    # create uniform sampling of a sphere
    indices = torch.arange(0, NUM_PTS, dtype=float) + 0.5
    Csphere = Csphere #1 full sphere, 2 top hemisphere (update to keep the same?)
    angles1 = torch.arccos(1 - 2*indices/(NUM_PTS*Csphere)) 
    angles2 = torch.pi * (1 + 5**0.5) * indices
    angles2 = torch.remainder(angles2, 360)

    # determine center and max radius in image_vol voxel cube for ray tracing
    shape_ct = torch.Tensor(list(image_vol.shape))
    max_r = float(torch.linalg.norm(torch.divide(shape_ct,2)))
    d_r = max_r/shape_ct.max()
    center_a = torch.divide(shape_ct,2)
    center_d = center_a + offset

    # for every ray along (theta_i, phi_i) determine the anchorpoint of skull
    for theta, phi in zip(angles1, angles2):
        if mode == 'outer':
            r=max_r
            outside_head=True
            while outside_head and r>0:
                xyz=torch.floor(center_d+torch.Tensor(
                                            [r*torch.sin(theta)*torch.cos(phi),
                                             r*torch.sin(theta)*torch.sin(phi),
                                             r*torch.cos(theta)])).int()
                if torch.all((xyz<shape_ct) & (xyz>=0)) \
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
                xyz=torch.floor(center_d+torch.Tensor(
                                            [r*torch.sin(theta)*torch.cos(phi),
                                             r*torch.sin(theta)*torch.sin(phi),
                                             r*torch.cos(theta)])).int()
                if torch.all((xyz<shape_ct) & (xyz>=0)) \
                        and (image_vol[xyz[0],xyz[1],xyz[2]] < THRESHOLD_MIN):
                    minFlag = True
                if torch.all((xyz<image_vol.shape) & (xyz>=0)) \
                        and (image_vol[xyz[0],xyz[1],xyz[2]] > THRESHOLD_MAX) \
                        and minFlag:
                    inside_head=False
                    minFlag=False
                    anchor_points.append(xyz)
                else:
                    r+=d_r
           
    # remove duplicate anchorpoints (if any)
    anchor_pts_clean = torch.unique(torch.stack(anchor_points, dim=0), dim=0)
    return anchor_pts_clean - \
           torch.ones_like(anchor_pts_clean)*center_a.reshape(1,-1)

def fit_affine_torch(anchor_pts_ct, 
                     anchor_pts_us, 
                     center, 
                     params,
                     bounds=[(None,None)]*6,
                     method='naive',
                     max_oper=1000,
                     lr=0.5, 
                     tol=1e-7):
    """
    ===========================================================================+
    Adam optimizer to search optimal alignment of two point clouds 
    anchor_pts_ct, anchor_pts_us by optimizing parameters

    For affine_transform_parameters see apply_affine()

    INPUT
    params                      (torch.Tensor)(float) affine transform 
                                parameters (6)
    anchor_pts_ct               (torch.Tensor)(int) CT anchor points
                                (NUM_PTS, 3)
    anchor_pts_us               (torch.Tensor)(int) Reference anchor points
                                (N, 3)
    center                      (torch.Tensor)(int) center of rotation (3,)
    method                      (str) "naive", "auction", "lin_sum"
    max_oper                    (int) maximum number of iterations
    lr                          (float) learning rate
    tol                         (float) tolerance

    OUTPUT
    params                      (torch.Tensor)(float) optimized parameters
    ===========================================================================+
    """
    # turn grads on parameters on and off on inputs
    params.requires_grad = True
    anchor_pts_ct.requires_grad = False
    anchor_pts_us.requires_grad = False
    center.requires_grad = False

    # configure optimiser
    adam = torch.optim.Adam([params, anchor_pts_ct, anchor_pts_us, center], 
                             lr=lr)
    L = OptimalTransportLoss(method=method)
    i = 0 
    prev = 1e16

    # run optimisation 
    for i in tqdm(range(0,max_oper)):

        adam.zero_grad()
        l = L.forward(params, anchor_pts_ct, anchor_pts_us, center)
        delta_loss = prev - l

        if i == max_oper-1 or delta_loss < tol:
            params_disp = params.detach()
            print("Loss: \t\t{:.2f}".format(l))

            if disp:
                print(f"Tolerance {tol} reached, return solution:")
                print(f"rot_x \t {params_disp[0]/10:.2f} [deg]")
                print(f"rot_y \t {params_disp[1]/10:.2f} [deg]")
                print(f"rot_z \t {params_disp[2]/10:.2f} [deg]")
                print(f"t_x \t {params_disp[3]:.2f} [vox]")
                print(f"t_y \t {params_disp[4]:.2f} [vox]")
                print(f"t_z \t {params_disp[5]:.2f} [vox]")
            break
        
        if i % 10 == 0 and disp:
            params_disp = params.detach()
            print("{} =========================".format(i))
            print("params: {}".format(params_disp))
            print("Loss: \t\t{:.2f}".format(l))
            print("Delta Loss: \t{}".format(delta_loss))
            print("============================")

        # update
        prev = l
        i += 1

        # step gradient
        L.backward()
        adam.step()

        # clamp bounds?
        for j in range(len(params)):
            if bounds[j][0] == None and bounds[j][1] == None:
                continue
            with torch.no_grad():
                params[j] = params[j].clamp(min=bounds[j][0],max=bounds[j][1])

    return params_disp, l

def grid_search_angle(pc_rtm, pc_sk, bounds, **kwargs):

    max_oper = kwargs.pop('max_oper', 100)
    N = kwargs.pop('num_gridpoints', 5)
    lr = kwargs.pop('learn_rate', 0.5)
    save = kwargs.pop('save', False)

    bdx = bounds[0]
    bdy = bounds[1]
    bdz = bounds[2]
    
    loss_mat = np.ones((N,N,N))*1e9
    config_mat = np.zeros((N,N,N,6))

    for i, r_i in enumerate(np.linspace(bdx[0],bdx[1],N)):
        for j, r_j in enumerate(np.linspace(bdy[0],bdy[1],N)):
            for k, r_k in enumerate(np.linspace(bdz[0],bdz[1],N)):
                
                print(f"{i},{j},{k}")

                #init
                start = torch.zeros(6)
                start[0] = r_i*10
                start[1] = r_j*10
                start[2] = r_k*10

                # bring CM back
                #start[3:] = start_tx

                # bounds
                dphi = 0
                dz = 50
                bounds = []
                for p in start[:3]:
                    bounds.append((p-dphi, p+dphi))
                for p in start[3:]:
                    bounds.append((p-dz, p+dz))
                
                # create
                A = affinePC_torch(torch.Tensor(pc_rtm), torch.Tensor(pc_sk))

                # fit
                A.fit(start=start, 
                        bounds=bounds,
                        method='naive',
                        max_oper=max_oper,
                        lr=lr)
                
                # save
                loss_mat[i,j,k] = A.loss
                opt = A.params.detach()
                config_mat[i,j,k,0] = opt[0]
                config_mat[i,j,k,1] = opt[1]
                config_mat[i,j,k,2] = opt[2]
                config_mat[i,j,k,3] = opt[3]
                config_mat[i,j,k,4] = opt[4]
                config_mat[i,j,k,5] = opt[5]

    # report
    if save:
        np.save('loss_gridsearch-fine.npy',loss_mat)
        np.save('config_gridsearch-fine.npy',config_mat)
    print(f'min loss: {loss_mat.min()}')
    i,j,k = np.unravel_index(loss_mat.argmin(), loss_mat.shape)
    print(f'opt config: {config_mat[i,j,k]}')

    return loss_mat.min(), config_mat[i,j,k]

if __name__=="__main__":

    # set up model
    shape = (287, 331, 260)
    extra = (40, 40, 40)
    absorbing = (30, 30, 30)
    spacing = (0.75e-3, 0.75e-3, 0.75e-3)
    space = Space(shape=shape,
                  extra=extra,
                  absorbing=absorbing,
                  spacing=spacing)
    start = 0.
    step = 0.15e-6
    num = 2000
    timey = Time(start=start,
                step=step,
                num=num)
    problem = Problem(name='alpha3D',
                      space=space, time=timey)
    true_model = ScalarField(name='true_model', grid=problem.grid)
    true_model_np = np.load('data/phan_ss_skull.npy')
    true_model = torch.Tensor(true_model_np)

    # trial transform parameters
    # center = torch.divide(torch.pad(true_model.data,40).shape, 2)
    theta_x = 0.1
    move_x = 30

    # get anchorpoints 
    pad = 40
    padding = (pad,pad,pad,pad,pad,pad)
    offset = torch.Tensor([0,0,100])
    """
    anchor_pts = extract_ct_anchor_points(torch.nn.functional.pad(true_model,
                                                    padding, value=1480.), 
                                                    NUM_PTS=1024,
                                                    threshold_max=2000,
                                                    Csphere=1.6,
                                                    offset=offset) # observed anchor points, very few but correctly positioned
    anchor_pts_REF = extract_ct_anchor_points(torch.nn.functional.pad(true_model,
                                                    padding, value=1480.), 
                                                    NUM_PTS=5000,
                                                    threshold_max=2000,
                                                    Csphere=1.6,
                                                    offset=offset)
    center = torch.mean(anchor_pts_REF, axis=0)
    anchor_pts_REF_rotshifted = apply_affine(anchor_pts_REF, torch.Tensor([theta_x, 0, 0, move_x, 0, 0]), center)
    

    # save starting points
    torch.save(anchor_pts, 'output/test_us_anchor_pts.pt')
    torch.save(anchor_pts_REF_rotshifted, 'output/test_ct_anchor_pts.pt')
    """

    # load starting points
    anchor_pts_ct_all=torch.load('output/test_ct_anchor_pts.pt')
    anchor_pts_us=torch.load('output/test_us_anchor_pts.pt')

    #anchor_pts_ct_all = torch.Tensor([[0,1,1], [1,2,0], [1,-1,-0.5], [3,4,-1]])
    #anchor_pts_us = torch.Tensor([[0,1,1], [1,2,0], [1,-1,-0.5]])
    center = torch.mean(anchor_pts_ct_all, axis=0)

    anchor_pts_ct_rot = apply_affine_torch(anchor_pts_ct_all, torch.Tensor([0, 0, 0, 0, 0, 0]), center)
    #center = torch.mean(anchor_pts_ct_rot, axis=0)

    fig = plot_PCs([anchor_pts_us, anchor_pts_ct_rot])
    fig.show()  
 
    # find optimal transform
    affPCUS_CT = affinePC_torch(anchor_pts_us, anchor_pts_ct_rot)
    params = affPCUS_CT.fit(start=torch.Tensor([0,0,0,0,0,0]),
                            method='lin_sum')
    affPCUS_CT.apply_aff()
    print("Best parameters: {}".format(params))

    # plot pointclouds
    fig = plot_PCs([anchor_pts_us, affPCUS_CT.pc_ct_r])
    fig.show()  
    
    # save transform
    #torch.save(affPCUS_CT.pc_ct_r, 'output/test_ct_anchor_pts_registeredThresh.pt')

