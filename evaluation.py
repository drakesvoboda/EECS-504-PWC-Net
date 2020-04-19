import numpy as np

def flow_error_map(F_gt, F_est):
    F_gt_du = F_gt[:,:,0]
    F_gt_dv = F_gt[:,:,1]
    F_gt_val = F_gt[:,:,2]

    F_est_du = F_est[:,:,0]
    F_est_dv = F_est[:,:,1]

    E_du = F_gt_du-F_est_du
    E_dv = F_gt_dv-F_est_dv
    E = np.sqrt(E_du*E_du+E_dv*E_dv)
    E[F_gt_val==0] = 0

    return E, F_gt_val

def flow_error(F_gt, F_est, tau=None):
    if tau is None:
        tau = np.asarray([3.0, 0.05])

    E, F_val = flow_error_map(F_gt, F_est)
    F_mag = np.sqrt(F_gt[:,:,0]*F_gt[:,:,0]+F_gt[:,:,1]*F_gt[:,:,1])+1e-16
    n_total = np.sum(F_val!=0)
    n_1 = F_val!=0
    n_2 = E > tau[0]
    n_3 = E/F_mag > tau[1]
    n_err = np.sum(n_1*n_2*n_3)
    f_err = n_err/n_total

    aepe = np.sum(E)/n_total

    return f_err, aepe
