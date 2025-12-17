import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import CubicSpline
from scipy.special import softmax
from scipy.optimize import minimize

class WeightedSimCCPoseRefiner:
    def __init__(self, K, dist, model_input_size):
        self.K = K
        self.dist = dist
        self.w_model, self.h_model = model_input_size
        self.splines_x = []
        self.splines_y = []
        self.weights = [] 
        self.simcc_scale_x = 1.0
        self.simcc_scale_y = 1.0

    def prepare_pdf_and_weights(self, logits_x, logits_y, sigma=2.0):
        """
        Generate PDF(via Spline) and weights.
        Boundary Filtering: Weight = 0 if Max Index is on the boundary
        """
        self.splines_x = []
        self.splines_y = []
        self.weights = []
        
        n_kpts, n_bins_x = logits_x.shape
        _, n_bins_y = logits_y.shape
        
        # Softmax
        probs_x_all = softmax(logits_x, axis=1)
        probs_y_all = softmax(logits_y, axis=1)
        
        # Confidence & Argmax
        conf_x = np.max(logits_x, axis=1)
        conf_y = np.max(logits_y, axis=1)
        
        idx_x = np.argmax(probs_x_all, axis=1)
        idx_y = np.argmax(probs_y_all, axis=1)

        # Weight Calculation (Soft Weighting + Boundary Filtering)
        raw_weights = np.minimum(conf_x, conf_y)
        
        # Boundary filtering
        is_boundary_x = (idx_x <= 5) | (idx_x >= n_bins_x - 6)
        is_boundary_y = (idx_y <= 5) | (idx_y >= n_bins_y - 6)
        is_boundary = is_boundary_x | is_boundary_y
        
        raw_weights[is_boundary] = 0.0
        
        self.weights = raw_weights
        
        # PDF (Spline)
        self.simcc_scale_x = n_bins_x / self.w_model
        self.simcc_scale_y = n_bins_y / self.h_model
        epsilon = 1e-6 

        for i in range(n_kpts):
            prob_x = gaussian_filter1d(probs_x_all[i], sigma=sigma)
            prob_y = gaussian_filter1d(probs_y_all[i], sigma=sigma)
            
            log_prob_x = np.log(np.clip(prob_x, epsilon, 1.0))
            log_prob_y = np.log(np.clip(prob_y, epsilon, 1.0))
            
            x_axis = np.arange(n_bins_x)
            y_axis = np.arange(n_bins_y)
            
            self.splines_x.append(CubicSpline(x_axis, log_prob_x))
            self.splines_y.append(CubicSpline(y_axis, log_prob_y))

    def objective_function(self, params, points_3d, crop_meta):
        """
        Weighted NLL Minimization
        """
        rvec, tvec = params[:3], params[3:]
        
        img_points, _ = cv2.projectPoints(points_3d, rvec, tvec, self.K, self.dist)
        img_points = img_points.squeeze()
        
        x1, y1 = crop_meta['crop_box'][:2]
        scale = crop_meta['scale_factor']
        if isinstance(scale, (list, np.ndarray)): scale = np.array(scale)
        
        model_pts = (img_points - np.array([x1, y1])) * scale
        simcc_pts_x = model_pts[:, 0] * self.simcc_scale_x
        simcc_pts_y = model_pts[:, 1] * self.simcc_scale_y
        
        weighted_nll_sum = 0.0
        n_bins_x = len(self.splines_x[0].x)
        n_bins_y = len(self.splines_y[0].x)
        penalty_base = 100.0

        for i in range(len(self.splines_x)):
            w = self.weights[i]
            if w < 1e-6: continue # Boundary Filtered

            ux, uy = simcc_pts_x[i], simcc_pts_y[i]
            
            if not (0 <= ux < n_bins_x):
                val_x = -(penalty_base + abs(ux - n_bins_x/2)*0.1)
            else:
                val_x = self.splines_x[i](ux)

            if not (0 <= uy < n_bins_y):
                val_y = -(penalty_base + abs(uy - n_bins_y/2)*0.1)
            else:
                val_y = self.splines_y[i](uy)
            
            weighted_nll_sum -= w * (val_x + val_y)

        return weighted_nll_sum

    def refine(self, rvec_init, tvec_init, points_3d, crop_meta):
        initial_params = np.concatenate([rvec_init.flatten(), tvec_init.flatten()])
        
        res = minimize(
            self.objective_function,
            initial_params,
            args=(points_3d, crop_meta),
            method='Nelder-Mead',
            options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-4, 'disp': False}
        )
        return res.x[:3], res.x[3:]