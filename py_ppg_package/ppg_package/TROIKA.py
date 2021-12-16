import typing

import numpy as np
from numpy.lib.histograms import _hist_bin_sqrt
from numpy.polynomial import Polynomial
from pprint import pprint
import pypg
from sklearn.metrics import pairwise_distances
from sklearn.cluster import AgglomerativeClustering
import scipy
from scipy.linalg import hankel
from scipy.optimize import minimize
from tqdm import tqdm

import pandas as pd

def ssa(ts: np.ndarray, L: int, perform_grouping: bool = True, wcorr_threshold: float = 0.3, ret_Wcorr: bool = False):
    """
    Performs SSA on ts
    https://www.kaggle.com/jdarcy/introducing-ssa-for-time-series-decomposition

    Parameters
    ----------
        ts : ndarray of shape (n_timestamps, )
            The time series to decompose
        L : int
            first dimension of the L-trajectory-matrix
        grouping : bool, default=True
            If True, perform grouping based on the w-correlations of the deconstructed time series
            using agglomerative hierarchical clustering with single linkage.
            If this parameter is True, the parameter distance_threshold must be set.
        wcorr_threshold : float, default=0.3
            The w-correlation threshold used with the agglomerative hierarchical clustering.
            Time series with at least this w-correlation will be grouped together.
            This parameter will be ignored if grouping is set to False.
        ret_Wcorr : bool, default=False
            Whether the resulting w-correlation matrix should be returned.
            If grouping is enabled, return the w-correlation matrix of the grouped time series.
            If grouping is disabled, return the w-correlation matrix of the ungrouped time series.
        

    Returns
    ----------
        Y : ndarray of shape (n_groups, n_timestamps) if grouping is enabled and (L, n_timestamps) if it is disabled.
        Wcorr : ndarray
            The Wcorrelation matrix.
            Wcorr will only be returned if ret_Wcorr is True
    """
    N = len(ts)
    K = N - L + 1

    L_trajectory_matrix = hankel(ts[:L], ts[L-1:]) # (L, K)
    U, Sigma, V = np.linalg.svd(L_trajectory_matrix) # (L, L); (d, ); (K, K)
    V = V.T # (K, K)
    d = len(Sigma)

    deconstructed_ts = []
    for i in range(d):
        X_elem = np.array(Sigma[i] * np.outer(U[:,i], V[:,i])) # (L, K)
        X_elem_rev = X_elem[::-1] # (L, K)
        ts_i = np.array([X_elem_rev.diagonal(i).mean() for i in range(-L+1, K)])
        deconstructed_ts.append(ts_i)
    deconstructed_ts = np.array(deconstructed_ts) # (d, L, K)
    
    if not perform_grouping and not ret_Wcorr:
        return deconstructed_ts
    

    w = np.concatenate((np.arange(1, L+1), np.full((K-L,), L), np.arange(L-1, 0, -1)))
    def wcorr(ts1: np.ndarray, ts2: np.ndarray) -> float:
        """
        weighted correlation of ts1 and ts2.
        w is precomputed for reuse.
        """
        w_covar = (w * ts1 * ts2).sum()
        ts1_w_norm = np.sqrt((w * ts1 * ts1).sum())
        ts2_w_norm = np.sqrt((w * ts2 * ts2).sum())
        
        return w_covar / (ts1_w_norm * ts2_w_norm)

    Wcorr_mat = pairwise_distances(deconstructed_ts, metric=wcorr)

    if not perform_grouping:
        return deconstructed_ts, Wcorr_mat

    Wcorr_mat_dist = 1 - Wcorr_mat
    distance_threshold = 1 - wcorr_threshold
    agg_clust = AgglomerativeClustering(affinity='precomputed', linkage='single',
                                        distance_threshold=distance_threshold, n_clusters=None)
    clust_labels = agg_clust.fit_predict(Wcorr_mat_dist)
    n_clusters = clust_labels.max() + 1
    grouped_ts = [np.sum(deconstructed_ts[clust_labels == cluster_id], axis=0) 
                  for cluster_id in range(n_clusters)]
    grouped_ts = np.array(grouped_ts)
    
    if not ret_Wcorr:
        return grouped_ts
    
    Wcorr_mat = pairwise_distances(grouped_ts, metric=wcorr)

    return grouped_ts, Wcorr_mat

class SSR:
    def __init__(self, M: int, N: int, f_s: float):
        # delta_f1 = 0.4 / f_s * N - 1
        # delta_f2 = 2 / f_s * N
        # I_phi = np.concatenate((
        #     np.arange(np.around(0.4 / f_s * N - delta_f1), np.around(5 / f_s * N + delta_f2) + 1, dtype=int),
        #     np.arange(np.around(N - 5 / f_s * N - delta_f2), np.around(N - 0.4 / f_s * N + delta_f1) + 1, dtype=int)
        # ))
        # self.Phi = np.zeros((M, N), dtype='complex128')
        # M_idx = np.arange(M)
        # self.Phi[:, I_phi] = np.e ** (1j * 2 * np.pi / N * np.outer(M_idx, I_phi))

        self.M = N
        self.N = N
        delta_f1 = 0.4 / f_s * N - 1
        delta_f2 = 2 / f_s * N
        # I_phi_low = np.arange(0, np.ceil(5 / f_s * N + delta_f2) + 1, dtype=int)
        # I_phi_high = np.arange(np.floor(N - 5 / f_s * N - delta_f2), N, dtype=int)
        # I_phi = np.concatenate([I_phi_low, I_phi_high])
        I_phi = np.arange(np.ceil(5 / f_s * N + delta_f2), np.floor(N - 5 / f_s * N - delta_f2), dtype=int)
        self.Phi = np.zeros((M, N), dtype='complex128')
        self.Phi[:,I_phi] = np.e ** (1j * 2 * np.pi / N * np.arange(M)[:, np.newaxis] * I_phi[np.newaxis, :])
        self.phi_pinv = np.linalg.pinv(self.Phi)

    def transform(self, y: np.ndarray):
        def print_current_target(xk):
            print(f"Sparsity: {np.linalg.norm(xk, ord=1)}, estimation quality: {np.linalg.norm(self.Phi @ xk - y, ord=2)}")

        from ppg_package import plot_spectrum, plot_ppg

        _, periodogram = scipy.signal.periodogram(y, nfft=self.N * 2 - 1)

        x0 = self.phi_pinv @ y
        print(f"Sparsity: {np.linalg.norm(x0, ord=1)}, estimation quality: {np.linalg.norm(self.Phi @ x0 - y, ord=2)}")

        constraints = {
            'type': 'eq',
            'fun': lambda x: np.linalg.norm(self.Phi @ x - y, ord=2)
        }
        optimize_result = minimize(lambda x: np.linalg.norm(x, ord=1),
                                   x0, method='SLSQP',
                                   options={'maxiter': 5},
                                   constraints=constraints,
                                   callback=print_current_target)
        s_k = optimize_result.x ** 2
        breakpoint()
        plot_spectrum(pd.DataFrame({'periodogram': periodogram,
                                    'optimize_result': optimize_result.x,
                                    's_k': s_k}))

        return s_k

class SpectralPeakTracker:
    def __init__(self, n_freq_bins=4096, ppg_sampling_freq=125, delta_s_rel=16/4069, eta=.3, tau=0.00048828125, theta=0.00146484375):
        self.n_freq_bins = n_freq_bins
        self.ppg_sampling_freq = ppg_sampling_freq
        
        # parameters for stage 2 (peak selection)
        self.delta_s = int(n_freq_bins * delta_s_rel)
        self.eta = eta
        
        # parameters for stage 3.1 (verification)
        self.tau = int(n_freq_bins * tau)
        self.theta = int(n_freq_bins * theta)

        self.history = [] # frequency indices

    def _get_N0_N1(self, spectrum):
        N_prev = self.history[-1]
        R_0_base_idx = N_prev - self.delta_s
        R_0_end_idx = N_prev + self.delta_s
        R_1_base_idx = 2 * (N_prev - self.delta_s - 1) + 1
        R_1_end_idx = 2 * (N_prev + self.delta_s - 1) + 1
        R_0_idx = np.arange(R_0_base_idx, R_0_end_idx + 1)
        R_1_idx = np.arange(R_1_base_idx, R_1_end_idx + 1)
        R_0 = R_0[0 <= R_0_idx < len(spectrum)]
        R_1 = R_1[0 <= R_1_idx < len(spectrum)]

        N_0 = np.argpartition(spectrum[R_0], -3)[-3:] + R_0_base_idx
        threshold = self.eta * np.max(spectrum[N_0])
        N_1 = np.argpartition(spectrum[R_1], -3)[-3:] + R_1_base_idx
        N_0 = N_0[spectrum[N_0] >= threshold] # N_0 cannot be empty since tau_peak is calculated based on R_0
        N_1 = N_1[spectrum[N_1] >= threshold] # N_1 can be empty

        return N_0, N_1

    def _get_N_hat(self, N_0, N_1):
        N_prev = self.history[-1]
        
        # Case 1
        N_hat = None
        for n_0 in N_0:
            for n_1 in N_1:
                if n_0 % n_1 == 0 or n_1 % n_0 == 0:
                    if N_hat is None or np.abs(N_hat - N_prev) > np.abs(n_0 - N_prev):
                        N_hat = n_0

        # Case 2
        if N_hat is None:
            Nf_set = np.concatenate((N_0, (N_1 - 1) / 2))
            N_hat_idx = np.argmin(np.abs(Nf_set - N_prev))
            N_hat = Nf_set[N_hat_idx]

        return N_hat

    def _verification_stage_1(self, N_hat):
        N_prev = self.history[-1]
        if N_hat - N_prev >= self.theta:
            N_cur = N_prev + self.tau
        elif N_hat - N_prev <= -self.tau:
            N_cur = N_prev - self.tau
        else:
            N_cur = N_hat

        return N_cur

    # def _verification_stage_2(self, N_cur):
    #     N_prev = self.history[-1]
    #     BPM_prev = N_cur / self.spectrum_len * 125 * 60
    #     history_BPM = self.histroy / self.spectrum_len * 125 * 60
    #     if len(history_BPM) < 3:
    #         missing_edge_len = len(history_BPM) - 3 / 2
    #         poly_y = np.pad(history_BPM,
    #                         (np.floor(missing_edge_len), np.ceil(missing_edge_len)),
    #                         mode='edge')
    #     else:
    #         poly_y = history_BPM
    #     poly_x = np.arange(len(history_BPM))
    #     polynomial = Polynomial.fit(poly_x, poly_y, 3)
    #     BPM_predict = polynomial(BPM_prev)
    #     if BPM_predict - BPM_prev >= 3:
    #         N_trend = 1
    #     elif BPM_predict - BPM_prev <= -3:
    #         N_trend = -1
    #     else:
    #         N_trend = 0
        
    #     N_cur = N_prev + 2 * N_trend
    #     return N_cur

    def transform_first(self, spectrum: np.ndarray):
        N_cur = np.argmax(spectrum)
        self.history.append(N_cur)
        
        return N_cur

    def transform(self, spectrum: np.ndarray):
        N_0, N_1 = self._get_N0_N1()
        N_hat = self._get_N_hat(N_0, N_1)
        N_cur = self._verification_stage_1(N_hat)
        self.history.append(N_cur)

        return N_cur

class Troika:
    def __init__(self, window_duration=8, step_duration=2, ppg_sampling_freq=125, acc_sampling_freq=125, cutoff_freqs=[0.4, 5]):
        self.window_duration = window_duration
        self.step_duration = step_duration
        self.sampling_freq = acc_sampling_freq
        self.cutoff_freqs = cutoff_freqs
        self.ppg_sampling_freq = ppg_sampling_freq
        self.acc_sampling_freq = acc_sampling_freq
        self.ppg_window_len = window_duration * ppg_sampling_freq
        self.acc_window_len = window_duration * acc_sampling_freq
        self.ppg_step_len = step_duration * ppg_sampling_freq
        self.acc_step_len = step_duration * acc_sampling_freq
        self.n_freq_bins = 4096
        self.ssr = SSR(self.ppg_window_len, self.n_freq_bins, self.ppg_sampling_freq)

    def _get_current_window_bounds(self, cur_window: int, n_ppg_samples: int, n_acc_samples: int):
        ppg_low_bound = (cur_window - 1) * self.ppg_step_len
        ppg_high_bound = min(ppg_low_bound + self.ppg_window_len, n_ppg_samples)
        acc_low_bound = (cur_window - 1) * self.acc_step_len
        acc_high_bound = min(acc_low_bound + self.acc_window_len, n_acc_samples)

        return (ppg_low_bound, ppg_high_bound), (acc_low_bound, acc_high_bound)

    def _get_dominant_frequencies(self, spectrum: np.ndarray, axis=-1, threshold=.5):
        """
        Given the frequency spectra of one or multiple signals, compute the dominant frequencies along a specified axis.

        Parameters
        ----------
            sig : ndarray
                The signals to compute the dominant frequencies on.
            axis : int, default=-1
                The axis along which to compute the dominant frequencies.
            threshold : float, default=0.5
                The threshold which divides dominant and non-dominant frequencies.
                A dominant frequency has a peak of amplitude of higher than threshold times the maximum amplitude in that spectrum

        Returns
        ----------
            dom_freqs : ndarray
                dom_freqs has the same shape as sig.
                Iff a value in sig corresponds to a dominant frequency, this value is set to True in dom_freqs.
        """

        max_amplitudes = np.max(spectrum, axis=axis, keepdims=True)
        dom_freqs = spectrum > threshold * max_amplitudes

        return dom_freqs

    def _temporal_difference(self, ts, k):
        """
        Perform the kth-order temporal-difference operation on the time series ts.
        The first-order temporal difference of a time series [h_1, h_2, ..., h_n] is
        another time series given by [h_2 - h_1, ..., h_n - h_(n-1)]
        and the kth order temporal difference is given by the first-order difference of the order k-1 difference.

        Parameters
        ----------
            ts : one-dimensional ndarray
                input array to compute the temporal difference on
            k : int
                the order

        Returns
        ----------
            ts_hat : ndarray of length len(ts) - k
                The computed temporal difference
        """
        res_len = len(ts) - k
        agg = ts[:res_len]
        for i in range(1, k+1):
            agg = ts[i:i+res_len] - agg
            
        return agg

    def _get_Facc(self, acc_window: np.ndarray, prev_window_hr_idx: int, delta: int = 10):
        _, acc_freqs = scipy.signal.periodogram(acc_window, nfft=self.n_freq_bins * 2 - 1)
        acc_dom_freqs = self._get_dominant_frequencies(acc_freqs)
        F_acc = np.logical_or.reduce(acc_dom_freqs)
        N_p = np.arange(prev_window_hr_idx, self.n_freq_bins + 1, prev_window_hr_idx)
        for idx in N_p:
            F_acc[idx-delta : idx+delta] = False
        
        return F_acc       

    def _filter_ssa_groups(self, ssa_groups: np.ndarray, F_acc: np.ndarray):
        ssa_groups_spectra = scipy.signal.periodogram(ssa_groups, nfft=self.n_freq_bins * 2 - 1)
        ssa_dom_freqs = self._get_dominant_frequencies(ssa_groups_spectra)
        group_filter = np.logical_or.reduce(np.logical_and(ssa_dom_freqs, F_acc), axis=1)
        group_filter = np.logical_not(group_filter)

        return ssa_groups[group_filter]

    def transform(self, ppg: np.ndarray, acc: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
            ppg : ndarray of shape (n_channels, n_timestamps)
                The PPG signal.
            acc : ndarray of shape (n_dimensions, n_timestamps)
                The accelaration data used for denoising.
        """
        n_ppg_samples = ppg.shape[-1]
        n_acc_samples = acc.shape[-1]

        # apply bandpass filter
        ppg = pypg.filters.butterfy(ppg, self.cutoff_freqs, self.sampling_freq) # (n_ppg_channels, n_timestamps)

        n_ppg_windows = int(np.ceil((n_ppg_samples - self.ppg_window_len) / self.ppg_step_len))
        n_acc_windows = int(np.ceil((n_acc_samples - self.acc_window_len) / self.acc_step_len))
        assert n_acc_windows == n_ppg_windows, "The given ppg data and ppg sampling frequency do not have the same number of windows as the given acc signal and acc sampling frequency"
        n_windows = n_acc_windows
        current_window = 1
        progress_bar = tqdm(total=n_windows, initial=current_window)

        spt = SpectralPeakTracker()

        while current_window <= n_windows:
            (ppg_l, ppg_h), (acc_l, acc_h) = self._get_current_window_bounds(current_window, n_ppg_samples, n_acc_samples)
            progress_bar.set_description(f"Calculating window {current_window}/{n_windows}")

            ppg_window = ppg[ppg_l:ppg_h]
            acc_window = acc[acc_l:acc_h]

            if current_window == 1:
                sparse_ppg_spectrum = self.ssr.transform(ppg_window)
                prev_window_hr_idx = spt.transform_first(sparse_ppg_spectrum)
                
                yield prev_window_hr_idx / self.n_freq_bins * self.ppg_sampling_freq
            else:
                F_acc = self._get_Facc(acc_window, prev_window_hr_idx)
                ssa_groups = ssa(ppg, 400, perform_grouping=True)
                filtered_ssa_groups = self._filter_ssa_groups(ssa_groups, F_acc)
                filtered_ppg = np.sum(filtered_ssa_groups, axis=0)
                temporal_difference = self._temporal_difference(filtered_ppg, 2)
                sparse_ppg_spectrum = self.ssr.transform(temporal_difference)
                prev_window_hr_idx = spt.transform(sparse_ppg_spectrum)
                
                yield prev_window_hr_idx / self.n_freq_bins * self.ppg_sampling_freq

            current_window += 1
            progress_bar.update()
            
        progress_bar.close()
