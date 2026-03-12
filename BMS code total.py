import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import tkinter as tk
from tkinter import filedialog
from tqdm import tqdm
from scipy.interpolate import interp1d, PchipInterpolator
from scipy.optimize import minimize, minimize_scalar, differential_evolution
from scipy.linalg import lu_factor, lu_solve
from scipy.integrate import quad
import time

# ==============================================================================
# CONFIGURATION & PATHS
# ==============================================================================
OCP_PATH = #Choose your path"

ENABLE_THERMAL_R_IMPACT = True   
ENABLE_THERMAL_Q_IMPACT = True   
ENABLE_BMS = True                
ENABLE_WARMUP = False
E_ACTIVATION_R = 30000.0         # Thermal sensitivity of R (J/mol), mean value 


# --- SYSTEM EFFICIENCIES ---
EFFICIENCY_DELIVERY_POINT = 0.99 
EFFICIENCY_TRANSFORMER = 0.98    
EFFICIENCY_PCS = 0.95            
EFF_TOTAL = EFFICIENCY_DELIVERY_POINT * EFFICIENCY_TRANSFORMER * EFFICIENCY_PCS # Global efficiency


# Aliases for flexible CSV column reading
COLUMN_ALIASES = {
    'Time': ['Time', 'time', 't', 'Test_Time(s)', 't_ch_power', 'Temps', 'Time [s]'],
    'Current': ['Current', 'current', 'I', 'i', 'Current(A)', 'I_in_A', 'I_ch_power', 'Current [A]'],
    'Voltage': ['Voltage', 'voltage', 'V', 'v', 'Voltage(V)', 'V_in_V', 'U_ch_power', 'Voltage [V]'],
    'Power': ['Power', 'power', 'P', 'p', 'Power(W)', 'P_in_W', 'P_ch_power', 'Puissance', 'Power [W]'],
    'Temp_Amb': ['T_amb', 'T_env', 'Temperature', 'T_Room_in_C', 'Amb_Temp'],
    'Temp_Bat': ['T_bat', 'T_cell', 'T_Cell', 'T_Bat_in_C', 'Temp_Bat', 'Temperature (C)_1']
}

# --- Pack Configuration ---
Ns = 14
Np = 2
FORCED_DT = 1.0
#P_BMS_IDLE_W = 0
P_BMS_IDLE_W = 0.08 #BMS consumption power

# --- Cell Physical Parameters ---
H_cell, W_cell, T_cell = 0.02722, 0.174, 0.0207
Area_cell = 2 * (H_cell * W_cell) + 2 * (H_cell * T_cell) + 2 * (W_cell * T_cell)
m_cell = 2.1
k_thermal_thickness = 0.91
R_cond_cell = (T_cell / 2) / (k_thermal_thickness * (H_cell * W_cell))

# --- Nominal Parameters ---
FACTOR_R_DYN = 1.0
R_HARNESS_CHARGE = 0.01
R_HARNESS_DISCHARGE = 0.00
QGes_Ah = 100
Q_ref_RC = 1.0189
SOC0_USER = 0.75
Q_oh_USER = 0.02
R0_cell_nom = 0.3/1000

# --- Statistical Parameters ---
sigma_rel_Q = 0.0039
sigma_rel_R = 0.022
R_weld_mean = 0
delta_R_weld = 0
R_busbar = 0

sigma_Q = sigma_rel_Q * QGes_Ah
sigma_R_cell = sigma_rel_R * R0_cell_nom
sigma_R_weld = (delta_R_weld / 1.645)
sigma_R_tot = np.sqrt(sigma_R_cell**2 + sigma_R_weld**2)
R_tot_mean = R0_cell_nom + R_busbar + R_weld_mean

show_bms_logs = False

# --- Model Parameters ---
n_particle = 30
r_dis_type = "weibull_cap"
COOLING_SYSTEM = "AIR"

if COOLING_SYSTEM == "AIR":
    Cp_fluid = 1006.0; Flow_rate = 0.05; H_CONVECTION = 5.0
else:
    Cp_fluid = 3500.0; Flow_rate = 0.20; H_CONVECTION = 800.0
    
Thermal_Coeff_Fluid = 1.0 / (Flow_rate * Cp_fluid)

# --- Weibull Parameters (Theta) ---
anode_optP = [14, 0.9, 2.5]
cathode_optP = [12, 0.97, 1.9]
ROCV_res_0 = [0.00018, 0.00005] 
ROCV_res = ROCV_res_0 

# --- Aging Targets (Integrated from Code 1) ---
target_years = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
target_soh   = np.array([]) #Fill with wanted values


# ==============================================================================
# PART 1: BMS CORE
# ==============================================================================
class BatteryManagementSystem:
    def __init__(self, config):
        self.Ns = config.get('Ns', 14)
        self.Np = config.get('Np', 2)
        self.capacity_ref = config.get('Capacity_Ah', 100.0)
        
        # Physical Limits
        self.V_cell_max = config.get('V_cell_max', 3.65)
        self.V_cell_min = config.get('V_cell_min', 2.50)
        
        self.I_pack_max_ch = config.get('I_charge_max', self.capacity_ref) * self.Np
        self.I_pack_max_dch = config.get('I_discharge_max', self.capacity_ref * 1.5) * self.Np
        
        # Thermal Parameters
        self.T_low_charge_limit = 0.0
        self.T_derating_start = 45.0
        self.T_cutoff = 60.0
        self.T_release = 55.0
        
        # SOC Strategy
        self.SOC_ch_target = 0.80      
        self.SOC_ch_ramp_start = 0.70  
        self.SOC_dch_ramp_start = 0.20 
        self.SOC_dch_floor = 0.05      
        
        self.is_tripped = False
        self.CCL = 0.0
        self.DCL = 0.0
        self.flag_voltage_limit = False

    def check_safety_state(self, T_pack):
        """ Checks if the battery pack has tripped thermal safety constraints """
        if T_pack >= self.T_cutoff:
            self.is_tripped = True
            return False
        if self.is_tripped:
            if T_pack <= self.T_release:
                self.is_tripped = False
                return True
            else:
                return False
        return True

    def compute_limits(self, T_pack, SOC_pack):
        """ Computes dynamic Charge Current Limit (CCL) and Discharge Current Limit (DCL) """
        if not self.check_safety_state(T_pack):
            self.CCL = 0.0; self.DCL = 0.0
            return 0.0, 0.0

        f_temp = 1.0
        if T_pack > self.T_derating_start:
            delta = self.T_cutoff - self.T_derating_start
            over = T_pack - self.T_derating_start
            f_temp = max(0.0, 1.0 - (over / delta))

        f_soc_ch = 1.0
        if SOC_pack >= self.SOC_ch_target: f_soc_ch = 0.0
        elif SOC_pack > self.SOC_ch_ramp_start:
            window = self.SOC_ch_target - self.SOC_ch_ramp_start
            f_soc_ch = 1.0 - ((SOC_pack - self.SOC_ch_ramp_start) / window)
        
        if T_pack < self.T_low_charge_limit: f_soc_ch = 0.0

        f_soc_dch = 1.0
        
        # 1. If we are at or below 5% -> COMPLETE CUTOFF (0 A)
        if SOC_pack <= self.SOC_dch_floor: 
            f_soc_dch = 0.0  
            
        # 2. If we are in the ramp zone (between 5% and 20%) -> PROGRESSIVE DERATING
        elif SOC_pack < self.SOC_dch_ramp_start:
            window = self.SOC_dch_ramp_start - self.SOC_dch_floor
            progress = (SOC_pack - self.SOC_dch_floor) / window
            # At 20%, progress = 1 -> f_soc = 1.0
            # At 5.01%, progress = 0 -> f_soc = 0.05 (just before cutoff)
            f_soc_dch = 0.05 + (0.95 * progress)

        self.CCL = self.I_pack_max_ch * min(f_temp, f_soc_ch)
        self.DCL = self.I_pack_max_dch * min(f_temp, f_soc_dch)
        
        return self.CCL, self.DCL

    def process_request(self, Val_Input, Mode_Input, U_pack_current, R_pack_estim):
        """ Processes user current/power request against calculated limits and current status """
        if Mode_Input == 'POWER':
            U_safe = max(U_pack_current, 2.0 * self.Ns)
            I_req = Val_Input / U_safe
        else:
            I_req = Val_Input

        if I_req >= 0: I_final = min(I_req, self.CCL)
        else: I_final = max(I_req, -self.DCL)

        R_safe = max(R_pack_estim, 1e-4)
        self.flag_voltage_limit = False
        
        if I_final > 0: # Charge
            V_max_pack = self.V_cell_max * self.Ns
            I_v_limit = (V_max_pack - U_pack_current) / R_safe
            if I_final > I_v_limit:
                I_final = max(0.0, I_v_limit)
                self.flag_voltage_limit = True
            
        elif I_final < 0: # Discharge
            V_min_pack = self.V_cell_min * self.Ns
            I_v_limit = (V_min_pack - U_pack_current) / R_safe
            if I_final < I_v_limit:
                I_final = min(0.0, I_v_limit)
                self.flag_voltage_limit = True

        return I_final

# ==============================================================================
# PART 2: ADVANCED PHYSICAL MODELS (THERMAL + ELECTRICAL)
# ==============================================================================

class ThermalAgingModel:
    """
    HYBRID MODEL:
    1. Thermal: Joule + Entropy (Reversible) + Convection
    2. Aging: Ekström Model (SEI Cracking with J, f, H)
    """
    def __init__(self, dt=1.0, T_init=25.0, T_core_init=None, m_cell=0.039, Cp=1100, Area=0.042, h=10):
        self.dt = dt
        self.T_surf = T_init + 273.15 
        self.T_amb_K = T_init + 273.15 
        
        if T_core_init is not None:
            self.T_core = T_core_init + 273.15
        else:
            self.T_core = self.T_surf
            
        self.m = m_cell
        self.Cp = Cp
        self.R_th = 1.0 / (h * Area)
        
        # Alias for compatibility
        self.T_cell = self.T_core
        
        # --- SOH / SEI Parameters (Ekström) ---
        self.Q_sei = 5.0e-4             # Initial SEI, value based on articles for EV batteries
        self.capa_nom_C = 377.0 * 3600.0
        self.Q_loss_acc = self.Q_sei / self.capa_nom_C
        self.SOH = 1.0 - self.Q_loss_acc
        self.time_elapsed = 0
        self.Ah_throughput = 0

        # Physical Constants
        self.F, self.R_gas = 96485.0, 8.314
        self.I1C = 300.0 # Reference 1C current for the model

        # --- EKSTRÖM PARAMETERS (J, f, H, alpha) ---
        self.Ea_sei = 45000.0      # SEI activation energy
        self.alpha = 0.6           # Transfer coefficient
        
        # Calibrated values for a certain run, to change after if you don't want to run optimization everytime
        self.J_ref = 1.44792e-04    # Kinetics
        self.f_ref = 3083.663       # Diffusion
        self.H_ref = 65.93         # Cracking Factor

    def get_entropic_coeff(self, soc):
        """ Entropic coefficient dU/dT (Typical LFP/Graphite) """
        soc_points =  [0.0,   0.1,   0.2,   0.3,   0.4,  0.5,  0.6,  0.7,  0.8,  0.9,  1.0]
        dudt_mV_K =   [-0.30, -0.25, -0.12, -0.05, 0.035, 0.06, 0.105, 0.105, 0.04, 0.05, 0.12]
        val_mV = np.interp(soc, soc_points, dudt_mV_K)
        return val_mV * 1e-3 # V/K

    def step(self, I, U_cell, U_ocv, SOC, T_amb_dynamic=None, U_anode_val=None):
        # ---------------------------------------------------------
        # 1. THERMAL PART
        # ---------------------------------------------------------
        T_env = (T_amb_dynamic + 273.15) if T_amb_dynamic is not None else self.T_amb_K

        Q_irr = np.abs(I * (U_cell - U_ocv))
        Q_rev = I * self.T_core * self.get_entropic_coeff(SOC)
        Q_gen = Q_irr + Q_rev

        if hasattr(self, 'R_cond') and hasattr(self, 'R_conv'):
            Q_cond = (self.T_core - self.T_surf) / self.R_cond
            Q_conv = (self.T_surf - T_env) / self.R_conv
            dT_core_dt = (Q_gen - Q_cond) / self.Cs_core
            dT_surf_dt = (Q_cond - Q_conv) / self.Cs_surf
            self.T_core += dT_core_dt * self.dt
            self.T_surf += dT_surf_dt * self.dt
            self.T_cell = self.T_core 
        else:
            Q_out = (self.T_cell - T_env) / self.R_th
            dT_dt = (Q_gen - Q_out) / (self.m * self.Cp)
            self.T_cell += dT_dt * self.dt
            self.T_core = self.T_cell
            self.T_surf = self.T_cell

        # ---------------------------------------------------------
        # 2. AGING PART
        # ---------------------------------------------------------
        self.time_elapsed += self.dt
        self.Ah_throughput += np.abs(I) * self.dt / 3600.0

        # --- SAFETY 1: Limit U_an to avoid overflow ---
        U_an = np.clip(U_anode_val if U_anode_val is not None else 0.08, 0.001, 2.0)

        T_factor = np.exp(self.Ea_sei/self.R_gas * (1/298.15 - 1/max(self.T_core, 250.0)))
        
        J = self.J_ref * T_factor
        f = self.f_ref / T_factor 

        K_crd = 0.0
        if I < -1.0: 
            K_crd = self.H_ref * (abs(I)/self.I1C)

        # --- SAFETY 2: Protected exponential calculation ---
        expo_val = (self.alpha * U_an * self.F) / (self.R_gas * self.T_core)
        term_cin = np.exp(np.clip(expo_val, -100, 100)) # Cap the exponential
        
        Q_safe = max(self.Q_sei, 1e-9)
        term_diff = (Q_safe * f * J) / self.I1C

        # SEI current (negative due to Lithium consumption)
        I_sei = -(1 + K_crd) * ((J * self.I1C) / (term_cin + term_diff))

        # SOH Update
        self.Q_sei += abs(I_sei) * self.dt
        self.Q_loss_acc = self.Q_sei / self.capa_nom_C
        self.SOH = max(0.05, 1.0 - self.Q_loss_acc)

        return self.T_surf - 273.15, self.SOH, Q_gen

class HalfCellSolver:
    def __init__(self, energy_curve_func, rc_param, theta, r_dis, Q_total_electrode, SOC0, n_particle):
        self.energy_curve = energy_curve_func
        self.n_particle = n_particle
        
        ROCVMean, ROCVdelta = float(theta[1]), float(theta[2])
        R_OCV = np.linspace(ROCVMean - ROCVdelta, ROCVMean + ROCVdelta, n_particle)
        
        if r_dis == "normal_cap":
            mu = ROCVMean; dx = (R_OCV[1]-R_OCV[0]); sigma = (mu-(R_OCV[0]-dx/2))/3.0
            dist_fun = lambda x: (1/(sigma*np.sqrt(2*np.pi)))*np.exp(-((x-mu)**2)/(2*sigma**2))
            edges = np.linspace(R_OCV[0]-dx/2, R_OCV[-1]+dx/2, n_particle+1)
            distCap = np.array([quad(dist_fun, edges[i], edges[i+1])[0] for i in range(n_particle)])
        else: 
            k_s = float(theta[3]); 
            R_max_range = max(ROCVdelta*2.0, 1e-6) 
            lam = np.power(-np.log(1-0.998), 1.0/k_s)/R_max_range
            cdf = lambda x,l,k: 1-np.exp(-np.power(l*np.maximum(x,0), k))
            edges = np.linspace(0, R_max_range, n_particle+1)
            distCap = cdf(edges[1:], lam, k_s) - cdf(edges[:-1], lam, k_s)

        currentCap = np.sum(distCap)
        scaling = Q_total_electrode / currentCap
        self.Q_OCVMax_nominal = distCap * scaling
        self.Q_OCVMax = self.Q_OCVMax_nominal.copy() 
        
        R_OCV_weighted = R_OCV / self.Q_OCVMax
        R_ges = 1.0 / np.sum(1.0/R_OCV_weighted)
        R_vec = (R_OCV_weighted/R_ges) * ROCVMean
        self.d_vec = np.zeros(n_particle); self.d_vec[-1] = R_vec[-1]

        A_R = np.diag(-R_vec[:-1]) + np.diag(R_vec[1:-1], 1); A_R[-1, :] -= R_vec[-1]
        dim_hys = 2 * n_particle - 1
        A_Hys = np.zeros((dim_hys, dim_hys))
        A_Hys[:n_particle-1, :n_particle-1] = A_R
        A_Hys[n_particle-1:2*n_particle-2, :n_particle-1] = np.eye(n_particle-1)
        A_Hys[-1, :n_particle-1] = -1.0
        b_Hys = np.zeros(dim_hys); b_Hys[n_particle-2] = R_vec[-1]; b_Hys[-1] = 1.0
        C_Hys = np.zeros((n_particle, dim_hys)); C_Hys[:n_particle-1, :n_particle-1] = np.diag(R_vec[:-1]); C_Hys[-1, :n_particle-1] = -R_vec[-1]

        self.use_rc = (theta[0] != -1)
        if self.use_rc:
            self.dim = dim_hys + 2
            self.A_base = np.zeros((self.dim, self.dim)); self.A_base[2:, 2:] = A_Hys
            self.b_base = np.zeros(self.dim); self.b_base[2:] = b_Hys
            self.C = np.zeros((n_particle, self.dim)); self.C[:, 0:2] = 1.0; self.C[:, 2:] = C_Hys
            self.M = np.diag(np.concatenate([np.ones(2), np.zeros(n_particle-1), np.ones(n_particle)]))
            self.nu = np.zeros(2*n_particle + 1)
            self.x = np.zeros(self.dim)
            self.x[n_particle+1:] = SOC0 * self.Q_OCVMax
        else:
            self.dim = dim_hys
            self.A_base = A_Hys; self.b_base = b_Hys; self.C = C_Hys
            self.M = np.diag(np.concatenate([np.zeros(n_particle-1), np.ones(n_particle)]))
            self.nu = np.zeros(2*n_particle - 1)
            self.x = np.zeros(self.dim)
            self.x[n_particle-1:] = SOC0 * self.Q_OCVMax

        soc_grid = rc_param[:,0]
        self.R0_p = PchipInterpolator(soc_grid, rc_param[:,1], extrapolate=True)
        self.R1_p = PchipInterpolator(soc_grid, rc_param[:,2], extrapolate=True)
        self.C1_p = PchipInterpolator(soc_grid, rc_param[:,3], extrapolate=True)
        self.R2_p = PchipInterpolator(soc_grid, rc_param[:,4], extrapolate=True)
        self.C2_p = PchipInterpolator(soc_grid, rc_param[:,5], extrapolate=True)
        
        self.U_electrode = 0.0
        self.SOC_global = SOC0

    def step(self, dt, I, SOH, T_kelvin, r_mult=1.0):
        # --- SWITCH 1: Temperature Impact on Capacity ---
        temp_cap_factor = 1.0
        if ENABLE_THERMAL_Q_IMPACT and T_kelvin < 293.15: # < 20°C
            delta_T = 293.15 - T_kelvin
            temp_cap_factor = max(0.5, 1.0 - (0.015 * delta_T))

        self.Q_OCVMax = self.Q_OCVMax_nominal * max(SOH, 0.1) * temp_cap_factor

        # --- SWITCH 2: Temperature Impact on Resistance (Arrhenius) ---
        scaling_rc = 1.0
        if ENABLE_THERMAL_R_IMPACT:
            # If T increases, scaling_rc decreases -> R decreases
            scaling_rc = np.exp((E_ACTIVATION_R / 8.314) * (1/T_kelvin - 1/298.15))
            scaling_rc = np.clip(scaling_rc, 0.1, 20.0)

        # Solver
        if self.use_rc: q_states = self.x[self.n_particle+1:]
        else: q_states = self.x[self.n_particle-1:]
        
        soc_loc = np.clip(q_states / self.Q_OCVMax, 0, 1)
        self.SOC_global = np.sum(q_states) / np.sum(self.Q_OCVMax)
        ocv_loc = self.energy_curve(soc_loc)
        
        # Update RC Matrices with temperature
        A = self.A_base.copy()
        b = self.b_base.copy()
        d_curr = self.d_vec.copy()
        rc_val_R0 = 0.02 * r_mult # Dynamic safety

        if self.use_rc:
            s_lookup = np.clip(self.SOC_global, 0, 1)
            
            # APPLICATION OF THERMAL FACTOR AND MULTIPLIER (Charge/Discharge) HERE
            rc_vals = np.array([self.R1_p(s_lookup), self.R2_p(s_lookup), self.R0_p(s_lookup)]) * scaling_rc * r_mult
            caps = np.array([self.C1_p(s_lookup), self.C2_p(s_lookup)])
            rc_val_R0 = rc_vals[2]
            
            A[0:2, 0:2] = np.diag(-1.0/(rc_vals[:2]*caps))
            b[0:2] = 1.0/caps
            d_curr += rc_vals[2]
            
            epsilon = 1e-5
            ocv_diff = (self.energy_curve(soc_loc+epsilon) - ocv_loc)/epsilon
            ocv_diff = np.clip(ocv_diff, 0, 50.0)
            
            term1 = -ocv_diff[:-1] / self.Q_OCVMax[:-1]
            term2 = ocv_diff[1:] / self.Q_OCVMax[1:]
            
            rows = np.arange(2, 2 + (self.n_particle - 1))
            J = A.copy()
            J[rows, np.arange(self.n_particle+1, 2*self.n_particle)] += term1
            J[rows, np.arange(self.n_particle+2, 2*self.n_particle+1)] += term2
        else:
            J = A.copy()
            self.nu[0:self.n_particle-1] = ocv_loc[1:] - ocv_loc[:-1]

        if self.use_rc: self.nu[2:self.n_particle+1] = ocv_loc[1:] - ocv_loc[:-1]

        r0 = A @ self.x + b * I + self.nu
        dx = lu_solve(lu_factor(self.M - dt * J), r0)
        self.x += dt * dx
        
        y_vec = self.C @ self.x + d_curr * I + ocv_loc
        self.U_electrode = y_vec[0]
        
        return self.U_electrode, self.SOC_global, rc_val_R0

class CellPDECM:
    def __init__(self, energy_curve_func, rc_param, theta, r_dis, QGes, SOC0, n_particle, m_cell, T_init=25.0, T_core_init=None, ocv_hysteresis_funcs=None):
        self.QGes_Ah = QGes
        self.n_particle = n_particle
        self.QGes_C = QGes * 3600.0
        self.Q_oh = 0.02
        n2p = 1.1

        self.SOC0_cat = (SOC0 * self.QGes_C + self.QGes_C * self.Q_oh) / (self.QGes_C + (self.QGes_C * self.Q_oh))
        self.SOC0_an = SOC0 / n2p
        
        # Initialization of advanced thermal model with T_core_init
        self.thermal = ThermalAgingModel(dt=1.0, T_init=T_init, T_core_init=T_core_init, m_cell=m_cell, h=H_CONVECTION, Area=Area_cell)
        self.T = T_init 
        self.U_cell = 0
        self.R_internal_current = 0.02 

        self.ocv_hysteresis_funcs = ocv_hysteresis_funcs
        self.use_hysteresis = (ocv_hysteresis_funcs is not None)

    def setup_dual_solvers(self, ocp_anode_func, ocp_cathode_func, rc_param, theta_anode, theta_cathode, r_dis):
        n2p = 1.1
        rc_an = rc_param.copy(); rc_an[:, 0] /= n2p
        self.anode = HalfCellSolver(ocp_anode_func, rc_param, theta_anode, r_dis, self.QGes_C * n2p, self.SOC0_an, self.n_particle)
        self.cathode = HalfCellSolver(ocp_cathode_func, rc_param, theta_cathode, r_dis, self.QGes_C * (1 + self.Q_oh), self.SOC0_cat, self.n_particle)

    def equilibrate_state(self, I_init, T_K_init):
            x_an_backup = self.anode.x.copy(); x_ca_backup = self.cathode.x.copy()
            huge_dt = 3600.0 * 10 
            current_soh = self.thermal.SOH
            
            self.anode.step(huge_dt, 0, current_soh, T_K_init)
            self.cathode.step(huge_dt, 0, current_soh, T_K_init)
            
            idx_an = self.anode.n_particle + 1 if self.anode.use_rc else self.anode.n_particle - 1
            idx_ca = self.cathode.n_particle + 1 if self.cathode.use_rc else self.cathode.n_particle - 1
            
            self.anode.x[idx_an:] = x_an_backup[idx_an:]
            self.cathode.x[idx_ca:] = x_ca_backup[idx_ca:]
            
            self.anode.SOC_global = np.sum(self.anode.x[idx_an:]) / np.sum(self.anode.Q_OCVMax)
            self.cathode.SOC_global = np.sum(self.cathode.x[idx_ca:]) / np.sum(self.cathode.Q_OCVMax)
            
            d_curr_an = self.anode.d_vec.copy()
            if self.anode.use_rc: d_curr_an[-1] += 0.0 
            ocv_an = self.anode.energy_curve(np.clip(self.anode.x[idx_an:] / self.anode.Q_OCVMax, 0, 1))
            self.anode.U_electrode = (self.anode.C @ self.anode.x + d_curr_an * I_init + ocv_an)[0]
            
            d_curr_ca = self.cathode.d_vec.copy()
            ocv_ca = self.cathode.energy_curve(np.clip(self.cathode.x[idx_ca:] / self.cathode.Q_OCVMax, 0, 1))
            self.cathode.U_electrode = (self.cathode.C @ self.cathode.x + d_curr_ca * I_init + ocv_ca)[0]
    
            self.U_cell = self.cathode.U_electrode + self.anode.U_electrode

    def step(self, dt, I, T_amb_dynamic, T_forcing_K=None, r_mult=1.0):
         self.thermal.dt = dt
         
         T_physics_K = T_forcing_K if (T_forcing_K is not None and not np.isnan(T_forcing_K)) else self.thermal.T_cell
         
         SOH = self.thermal.SOH
         
         # 1. Electrical Calculation
         U_an, SOC_an, R0_an = self.anode.step(dt, I, SOH, T_physics_K, r_mult)
         U_ca, SOC_ca, R0_ca = self.cathode.step(dt, I, SOH, T_physics_K, r_mult)
         
         OCV_an_mean = self.anode.energy_curve(self.anode.SOC_global)
         OCV_ca_mean = self.cathode.energy_curve(self.cathode.SOC_global)
         OCV_global_physics = OCV_ca_mean + OCV_an_mean
         U_cell_physics = U_ca + U_an 

         if self.use_hysteresis:
             current_soc = self.cathode.SOC_global 
             if I < -1e-5: 
                 OCV_corrected = self.ocv_hysteresis_funcs['charge'](current_soc)
             else: 
                 OCV_corrected = self.ocv_hysteresis_funcs['discharge'](current_soc)
             
             OCV_final = float(OCV_corrected)
             Overpotentials = U_cell_physics - OCV_global_physics
             self.U_cell = OCV_final + Overpotentials
         else:
             OCV_final = OCV_global_physics
             self.U_cell = U_cell_physics
         
         # 2. Thermal Calculation
         T_surf_sim, SOH_new, Q_gen = self.thermal.step(
             I=I, 
             U_cell=self.U_cell, 
             U_ocv=OCV_final, 
             SOC=self.cathode.SOC_global, 
             T_amb_dynamic=T_amb_dynamic, 
             U_anode_val=-U_an
         )
         self.T = T_surf_sim
         SOC_final = self.cathode.SOC_global * (1 + self.Q_oh) - self.Q_oh
         R_app = R0_an + R0_ca
         
         return self.U_cell, T_surf_sim, SOC_final, Q_gen, R_app, SOH_new

# ==============================================================================
# PART 3: HELPER FUNCTIONS
# ==============================================================================

def load_ocp_data():
    try:
        df_an = pd.read_excel(os.path.join(OCP_PATH, "param_ocpn_graphite.xlsx"))
        dol_an = df_an.iloc[:, 0].values; pot_an = df_an.iloc[:, 1].values
        
        df_ca = pd.read_excel(os.path.join(OCP_PATH, "param_ocpn_lfp.xlsx"))
        dol_ca = df_ca.iloc[:, 0].values; pot_ca_raw = df_ca.iloc[:, 1].values
        return dol_an, pot_an, dol_ca, pot_ca_raw
    except Exception as e:
        print(f"OCP LOADING ERROR : {e}")
        print(f"Check the path : {OCP_PATH}")
        sys.exit()

def build_rc_param_lfp_exact():
    soc_grid = np.array([
        0.009997615496963, 0.019996030588378, 0.029993985128756, 0.039991749649365,
        0.049989565613247, 0.099978753261363, 0.149967825852979, 0.199956926120683,
        0.249945899426573, 0.299935158039249, 0.349924702963101, 0.399914266515770,
        0.449903482749856, 0.499892579259504, 0.549882122104548, 0.599871324846072,
        0.649860648819611, 0.699849921098065, 0.749839165190624, 0.799828483403504,
        0.849817983475960, 0.899807648688257, 0.949797218653482, 0.959838070231234,
        0.969878839839737, 0.979919643869179, 0.989960500123429, 0.996927213134967
    ])
    R0 = np.full_like(soc_grid, R0_cell_nom) 
    R1 = np.array([
        0.005145462248723, 0.006008493599129, 0.006448055559971, 0.005856821760000,
        0.007308262089632, 0.007509648472672, 0.006164100775371, 0.006538003055329,
        0.007175258860108, 0.007339109630299, 0.008713559269369, 0.007868335767674,
        0.0079, 0.008077902549460, 0.008233438629941, 0.008372608283425,
        0.008866265791057, 0.008256499800489, 0.008302914879219, 0.009460612752251,
        0.009128089741437, 0.009279168633957, 0.005332737116468, 0.010406334248734,
        0.010249150279031, 0.010786550777466, 0.012186139575975, 0.014386444420050
    ])
    C1 = np.array([
        5.409584584248230e+02, 7.181581009417870e+02, 9.974374112660480e+02, 9.526685246154360e+02,
        1.235915935700990e+03, 1.451946136460340e+03, 1.109099044939890e+03, 1.148103647912950e+03,
        1.255404113344680e+03, 1.296209847240250e+03, 1.437193241447520e+03, 1.422460430375690e+03,
        1350,      1.292942337654670e+03, 1.275722141773700e+03, 1.194925547740810e+03,
        1.240362206587940e+03, 1.114621235211460e+03, 1.116076559898230e+03, 1.183985227204640e+03,
        1.175416673097070e+03, 1.137201442510130e+03, 1090,      1.047977398745050e+03,
        9.778624546175570e+02, 9.317004132974620e+02, 7.255625094020860e+02, 5.482990469173320e+02
    ])
    R2 = np.array([
        0.007312894428095, 0.006354488548771, 0.005325996879846, 0.005823810672671,
        0.004087967375016, 0.004481377717510, 0.005626882552544, 0.005003482522166,
        0.004482482114493, 0.004240140675009, 0.003032292840389, 0.003553585671536,
        0.007532220410880, 0.004049237905473, 0.004068814660232, 0.004373882384749,
        0.003975004623248, 0.004815227674462, 0.004701806157452, 0.003643303739943,
        0.003854993719911, 0.003989690732903, 0.008448169222566, 0.004062776625450,
        0.004607083438438, 0.004601160745784, 0.005558162543998, 0.006825884591911
    ])
    C2 = np.array([
        4.191673495019170e+03, 6.264774820699370e+03, 9.112625355453310e+03, 7.485573992374370e+03,
        1.521575292644640e+04, 1.649876972238050e+04, 9.778000664055930e+03, 1.190362863083720e+04,
        1.635688227321210e+04, 1.797873703920390e+04, 3.994897731885730e+04, 2.644552696009330e+04,
        5.600714310220200e+03, 2.255442328382470e+04, 2.319008172823190e+04, 2.240919320956000e+04,
        2.802266166036970e+04, 1.745860859554570e+04, 1.717441884929220e+04, 2.993944583738240e+04,
        2.472276521509160e+04, 2.540695892681510e+04, 2.6e+4, 2.907973291941980e+04,
        2.209831152237530e+04, 2.315795229955040e+04, 1.768556454942660e+04, 1.247294202184850e+04
    ])
    return np.column_stack([soc_grid, R0, R1, C1, R2, C2])

def get_data_from_csv(path, aliases, read_power_mode=False, power_factor=1.0):
    if not path or not os.path.exists(path): 
        print("Error: Invalid path."); return None
    try:
        print(f"Reading file: {os.path.basename(path)}")
        if path.endswith('.csv'):
            try: df = pd.read_csv(path, sep=None, engine='python')
            except: df = pd.read_csv(path, sep=',')
        else:
            df = pd.read_excel(path)

        col_map = {}
        for target_key, possible_names in aliases.items():
            for name in possible_names:
                matches = [col for col in df.columns if col.lower() == name.lower()]
                if matches: col_map[target_key] = matches[0]; break
        
        if 'Time' not in col_map:
            print(f"Error: 'Time' not found. Available columns: {list(df.columns)}")
            return None

        def clean_to_num(series):
            return pd.to_numeric(series.astype(str).str.replace(',', '.', regex=False), errors='coerce')

        t_raw = df[col_map['Time']]
        t_numeric = clean_to_num(t_raw)
        
        if t_numeric.notna().sum() > (len(t_raw) * 0.9):
            t_sec = t_numeric.values
        else:
            t_date = pd.to_datetime(t_raw, dayfirst=True, errors='coerce')
            t_sec = (t_date - t_date.dropna().iloc[0]).dt.total_seconds().values

        t_sec = t_sec - np.nanmin(t_sec)
        v_val = clean_to_num(df[col_map['Voltage']]).values if 'Voltage' in col_map else np.full(len(df), np.nan)
        
        if read_power_mode:
            if 'Power' not in col_map: 
                print("Power mode active but Power column missing."); return None
            p_val = clean_to_num(df[col_map['Power']]).values * power_factor
            
            nominal_voltage = 3.2 * Ns
            v_safe = np.copy(v_val)
            v_safe[np.isnan(v_safe) | (v_safe == 0)] = nominal_voltage
            i_val = p_val / v_safe
        else:
            if 'Current' not in col_map: 
                print("Current mode active but Current column missing."); return None
            i_val = clean_to_num(df[col_map['Current']]).values

        t_bat_val = clean_to_num(df[col_map['Temp_Bat']]).values if 'Temp_Bat' in col_map else np.full(len(df), np.nan)
        if 'Temp_Amb' in col_map:
            t_amb_val = clean_to_num(df[col_map['Temp_Amb']]).values
        else:
            t_amb_val = np.full(len(df), 25.0)

        mask = ~np.isnan(t_sec) & ~np.isnan(i_val)
        return t_sec[mask], i_val[mask], v_val[mask], t_amb_val[mask], t_bat_val[mask]

    except Exception as e:
        print(f"CSV reading exception: {e}"); return None

def ocp_anode(sto, pot, pot_dol):
    return -np.interp(sto, pot_dol, pot)

def ocp_cathode(sto, pot, pot_dol):
    return np.interp(sto, pot_dol, pot)

def test_profile_multi_pulse_torture(capacity_pack):
    """
    Simplified and readable torture profile (Total duration: 2 hours).
    Alternates long charge/discharge phases (macro-cycles)
    and retains a targeted phase of strong dynamic pulses.
    """
    #Deep and sustained discharge (40 min at -1C)
    p1 = np.full(40 * 60, -1.0 * capacity_pack)
    
    #Rest (10 min)
    p2 = np.zeros(10 * 60)
    
    #Strong and sustained charge (30 min at 1.2C)
    p3 = np.full(30 * 60, 1.2 * capacity_pack)
    
    #Rest (10 min)
    p4 = np.zeros(10 * 60)
    
    #Aggressive dynamic pulse phase (15 min)
    pulse_sym = np.concatenate([np.full(15, 2.0 * capacity_pack), np.full(15, -2.0 * capacity_pack)])
    p5 = np.tile(pulse_sym, (15 * 60) // 30) # Repeated 30 times (900s)
    
    #Final rest (15 min)
    p6 = np.zeros(15 * 60)
    
    # Combination
    val_prof = np.concatenate([p1, p2, p3, p4, p5, p6])
    t_prof = np.arange(len(val_prof))
    
    # Ambient temperature fixed at 25°
    temp_prof = np.full(len(val_prof), 25.0) 
    
    return t_prof, val_prof, temp_prof

def generate_bms_action_profile(capacity_pack):
    
    t_morning = 4 * 3600
    p1 = np.tile(np.concatenate([
        np.full(60, -1.8 * capacity_pack), 
        np.full(30, 0.8 * capacity_pack),
        np.full(30, 0.0)                   
    ]), t_morning // 120)

    t_noon = int(1.5 * 3600)
    p2 = np.full(t_noon, 2.8 * capacity_pack)

    t_afternoon = 5 * 3600
    p3 = np.full(t_afternoon, -1.2 * capacity_pack)

    t_evening = int(3.5 * 3600)
    p4 = np.tile(np.concatenate([
        np.full(300, -0.5 * capacity_pack),
        np.full(300, 0.5 * capacity_pack)
    ]), t_evening // 600)

    t_night = 10 * 3600
    p5 = np.zeros(t_night)

    val_prof = np.concatenate([p1, p2, p3, p4, p5])
    t_prof = np.arange(len(val_prof))
    
    temp_prof = 20 + 10 * np.sin(2 * np.pi * t_prof / 86400 - np.pi/2)
    
    return t_prof, val_prof, temp_prof

def test_profile_charge(capacity_pack):
    """ Forces a continuous charge for 1 hour """
    t_prof = np.arange(3600)
    val_prof = np.full(3600, 1.0 * capacity_pack) 
    temp_prof = np.full(3600, 25.0) 
    return t_prof, val_prof, temp_prof

def test_profile_discharge(capacity_pack):
    """ Forces a strong continuous discharge """
    t_prof = np.arange(1800)
    val_prof = np.full(1800, -1.4 * capacity_pack)
    temp_prof = np.full(1800, 25.0)
    return t_prof, val_prof, temp_prof

def test_profile_thermal(capacity_pack):
    """ Heats the battery with aggressive cycles """
    t_prof = np.arange(5000)
    p1 = np.full(100, 2.5 * capacity_pack)
    p2 = np.full(100, -2.5 * capacity_pack)
    val_prof = np.tile(np.concatenate([p1, p2]), 25)
    temp_prof = np.full(5000, 48.0) 
    return t_prof, val_prof, temp_prof

def test_profile_physical_hysteresis(capacity_pack):
    """ Pulses with rest to observe multi-particle relaxation """
    t_prof = np.arange(7200) 
    p_dchg = np.full(300, -0.2 * capacity_pack)
    p_rest = np.zeros(900)
    p_chg  = np.full(300, 0.2 * capacity_pack)
    cycle = np.concatenate([p_dchg, p_rest, p_chg, p_rest])
    val_prof = np.tile(cycle, 3) 
    temp_prof = np.full(len(val_prof), 25.0)
    return t_prof, val_prof, temp_prof

def generate_48h_duty_cycle(capacity_pack):
    """ Generates a 48-hour profile """
    t_dch = 2 * 3600; t_rest1 = 6 * 3600; t_ch = 2 * 3600; t_rest2 = 6 * 3600
    I_dch = -0.4 * capacity_pack; I_ch = 0.4 * capacity_pack
    cycle_16h = np.concatenate([np.full(t_dch, I_dch), np.full(t_rest1, 0.0), np.full(t_ch, I_ch), np.full(t_rest2, 0.0)])
    val_prof = np.tile(cycle_16h, 3)
    t_prof = np.arange(len(val_prof))
    temp_prof = np.full_like(t_prof, 25.0) 
    return t_prof, val_prof, temp_prof

def test_profile_full_cycle(capacity_pack):
    """
    Standard Full Cycle Test: 
    One big charge, rest, one big discharge, rest.
    Great for checking total capacity and thermal build-up across a full sweep.
    """
    #Initial Rest (5 min)
    p1 = np.zeros(5 * 60)
    
    #Full Charge at 1C (75 min)
    p2 = np.full(75 * 60, 1.0 * capacity_pack)
    
    #Rest/Relaxation (30 min)
    p3 = np.zeros(30 * 60)
    
    #Full Discharge at -1C (75 min)
    p4 = np.full(75 * 60, -1.0 * capacity_pack)
    
    #Final Rest (30 min)
    p5 = np.zeros(30 * 60)
    
    val_prof = np.concatenate([p1, p2, p3, p4, p5])
    t_prof = np.arange(len(val_prof))
    temp_prof = np.full(len(val_prof), 25.0)
    
    return t_prof, val_prof, temp_prof

def test_profile_asymmetric_staircase(capacity_pack):
    """
    "Staircase and Asymmetry" test profile (Total duration: 2 hours).
    Designed to evaluate voltage response at different C-rates (steps)
    and test thermal heating via asymmetric cycling.
    """
    #Discharge staircase (30 min)
    #3 steps of 10 min: -0.5C, -1C, -1.5C
    p1_1 = np.full(10 * 60, -0.5 * capacity_pack)
    p1_2 = np.full(10 * 60, -1.0 * capacity_pack)
    p1_3 = np.full(10 * 60, -1.5 * capacity_pack)
    p1 = np.concatenate([p1_1, p1_2, p1_3])
    
    #Rest (10 min)
    p2 = np.zeros(10 * 60)
    
    #Charge staircase (30 min)
    #3 steps of 10 min: +0.5C, +1C, +1.5C
    p3_1 = np.full(10 * 60, 0.5 * capacity_pack)
    p3_2 = np.full(10 * 60, 1.0 * capacity_pack)
    p3_3 = np.full(10 * 60, 1.5 * capacity_pack)
    p3 = np.concatenate([p3_1, p3_2, p3_3])
    
    #Rest (10 min)
    p4 = np.zeros(10 * 60)
    
    #Asymmetric cycling (30 min)
    cycle_asym = np.concatenate([np.full(60, 2.0 * capacity_pack), np.full(120, -1.0 * capacity_pack)])
    p5 = np.tile(cycle_asym, 10)
    
    #Final rest (10 min)
    p6 = np.zeros(10 * 60)
    
    val_prof = np.concatenate([p1, p2, p3, p4, p5, p6])
    t_prof = np.arange(len(val_prof))
    
    # Ambient temperature fixed at 25°C
    temp_prof = np.full(len(val_prof), 25.0) 
    
    return t_prof, val_prof, temp_prof

def plot_aging_comparison(t_sim, d1, d12, month_ax, soh_ax):
    fig, ax = plt.subplots(3, 2, figsize=(14, 12))
    t_hrs = t_sim / 3600 
    ax[0, 0].plot(month_ax, soh_ax, 'bo-', linewidth=2)
    ax[0, 0].set_title("SOH Evolution")
    ax[0, 0].set_ylabel("SOH [%]"); ax[0, 0].grid(True)
    ax[1, 0].plot(t_hrs, d1['V'], 'g', label="Month 1 (New)", alpha=0.6)
    ax[1, 0].plot(t_hrs, d12['V'], 'r', label="Last Month (Aged)")
    ax[1, 0].set_title("Pack Voltage Comparison (Over 48h)")
    ax[1, 0].set_ylabel("Voltage [V]"); ax[1, 0].legend(); ax[1, 0].grid(True)
    ax[0, 1].plot(t_hrs, d1['T'], 'g', alpha=0.6)
    ax[0, 1].plot(t_hrs, d12['T'], 'r')
    ax[0, 1].set_title("Temperature Comparison (°C)")
    ax[0, 1].set_ylabel("Temp [°C]"); ax[0, 1].grid(True)
    ax[1, 1].plot(t_hrs[:7200], d1['V'][:7200], 'g')
    ax[1, 1].plot(t_hrs[:7200], d12['V'][:7200], 'r')
    ax[1, 1].set_title("Zoom: Power Fade")
    ax[1, 1].set_ylabel("V"); ax[1, 1].grid(True)
    ax[2, 0].plot(t_hrs, d1['SOC'], 'g', alpha=0.6)
    ax[2, 0].plot(t_hrs, d12['SOC'], 'r')
    ax[2, 0].set_title("SOC Drift")
    ax[2, 0].set_ylabel("%"); ax[2, 0].set_xlabel("Time [h]"); ax[2, 0].grid(True)
    ax[2, 1].plot(t_hrs, d1['I'], 'k')
    ax[2, 1].set_title("Current Profile")
    ax[2, 1].set_ylabel("Current [A]"); ax[2, 1].set_xlabel("Time [h]"); ax[2, 1].grid(True)
    plt.tight_layout()
    plt.show()

# ==============================================================================
# PART 3.5: LONG TERM AGING
# ==============================================================================

def run_aging_optimization():
    print("\n" + "="*60)
    print("PHASE 0: PARAMETER OPTIMIZATION (FIX VARIABLE ERROR)")
    print("="*60)

    # 1. Data preparation (Same as BMS)
    dol_an, pot_an, _, _ = load_ocp_data()
    f_ocp_anode_interp = interp1d(dol_an, pot_an, fill_value="extrapolate")

    Q_pack = QGes_Ah * Np 
    capa_c = Q_pack * 3600.0
    
    # --- TIME CONSTANTS DEFINITION ---
    cycles_per_day = 1.5
    c_rate_charge = 0.4  # The 48h profile charges at 0.4C
    
    # t_charge_day : time spent charging per 24h (in seconds)
    # (1.5 cycles / 0.4 C) * 3600 = 13500 seconds (i.e. 3.75h)
    t_charge_day = (cycles_per_day / c_rate_charge) * 3600 

    def objective(params):
        J, f, H = params
        if J < 1e-9 or f < 10: return 1e9
        
        q_sei_sim = 5.0e-4 # Initial SEI BMS
        soh_history = []
        
        # --- ANODE STRESS CALCULATION ---
        # Averaging stress between SOC 0.75 (BMS start) and 0.95 (end of charge)
        soc_points = np.linspace(0.75, 0.95, 20)
        u_an_points = f_ocp_anode_interp(soc_points)
        u_an_points = np.maximum(0.005, u_an_points)
        
        # Average kinetic term (exponential)
        kinetic_terms = np.exp((0.6 * u_an_points * 96485.0) / (8.314 * 298.15))
        term_kin_effective = np.mean(kinetic_terms)

        current_days = 0
        for yr in target_years:
            days_to_sim = (yr * 365) - current_days
            
            # Simulation in 10-day blocks
            for _ in range(int(days_to_sim/10)):
                # Ekström denominator
                denominator = term_kin_effective + (q_sei_sim * f * J / Q_pack)
                
                # SEI current (Physics: I < 0 for charge)
                # Using 0.4 C-rate for H factor
                I_sei = -(1 + H * c_rate_charge) * (J * Q_pack / denominator)
                
                # Daily loss applied only during charging hours
                loss_10_days = abs(I_sei) * t_charge_day * 10
                q_sei_sim += loss_10_days
            
            # SOH calculation consistent with BMS
            soh = 1.0 - (q_sei_sim / capa_c)
            soh_history.append(soh)
            current_days = yr * 365
            
        rmse = np.sqrt(np.mean((np.array(soh_history) - target_soh)**2))
        return rmse

    print(f"Optimization in progress for {Q_pack}Ah...")
    res = minimize(objective, [2.0e-4, 600, 10], method='Nelder-Mead', tol=1e-4)
    
    print(f"\nPARAMETERS FOUND:")
    print(f"J_ref = {res.x[0]:.4e}")
    print(f"f_ref = {res.x[1]:.2f}")
    print(f"H_ref = {res.x[2]:.2f}")
    
    return res.x

def run_long_term_aging_simulation(calib_params, aging_params=None):
    print("\n" + "="*60)
    print("PHASE 3: LONG-TERM AGING SIMULATION (EXPERIMENTAL COMPARISON)")
    print("="*60)

    dol_an, pot_an, dol_ca, pot_ca_raw = load_ocp_data()

    # --- 1. Initial configuration (1 Representative Cell) ---
    if calib_params is None: calib_params = [1.0, 1.0, SOC0_USER, 1.0, 0.0, 1.0]
    r_chg, r_dchg, soc_start, cap_factor, v_offset, rc_factor = calib_params

    Q_final = QGes_Ah * cap_factor * Np  
    pot_ca_final = pot_ca_raw
    scaling_factor = Q_ref_RC / Q_final
    
    # Setup Theta/RC
    frac_an = anode_optP[0] / (cathode_optP[0] + anode_optP[0])
    frac_ca = cathode_optP[0] / (cathode_optP[0] + anode_optP[0])
    theta_an = [-1, frac_an*ROCV_res[0], anode_optP[1]*ROCV_res[1], anode_optP[2]*0.95]
    theta_ca = [0,  frac_ca*ROCV_res[0], cathode_optP[1]*ROCV_res[1], cathode_optP[2]*0.92]
    if theta_an[3] == 0: theta_an[3] = 2.0
    if theta_ca[3] == 0: theta_ca[3] = 2.0
    
    rc_params = build_rc_param_lfp_exact()
    rc_params[:, 1] *= scaling_factor * FACTOR_R_DYN 
    rc_params[:, [2, 4]] *= scaling_factor * rc_factor
    rc_params[:, [3, 5]] /= (scaling_factor * rc_factor)
    
    f_an_final = lambda x: ocp_anode(x, pot_an, dol_an)
    f_ca_final = lambda x: ocp_cathode(x, pot_ca_final, dol_ca)

    print("Initializing macroscopic cell...")
    cell = CellPDECM(None, None, None, r_dis_type, Q_final, soc_start, 5, m_cell * Np, 25.0, 25.0)
    cell.thermal.Area = Area_cell * Np
    cell.thermal.R_cond = R_cond_cell / Np
    cell.thermal.R_conv = 1.0 / (H_CONVECTION * Area_cell * Np)
    cell.thermal.Cs_core = (m_cell * Np) * 0.9 * 1100
    cell.thermal.Cs_surf = (m_cell * Np) * 0.1 * 900
    
    if aging_params is not None:
        cell.thermal.J_ref = aging_params[0]
        cell.thermal.f_ref = aging_params[1]
        cell.thermal.H_ref = aging_params[2]
    
    cell.thermal.time_elapsed = 3600.0 * 24 * 30
    cell.thermal.Ah_throughput = 200.0
    
    cell.setup_dual_solvers(f_an_final, f_ca_final, rc_params, theta_an, theta_ca, r_dis_type)
    cell.equilibrate_state(0, 25.0 + 273.15)

    # --- 2. Generation of 48h Pattern ---
    t_sim, val_prof, T_amb_sim = generate_48h_duty_cycle(Q_final)
    dt = np.mean(np.diff(t_sim)) if len(t_sim) > 1 else 1.0

    # --- 3. Months Loop ---
    nb_months = 240 # 20 years
    print(f"\nLaunching time jump simulation over {nb_months//12} years (1 Month = 1 computed 48h block)...")
    
    month_list = [0]
    soh_list = [cell.thermal.SOH * 100]
    
    data_month_1 = {'V': [], 'T': [], 'SOC': [], 'I': []}
    data_month_end = {'V': [], 'T': [], 'SOC': [], 'I': []}
    
    for month in tqdm(range(1, nb_months + 1), desc="SOH Progress"):
        T_reset = 25.0 + 273.15 
        cell.thermal.T_core = T_reset
        cell.thermal.T_surf = T_reset
        cell.T = 25.0 
        
        soh_start = cell.thermal.SOH
        
        # Simulating the 48h block
        for k in range(len(t_sim)):
            current_r_mult = r_chg if val_prof[k] >= 0 else r_dchg
            u, t_surf, soc_f, q_gen, r_app, soh_n = cell.step(dt, -val_prof[k], T_amb_sim[k], r_mult=current_r_mult)
            
            if month == 1 or month == nb_months:
                target = data_month_1 if month == 1 else data_month_end
                target['V'].append(u * Ns) 
                target['T'].append(t_surf)
                target['SOC'].append(soc_f * 100)
                target['I'].append(val_prof[k])
        
        soh_end = cell.thermal.SOH
        delta_soh = soh_start - soh_end 
        
        # Extrapolate for the rest of the month (1 month = 15 x 48h approx, 1 run + 14 jumps)
        extra_loss = (delta_soh * 14) * cell.thermal.capa_nom_C
        cell.thermal.Q_sei += extra_loss
        
        cell.thermal.Q_loss_acc = cell.thermal.Q_sei / cell.thermal.capa_nom_C
        cell.thermal.SOH = max(0.05, 1.0 - cell.thermal.Q_loss_acc)
        
        cell.thermal.time_elapsed += (48 * 3600) * 14
        Ah_48h = np.sum(np.abs(val_prof)) * dt / 3600.0
        cell.thermal.Ah_throughput += (Ah_48h * 14)
        
        month_list.append(month)
        soh_list.append(cell.thermal.SOH * 100)

    # --- ERROR CALCULATION ---
    sim_years = np.array(month_list) / 12.0
    soh_sim_interp = np.interp(target_years, sim_years, np.array(soh_list) / 100.0)
    
    rmse_soh = np.sqrt(np.mean((soh_sim_interp - target_soh)**2))
    mae_soh = np.mean(np.abs(soh_sim_interp - target_soh))
    
    print(f"\n📊 RESULTS: RMSE = {rmse_soh*100:.3f}% | MAE = {mae_soh*100:.3f}%")

    # --- 4. DISPLAY ---
    plt.figure(figsize=(10, 6))
    
    
    year_indices = [i for i, m in enumerate(month_list) if m % 12 == 0]
    bars_years = sim_years[year_indices]
    bars_soh = np.array(soh_list)[year_indices]

    plt.bar(bars_years, bars_soh, width=0.6, color='steelblue', edgecolor='black', label="Simulation SOH")
    plt.ylim(min(bars_soh) - 2, 102)
    plt.xticks(bars_years)

    plt.title(f"20-year aging\nRMSE: {rmse_soh*100:.3f}% | MAE: {mae_soh*100:.3f}%")
    plt.xlabel("Years")
    plt.ylabel("SOH [%]")
    plt.legend()
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    plot_aging_comparison(t_sim, data_month_1, data_month_end, month_list, soh_list)



# ==============================================================================
# PART 4: CALIBRATION
# ==============================================================================

def run_calibration_routine():
    print("\n" + "="*60)
    print("PHASE 1: CALIBRATION (SIMPLE CURVE FITTING MODE)")
    print("="*60)
    
    dol_an, pot_an, dol_ca, pot_ca_raw = load_ocp_data()
    
    root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
    path_calib = filedialog.askopenfilename(title="Calibration File", filetypes=[("Data", "*.csv *.xlsx")])
    root.destroy()
    if not path_calib: return None
        
    mode_str = input("File in (C)urrent or (P)ower? [C/P]: ").upper()
    read_pwr = (mode_str == 'P')
    
    res = get_data_from_csv(path_calib, COLUMN_ALIASES, read_power_mode=read_pwr)
    if not res: return None
    t_csv, I_csv, U_exp, T_amb_csv, _ = res
    
    step_idx = max(1, int(len(t_csv) / 500))
    idx = np.arange(0, len(t_csv), step_idx)
    
    t_opt = t_csv[idx]
    U_opt = U_exp[idx]
    T_opt = T_amb_csv[idx]
    
    I_opt = []
    for i in range(len(idx)):
        start = idx[i]
        end = idx[i+1] if i+1 < len(idx) else len(t_csv)
        I_opt.append(np.mean(I_csv[start:end]))
    I_opt = np.array(I_opt)

    def objective_curve_fitting(params):
        try:
            r_chg, r_dchg, soc0, cap_f, an_shift, rc_f, v_offset = params
            
            if not (0.05 <= soc0 <= 0.95): return 1e9
            
            Q_curr = QGes_Ah * cap_f
            scaling_factor = Q_ref_RC / Q_curr
            
            frac_an = anode_optP[0] / (cathode_optP[0] + anode_optP[0])
            frac_ca = cathode_optP[0] / (cathode_optP[0] + anode_optP[0])
            
            rocv_mean_an = max(1e-6, frac_an * ROCV_res[0] + an_shift*0.01)
            
            theta_an = [-1, rocv_mean_an, anode_optP[1] * ROCV_res[1], anode_optP[2] * 0.95]
            theta_ca = [0, frac_ca * ROCV_res[0], cathode_optP[1] * ROCV_res[1], cathode_optP[2] * 0.92]
            
            rc = build_rc_param_lfp_exact()
            rc[:, 1] *= scaling_factor * 1.0 
            rc[:, [2, 4]] *= scaling_factor * rc_f
            rc[:, [3, 5]] /= (scaling_factor * rc_f)
            
            f_an = lambda x: ocp_anode(x, pot_an, dol_an)
            f_ca = lambda x: ocp_cathode(x, pot_ca_raw, dol_ca)
            
            cell = CellPDECM(None, None, None, r_dis_type, Q_curr, soc0, 5, m_cell, T_opt[0], T_opt[0])
            cell.setup_dual_solvers(f_an, f_ca, rc, theta_an, theta_ca, r_dis_type)
            
            cell.equilibrate_state(0.0, T_opt[0] + 273.15)
            
            # Warm-up for calibration
            if ENABLE_WARMUP:
                for _ in range(30):
                    r_m = r_chg if I_opt[0] >= 0 else r_dchg
                    cell.step(10.0, I_opt[0] / Np, T_opt[0], r_mult=r_m)
                    
            u_sim_raw = []
            for k in range(len(t_opt)):
                dt_step = (t_opt[k] - t_opt[k-1]) if k > 0 else 1.0
                dt_step = max(0.1, dt_step) 
                
                i_val = I_opt[k] / Np
                
                current_r_mult = r_chg if i_val >= 0 else r_dchg
                u, _, _, _, r_dyn, _ = cell.step(dt_step, i_val, T_opt[k], r_mult=current_r_mult)
                
                u_sim_raw.append((u * Ns) + (I_opt[k] * R_HARNESS_CHARGE))
            
            u_sim_raw = np.array(u_sim_raw)
            if np.any(np.isnan(u_sim_raw)): return 1e9 
            
            u_sim = u_sim_raw + v_offset
            
            error = u_sim - U_opt
            weights = np.ones(len(U_opt))
            
            margin = max(1, int(len(U_opt) * 0.05))
            weights[:margin] = 100.0
            weights[-margin:] = 100.0
            
            rmse = np.sqrt(np.mean(weights * error**2))
            return rmse
            
        except Exception as e:
            return 1e9

    bounds = [
        (0.5, 5.0),    # 0: r_chg 
        (0.5, 5.0),    # 1: r_dchg 
        (0.05, 0.999),  # 2: soc0 
        (0.8, 1.5),    # 3: cap_f
        (-0.02, 0.02), # 4: an_shift
        (0.5, 15.0),   # 5: rc_f 
        (-3, 3)        # 6: v_offset_optim 
    ]
    
    t_start_calib = time.time()
    result = differential_evolution(
        objective_curve_fitting, 
        bounds, 
        maxiter=50, 
        popsize=15, 
        disp=True, 
        polish=True,
        seed = 42
    )
    t_end_calib = time.time()
    calib_duration = t_end_calib - t_start_calib
    print(f"\n⏱️ COMPUTATION TIME: {calib_duration:.2f} seconds")
    
    x = result.x
    print(f"\n--- OPTIMIZED RESULTS ---")
    print(f"SOC0: {x[2]*100:.2f}% | Cap_f: {x[3]:.4f} | R_chg: {x[0]:.2f}x | R_dchg: {x[1]:.2f}x | Offset: {x[6]:.3f}V | RC_f: {x[5]:.3f}")
    
    return [ x[0], x[1], x[2], x[3], x[6], x[5] ]


def run_sensitivity_analysis(calib_params):
    print("\n" + "="*60)
    print("🔬 CROSS SENSITIVITY ANALYSIS (DETAILED MODE)")
    print("="*60)
    
    # 1. Loading input data
    root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
    path_csv = filedialog.askopenfilename(title="Reference file for analysis", filetypes=[("Data", "*.csv *.xlsx")])
    root.destroy()
    if not path_csv:
        print("Analysis cancelled.")
        return
        
    res = get_data_from_csv(path_csv, COLUMN_ALIASES, read_power_mode=False)
    if not res: return
    t_csv, I_csv, U_exp, T_amb_csv, _ = res
    
    dol_an, pot_an, dol_ca, pot_ca_raw = load_ocp_data()
    
    if calib_params is None: 
        calib_params = [1.0, 1.0, SOC0_USER, 1.0, 0.0, 1.0]
    r_chg, r_dchg, soc_start, cap_factor, v_offset, rc_factor = calib_params
    
    Q_final = QGes_Ah * cap_factor
    scaling_factor = Q_ref_RC / Q_final
    frac_an = anode_optP[0] / (cathode_optP[0] + anode_optP[0])
    frac_ca = cathode_optP[0] / (cathode_optP[0] + anode_optP[0])
    theta_an = [-1, frac_an*ROCV_res[0], anode_optP[1]*ROCV_res[1], anode_optP[2]*0.95]
    theta_ca = [0,  frac_ca*ROCV_res[0], cathode_optP[1]*ROCV_res[1], cathode_optP[2]*0.92]
    
    rc_params = build_rc_param_lfp_exact()
    rc_params[:, 1] *= scaling_factor * 1.0 
    rc_params[:, [2, 4]] *= scaling_factor * rc_factor
    rc_params[:, [3, 5]] /= (scaling_factor * rc_factor)
    
    f_an = lambda x: ocp_anode(x, pot_an, dol_an)
    f_ca = lambda x: ocp_cathode(x, pot_ca_raw, dol_ca)

    # Internal evaluation function in DETAILED MODE
    def evaluate_model_detailed(n_part, dt_val):
        t_resampled = np.arange(0, t_csv[-1], dt_val)
        I_resampled = np.interp(t_resampled, t_csv, I_csv)
        T_resampled = np.interp(t_resampled, t_csv, T_amb_csv)
        U_exp_resampled = np.interp(t_resampled, t_csv, U_exp)
        
        # Creation of statistical dispersion
        rng = np.random.default_rng(42)
        R_mean_optim = R_tot_mean * 1.0  # Pure mean without multiplier
        Q_dist = rng.normal(Q_final, sigma_Q, (Ns, Np))
        R_dist = np.maximum(rng.normal(R_mean_optim, sigma_R_tot, (Ns, Np)), 1e-4)
        
        # Instantiation of the complete grid (Detailed Mode)
        cells_grid = []
        for s in range(Ns):
            row = []
            for p in range(Np):
                c = CellPDECM(None, None, None, r_dis_type, Q_dist[s,p], soc_start, n_part, m_cell, T_resampled[0], T_resampled[0])
                c.thermal.Area = Area_cell; c.thermal.R_cond = R_cond_cell
                c.thermal.R_conv = 1.0/(H_CONVECTION*Area_cell)
                c.thermal.Cs_core = m_cell*0.9*1100; c.thermal.Cs_surf = m_cell*0.1*900
                c.setup_dual_solvers(f_an, f_ca, rc_params, theta_an, theta_ca, r_dis_type)
                c.R_internal_current = R_dist[s,p]
                c.equilibrate_state(0, T_resampled[0] + 273.15)
                row.append(c)
            cells_grid.append(row)
            
        # Warm-up
        i_warmup = I_resampled[0] / Np
        if ENABLE_WARMUP:
            for _ in range(30):
                for s in range(Ns):
                    for p in range(Np):
                        current_r_mult_warmup = r_chg if i_warmup >= 0 else r_dchg
                        cells_grid[s][p].step(10.0, i_warmup, T_resampled[0], r_mult=current_r_mult_warmup)
                        
        # Main Simulation
        u_sim_raw = []
        local_offset = 0.0
        
        for k in range(len(t_resampled)):
            I_app = I_resampled[k]
            i_cell = I_app / Np
            Ta = T_resampled[k]
            step_u = []
            
            for s in range(Ns):
                for p in range(Np):
                    current_r_mult = r_chg if i_cell >= 0 else r_dchg
                    
                
                    u_corr, _, _, _, r_dyn, _ = cells_grid[s][p].step(dt_val, i_cell, Ta, r_mult=current_r_mult)
                    
                    # Dispersion
                    delta_R = (R_dist[s,p] - R_mean_optim) * current_r_mult
                    cells_grid[s][p].R_internal_current = r_dyn + delta_R
                    
                    u_final = u_corr + (i_cell * delta_R)
                    step_u.append(u_final)
                    
            u_mat = np.array(step_u).reshape(Ns, Np)
            
        
            U_pack_brut = np.sum(np.mean(u_mat, axis=1)) + (I_app * R_HARNESS_CHARGE)
            
            
            if k == 0:
                if not np.isnan(U_exp_resampled[k]):
                    local_offset = U_exp_resampled[k] - U_pack_brut
                else:
                    local_offset = v_offset # Security if real data is empty
            
            
            u_sim_raw.append(U_pack_brut + local_offset)
            
        u_sim = np.array(u_sim_raw)
        
        # --- RMSE CALCULATION  ---
        valid_mask = ~np.isnan(U_exp_resampled)
        if np.any(valid_mask):
            rmse = np.sqrt(np.mean((u_sim[valid_mask] - U_exp_resampled[valid_mask])**2))
        else:
            rmse = 999.0
            
        return rmse

    # --- 2. Definition of the test grid ---
    n_particles_list = [2, 5, 10, 15, 20, 25, 30, 50]
    dt_list = [0.5, 1.0, 2.0, 5.0, 10.0, 20 , 30, 60]
    
    rmse_matrix = np.zeros((len(n_particles_list), len(dt_list)))
    time_matrix = np.zeros((len(n_particles_list), len(dt_list)))
    
    print(f"Launching {len(n_particles_list) * len(dt_list)} Detailed simulations ({Ns}x{Np} cells)...")
    
    # --- 3. Calculation loop ---
    total_iters = len(n_particles_list) * len(dt_list)
    with tqdm(total=total_iters, desc="Creating Heatmaps") as pbar:
        for i, n in enumerate(n_particles_list):
            for j, dt_val in enumerate(dt_list):
                t_start = time.time()
                rmse = evaluate_model_detailed(n, dt_val)
                t_exec = time.time() - t_start
                
                rmse_matrix[i, j] = rmse
                time_matrix[i, j] = t_exec
                
                pbar.set_postfix({'n': n, 'dt': dt_val, 'Err(V)': f"{rmse:.3f}", 'Time(s)': f"{t_exec:.1f}"})
                pbar.update(1)

    # --- 4. Graphical Display ---
    fig = plt.figure(figsize=(18, 12))
    
    # Graph 1: 2D Heatmap (RMSE)
    ax1 = fig.add_subplot(2, 2, 1)
    im1 = ax1.imshow(rmse_matrix, cmap='viridis_r', aspect='auto', origin='lower')
    ax1.set_xticks(np.arange(len(dt_list))); ax1.set_yticks(np.arange(len(n_particles_list)))
    ax1.set_xticklabels(dt_list); ax1.set_yticklabels(n_particles_list)
    ax1.set_xlabel("Time step dt [s]"); ax1.set_ylabel("Number of particles (n)")
    ax1.set_title("Heatmap: RMSE Error [V]")
    for i in range(len(n_particles_list)):
        for j in range(len(dt_list)):
            ax1.text(j, i, f"{rmse_matrix[i, j]:.3f}", ha="center", va="center", 
                     color="r" if rmse_matrix[i, j] < np.median(rmse_matrix) else "b")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # Graph 2: 2D Heatmap (Computation time)
    ax2 = fig.add_subplot(2, 2, 2)
    im2 = ax2.imshow(time_matrix, cmap='magma_r', aspect='auto', origin='lower')
    ax2.set_xticks(np.arange(len(dt_list))); ax2.set_yticks(np.arange(len(n_particles_list)))
    ax2.set_xticklabels(dt_list); ax2.set_yticklabels(n_particles_list)
    ax2.set_xlabel("Time step dt [s]"); ax2.set_ylabel("Number of particles (n)")
    ax2.set_title("Heatmap: Execution Computation Time [seconds]")
    for i in range(len(n_particles_list)):
        for j in range(len(dt_list)):
            ax2.text(j, i, f"{time_matrix[i, j]:.1f}s", ha="center", va="center", 
                     color="r" if time_matrix[i, j] < np.median(time_matrix) else "b")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)


    X, Y = np.meshgrid(np.arange(len(dt_list)), np.arange(len(n_particles_list)))

    # Graph 3: 3D Surface (RMSE)
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    surf3 = ax3.plot_surface(X, Y, rmse_matrix, cmap='viridis_r', edgecolor='none', alpha=0.9)
    ax3.set_xticks(np.arange(len(dt_list))); ax3.set_xticklabels(dt_list)
    ax3.set_yticks(np.arange(len(n_particles_list))); ax3.set_yticklabels(n_particles_list)
    ax3.set_xlabel('Time step dt [s]'); ax3.set_ylabel('Particles (n)'); ax3.set_zlabel('RMSE [V]')
    ax3.set_title("3D Surface: Error Convergence")
    ax3.view_init(elev=30, azim=225) # Optimal viewing angle

    # Graph 4: 3D Surface (Computation time)
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    surf4 = ax4.plot_surface(X, Y, time_matrix, cmap='magma_r', edgecolor='none', alpha=0.9)
    ax4.set_xticks(np.arange(len(dt_list))); ax4.set_xticklabels(dt_list)
    ax4.set_yticks(np.arange(len(n_particles_list))); ax4.set_yticklabels(n_particles_list)
    ax4.set_xlabel('Time step dt [s]'); ax4.set_ylabel('Particles (n)'); ax4.set_zlabel('Execution Time [s]')
    ax4.set_title("3D Surface: Computational Cost")
    ax4.view_init(elev=30, azim=225)

    plt.tight_layout()
    plt.show()

# ==============================================================================
# PART 5: SIMULATION WITH BMS CONTROL
# ==============================================================================
def run_bms_simulation(calib_params):
    print("\n" + "="*60)
    print("PHASE 2: SIMULATION WITH ACTIVE BMS & LIVE LOGS")
    print("="*60)
    
    dol_an, pot_an, dol_ca, pot_ca_raw = load_ocp_data()

    print("\nPack physics detail level:")
    print("  [1] DETAILED: Simulates each cell (Random R/Q variations) -> SLOW")
    print("  [2] FAST: Simulates a single 'average' cell x Ns/Np       -> VERY FAST")
    choice_physics = input("Choice [1/2]: ")
    use_fast_mode = (choice_physics == '2')

    if calib_params is None: 
        calib_params = [1.0, 1.0, SOC0_USER, 1.0, 0.0, 1.0]
        
    r_chg, r_dchg, soc_start, cap_factor, v_offset, rc_factor = calib_params
    
    Q_final = QGes_Ah * cap_factor
    pot_ca_final = pot_ca_raw
    scaling_factor = Q_ref_RC / Q_final
    
    frac_an = anode_optP[0] / (cathode_optP[0] + anode_optP[0])
    frac_ca = cathode_optP[0] / (cathode_optP[0] + anode_optP[0])
    theta_an = [-1, frac_an*ROCV_res[0], anode_optP[1]*ROCV_res[1], anode_optP[2]*0.95]
    theta_ca = [0,  frac_ca*ROCV_res[0], cathode_optP[1]*ROCV_res[1], cathode_optP[2]*0.92]
    if theta_an[3] == 0: theta_an[3] = 2.0
    if theta_ca[3] == 0: theta_ca[3] = 2.0
    
    rc_params = build_rc_param_lfp_exact()
    rc_params[:, 1] *= scaling_factor * FACTOR_R_DYN
    rc_params[:, [2, 4]] *= scaling_factor * rc_factor
    rc_params[:, [3, 5]] /= (scaling_factor * rc_factor)
    
    f_an_final = lambda x: ocp_anode(x, pot_an, dol_an)
    f_ca_final = lambda x: ocp_cathode(x, pot_ca_final, dol_ca)
    
    cells_grid = []
    single_cell_ref = None 

    rng = np.random.default_rng(42)
    R_mean_base = R_tot_mean

    print("\nSimulation Profile Choice:")
    print("  [F] File (Real data)")
    print("  [G] Torture Test (Full profile over 24h)")
    print("  --- BMS Validation Tests ---")
    print("  [1] Test: Charge Firewall (Blocks at 80% SOC)")
    print("  [2] Test: Discharge Firewall (Clips at 5% SOC)")
    print("  [3] Test: Thermal Firewall (Overheating & Cut-off at 60°C)")
    print("  [4] Test: Physical Hysteresis (Multi-particle relaxation)")
    print("  [5] Test: Readable Torture (Macro-cycles)")
    print("  [6] Test: Staircase and Asymmetric Cycling (Limits & Thermal)")
    print("  [7] Test: Full Cycle (Large charge followed by large discharge)")
    choice = input("Choice: ").upper()
    
    if choice == 'F':
        root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
        path = filedialog.askopenfilename(title="Profile File", filetypes=[("Data", "*.csv *.xlsx")])
        root.destroy()
        if not path: sys.exit()
        
        mode_str = input("(C)urrent or (P)ower Mode? [C/P]: ").upper()
        read_pwr = (mode_str == 'P')
        
        res = get_data_from_csv(path, COLUMN_ALIASES, read_power_mode=read_pwr)
        t_sim, I_input_sim, v_ref, T_amb_sim, T_bat_sim = res
        val_prof = I_input_sim 
        mode_bms = 'CURRENT'
        title_graph = f"File: {os.path.basename(path)}"
    elif choice == '1':
        soc_start = 0.78
        cap_pack_est = Q_final * Np
        print(f"-> Generating Test Safety Charge (Initial SOC 78%)...")
        t_sim, val_prof, T_amb_sim = test_profile_charge(cap_pack_est)
        mode_bms = 'CURRENT'
        v_ref = np.full_like(t_sim, np.nan)
        T_bat_sim = np.full_like(t_sim, 25.0)
        title_graph = "Test: Safety Charge (80%)"
    elif choice == '2':
        soc_start = 0.08
        cap_pack_est = Q_final * Np
        print(f"-> Generating Test Safety Discharge (Initial SOC 8%)...")
        t_sim, val_prof, T_amb_sim = test_profile_discharge(cap_pack_est)
        mode_bms = 'CURRENT'
        v_ref = np.full_like(t_sim, np.nan)
        T_bat_sim = np.full_like(t_sim, 25.0)
        title_graph = "Test: Safety Discharge (5%)"
    elif choice == '3':
        soc_start = 0.50
        cap_pack_est = Q_final * Np
        print(f"-> Generating Test Safety Thermal (Initial SOC 50%, T_amb=48°C)...")
        t_sim, val_prof, T_amb_sim = test_profile_thermal(cap_pack_est)
        mode_bms = 'CURRENT'
        v_ref = np.full_like(t_sim, np.nan)
        T_bat_sim = np.full_like(t_sim, 48.0)
        title_graph = "Test: Thermal Safety (Cut-off 60°C)"
    elif choice == '4':
        soc_start = 0.50
        cap_pack_est = Q_final * Np
        print(f"-> Generating Test Hysteresis (Initial SOC 50%)...")
        t_sim, val_prof, T_amb_sim = test_profile_physical_hysteresis(cap_pack_est)
        mode_bms = 'CURRENT'
        v_ref = np.full_like(t_sim, np.nan)
        T_bat_sim = np.full_like(t_sim, 25.0)
        title_graph = "Test: Hysteresis"
    elif choice == '5': 
        soc_start = 0.50
        cap_pack_est = Q_final * Np
        print(f"-> Generating Test Multi-Pulse Torture (Initial SOC 50%)...")
        t_sim, val_prof, T_amb_sim = test_profile_multi_pulse_torture(cap_pack_est)
        mode_bms = 'CURRENT'
        v_ref = np.full_like(t_sim, np.nan)
        T_bat_sim = np.full_like(t_sim, 30.0)
        title_graph = "Test: Multi-Pulse (Torture)"
    elif choice == '6': 
        soc_start = 0.50 # Start at 50% to be perfectly in the middle
        cap_pack_est = Q_final * Np
        print(f"-> Generating Test Asymmetric Staircase...")
        t_sim, val_prof, T_amb_sim = test_profile_asymmetric_staircase(cap_pack_est)
        mode_bms = 'CURRENT'
        v_ref = np.full_like(t_sim, np.nan)
        T_bat_sim = np.full_like(t_sim, 25.0)
        title_graph = "Test: Steps and asymmetric cycling"
        
    elif choice == '7': 
        # Start practically empty (5%) so the big charge has room to work
        soc_start = 0.05 
        cap_pack_est = Q_final * Np
        print(f"-> Generating Test Full Cycle (Initial SOC 5%)...")
        t_sim, val_prof, T_amb_sim = test_profile_full_cycle(cap_pack_est)
        mode_bms = 'CURRENT'
        v_ref = np.full_like(t_sim, np.nan)
        T_bat_sim = np.full_like(t_sim, 25.0)
        title_graph = "Test: Full Cycle (Charge -> Rest -> Discharge)"
    else:
        cap_pack_est = Q_final * Np
        print(f"-> Generating scenario for Pack ~{cap_pack_est:.0f}Ah...")
        t_sim, val_prof, T_amb_sim = generate_bms_action_profile(cap_pack_est)
        mode_bms = 'CURRENT'
        v_ref = np.full_like(t_sim, np.nan)
        T_bat_sim = np.full_like(t_sim, 25.0)
        title_graph = "Torture Test (Multi-Phase)"

    T_init_sim = T_bat_sim[0] if len(T_bat_sim)>0 else 25.0

    T_core_init_sim = T_init_sim
    if len(T_bat_sim) > 0 and not np.isnan(T_bat_sim[0]):
        R_cond_c = R_cond_cell
        R_conv_c = 1.0 / (H_CONVECTION * Area_cell)
        
    
        T_core_init_sim = T_init_sim + (R_cond_c / R_conv_c) * (T_init_sim - T_amb_sim[0])
        print(f"🌡️ Inverse Thermal Initialization: T_surf = {T_init_sim:.2f}°C --> Deduced T_core = {T_core_init_sim:.2f}°C")
    

    if use_fast_mode:
        print("Instantiation: FAST MODE (1 Representative Cell)")
        c = CellPDECM(None, None, None, r_dis_type, Q_final, soc_start, n_particle, m_cell, T_init_sim, T_core_init_sim)
        c.thermal.Area = Area_cell; c.thermal.R_cond = R_cond_cell
        c.thermal.R_conv = 1.0/(H_CONVECTION*Area_cell)
        c.thermal.Cs_core = m_cell*0.9*1100; c.thermal.Cs_surf = m_cell*0.1*900
        c.setup_dual_solvers(f_an_final, f_ca_final, rc_params, theta_an, theta_ca, r_dis_type)
        c.R_internal_current = R_mean_base
        c.equilibrate_state(0, T_init_sim + 273.15)
        single_cell_ref = c
    else:
        print("Instantiation: DETAILED MODE (Ns x Np grid with dispersion)")
        Q_dist = rng.normal(Q_final, sigma_Q, (Ns, Np))
        R_dist_base = np.maximum(rng.normal(R_mean_base, sigma_R_tot, (Ns, Np)), 1e-4)
        
        for s in range(Ns):
            row = []
            for p in range(Np):
                c = CellPDECM(None, None, None, r_dis_type, Q_dist[s,p], soc_start, n_particle, m_cell, T_init_sim, T_core_init_sim)
                c.thermal.Area = Area_cell; c.thermal.R_cond = R_cond_cell
                c.thermal.R_conv = 1.0/(H_CONVECTION*Area_cell)
                c.thermal.Cs_core = m_cell*0.9*1100; c.thermal.Cs_surf = m_cell*0.1*900
                c.setup_dual_solvers(f_an_final, f_ca_final, rc_params, theta_an, theta_ca, r_dis_type)
                c.R_internal_current = R_dist_base[s,p]
                c.equilibrate_state(0, T_init_sim + 273.15)
                row.append(c)
            cells_grid.append(row)

    if calib_params is not None and v_offset != 0.0:
        v_offset_pack = v_offset 
        print(f"\n-> Using calibrated AI offset: {v_offset_pack:.3f} V")
    else:
        if len(v_ref) > 0 and not np.isnan(v_ref[0]):
            if use_fast_mode:
                u_base = single_cell_ref.U_cell * Ns
            else:
                u_mat = np.array([[c.U_cell for c in row] for row in cells_grid])
                u_base = np.sum(np.mean(u_mat, axis=1))
                
            v_exp_start = np.nanmean(v_ref[:min(10, len(v_ref))]) 
            v_offset_pack = v_exp_start - u_base
            print(f"\n-> Auto-Alignment: Applying a Pack offset of {v_offset_pack:.3f} V")
        else:
            v_offset_pack = 0.0

    bms_conf = {
        'Ns': Ns, 'Np': Np,
        'Capacity_Ah': Q_final,
        'V_cell_max': 3.65, 'V_cell_min': 2.0,
        'I_charge_max': 200.0, 'I_discharge_max': 300.0
    }
    bms = BatteryManagementSystem(bms_conf)
    
    steps = len(t_sim)
    dt = np.mean(np.diff(t_sim)) if steps > 1 else 1.0
    
    res_V, res_I_app, res_I_req = [], [], []
    res_T, res_SOC, res_SOH = [], [], []
    res_Limit_Ch, res_Limit_Dch = [], []
    res_T_surf = []
    res_V_cell_min, res_V_cell_max = [], []
    
    U_pack = 3.3 * Ns
    
    print(f"Starting simulation (ACTIVE BMS = {ENABLE_BMS})...")

    # ========================================================
    # CODE 2 WARM-UP (Electrical only)
    # ========================================================
    if ENABLE_WARMUP:
        print("🔥 Model warm-up (initial stabilization of RCs)...")
        I_warmup = val_prof[0]
        Ta_warmup = T_amb_sim[0]
        dt_warmup = 10.0      
        nb_steps_warmup = 30  

        I_bms_loss_warmup = P_BMS_IDLE_W / max(U_pack, 10.0) if ENABLE_BMS else 0.0
        I_phys_warmup = I_warmup - I_bms_loss_warmup
        I_cell_warmup = I_phys_warmup / Np
        r_mult_warmup = r_chg if I_phys_warmup >= 0 else r_dchg

        if use_fast_mode:
            for _ in range(nb_steps_warmup):
                single_cell_ref.step(dt_warmup, I_cell_warmup, Ta_warmup, r_mult=r_mult_warmup)
                
            single_cell_ref.thermal.T_surf = T_init_sim + 273.15
            single_cell_ref.thermal.T_core = T_core_init_sim + 273.15
            single_cell_ref.thermal.T_cell = single_cell_ref.thermal.T_core
            single_cell_ref.T = T_init_sim
        else:
            for _ in range(nb_steps_warmup):
                for s in range(Ns):
                    for p in range(Np):
                        cells_grid[s][p].step(dt_warmup, I_cell_warmup, Ta_warmup, r_mult=r_mult_warmup)
                        
            for s in range(Ns):
                for p in range(Np):
                    cells_grid[s][p].thermal.T_surf = T_init_sim + 273.15
                    cells_grid[s][p].thermal.T_core = T_core_init_sim + 273.15
                    cells_grid[s][p].thermal.T_cell = cells_grid[s][p].thermal.T_core
                    cells_grid[s][p].T = T_init_sim
    else:
        print("⏭️ Model warm-up bypassed. Starting exactly at requested SOC.")
    

    t_start_sim = time.time()
    for k in tqdm(range(steps), desc="BMS Simulation", unit="step"):
        val_req_grid = val_prof[k] # What the grid requests
        Ta = T_amb_sim[k]

        if val_req_grid >= 0:
            # Charging: battery receives less than what the grid sends
            val_req_bat = val_req_grid * EFF_TOTAL  
        else:
            # Discharging: battery must provide more to satisfy the grid
            val_req_bat = val_req_grid / EFF_TOTAL
        
        if use_fast_mode:
            T_max = single_cell_ref.thermal.T_core - 273.15     
            T_surf_val = single_cell_ref.thermal.T_surf - 273.15 
            SOC_avg = single_cell_ref.cathode.SOC_global
            SOH_min = single_cell_ref.thermal.SOH
        else:
            all_tc = [c.thermal.T_core - 273.15 for row in cells_grid for c in row]
            all_ts = [c.thermal.T_surf - 273.15 for row in cells_grid for c in row]
            T_max = np.max(all_tc)
            T_surf_val = np.max(all_ts) 
            SOC_avg = np.mean([c.cathode.SOC_global for row in cells_grid for c in row])
            SOH_min = np.min([c.thermal.SOH for row in cells_grid for c in row])
        
        ccl, dcl = bms.compute_limits(T_max, SOC_avg)
        
        R_physique_statique = R_mean_base * ((r_chg+r_dchg)/2) * (Ns/Np) + R_HARNESS_CHARGE
        R_est_bms = R_physique_statique * 1.5 
        
        if ENABLE_BMS:
            I_app = bms.process_request(val_req_bat, mode_bms, U_pack, R_est_bms)
            I_bms_loss = P_BMS_IDLE_W / max(U_pack, 10.0)
        else:
            I_app = val_req_bat 
            I_bms_loss = 0.0
            bms.flag_voltage_limit = False
            
        I_physique_total = I_app - I_bms_loss
        current_r_mult = r_chg if I_physique_total >= 0 else r_dchg
        
        if use_fast_mode:
            I_cell = I_physique_total / Np
            u, _, _, _, r_dyn, _ = single_cell_ref.step(dt, I_cell, Ta, r_mult=current_r_mult)
            single_cell_ref.R_internal_current = r_dyn
            
            U_pack = (u * Ns) + (I_app * R_HARNESS_CHARGE) + v_offset_pack
            
            res_V_cell_min.append(u)
            res_V_cell_max.append(u)
        else:
            step_u = []
            
            for s in range(Ns):
                conductances = []
                for p in range(Np):
                    r_cellule = R_dist_base[s, p] * current_r_mult
                    r_safe = max(r_cellule, 1e-6) 
                    conductances.append(1.0 / r_safe)
                
                G_totale_etage = sum(conductances)
                
                for p in range(Np):
                    fraction_courant = conductances[p] / G_totale_etage
                    I_cell_dynamique = I_physique_total * fraction_courant
                    
                    u_theorique, _, _, _, r_dyn, _ = cells_grid[s][p].step(dt, I_cell_dynamique, Ta, r_mult=current_r_mult)
                    
                    delta_R = (R_dist_base[s, p] - R_mean_base) * current_r_mult
                    u_reel = u_theorique + (I_cell_dynamique * delta_R)
                    
                    cells_grid[s][p].R_internal_current = r_dyn + delta_R
                    step_u.append(u_reel)
            
            u_mat = np.array(step_u).reshape(Ns, Np)
            U_pack = np.sum(np.mean(u_mat, axis=1)) + (I_app * R_HARNESS_CHARGE) + v_offset_pack
            
            res_V_cell_min.append(np.min(u_mat))
            res_V_cell_max.append(np.max(u_mat))

        res_V.append(U_pack)
        res_I_app.append(I_app)
        res_T.append(T_max)
        res_T_surf.append(T_surf_val) 
        res_SOC.append(SOC_avg)
        res_Limit_Ch.append(ccl)
        res_Limit_Dch.append(-dcl)
        res_SOH.append(SOH_min) 
        
        if mode_bms == 'POWER': 
            req_equiv = val_req_bat / max(U_pack, 10.0)
        else: 
            req_equiv = val_req_bat
            
        res_I_req.append(req_equiv)

    t_end_sim = time.time()
    sim_duration = t_end_sim - t_start_sim
    print(f"\n⏱️ COMPUTATION TIME: {sim_duration:.2f} seconds")

    res_V = np.array(res_V, dtype=float)
    v_ref = np.array(v_ref, dtype=float)
    res_SOC_arr = np.array(res_SOC) * 100

    v_ref[v_ref < -1e9] = np.nan
    res_V[res_V < -1e9] = np.nan
    res_V[res_V > 1e9] = np.nan 

    show_exp = True
    if np.all(np.isnan(v_ref)) or np.nanmax(np.abs(np.nan_to_num(v_ref))) < 0.1:
        show_exp = False
    
    print("\n" + "-"*40)
    print("📊 SIMULATION RESULTS:")
    
    if choice == 'F':
        if show_exp:
            valid_v = ~np.isnan(v_ref) & ~np.isnan(res_V)
            if np.any(valid_v):
                rmse_V = np.sqrt(np.mean((res_V[valid_v] - v_ref[valid_v])**2))
                mean_V = np.mean(v_ref[valid_v])
                rmse_V_pct = (rmse_V / mean_V) * 100 if mean_V != 0 else 0
                print(f"📉 Voltage RMSE      : {rmse_V:.6f} V  ({rmse_V_pct:.6f} %)")
        else:
            print("📉 Voltage RMSE      : Not calculable (Missing data)")
            
        valid_t = ~np.isnan(T_bat_sim) & ~np.isnan(np.array(res_T_surf))
        if np.any(valid_t):
            res_T_surf_arr = np.array(res_T_surf, dtype=float)
            T_bat_sim_arr = np.array(T_bat_sim, dtype=float)
            rmse_T = np.sqrt(np.mean((res_T_surf_arr[valid_t] - T_bat_sim_arr[valid_t])**2))
            mean_T = np.mean(T_bat_sim[valid_t])
            rmse_T_pct = (rmse_T / mean_T) * 100 if mean_T != 0 else 0
            print(f"🌡️ Temperature RMSE  : {rmse_T:.6f} °C  ({rmse_T_pct:.6f} %)")
        else:
            print("🌡️ Temperature RMSE  : Not calculable (Missing data)")
    else:
        print("Generated profile (Torture Test): No reference experimental data.")
        
    print("-" * 40)
    
    # =========================================================
    # ROUND TRIP EFFICIENCY (RTE) & ENERGY COMPARISON
    # =========================================================
    arr_V = np.array(res_V)         
    arr_I = np.array(res_I_app)     
    puissance_W = arr_V * arr_I
    
    charge_energy_wh = np.sum(puissance_W[arr_I > 0]) * dt / 3600.0
    discharge_energy_wh = np.sum(np.abs(puissance_W[arr_I < 0])) * dt / 3600.0
    
    charge_energy_wh_grid = charge_energy_wh / EFF_TOTAL
    discharge_energy_wh_grid = discharge_energy_wh * EFF_TOTAL

    print("\n⚡ ENERGY BALANCE (BATTERY TERMINALS):")
    print(f"   -> Absorbed by Battery (Charge)     : {charge_energy_wh:.2f} Wh")
    print(f"   -> Restored by Battery (Discharge)  : {discharge_energy_wh:.2f} Wh")

    print("\n🌍 ENERGY BALANCE (GRID LEVEL - Including Losses):")
    print(f"   -> Taken from Grid (Charge)         : {charge_energy_wh_grid:.2f} Wh")
    print(f"   -> Delivered to Grid (Discharge)    : {discharge_energy_wh_grid:.2f} Wh")
    if charge_energy_wh > 0:
        raw_rte = (discharge_energy_wh / charge_energy_wh) * 100.0
        print(f"   📊 Raw Round Trip Efficiency        : {raw_rte:.2f} %")
        
        
        soc_start_test = res_SOC[0]
        soc_end_test = res_SOC[-1]
        delta_soc = soc_end_test - soc_start_test
        
        if abs(delta_soc) > 0.005:
            nominal_pack_voltage = 3.2 * Ns  
            pack_capacity_wh = Q_final * Np * nominal_pack_voltage
            delta_stored_energy_wh = delta_soc * pack_capacity_wh
            corrected_rte = ((discharge_energy_wh + delta_stored_energy_wh) / charge_energy_wh) * 100.0
            
            print(f"   ⚠️ Open cycle detected! (Delta SOC: {delta_soc*100:+.2f}%)")
            print(f"   -> Change in stored internal energy: {delta_stored_energy_wh:+.2f} Wh")
            print(f"   🏆 CORRECTED Round Trip Efficiency : {corrected_rte:.2f} %")
        else:
            print(f"   🏆 Round Trip Efficiency (Energy): {raw_rte:.2f} %")
    else:
        print("   -> RTE not calculable (No charge detected).")

    # --- EXPERIMENTAL ENERGY COMPARISON ---
    if choice == 'F' and show_exp:
        print("\n🔬 ENERGY BALANCE (EXPERIMENTAL VS SIMULATION):")
        
        valid_mask = ~np.isnan(v_ref)
        
        if np.any(valid_mask):
            v_exp_valid = v_ref[valid_mask]
            i_exp_valid = val_prof[valid_mask]
            p_exp_W = v_exp_valid * i_exp_valid
            
            e_charge_exp_Wh = np.sum(p_exp_W[i_exp_valid > 0]) * dt / 3600.0
            e_decharge_exp_Wh = np.sum(np.abs(p_exp_W[i_exp_valid < 0])) * dt / 3600.0
            
            # Calculate error percentages
            err_charge_pct = ((charge_energy_wh - e_charge_exp_Wh) / e_charge_exp_Wh) * 100 if e_charge_exp_Wh != 0 else 0
            err_decharge_pct = ((discharge_energy_wh - e_decharge_exp_Wh) / e_decharge_exp_Wh) * 100 if e_decharge_exp_Wh != 0 else 0
            
            print(f"   [CHARGE]    Simulated: {charge_energy_wh:.2f} Wh | Experimental: {e_charge_exp_Wh:.2f} Wh | Diff: {charge_energy_wh - e_charge_exp_Wh:+.2f} Wh ({err_charge_pct:+.2f}%)")
            print(f"   [DISCHARGE] Simulated: {discharge_energy_wh:.2f} Wh | Experimental: {e_decharge_exp_Wh:.2f} Wh | Diff: {discharge_energy_wh - e_decharge_exp_Wh:+.2f} Wh ({err_decharge_pct:+.2f}%)")
            
            if e_charge_exp_Wh > 0:
                raw_rte_exp = (e_decharge_exp_Wh / e_charge_exp_Wh) * 100.0
                print(f"   📊 Raw Experimental RTE: {raw_rte_exp:.2f} %")
        else:
             print("   -> Cannot calculate experimental energy (missing voltage data).")
        
    print("\nGenerating BMS graphs...")
    fig, ax = plt.subplots(4, 1, figsize=(12, 16), sharex=True)

    arr_t = np.array(t_sim)
    arr_req = np.array(res_I_req)
    arr_app = np.array(res_I_app)
    arr_V = np.array(res_V)

    ax[0].set_title(f"BMS ACTION: {title_graph} ({'FAST' if use_fast_mode else 'DETAILED'} Mode)", fontsize=14, fontweight='bold')
    ax[0].plot(arr_t, arr_req, 'k--', linewidth=1.5, alpha=0.6, label='User Demand')
    ax[0].plot(arr_t, arr_app, 'b', linewidth=2, label='Actual Current (BMS)')
    mask_cv = (arr_req > 0) & (arr_app < (arr_req - 0.1))
    ax[0].fill_between(arr_t, arr_app, arr_req, where=mask_cv, color='cyan', alpha=0.5, hatch='//', label='CV Zone')
    mask_prot = (arr_req < 0) & (arr_app > (arr_req + 0.1))
    ax[0].fill_between(arr_t, arr_app, arr_req, where=mask_prot, color='red', alpha=0.4, label='Protection')
    ax[0].set_ylabel("Current [A]"); ax[0].legend(loc='upper right'); ax[0].grid(True)

    ax[1].plot(arr_t, arr_V, 'purple', linewidth=2, label='Pack Voltage (Sim)')
    if show_exp:
        ax[1].plot(arr_t, v_ref, 'k:', linewidth=1.5, alpha=0.7, label='Experimental')
    
    v_max_pack = bms.V_cell_max * Ns
    v_min_pack = bms.V_cell_min * Ns
    V_threshold = 0.5 * Ns 
    
    if np.nanmax(arr_V) > (v_max_pack - V_threshold):
        ax[1].axhline(v_max_pack, color='r', linestyle='--', linewidth=1, label='Max')
        
    if np.nanmin(arr_V) < (v_min_pack + V_threshold):
        ax[1].axhline(v_min_pack, color='r', linestyle='--', linewidth=1, label='Min')

    ax[1].set_ylabel("Voltage [V]")
    
    if show_exp:
        v_concat = np.concatenate([arr_V, v_ref[~np.isnan(v_ref)]])
        y_min, y_max = np.min(v_concat), np.max(v_concat)
        margin = (y_max - y_min) * 0.1
        ax[1].set_ylim(y_min - margin, y_max + margin)
    else:
        ax[1].set_ylim(v_min_pack - 0.5, v_max_pack + 0.5)
        
    if choice == 'F' and show_exp and 'rmse_V' in locals():
        text_str = f"RMSE: {rmse_V:.3f}V\nErr: {rmse_V_pct:.2f}%"
        ax[1].text(0.02, 0.95, text_str, transform=ax[1].transAxes, 
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax[1].legend(loc='upper left')
    ax[1].grid(True)

    # --- HYSTERESIS VISUALIZATION  --- 
    if choice == '4':
        target_soc = 50.0 
        
        idx_charging = np.where(arr_app > 5)[0]
        idx_discharging = np.where(arr_app < -5)[0]
        
        if len(idx_charging) > 0 and len(idx_discharging) > 0:
            closest_chg_idx = idx_charging[np.argmin(np.abs(res_SOC_arr[idx_charging] - target_soc))]
            closest_dch_idx = idx_discharging[np.argmin(np.abs(res_SOC_arr[idx_discharging] - target_soc))]
            
            t_chg = arr_t[closest_chg_idx]
            t_dch = arr_t[closest_dch_idx]
            v_chg = arr_V[closest_chg_idx]
            v_dch = arr_V[closest_dch_idx]
            
            axins = ax[1].inset_axes([0.65, 0.1, 0.3, 0.4]) 
            axins.plot(res_SOC_arr, arr_V, 'purple', linewidth=1.5)
            
            axins.set_xlim(target_soc - 0.5, target_soc + 0.5)
            axins.set_ylim(min(v_chg, v_dch) - 0.5, max(v_chg, v_dch) + 0.5)
            
            axins.set_title("Hysteresis Zoom", fontsize=9)
            axins.set_xlabel("SOC [%]", fontsize=8)
            axins.tick_params(labelsize=8)
            axins.grid(True, linestyle=':', alpha=0.6)
            
            axins.annotate('', 
                         xy=(target_soc, v_chg), 
                         xytext=(target_soc, v_dch),
                         arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
            
            delta_v_gap = abs(v_chg - v_dch)
            axins.text(target_soc + 0.05, (v_chg + v_dch) / 2, 
                     f'$\\Delta$V = {delta_v_gap:.2f}V\n(Same SOC)', 
                     va='center', ha='left', fontsize=8, 
                     bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.8))

    ax[2].plot(arr_t, res_SOC_arr, 'g', linewidth=2, label='SOC %')
    ax[2].axhline(bms.SOC_ch_ramp_start*100, color='orange', linestyle=':')
    ax[2].axhline(bms.SOC_dch_ramp_start*100, color='orange', linestyle=':')
    ax[2].set_ylabel("SOC [%]"); ax[2].legend(loc='right'); ax[2].grid(True)

    ax[3].plot(arr_t, res_T, 'orange', linewidth=2, label='Simulated (Core)')
    ax[3].plot(arr_t, res_T_surf, 'red', linestyle='--', linewidth=1.5, label='Simulated (Surface)')
    
    if show_exp and not np.all(np.isnan(T_bat_sim)):
        ax[3].plot(arr_t, T_bat_sim, 'k:', linewidth=1.5, alpha=0.8, label='Experimental (Surface)')

    T_threshold = 5.0 
    if np.nanmax(res_T) > (45 - T_threshold):
        ax[3].axhline(45, color='y', linestyle='--', label='Warning (45°C)')
    if np.nanmax(res_T) > (60 - T_threshold):
        ax[3].axhline(60, color='r', linewidth=2, label='Cutoff (60°C)')
        
    ax[3].set_ylabel("Temp [°C]")
    ax[3].set_xlabel("Time [seconds]")
    
    if choice == 'F' and 'rmse_T' in locals():
        text_str = f"RMSE: {rmse_T:.2f}°C\nErr: {rmse_T_pct:.2f}%"
        ax[3].text(0.02, 0.95, text_str, transform=ax[3].transAxes, 
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax[3].legend(loc='upper right') 
    ax[3].grid(True)

    plt.tight_layout()
    plt.show()


    if not use_fast_mode:
        plt.figure(figsize=(10, 4))
        
        plt.plot(arr_t, res_V_cell_max, 'r', linewidth=1.5, label='Most charged cell (V_max)')
        plt.plot(arr_t, res_V_cell_min, 'b', linewidth=1.5, label='Weakest cell (V_min)')
        plt.fill_between(arr_t, res_V_cell_min, res_V_cell_max, color='gray', alpha=0.3, label='Delta V (Imbalance)')
        
        delta_v_array = np.array(res_V_cell_max) - np.array(res_V_cell_min)
        max_delta_v = np.max(delta_v_array)
        max_idx = np.argmax(delta_v_array)
        t_max_delta = arr_t[max_idx]
        
        text_str = f"MAX $\\Delta$V: {max_delta_v * 1000:.1f} mV"
        plt.text(0.02, 0.95, text_str, transform=plt.gca().transAxes, 
                 fontsize=11, fontweight='bold', color='darkred',
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='red'))
        
        plt.plot([t_max_delta, t_max_delta], [res_V_cell_min[max_idx], res_V_cell_max[max_idx]], 
                 'k--', linewidth=1.5, label='Critical moment')

        plt.title("Internal Pack Dispersion: Gap between best and worst cell", fontweight='bold')
        plt.xlabel("Time [seconds]")
        plt.ylabel("Cell Voltage [V]")
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    print("\n" + "="*50)
    print(">>> COMPLETE BMS SIMULATOR (LOGIC + PHYSICS) <<<")
    print("="*50)
    
    SAVED_PARAMS = {
        'R_chg':  4.71,    # Charge resistance multiplier
        'R_dchg': 2.13,    # Discharge resistance multiplier
        'SOC0':   0.4725,  # Initial SOC 
        'Cap_f':  0.8237,  # Capacity multiplier
        'Offset': 1.669,   # Global voltage offset
        'RC_f':   2.165    # RC dynamics multiplier
    }
    
    saved_params = [
        SAVED_PARAMS['R_chg'], 
        SAVED_PARAMS['R_dchg'], 
        SAVED_PARAMS['SOC0'], 
        SAVED_PARAMS['Cap_f'], 
        SAVED_PARAMS['Offset'], 
        SAVED_PARAMS['RC_f']
    ]
  
    # AGING PARAMETER OPTIMIZATION PROMPT
    opt_params_soh = None
    do_opt = input("\nRun aging parameter optimization? [Y/N]: ").upper()
    if do_opt == 'Y':
        opt_params_soh = run_aging_optimization()

    # ELECTRICAL PARAMETER CALIBRATION
    do_calib = input("\nRun electrical calibration before simulation? [Y/N]: ").upper()
    
    if do_calib == 'Y':
        params = run_calibration_routine()
        if params is None: 
            print("-> Calibration cancelled. Using saved parameters.")
            params = saved_params
    else:
        print("\n-> Direct mode: Using saved optimized parameters.")
        params = saved_params
        
    print("\nWhat do you want to simulate?")
    print("  [1] Classic BMS Simulation (File or Torture Test)")
    print("  [2] LONG TERM Aging (20 Years - Cycle Jumping at 1.5 cycles/day)")
    print("  [3] Sensitivity Analysis (n_particle & dt)")
    choix_simu = input("Choice [1/2/3]: ")
    
    if choix_simu == '2':
        run_long_term_aging_simulation(params, opt_params_soh)
    elif choix_simu == '3':
        run_sensitivity_analysis(params)
    else:
        run_bms_simulation(params)
