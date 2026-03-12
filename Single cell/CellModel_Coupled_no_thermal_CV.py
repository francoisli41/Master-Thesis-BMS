import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize_scalar
from scipy.linalg import lu_factor, lu_solve
from scipy.integrate import quad

# ==============================================================================
# 1. MODÈLE THERMIQUE ET VIEILLISSEMENT (AMÉLIORÉ - Sources: Wang et al. 2011, Lu et al. 2013)
# ==============================================================================
class ThermalAgingModel:
    def __init__(self, dt, T_init=25.0, m_cell=3.0, Cp=1100, Area=0.1, h=10):
        self.dt = dt
        # Températures en Kelvin
        self.T_core = T_init + 273.15 
        self.T_surf = T_init + 273.15 
        self.T_amb_K = T_init + 273.15 
        
        # Paramètres thermiques de la cellule
        self.Cs_core = (m_cell * 0.90) * Cp 
        self.Cs_surf = (m_cell * 0.10) * 900 
        self.R_conv = 1.0 / (h * Area)
        self.R_cond = 0.4 

        # État de Santé (SOH)
        self.SOH = 1.0
        self.Q_loss_acc = 0.0
        
        # Compteurs pour l'intégration (Temps et Débit Ah)
        self.time_elapsed = 0.0      # Secondes totales
        self.Ah_throughput = 0.0     # Ampères-heures totaux échangés
        
        # --- NOUVEAUX PARAMÈTRES LFP (Moins pessimistes) ---
        self.R_gas = 8.314
        
        # Énergies d'activation (J/mol) - Source: Wang et al. 2011
        # Le LFP est thermiquement plus stable que le NCA/NMC.
        # Ea_cal ~ 20-25 kJ/mol (au lieu de 24.5k)
        # Ea_cyc ~ 20-22 kJ/mol (au lieu de 31.7k qui était trop élevé/pessimiste)
        self.Ea_cal = 21000.0  
        self.Ea_cyc = 22000.0
        
        # Facteurs pré-exponentiels (k_ref)
        # Calibrés pour une perte de ~20% sur 10 ans (cal) ou 3000-5000 cycles.
        # k_cal_ref est ajusté pour une loi en racine carrée du temps (h^0.5).
        # k_cyc_ref est ajusté pour une loi en racine carrée des Ah (Ah^0.5).
        self.k_cal_ref = 5.0e-5  
        self.k_cyc_ref = 3.5e-4 

    def get_entropic_coeff(self, soc, current):
        """
        Coefficient entropique (dU/dT) pour le calcul de la chaleur réversible.
        Important pour la précision thermique, même si l'impact sur le vieillissement est indirect.
        """
        soc_points = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
        dudt_charge = [-0.11, -0.03, -0.13, -0.10, -0.12, -0.22, 0.30, 0.295, 0.30, 0.27, 0.25, 0.16, 0.03, 0.105, 0.095, 0.10, 0.12, 0.19]
        dudt_discharge = [-0.17, -0.20, -0.22, -0.18, -0.20, 0.035, 0.04, 0.06, 0.07, 0.08, 0.075, 0.015, -0.13, -0.12, -0.11, -0.09, -0.06, 0.04]

        if current < -1e-5: target_curve = dudt_charge
        else: target_curve = dudt_discharge
        
        val_mV = np.interp(soc, soc_points, target_curve)
        return val_mV * 1e-3

    def get_soc_stress(self, soc):
        """
        Remplace get_anode_stress. 
        Le LFP est robuste, le stress majeur est le stockage à haut SOC (>90%).
        Basé sur Schimpe et al.
        """
        # Fonction douce : faible impact à 50%, impact croissant vers 100%
        # exp(3 * (SOC - 0.5)) donne un facteur ~1 à 50% et ~4.5 à 100%
        base_stress = np.exp(3.0 * (soc - 0.5))
        return base_stress

    def step(self, I, U_cell, U_ocv, SOC, T_amb_dynamic=None, U_anode_val=None):
        # 1. Gestion Thermique (Physique)
        T_env = T_amb_dynamic + 273.15 if T_amb_dynamic is not None else self.T_amb_K
        
        Q_irr = np.abs(I * (U_cell - U_ocv))
        coeff_entropic = self.get_entropic_coeff(SOC, current=I)
        Q_rev = -I * self.T_core * coeff_entropic 
        Q_gen = Q_irr + Q_rev
        
        Q_core_to_surf = (self.T_core - self.T_surf) / self.R_cond
        Q_surf_to_amb = (self.T_surf - T_env) / self.R_conv
        
        dT_core = (Q_gen - Q_core_to_surf) / self.Cs_core * self.dt
        self.T_core += dT_core
        dT_surf = (Q_core_to_surf - Q_surf_to_amb) / self.Cs_surf * self.dt
        self.T_surf += dT_surf
        
        # Mise à jour des compteurs globaux
        self.time_elapsed += self.dt
        self.Ah_throughput += np.abs(I) * self.dt / 3600.0
        
        # 2. Calcul du Vieillissement (Amélioré)
        # On utilise T_core pour la cinétique de réaction
        T_physics = max(self.T_core, 220.0) # Sécurité numérique
        
        # Facteurs Arrhenius
        # Note: 298.15K est la température de référence incluse dans les paramètres k_ref
        arr_term_cal = np.exp(-self.Ea_cal / self.R_gas * (1/T_physics - 1/298.15))
        arr_term_cyc = np.exp(-self.Ea_cyc / self.R_gas * (1/T_physics - 1/298.15))
        
        # Facteur de Stress SOC (Calendaire)
        stress_soc = self.get_soc_stress(SOC)
        
        # Calcul des dérivés de perte (dQ/dt)
        # Modèle: Q_loss = k * t^0.5  => dQ = 0.5 * k * t^-0.5 * dt
        
        # A. Perte Calendaire (Temps)
        # On évite la division par zéro avec max(..., 1 seconde)
        time_hours = max(self.time_elapsed / 3600.0, 1.0/3600.0)
        rate_cal = (self.k_cal_ref * stress_soc * arr_term_cal) / (2 * np.sqrt(time_hours))
        d_Qcal = rate_cal * (self.dt / 3600.0) # Conversion dt en heures
        
        # B. Perte Cyclage (Ah throughput)
        # Modèle: Q_loss = k * Ah^0.5 => dQ = 0.5 * k * Ah^-0.5 * dAh
        Ah_total = max(self.Ah_throughput, 1e-6)
        rate_cyc = (self.k_cyc_ref * arr_term_cyc) / (2 * np.sqrt(Ah_total))
        d_Ah = np.abs(I) * self.dt / 3600.0
        d_Qcyc = rate_cyc * d_Ah
        
        # Somme des pertes (Hypothèse d'indépendance des mécanismes)
        self.Q_loss_acc += (d_Qcal + d_Qcyc)
        
        # Saturation SOH
        self.SOH = max(0.05, 1.0 - self.Q_loss_acc)
        
        # Retourne exactement les mêmes paramètres qu'avant
        return self.T_surf - 273.15, self.SOH, Q_gen


# ==============================================================================
# 2. DEMI-CELLULE (INCHANGÉ)
# ==============================================================================
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

    def step(self, dt, I, SOH, T_kelvin):
        self.Q_OCVMax = self.Q_OCVMax_nominal * max(SOH, 0.1)
        scaling = np.exp((25000.0/8.314) * (1/T_kelvin - 1/298.15))

        if self.use_rc: q_states = self.x[self.n_particle+1:]
        else: q_states = self.x[self.n_particle-1:]
        
        soc_loc = np.clip(q_states / self.Q_OCVMax, 0, 1)
        self.SOC_global = np.sum(q_states) / np.sum(self.Q_OCVMax)
        ocv_loc = self.energy_curve(soc_loc)
        
        epsilon = 1e-5
        ocv_diff = (self.energy_curve(soc_loc+epsilon) - ocv_loc)/epsilon
        ocv_diff = np.clip(ocv_diff, 0, 50.0)

        if self.use_rc: self.nu[2:self.n_particle+1] = ocv_loc[1:] - ocv_loc[:-1]
        else: self.nu[0:self.n_particle-1] = ocv_loc[1:] - ocv_loc[:-1]

        A = self.A_base.copy()
        b = self.b_base.copy()
        d_curr = self.d_vec.copy()

        rc_val_R0 = 0.02
        if self.use_rc:
            s_lookup = np.clip(self.SOC_global, 0, 1)
            rc_vals = np.array([self.R1_p(s_lookup), self.R2_p(s_lookup), self.R0_p(s_lookup)]) * scaling
            caps = np.array([self.C1_p(s_lookup), self.C2_p(s_lookup)])
            rc_val_R0 = rc_vals[2]
            
            A[0:2, 0:2] = np.diag(-1.0/(rc_vals[:2]*caps))
            b[0:2] = 1.0/caps
            d_curr += rc_vals[2]
            
            term1 = -ocv_diff[:-1] / self.Q_OCVMax[:-1]
            term2 = ocv_diff[1:] / self.Q_OCVMax[1:]
            
            rows = np.arange(2, 2 + (self.n_particle - 1))
            J = A.copy()
            J[rows, np.arange(self.n_particle+1, 2*self.n_particle)] += term1
            J[rows, np.arange(self.n_particle+2, 2*self.n_particle+1)] += term2
        else:
            J = A.copy()

        r0 = A @ self.x + b * I + self.nu
        dx = lu_solve(lu_factor(self.M - dt * J), r0)
        self.x += dt * dx
        
        y_vec = self.C @ self.x + d_curr * I + ocv_loc
        self.U_electrode = y_vec[0]
        
        return self.U_electrode, self.SOC_global, rc_val_R0


# ==============================================================================
# 3. CELLULE COMPLÈTE (CellPDECM) - INCHANGÉ (Compatible nouveau modèle)
# ==============================================================================
class CellPDECM:
    def __init__(self, energy_curve_func, rc_param, theta, r_dis, QGes, SOC0, n_particle, m_cell, T_init=25.0, 
                 ocv_hysteresis_funcs=None):
        self.QGes_Ah = QGes
        self.n_particle = n_particle
        self.QGes_C = QGes * 3600.0
        self.Q_oh = 0.02
        n2p = 1.1

        self.SOC0_cat = (SOC0 * self.QGes_C + self.QGes_C * self.Q_oh) / (self.QGes_C + (self.QGes_C * self.Q_oh))
        self.SOC0_an = SOC0 / n2p
        
        self.thermal = ThermalAgingModel(dt=1.0, T_init=T_init, m_cell=m_cell)
        self.T = T_init 
        self.U_cell = 0
        self.R_internal_current = 0.02 

        self.ocv_hysteresis_funcs = ocv_hysteresis_funcs
        self.use_hysteresis = (ocv_hysteresis_funcs is not None)

    def setup_dual_solvers(self, ocp_anode_func, ocp_cathode_func, rc_param, theta_anode, theta_cathode, r_dis):
        n2p = 1.1
        rc_an = rc_param.copy(); rc_an[:, 0] /= n2p
        self.anode = HalfCellSolver(ocp_anode_func, rc_an, theta_anode, r_dis, self.QGes_C * n2p, self.SOC0_an, self.n_particle)
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

    def step(self, dt, I, T_amb_dynamic, T_forcing_K=None):
         self.thermal.dt = dt
         T_physics_K = T_forcing_K if (T_forcing_K is not None and not np.isnan(T_forcing_K)) else self.thermal.T_core
         SOH = self.thermal.SOH
         
         U_an, SOC_an, R0_an = self.anode.step(dt, I, SOH, T_physics_K)
         U_ca, SOC_ca, R0_ca = self.cathode.step(dt, I, SOH, T_physics_K)
         
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
         
         # Appel au modèle thermique mis à jour
         # Note: U_anode_val est passé mais ignoré dans la nouvelle formule de stress
         T_surf_sim, SOH_new, Q_gen = self.thermal.step(
             I=I, U_cell=self.U_cell, U_ocv=OCV_final, SOC=self.cathode.SOC_global, 
             T_amb_dynamic=T_amb_dynamic, U_anode_val=-U_an
         )
         self.T = T_surf_sim
         SOC_final = self.cathode.SOC_global * (1 + self.Q_oh) - self.Q_oh
         R_app = R0_an + R0_ca
         
         return self.U_cell, T_surf_sim, SOC_final, Q_gen, R_app, SOH_new