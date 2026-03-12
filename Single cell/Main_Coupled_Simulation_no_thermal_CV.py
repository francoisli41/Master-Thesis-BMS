import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd
import os
import sys
import time
import tkinter as tk
from tkinter import filedialog
from tqdm import tqdm
from scipy.interpolate import CubicSpline, PchipInterpolator, interp1d
from scipy.optimize import minimize_scalar

# --- IMPORT DE LA CLASSE CELLULE ---
from CellModel_Coupled_no_thermal_CV import CellPDECM

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================

COLUMN_ALIASES = {
    'Time': ['Time', 'time', 't', 'Test_Time(s)', 't_ch_power', 'Temps', 'Time [s]', 't_dc_power'],
    'Current': ['Current', 'current', 'I', 'i', 'Current(A)', 'I_in_A', 'I_ch_power', 'Current [A]', 'I_dc_power'],
    'Voltage': ['Voltage', 'voltage', 'V', 'v', 'Voltage(V)', 'V_in_V', 'U_ch_power', 'Voltage [V]', 'U_dc_power'],
    'Temp_Amb': ['T_amb', 'T_env', 'Temperature', 'T_Room_in_C', 'Amb_Temp'],
    'Temp_Bat': ['T_bat', 'T_cell', 'T_Cell', 'T_Bat_in_C', 'Temp_Bat', 'Temperature (C)_1']
}

DATA_SOURCE_TYPE = "FILE" 
Ns = 14
Np = 2
FORCED_DT = 1        
# --- MODIFICATIONS CLÉS POUR FIXER LES PICS ET LA MONTÉE DE TENSION ---
FACTOR_R_DYN = 1          # Légèrement réduit pour adoucir les variations
R_HARNESS_CHARGE = 0.00     # RÉDUIT AU MINIMUM (pour éliminer les 2.4V d'erreur artificielle)
R_HARNESS_DISCHARGE = 0.00  # RÉDUIT AU MINIMUM
QGes_Ah = 100                 

Q_ref_RC = 1.0189        
SOC0_USER = 0.0
Q_oh_USER = 0.02
R0_cell_nom = 0.4/1000     

H_cell = 0.218  # 218 mm
W_cell = 0.141  # 141 mm
T_cell = 0.066  # 66 mm (Thickness)   
Area_cell = 2 * (H_cell * W_cell) + 2 * (H_cell * T_cell) + 2 * (W_cell * T_cell)
m_cell = 3.1             
k_thermal_thickness = 0.91
R_cond_cell = (T_cell / 2) / (k_thermal_thickness * (H_cell * W_cell))     

ACTIVATE_CV_LIMITER = True
V_CELL_MAX =  3.65
SEUIL_ACTIVATION = 3.9
facteur_gain = 0.8

ACTIVATE_SOC_CALIBRATION = True 
OCV_SOURCE = "FILE" 

n_particle = 30
r_dis_type = "weibull_cap" 
COOLING_SYSTEM = "AIR" 

sigma_rel_Q = 0.0039
sigma_rel_R = 0.022
#R_weld_mean = (0.318 - 0.167) / 2 * 1e-3
R_weld_mean = 0
delta_R_weld=0
# delta_R_weld = 0.1 * 1e-3
# R_busbar = 0.00002
R_busbar = 0

sigma_Q = sigma_rel_Q * QGes_Ah
sigma_R_cell = sigma_rel_R * R0_cell_nom
sigma_R_weld = (delta_R_weld / 1.645)
sigma_R_tot = np.sqrt(sigma_R_cell**2 + sigma_R_weld**2)
R_tot_mean = R0_cell_nom + R_busbar + R_weld_mean
Total_Cells = Ns * Np

if COOLING_SYSTEM == "AIR":
    Cp_fluid = 1006.0        
    Flow_rate = 0.05         
    H_CONVECTION = 5 
else:
    Cp_fluid = 3500.0        
    Flow_rate = 0.20         
    H_CONVECTION = 800.0     
    
Thermal_Coeff_Fluid = 1.0 / (Flow_rate * Cp_fluid)
BASE_PATH = "." 
OCP_PATH = #your OCP path

# ==============================================================================
# 3. HELPERS
# ==============================================================================
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
        0.003978932415778, 0.008077902549460, 0.008233438629941, 0.008372608283425,
        0.008866265791057, 0.008256499800489, 0.008302914879219, 0.009460612752251,
        0.009128089741437, 0.009279168633957, 0.005332737116468, 0.010406334248734,
        0.010249150279031, 0.010786550777466, 0.012186139575975, 0.014386444420050
    ])
    C1 = np.array([
        5.409584584248230e+02, 7.181581009417870e+02, 9.974374112660480e+02, 9.526685246154360e+02,
        1.235915935700990e+03, 1.451946136460340e+03, 1.109099044939890e+03, 1.148103647912950e+03,
        1.255404113344680e+03, 1.296209847240250e+03, 1.437193241447520e+03, 1.422460430375690e+03,
        0.024097227883638,      1.292942337654670e+03, 1.275722141773700e+03, 1.194925547740810e+03,
        1.240362206587940e+03, 1.114621235211460e+03, 1.116076559898230e+03, 1.183985227204640e+03,
        1.175416673097070e+03, 1.137201442510130e+03, 0.033193634582519,      1.047977398745050e+03,
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
        2.472276521509160e+04, 2.540695892681510e+04, 5.358353159219460e+03, 2.907973291941980e+04,
        2.209831152237530e+04, 2.315795229955040e+04, 1.768556454942660e+04, 1.247294202184850e+04
    ])
    return np.column_stack([soc_grid, R0, R1, C1, R2, C2])

def get_data_from_csv(path, aliases):
    if not path or not os.path.exists(path): 
        print("Erreur : Chemin invalide."); return None, None, None, None, None
    try:
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
        
        if 'Time' not in col_map or 'Current' not in col_map: 
            print("Erreur: Colonnes Time ou Current manquantes."); return None, None, None, None, None

        def clean_to_num(series):
            return pd.to_numeric(series.astype(str).str.replace(',', '.', regex=False), errors='coerce')

        t_raw = df[col_map['Time']]
        t_numeric = clean_to_num(t_raw)

        if t_numeric.notna().sum() > (len(t_raw) * 0.9):
            t_sec = t_numeric.values
        else:
            t_date = pd.to_datetime(t_raw, errors='coerce')
            t_sec = (t_date - t_date.dropna().iloc[0]).dt.total_seconds().values

        t_sec = t_sec - np.nanmin(t_sec)
        i_val = clean_to_num(df[col_map['Current']]).values
        v_val = clean_to_num(df[col_map['Voltage']]).values if 'Voltage' in col_map else np.full(len(df), np.nan)
        t_bat_val = clean_to_num(df[col_map['Temp_Bat']]).values if 'Temp_Bat' in col_map else np.full(len(df), np.nan)
        
        if 'Temp_Amb' in col_map:
            t_amb_val = clean_to_num(df[col_map['Temp_Amb']]).values
        else:
            first_t = t_bat_val[~np.isnan(t_bat_val)][0] if np.any(~np.isnan(t_bat_val)) else 25.0
            t_amb_val = np.full(len(df), first_t)

        mask = ~np.isnan(t_sec) & ~np.isnan(i_val)
        df_clean = pd.DataFrame({
            't': t_sec[mask], 'i': i_val[mask], 'v': v_val[mask], 
            'Ta': t_amb_val[mask], 'Tb': t_bat_val[mask]
        })
        df_clean = df_clean.sort_values('t').drop_duplicates('t')
        
        return df_clean['t'].values, df_clean['i'].values, df_clean['v'].values, df_clean['Ta'].values, df_clean['Tb'].values

    except Exception as e:
        print(f"EXCEPTION lors de la lecture CSV : {e}")
        return None, None, None, None, None

def calibrate_soc_multipoint(t_csv, I_csv, T_csv, U_exp_csv, 
                             cell_class, cell_init_args, setup_args, 
                             R_dist_val, Ns_pack, Np_pack, R_harness, window_seconds=60.0):
    print(f"\n--- CALIBRATION SOC INITIAL (Recherche Globale) ---")
    t_start = t_csv[0]; t_end = t_start + window_seconds
    t_cal = np.arange(t_start, t_end, 1.0) 
    
    f_I = interp1d(t_csv, I_csv, kind='linear', fill_value="extrapolate")
    f_T = interp1d(t_csv, T_csv, kind='linear', fill_value="extrapolate")
    f_U = interp1d(t_csv, U_exp_csv, kind='linear', fill_value="extrapolate")
    
    I_cal_cell = f_I(t_cal); T_cal = f_T(t_cal); U_target_pack = f_U(t_cal)

    def get_rmse(soc_test):
        init_kwargs = cell_init_args.copy()
        init_kwargs['SOC0'] = soc_test
        temp_cell = cell_class(**init_kwargs)
        temp_cell.thermal.Area = Area_cell
        temp_cell.thermal.R_cond = R_cond_cell
        temp_cell.thermal.Cs_core = (m_cell * 0.90) * 1100
        temp_cell.thermal.Cs_surf = (m_cell * 0.10) * 900
        
        temp_cell.setup_dual_solvers(**setup_args)
        temp_cell.R_internal_current = R_dist_val
        if len(T_cal) > 0: 
            temp_cell.equilibrate_state(I_cal_cell[0], T_cal[0] + 273.15)
            
        error_sq_sum = 0.0
        for k in range(len(t_cal)):
            dt = 1.0
            u_cell, _, _, _, r_app = temp_cell.step(dt, I_cal_cell[k], T_cal[k])
            temp_cell.R_internal_current = r_app + (R_dist_val - R0_cell_nom)
            u_stack = u_cell * Ns_pack
            i_pack = I_cal_cell[k] * Np_pack
            u_pack_sim = u_stack + (i_pack * R_harness) 
            error_sq_sum += (u_pack_sim - U_target_pack[k])**2
        return np.sqrt(error_sq_sum / len(t_cal))

    print("  -> Étape 1 : Balayage large...")
    soc_grid = np.linspace(0.0, 1.0, 21)
    errors = [get_rmse(s) for s in soc_grid]
    best_idx = np.argmin(errors); best_soc_coarse = soc_grid[best_idx]
    
    print(f"  -> Étape 2 : Affinage...")
    # search_min = max(0.0, best_soc_coarse - 0.05) 
    # search_max = min(1.0, best_soc_coarse + 0.05)
    # res = minimize_scalar(get_rmse, bounds=(search_min, search_max), method='bounded', options={'xatol': 1e-5})
    bounds_limit = (0.05, 0.95) 
    res = minimize_scalar(get_rmse, bounds=bounds_limit, method='bounded', options={'xatol': 1e-5})
    
    return res.x if res.success else best_soc_coarse

def ocp_anode(sto, pot, pot_dol):
    val = -np.interp(sto, pot_dol, pot)
    return val

def ocp_cathode(sto, pot, pot_dol):
    return np.interp(sto, pot_dol, pot)

def generate_dummy_data_profile():
    t = np.arange(0, 3600, 1)
    i = np.full_like(t, 20.0) 
    v = np.full_like(t, 48.0)
    ta = np.full_like(t, 25.0)
    tb = np.full_like(t, 25.0)
    return t, i, v, ta, tb

# ==============================================================================
# 4. SIMULATION (MAIN)
# ==============================================================================
if __name__ == "__main__":
    print(f"--- SIMULATION PACK LIAO 100Ah - THERMIQUE 2 ÉTATS - VALIDATION ---")
    
    if DATA_SOURCE_TYPE == "FILE":
        root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
        path_file = filedialog.askopenfilename(title="Fichier Données Expérimentales", filetypes=[("Data", "*.csv *.xlsx *.xls")])
        root.destroy()
        if not path_file:
            print("Aucun fichier. Passage DUMMY.")
            t_csv, I_input_csv, U_pack_exp, T_amb_csv, T_bat_exp = generate_dummy_data_profile()
        else:
            t_csv, I_input_csv, U_pack_exp, T_amb_csv, T_bat_exp = get_data_from_csv(path_file, COLUMN_ALIASES)
            if t_csv is None: sys.exit()
    else:
        t_csv, I_input_csv, U_pack_exp, T_amb_csv, T_bat_exp = generate_dummy_data_profile()
        
    file_an = os.path.join(OCP_PATH, "param_ocpn_graphite.xlsx")
    file_ca = os.path.join(OCP_PATH, "param_ocpn_lfp.xlsx")
    
    # Lecture Anode
    df_an = pd.read_excel(file_an)
    dol_an = df_an.iloc[:, 0].values
    pot_an = df_an.iloc[:, 1].values 
    
    # Lecture Cathode
    df_ca = pd.read_excel(file_ca)
    dol_ca = df_ca.iloc[:, 0].values 
    pot_ca = df_ca.iloc[:, 1].values
    
    anode_optP = [14, 0.9, 2.5]
    cathode_optP = [12, 0.97, 1.9] 
    #ROCV_res_0 = [0.000359, 0.00005]  
    ROCV_res_0 = [0.0018, 0.00005]
    ROCV_res = ROCV_res_0
    scaling_factor = Q_ref_RC / QGes_Ah

    frac_an = anode_optP[0] / (cathode_optP[0] + anode_optP[0])
    frac_ca = cathode_optP[0] / (cathode_optP[0] + anode_optP[0])

    theta_an_base = [-1, frac_an*ROCV_res[0], anode_optP[1]*ROCV_res[1], anode_optP[2]*0.95]
    theta_ca_base = [0,  frac_ca*ROCV_res[0], cathode_optP[1]*ROCV_res[1], cathode_optP[2]*0.92]

    rc_param_0 = build_rc_param_lfp_exact()
    rc_params = rc_param_0.copy()
    rc_params[:, 1] *= scaling_factor * FACTOR_R_DYN         
    rc_params[:, [2, 4]] *= scaling_factor * FACTOR_R_DYN   
    rc_params[:, [3, 5]] /= scaling_factor        

    ocp_an_func = lambda x: ocp_anode(x, pot_an, dol_an)
    ocp_ca_func = lambda x: ocp_cathode(x, pot_ca, dol_ca)

    rng = np.random.default_rng(42)
    Q_dist = rng.normal(QGes_Ah, sigma_Q, (Ns, Np))
    R_dist = np.maximum(rng.normal(R_tot_mean, sigma_R_tot, (Ns, Np)), 0.0001)
    
    T_init_val = T_bat_exp[0] if (len(T_bat_exp) > 0 and not np.isnan(T_bat_exp[0])) else 25.0
    SOC0_CALIBRATED = SOC0_USER

    # --- AUTO-CALIBRATION DU SOC ---
    has_volt_exp = (len(U_pack_exp) > 0 and not np.all(np.isnan(U_pack_exp)))
    if ACTIVATE_SOC_CALIBRATION and has_volt_exp and DATA_SOURCE_TYPE == "FILE":
        print("Lancement Calibration SOC...")
        cell_init_params = {
            'energy_curve_func': None, 'rc_param': None, 'theta': None,
            'r_dis': r_dis_type, 'QGes': Q_dist[0, 0], 
            'SOC0': SOC0_USER, 'n_particle': n_particle, 'm_cell': m_cell,
            'T_init': T_init_val
        }
        setup_params = {
            'ocp_anode_func': ocp_an_func, 'ocp_cathode_func': ocp_ca_func,
            'rc_param': rc_params, 'theta_anode': theta_an_base,
            'theta_cathode': theta_ca_base, 'r_dis': r_dis_type
        }
        I_calib_source = I_input_csv / Np 
        SOC0_CALIBRATED = calibrate_soc_multipoint(
             t_csv, I_calib_source, T_amb_csv, U_pack_exp,
             CellPDECM, cell_init_params, setup_params,
             R_dist[0,0], Ns_pack=Ns, Np_pack=Np, 
             R_harness= R_HARNESS_CHARGE, window_seconds=60.0
         )
    
    # Instanciation
    cells_grid = [] 
    for s in range(Ns):
        row_cells = []
        for p in range(Np):
            cell = CellPDECM(None, None, None, r_dis_type, Q_dist[s, p], SOC0_CALIBRATED, n_particle, m_cell, T_init_val)
            
            cell.thermal.Area = Area_cell
            cell.thermal.R_cond = R_cond_cell
            cell.thermal.R_conv = 1.0 / (H_CONVECTION * Area_cell)
            cell.thermal.Cs_core = (m_cell * 0.90) * 1100 
            cell.thermal.Cs_surf = (m_cell * 0.10) * 900
            
            cell.setup_dual_solvers(ocp_an_func, ocp_ca_func, rc_params, theta_an_base, theta_ca_base, r_dis_type)
            cell.R_internal_current = R_dist[s, p] 
            row_cells.append(cell)
        cells_grid.append(row_cells)

    # Simulation setup
    t_sim = np.arange(t_csv[0], t_csv[-1], FORCED_DT) 
    steps = len(t_sim)
    
    f_interp_I = interp1d(t_csv, I_input_csv, kind='linear', fill_value="extrapolate")
    I_input_sim = f_interp_I(t_sim)
    f_interp_Tamb = interp1d(t_csv, T_amb_csv, kind='linear', fill_value="extrapolate")
    T_amb_sim = f_interp_Tamb(t_sim)
    T_bat_forced_sim = interp1d(t_csv, T_bat_exp, kind='linear', fill_value="extrapolate")(t_sim) if len(T_bat_exp) > 0 else np.full_like(t_sim, np.nan)

    I_cell_init = I_input_sim[0] / Np
    for s in range(Ns):
        for p in range(Np):
            cells_grid[s][p].equilibrate_state(I_cell_init, T_init_val + 273.15)

    # Stockage résultats
    res_T_surf = np.zeros(steps) 
    res_T_core = np.zeros(steps) 
    res_U_pack = np.zeros(steps)
    res_I_applied_pack = np.zeros(steps)
    res_SOC = np.zeros(steps) # <--- Nouveau tableau pour stocker le SOC

    print(f"Démarrage simulation...")
    U_pack_prev = 3.2 * Ns
    I_pack_prev = I_input_sim[0]
    cv_mode_active = False
    charge_finished = False
    I_CUTOFF = 0.04 * QGes_Ah 
    OCV_estim = U_pack_prev
    
    for k in tqdm(range(steps), desc="Simulation", unit="step"):
        dt = FORCED_DT 
        I_target_csv = I_input_sim[k]
        V_TARGET = V_CELL_MAX * Ns
        
        if I_target_csv >= 0: # Charge
            R_active_harness = R_HARNESS_CHARGE
        else: # Décharge
            R_active_harness = R_HARNESS_DISCHARGE
            
        R_avg_internal = np.mean([c.R_internal_current for row in cells_grid for c in row])
        R_base = (R_avg_internal * Ns / Np) + R_active_harness
        
        V_potential = (U_pack_prev / Ns)
        R_dynamic_adder = 0.0
        
        if V_potential > SEUIL_ACTIVATION:
            delta_v = V_potential - SEUIL_ACTIVATION
            R_dynamic_adder = delta_v * facteur_gain
        
        if cv_mode_active and I_pack_prev > 0.05:
                U_base = OCV_estim + (R_base * I_pack_prev)
                missing_v = (V_CELL_MAX * Ns) - U_base
                if missing_v > 0:
                    R_target = missing_v / I_pack_prev
                    R_dynamic_adder = max(R_dynamic_adder, R_target)

        if I_target_csv < 0: R_dynamic_adder = 0.0 
        R_total_effective = R_base
        OCV_estim = U_pack_prev - (I_pack_prev * R_total_effective)

        if I_target_csv <= 0: 
            I_final = I_target_csv
            cv_mode_active = False 
        elif charge_finished:
            I_final = 0.0
        else:
            I_maintain_voltage = (V_TARGET - OCV_estim) / max(R_total_effective, 1e-6)
            V_predicted = OCV_estim + (I_target_csv * R_total_effective)
            
            if cv_mode_active or (V_predicted >= V_TARGET):
                cv_mode_active = True
                I_final = max(0.0, min(I_target_csv, I_maintain_voltage))
                if I_final < I_CUTOFF:
                    charge_finished = True
                    I_final = 0.0
            else:
                cv_mode_active = False
                I_final = I_target_csv

        res_I_applied_pack[k] = I_final
        I_pack_prev = I_final

        I_input_total_corrected = I_final 
        T_air_current = T_amb_sim[k]
        T_forced_val_K = (T_bat_forced_sim[k] + 273.15) if not np.isnan(T_bat_forced_sim[k]) else None
        
        conductances = [[1.0 / max(c.R_internal_current, 1e-6) for c in row] for row in cells_grid]
        I_cells_grid = np.zeros((Ns, Np))
        for s in range(Ns):
            G_row = np.array(conductances[s]); G_sum = np.sum(G_row)
            I_cells_grid[s, :] = I_input_total_corrected * (G_row / G_sum) if G_sum > 0 else I_input_total_corrected/Np

        temps_surf_step, temps_core_step, u_step, soc_step = [], [], [], []
        
        for s in range(Ns):
            Q_dissipated_module = 0.0 
            for p in range(Np):
                cell = cells_grid[s][p]
                U, T_surf, SOC, Q_gen, R_app = cell.step(dt, I_cells_grid[s, p], T_air_current, T_forcing_K=T_forced_val_K)
                
                T_core = cell.thermal.T_core - 273.15
                cell.R_internal_current = R_app + (R_dist[s, p] - R0_cell_nom)
                Q_dissipated_module += Q_gen
                
                temps_surf_step.append(T_surf)
                temps_core_step.append(T_core)
                u_step.append(U)
                soc_step.append(SOC)
                
            T_air_current += max(Q_dissipated_module, 0) * Thermal_Coeff_Fluid
            
        res_T_surf[k] = np.mean(temps_surf_step)
        res_T_core[k] = np.mean(temps_core_step)
        res_SOC[k] = np.mean(soc_step) # Moyenne du SOC du pack
        
        u_mat = np.array(u_step).reshape(Ns, Np)
        raw_U_pack = np.sum(np.mean(u_mat, axis=1))
        res_U_pack[k] = raw_U_pack + (I_input_total_corrected * R_active_harness) + (I_input_total_corrected * R_dynamic_adder)
        U_pack_prev = res_U_pack[k]

    # ==============================================================================
    # 5. COMPARAISON GRAPHIQUE ET CALCUL D'ERREUR (%)
    # ==============================================================================
    
    # Préparation Données Expérimentales (Alignement temporel)
    has_temp_exp = (len(T_bat_exp) > 0 and not np.all(np.isnan(T_bat_exp)))
    has_volt_exp = (len(U_pack_exp) > 0 and not np.all(np.isnan(U_pack_exp)))
    
    U_exp_aligned = interp1d(t_csv, U_pack_exp, bounds_error=False, fill_value=np.nan)(t_sim) if has_volt_exp else None
    T_exp_aligned = interp1d(t_csv, T_bat_exp, bounds_error=False, fill_value=np.nan)(t_sim) if has_temp_exp else None
    
    # --- CALCULS STATISTIQUES (RMSE & %) ---
    
    # 1. Tension
    rmse_v, mape_v = 0.0, 0.0
    if has_volt_exp:
        # Masque pour éviter les NaNs et les divisions par zéro
        mask_v = ~np.isnan(U_exp_aligned) & (np.abs(U_exp_aligned) > 0.1)
        if np.sum(mask_v) > 0:
            diff_v = res_U_pack[mask_v] - U_exp_aligned[mask_v]
            rmse_v = np.sqrt(np.mean(diff_v**2))
            # MAPE (Mean Absolute Percentage Error)
            mape_v = np.mean(np.abs(diff_v / U_exp_aligned[mask_v])) * 100

    # 2. Température
    rmse_t, mape_t = 0.0, 0.0
    if has_temp_exp:
        # Masque (Attention: MAPE en Celsius peut être trompeur si T proche de 0°C)
        mask_t = ~np.isnan(T_exp_aligned) & (np.abs(T_exp_aligned) > 1.0)
        if np.sum(mask_t) > 0:
            diff_t = res_T_surf[mask_t] - T_exp_aligned[mask_t]
            rmse_t = np.sqrt(np.mean(diff_t**2))
            mape_t = np.mean(np.abs(diff_t / T_exp_aligned[mask_t])) * 100

    # --- AFFICHAGE CONSOLE ---
    print("-" * 50)
    print(f"RÉSULTATS DE VALIDATION (SIMULATION vs RÉALITÉ)")
    print("-" * 50)
    
    # --- AJOUT : AFFICHAGE DU SOC INITIAL ---
    print(f"SOC INITIAL UTILISÉ : {SOC0_CALIBRATED*100:.2f} %")
    if ACTIVATE_SOC_CALIBRATION:
        print(f" (Valeur calibrée automatiquement)")
    else:
        print(f" (Valeur imposée par l'utilisateur)")
    print("-" * 30)
    
    if has_volt_exp: 
        print(f"TENSION Pack :")
        print(f"  -> RMSE (Erreur Moyenne) : {rmse_v:.4f} V")
        print(f"  -> MAPE (Erreur Relative): {mape_v:.3f} %")
    else:
        print("TENSION : Pas de données expérimentales pour comparer.")
        
    print("-" * 30)
    
    if has_temp_exp: 
        print(f"TEMPÉRATURE Surface :")
        print(f"  -> RMSE (Erreur Moyenne) : {rmse_t:.4f} °C")
        print(f"  -> MAPE (Erreur Relative): {mape_t:.3f} %")
    else:
        print("TEMPÉRATURE : Pas de données expérimentales pour comparer.")
    print("-" * 50)

    # --- PLOTS ---
    plt.figure(figsize=(12, 12)) # Agrandissement pour 4 subplots
    
    # Graphique 1 : Tension
    plt.subplot(4, 1, 1)
    plt.plot(t_sim, res_U_pack, color='purple', linewidth=2, label='Simulé')
    if has_volt_exp:
        plt.plot(t_sim, U_exp_aligned, color='green', linestyle='--', alpha=0.7, label='Expérimental')
        plt.title(f"Tension Pack (Err: {mape_v:.2f}%)")
    else:
        plt.title("Tension Pack")
    plt.ylabel("Tension [V]"); plt.legend(); plt.grid(True)

    # Graphique 2 : Courant
    plt.subplot(4, 1, 2)
    plt.plot(t_sim, I_input_sim, color='gray', linestyle=':', label='Consigne Fichier')
    plt.plot(t_sim, res_I_applied_pack, color='blue', linewidth=1.5, label='Réel Simulé')
    plt.ylabel("Courant [A]"); plt.legend(); plt.grid(True)
    plt.title("Courant (Limitation CV active)")
    
    # Graphique 3 : Température (Comparaison Surface)
    plt.subplot(4, 1, 3)
    plt.plot(t_sim, res_T_core, color='red', linestyle='-.', linewidth=1.5, label='Simu T_Cœur (Interne)')
    plt.plot(t_sim, res_T_surf, color='orange', linewidth=2, label='Simu T_Surface')
    
    if has_temp_exp:
        plt.plot(t_sim, T_exp_aligned, color='green', linestyle='--', alpha=0.7, label='Exp T_Capteur')
        plt.title(f"Thermique (Err Surface: {mape_t:.2f}%)")
    else:
        plt.title(f"Thermique 2-États (R_cond={R_cond_cell})")
    
    plt.ylabel("Temp. [°C]"); plt.legend(); plt.grid(True)

    # Graphique 4 : SOC (NOUVEAU)
    plt.subplot(4, 1, 4)
    plt.plot(t_sim, res_SOC * 100, color='blue', linewidth=2, label='SOC Moyen')
    plt.ylabel("SOC [%]"); plt.xlabel("Temps [s]"); plt.legend(); plt.grid(True)
    plt.title("État de Charge (SOC)")
    plt.ylim(0, 105) # Pour bien voir la fin de charge

    plt.tight_layout()
    plt.show()