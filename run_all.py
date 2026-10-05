import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
from sklearn.ensemble import ExtraTreesRegressor
import lightgbm as lgb

FILE_NAME = "Volve production data.xlsx"

if not os.path.exists(FILE_NAME):
    raise FileNotFoundError(f"Could not find '{FILE_NAME}' in {os.getcwd()}. Place the Excel file in this folder!")

print("--- Step 1: Loading & Inspecting Volve Dataset ---")
excel_obj = pd.ExcelFile(FILE_NAME)
sheet = "Daily Production Data" if "Daily Production Data" in excel_obj.sheet_names else excel_obj.sheet_names[0]
df_raw = pd.read_excel(FILE_NAME, sheet_name=sheet)
print(f"Loaded '{sheet}' successfully. Total rows: {len(df_raw)}")

dp_col = 'DP_CHOKE_SIZE' if 'DP_CHOKE_SIZE' in df_raw.columns else ('AVG_DP_CHOKE_P' if 'AVG_DP_CHOKE_P' in df_raw.columns else 'AVG_DP_TUBING')

df_raw['CLEAN_WELL_NAME'] = df_raw['NPD_WELL_BORE_NAME'].astype(str).str.replace('NO ', '').str.strip()
target_wells = ['15/9-F-14', '15/9-F-12', '15/9-F-11', '15/9-F-1 C']
df = df_raw[df_raw['CLEAN_WELL_NAME'].isin(target_wells)].copy()

df = df[
    (df['ON_STREAM_HRS'] >= 15) & 
    (df['AVG_CHOKE_SIZE_P'] > 0) & 
    (df['BORE_OIL_VOL'] > 0) & 
    (df['AVG_WHP_P'] > 0)
].copy()

df['TOTAL_LIQ_VOL'] = df['BORE_OIL_VOL'] + df['BORE_WAT_VOL']

df['CHOKE_64TH'] = (df['AVG_CHOKE_SIZE_P'] / 100.0) * 128.0
df['P_UP_BAR'] = df['AVG_WHP_P']
df['DP_BAR'] = df[dp_col].fillna(df['AVG_WHP_P'] * 0.3)
df['P_DN_BAR'] = np.maximum(df['P_UP_BAR'] - df['DP_BAR'], 1.0)
df['TEMP_C'] = df['AVG_WHT_P'] if 'AVG_WHT_P' in df.columns else df['AVG_DOWNHOLE_TEMPERATURE']
df['GLR'] = df['BORE_GAS_VOL'] / np.maximum(df['TOTAL_LIQ_VOL'], 1e-4)

df['PRESSURE_RATIO'] = df['P_DN_BAR'] / df['P_UP_BAR']
water_cut = df['BORE_WAT_VOL'] / df['TOTAL_LIQ_VOL']
rho_liq = (1.0 - water_cut) * 830.0 + (water_cut * 1000.0)
df['RHO_MIX'] = rho_liq / (1.0 + (df['GLR'] * 0.005))
df['Q_HYDRO'] = (df['CHOKE_64TH']**2) * np.sqrt(np.maximum(df['DP_BAR'] * 1e5, 0) / np.maximum(df['RHO_MIX'], 10.0))

for col in ['TOTAL_LIQ_VOL', 'P_UP_BAR', 'CHOKE_64TH', 'TEMP_C']:
    m, s = df[col].mean(), df[col].std()
    df = df[(df[col] >= m - 3*s) & (df[col] <= m + 3*s)]

print(f"Cleaned dataset rows: {len(df)}")
print("Well counts:\n", df['CLEAN_WELL_NAME'].value_counts())

print("\n--- Step 2: Running Stanford CS229 Baselines ---")
def gilbert_pred(p_up_psi, choke_64th, glr_scf):
    glr_safe = np.maximum(glr_scf, 1.0)
    q_bpd = (p_up_psi * (choke_64th**1.89)) / (10.0 * (glr_safe**0.546))
    return q_bpd * 0.158987

p_psi = df['P_UP_BAR'] * 14.5038
glr_scf = df['GLR'] * 5.615
df['GILBERT_PRED'] = gilbert_pred(p_psi, df['CHOKE_64TH'], glr_scf)

print(f"[Gilbert Correlation] R2: {r2_score(df['TOTAL_LIQ_VOL'], df['GILBERT_PRED']):.3f} | MAPE: {mean_absolute_percentage_error(df['TOTAL_LIQ_VOL'], df['GILBERT_PRED']) * 100:.2f}%")

raw_features = ['P_UP_BAR', 'DP_BAR', 'CHOKE_64TH', 'TEMP_C', 'GLR']
X, y = df[raw_features], df['TOTAL_LIQ_VOL']
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

et_random = ExtraTreesRegressor(n_estimators=100, random_state=42)
et_random.fit(X_tr, y_tr)
print(f"[Stanford Extra Trees 80/20] Train R2: {r2_score(y_tr, et_random.predict(X_tr)):.3f} | Test R2: {r2_score(y_te, et_random.predict(X_te)):.3f}")

print("\n--- Step 3: Cross-Well Generalization Test (Train: F-14 -> Test: F-12) ---")
novel_features = ['P_UP_BAR', 'DP_BAR', 'CHOKE_64TH', 'TEMP_C', 'GLR', 'PRESSURE_RATIO', 'Q_HYDRO']
monotone_constraints = [0, 1, 1, 0, 0, 0, 1]

train_mask = df['CLEAN_WELL_NAME'] == '15/9-F-14'
test_mask = df['CLEAN_WELL_NAME'] == '15/9-F-12'

X_tr_cw, y_tr_cw = df.loc[train_mask, novel_features], df.loc[train_mask, 'TOTAL_LIQ_VOL']
X_te_cw, y_te_cw = df.loc[test_mask, novel_features], df.loc[test_mask, 'TOTAL_LIQ_VOL']

et_cross = ExtraTreesRegressor(n_estimators=100, random_state=42)
et_cross.fit(X_tr_cw[raw_features], y_tr_cw)
pred_et_cw = et_cross.predict(X_te_cw[raw_features])

lgb_model = lgb.LGBMRegressor(
    n_estimators=150, 
    learning_rate=0.05, 
    max_depth=5, 
    monotone_constraints=monotone_constraints, 
    random_state=42, 
    verbose=-1
)
lgb_model.fit(X_tr_cw, y_tr_cw)
pred_lgb_cw = lgb_model.predict(X_te_cw)

print(f"Stanford Extra Trees Transfer R2: {r2_score(y_te_cw, pred_et_cw):.3f} | MAPE: {mean_absolute_percentage_error(y_te_cw, pred_et_cw)*100:.2f}%")
print(f"Novel Physics-Constrained LGBM R2: {r2_score(y_te_cw, pred_lgb_cw):.3f} | MAPE: {mean_absolute_percentage_error(y_te_cw, pred_lgb_cw)*100:.2f}%")

print("\n--- Step 4: Physical Monotonicity Sweep Audit ---")
test_chokes = np.linspace(10, 100, 100)
synthetic_profile = pd.DataFrame({
    'P_UP_BAR': [100.0] * 100,
    'DP_BAR': [40.0] * 100,
    'CHOKE_64TH': test_chokes,
    'TEMP_C': [70.0] * 100,
    'GLR': [150.0] * 100,
    'PRESSURE_RATIO': [0.6] * 100,
    'Q_HYDRO': (test_chokes**2) * np.sqrt(40.0 * 1e5 / 750.0)
})

et_syn = et_cross.predict(synthetic_profile[raw_features])
lgb_syn = lgb_model.predict(synthetic_profile)

print(f"Physical Monotonicity Inversions (Stanford Extra Trees): {np.sum(np.diff(et_syn) < 0)} violations")
print(f"Physical Monotonicity Inversions (Novel Physics Model): {np.sum(np.diff(lgb_syn) < 0)} violations")
