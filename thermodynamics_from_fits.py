import pandas as pd
import numpy as np

# params_path = "citations_fit_params.csv"
params_path = "social_media_params.csv"
params_df = pd.read_csv(params_path)

def safe_integral(alpha, beta, sigma, func=None, w_max=2000, dw=0.1):
    """Safely integrate w^alpha exp(-beta*(sigma*exp(-w/sigma) + w)) * func(w)."""
    w_vals = np.arange(dw, w_max, dw)
    logf = alpha * np.log(w_vals) - beta * (sigma * np.exp(-w_vals/sigma) + w_vals)

    # Scaling to prevent overflow/underflow
    logf_max = np.max(logf)
    f_scaled = np.exp(logf - logf_max)

    if func is not None:
        f_scaled *= func(w_vals)

    integral_scaled = np.trapezoid(f_scaled, w_vals)
    return integral_scaled * np.exp(logf_max)

def thermodynamic_quantities(alpha, beta, sigma, w_max=9000, dw=0.01):
    w_vals = np.arange(dw, w_max, dw)
    pdf_log = alpha * np.log(w_vals) - beta * (sigma * np.exp(-w_vals/sigma) + w_vals)
    pdf_log -= np.max(pdf_log)
    pdf = np.exp(pdf_log)
    pdf /= np.trapezoid(pdf, w_vals)  # normalize

    # Partition function etc.
    Z = safe_integral(alpha, beta, sigma, None, w_max, dw)
    if Z <= 0 or np.isnan(Z) or np.isinf(Z):
        return [np.nan]*9  # keep outputs consistent

    E_mean = safe_integral(alpha, beta, sigma,
                           lambda w: w + sigma * np.exp(-w/sigma),
                           w_max, dw) / Z
    lnw_mean = safe_integral(alpha, beta, sigma,
                             lambda w: np.log(w),
                             w_max, dw) / Z
    E2_mean = safe_integral(alpha, beta, sigma,
                            lambda w: (w + sigma * np.exp(-w/sigma))**2,
                            w_max, dw) / Z
    lnw2_mean = safe_integral(alpha, beta, sigma,
                              lambda w: (np.log(w))**2,
                              w_max, dw) / Z

    varE = E2_mean - E_mean**2
    varlnw = lnw2_mean - lnw_mean**2

    # Thermodynamic quantities
    S = np.log(Z) + beta * E_mean - alpha * lnw_mean
    C = (beta**2) * varE
    chi_alpha = varlnw

    # Gompertz contribution
    num = safe_integral(alpha, beta, sigma,
                        lambda w: sigma * np.exp(-w/sigma),
                        w_max, dw)
    den = safe_integral(alpha, beta, sigma,
                        lambda w: w + sigma * np.exp(-w/sigma),
                        w_max, dw)
    gompertz_index = num / den if den > 0 else np.nan

    # --- Bowley’s quantile skewness ---
    cdf = np.cumsum(pdf) * (w_vals[1]-w_vals[0])
    def q(p):
        return np.interp(p, cdf, w_vals)

    Q1, Q2, Q3 = q(0.25), q(0.5), q(0.75)
    bowley_skew = ((Q3 + Q1 - 2*Q2) / (Q3 - Q1)) if (Q3>Q1) else np.nan

    # --- FWHM asymmetry ---
    peak_idx = np.argmax(pdf)
    peak_val = pdf[peak_idx]
    half_max = peak_val / 2.0

    # --- left half-width (interpolated) ---
    left_idx = np.where(pdf[:peak_idx] <= half_max)[0]
    if len(left_idx) > 0:
        i = left_idx[-1]
        W_L = w_vals[peak_idx] - np.interp(
            half_max, [pdf[i], pdf[i+1]], [w_vals[i], w_vals[i+1]]
        )
    else:
        W_L = w_vals[peak_idx]  # fallback: distance to w=0

    # --- right half-width (interpolated) ---
    right_idx = np.where(pdf[peak_idx:] <= half_max)[0]
    if len(right_idx) > 0:
        i = right_idx[0] + peak_idx  # first index below half_max after peak
        W_R = np.interp(
            half_max,
            [pdf[i-1], pdf[i]],
            [w_vals[i-1], w_vals[i]]
        ) - w_vals[peak_idx]
    else:
        W_R = np.nan

    # asymmetry ratio
    fwhm_asym = (W_R / W_L) if (W_L > 0 and not np.isnan(W_L) and not np.isnan(W_R)) else np.nan

    return Z, E_mean, lnw_mean, S, C, chi_alpha, gompertz_index, bowley_skew, fwhm_asym


# Process all datasets
results = []
for _, row in params_df.iterrows():
    wmax  = row["wmax"]
    name  = row["series"]
    M0    = abs(row["Mo"])
    beta  = abs(row["b"] )
    alpha = abs(row["a"] )
    sigma = abs(row["s"] )
       
    Z, E_mean, lnw_mean, S, C, chi_alpha, gompertz_index, bowley_skew, fwhm_asym = thermodynamic_quantities(alpha, beta, sigma, w_max=wmax)
    T = 1.0 / beta if beta > 0 else np.nan

    results.append([
        name, M0 ,alpha, beta, sigma, T, Z, E_mean, lnw_mean, S, 
        C, chi_alpha, gompertz_index, bowley_skew, fwhm_asym
    ])

results_df = pd.DataFrame(results, columns=[
    "series", "M0", "alpha", "beta", "sigma", "T", "Z", 
    "E_mean", "lnw_mean", "S", "C (Heat-Capacity)", 
    "chi_alpha (susceptibility)", "gompertz_index (thermo)",
    "bowley_skewness", "fwhm_asymmetry"
])

# results_df.to_csv("citation_thermo_results.csv", index=False)
results_df.to_csv("social_thermo_results.csv", index=False)
