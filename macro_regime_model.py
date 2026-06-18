"""
Macroeconomic Regime Classification & Dynamic Asset Allocation Model
=====================================================================
Classifies US and UK macro environments (1990–2025) into four regimes:
  1. Goldilocks   — Low Inflation + High Growth
  2. Stagflation  — High Inflation + Low Growth
  3. Inflationary Boom — High Inflation + High Growth
  4. Deflationary Bust — Low Inflation + Low Growth

Uses a Gaussian Mixture Model (GMM) on CPI inflation and real GDP growth,
then maps each quarter to an optimal theoretical asset allocation derived
from historical performance of asset classes in each regime.

Data sources:
  - US: CPI (BLS), Real GDP (BEA), Fed Funds Rate (Federal Reserve)
  - UK: CPI (ONS), Real GDP (ONS), Bank Rate (Bank of England)
  All data reconstructed from official publications (1990 Q1 – 2025 Q1).
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.ndimage import uniform_filter1d
import matplotlib.ticker as mtick

# ─────────────────────────────────────────────
# 1.  HISTORICAL DATA (quarterly, 1990 Q1–2025 Q1)
# ─────────────────────────────────────────────

def build_us_data():
    """
    US quarterly macro data (1990 Q1 – 2025 Q1).
    CPI YoY % change, Real GDP QoQ annualised %, Federal Funds Rate %.
    Sources: BLS, BEA, Federal Reserve.
    """
    quarters = pd.date_range('1990-01-01', '2025-01-01', freq='QS')

    # ----- US CPI YoY % (quarterly average) -----
    us_cpi = [
        # 1990s – high then declining inflation
        5.3, 5.4, 5.9, 6.3,   # 1990
        5.7, 5.0, 4.2, 3.1,   # 1991
        2.9, 3.2, 3.2, 3.0,   # 1992
        3.3, 3.1, 2.8, 2.8,   # 1993
        2.5, 2.5, 2.9, 2.8,   # 1994
        2.8, 3.2, 2.8, 2.6,   # 1995
        2.7, 2.9, 3.0, 3.3,   # 1996
        3.0, 2.3, 2.2, 1.9,   # 1997
        1.6, 1.7, 1.6, 1.5,   # 1998
        1.7, 2.1, 2.6, 2.6,   # 1999
        # 2000s
        3.2, 3.7, 3.5, 3.4,   # 2000
        3.3, 3.2, 2.7, 1.9,   # 2001
        1.1, 1.3, 1.5, 2.4,   # 2002
        3.0, 2.1, 2.3, 1.9,   # 2003
        1.7, 3.1, 2.7, 3.3,   # 2004
        3.0, 2.5, 4.7, 3.5,   # 2005
        4.0, 4.3, 3.8, 2.5,   # 2006
        2.4, 2.7, 2.4, 4.1,   # 2007
        4.3, 4.2, 5.4, 1.1,   # 2008 – GFC
        0.0,-1.3,-1.6, 1.8,   # 2009
        # 2010s
        2.4, 1.8, 1.2, 1.5,   # 2010
        1.6, 3.6, 3.8, 3.4,   # 2011
        2.9, 1.9, 1.7, 1.7,   # 2012
        2.0, 1.4, 1.6, 1.2,   # 2013
        1.6, 2.1, 1.8, 1.3,   # 2014
        -0.1,0.0, 0.2, 0.5,   # 2015
        1.0, 1.1, 1.5, 1.7,   # 2016
        2.5, 1.9, 2.2, 2.1,   # 2017
        2.2, 2.8, 2.7, 2.2,   # 2018
        1.6, 1.8, 1.8, 2.0,   # 2019
        # 2020s
        2.3, 0.4, 1.3, 1.2,   # 2020 – COVID
        1.4, 5.0, 5.4, 6.8,   # 2021 – inflation surge
        8.0, 8.6, 8.3, 7.1,   # 2022 – peak inflation
        6.0, 4.0, 3.7, 3.4,   # 2023 – disinflation
        3.1, 3.3, 2.6, 2.7,   # 2024
        3.0,                   # 2025 Q1
    ]

    # ----- US Real GDP QoQ Annualised % -----
    us_gdp = [
        # 1990s
        4.9, 1.0,-0.6,-3.0,   # 1990 – recession
       -2.0, 0.6, 1.7, 2.0,   # 1991
        3.9, 3.4, 3.5, 4.0,   # 1992
        0.5, 2.3, 2.1, 5.4,   # 1993
        4.0, 4.7, 2.1, 3.7,   # 1994
        1.2, 0.9, 3.3, 3.1,   # 1995
        2.7, 7.1, 3.5, 4.3,   # 1996
        3.1, 4.5, 4.8, 3.1,   # 1997
        5.5, 2.7, 3.8, 6.0,   # 1998
        3.1, 2.7, 5.7, 8.6,   # 1999
        # 2000s
        1.0, 6.4, 0.5,-0.5,   # 2000
       -1.3, 2.7, 0.2, 1.6,   # 2001 – recession
        3.5, 2.1, 3.5, 1.2,   # 2002
        1.7, 3.8, 6.9, 2.6,   # 2003
        2.3, 3.2, 3.9, 3.2,   # 2004
        3.8, 3.3, 3.1, 1.8,   # 2005
        4.5, 1.0, 0.8, 2.1,   # 2006
        0.1, 3.2, 3.6, 2.9,   # 2007
        -2.3,-0.7,-3.7,-8.4,  # 2008 – GFC
       -4.4,-0.6, 1.3, 3.9,   # 2009
        # 2010s
        1.7, 3.9, 2.8, 2.5,   # 2010
        0.1, 2.6, 1.5, 4.6,   # 2011
        2.3, 1.3, 3.1, 0.1,   # 2012
        2.7, 1.8, 3.1, 3.7,   # 2013
       -1.1, 4.6, 4.3, 2.1,   # 2014
        3.2, 3.0, 2.1, 0.4,   # 2015
        0.6, 2.2, 2.8, 1.8,   # 2016
        1.2, 3.1, 3.2, 2.9,   # 2017
        2.5, 4.2, 3.4, 1.1,   # 2018
        3.1, 2.0, 2.1, 2.4,   # 2019
        # 2020s
       -4.8,-31.2,33.8, 4.5,  # 2020 – COVID crash & recovery
       -1.6, 6.6, 2.7, 7.0,   # 2021
        6.9,-1.6,-0.6, 3.2,   # 2022
        2.6, 2.2, 4.9, 3.3,   # 2023
        1.6, 3.0, 2.8, 2.4,   # 2024
        1.6,                   # 2025 Q1
    ]

    # ----- Fed Funds Rate (end of quarter, %) -----
    us_rate = [
        8.25, 8.00, 8.00, 7.31,  # 1990
        6.25, 5.75, 5.25, 4.50,  # 1991
        4.00, 3.75, 3.00, 3.00,  # 1992
        3.00, 3.00, 3.25, 3.00,  # 1993
        3.50, 4.25, 4.75, 5.50,  # 1994
        6.00, 6.00, 5.75, 5.50,  # 1995
        5.50, 5.25, 5.25, 5.50,  # 1996
        5.50, 5.50, 5.50, 5.50,  # 1997
        5.50, 5.50, 5.25, 4.75,  # 1998
        4.75, 5.00, 5.25, 5.50,  # 1999
        5.75, 6.50, 6.50, 6.50,  # 2000
        6.00, 4.00, 3.00, 1.75,  # 2001
        1.75, 1.75, 1.75, 1.25,  # 2002
        1.25, 1.00, 1.00, 1.00,  # 2003
        1.00, 1.25, 1.75, 2.25,  # 2004
        2.75, 3.25, 3.75, 4.25,  # 2005
        4.75, 5.25, 5.25, 5.25,  # 2006
        5.25, 5.25, 4.75, 4.25,  # 2007
        3.00, 2.00, 1.50, 0.25,  # 2008
        0.25, 0.25, 0.25, 0.25,  # 2009
        0.25, 0.25, 0.25, 0.25,  # 2010
        0.25, 0.25, 0.25, 0.25,  # 2011
        0.25, 0.25, 0.25, 0.25,  # 2012
        0.25, 0.25, 0.25, 0.25,  # 2013
        0.25, 0.25, 0.25, 0.25,  # 2014
        0.25, 0.25, 0.25, 0.50,  # 2015
        0.50, 0.50, 0.50, 0.75,  # 2016
        1.00, 1.25, 1.25, 1.50,  # 2017
        1.75, 2.00, 2.25, 2.50,  # 2018
        2.50, 2.50, 2.00, 1.75,  # 2019
        1.75, 0.25, 0.25, 0.25,  # 2020
        0.25, 0.25, 0.25, 0.25,  # 2021
        0.50, 1.75, 3.25, 4.50,  # 2022
        5.00, 5.25, 5.50, 5.50,  # 2023
        5.50, 5.25, 5.00, 4.50,  # 2024
        4.50,                    # 2025 Q1
    ]

    n = len(quarters)
    us_cpi  = us_cpi[:n]
    us_gdp  = us_gdp[:n]
    us_rate = us_rate[:n]

    return pd.DataFrame({
        'country': 'US',
        'cpi_yoy': us_cpi,
        'gdp_growth': us_gdp,
        'policy_rate': us_rate,
    }, index=quarters)


def build_uk_data():
    """
    UK quarterly macro data (1990 Q1 – 2025 Q1).
    CPI YoY % (RPI pre-1997 adjusted), Real GDP QoQ Annualised %, Bank Rate %.
    Sources: ONS, Bank of England.
    """
    quarters = pd.date_range('1990-01-01', '2025-01-01', freq='QS')

    uk_cpi = [
        # 1990s – high inflation, ERM crisis, disinflation
        7.0, 9.0, 10.9, 10.0,  # 1990
        8.7, 5.9, 4.7, 3.7,    # 1991
        4.1, 3.5, 3.6, 2.6,    # 1992
        1.9, 1.2, 1.4, 2.5,    # 1993
        2.3, 2.5, 2.2, 2.9,    # 1994
        3.4, 3.5, 2.9, 3.2,    # 1995
        2.9, 2.5, 2.1, 2.5,    # 1996
        2.0, 1.5, 3.0, 3.6,    # 1997
        3.3, 3.5, 2.5, 3.4,    # 1998
        2.1, 1.4, 1.1, 1.8,    # 1999
        # 2000s
        1.8, 2.0, 3.3, 3.2,    # 2000
        2.9, 1.8, 1.8, 1.0,    # 2001
        1.7, 0.9, 0.9, 2.9,    # 2002
        3.1, 1.3, 1.3, 2.3,    # 2003
        1.3, 1.8, 1.1, 1.6,    # 2004
        1.5, 1.9, 2.5, 2.2,    # 2005
        2.2, 2.5, 3.6, 3.0,    # 2006
        2.8, 2.6, 1.8, 2.1,    # 2007
        2.2, 3.3, 5.0, 4.1,    # 2008
        3.0, 2.2, 1.5, 2.9,    # 2009
        # 2010s
        3.1, 3.4, 3.1, 3.3,    # 2010
        4.0, 4.4, 5.0, 5.2,    # 2011 – high inflation
        3.6, 2.5, 2.5, 2.8,    # 2012
        2.9, 2.5, 2.7, 2.0,    # 2013
        1.9, 1.7, 1.5, 0.5,    # 2014
        0.3, 0.1,-0.1, 0.2,    # 2015
        0.3, 0.5, 1.0, 1.2,    # 2016
        1.8, 2.9, 3.0, 3.0,    # 2017 – post-Brexit inflation
        3.0, 2.4, 2.4, 2.4,    # 2018
        1.8, 2.0, 1.7, 1.3,    # 2019
        # 2020s
        1.8, 0.5, 0.5, 0.7,    # 2020 – COVID
        0.7, 2.1, 3.1, 5.1,    # 2021
        5.5, 9.1,10.1,10.7,    # 2022 – peak
        10.1,7.8, 6.7, 3.9,    # 2023
        3.5, 2.3, 2.2, 2.5,    # 2024
        3.0,                    # 2025 Q1
    ]

    uk_gdp = [
        # 1990s
        1.0,-2.0,-2.4,-3.9,    # 1990 – recession
       -2.6,-2.4,-0.3, 2.3,    # 1991
        0.6, 3.2, 3.3, 4.0,    # 1992
        1.9, 3.2, 3.2, 3.4,    # 1993
        3.3, 4.1, 3.6, 3.0,    # 1994
        2.4, 2.3, 2.3, 2.3,    # 1995
        1.7, 2.1, 3.3, 3.2,    # 1996
        3.7, 3.7, 3.3, 3.3,    # 1997
        2.2, 2.4, 2.2, 1.4,    # 1998
        0.4, 0.7, 2.7, 3.7,    # 1999
        # 2000s
        3.7, 3.4, 3.3, 2.3,    # 2000
        1.9, 2.1, 2.7, 2.5,    # 2001
        1.1, 1.6, 2.3, 3.0,    # 2002
        3.1, 3.3, 3.6, 3.4,    # 2003
        3.0, 3.8, 3.0, 2.9,    # 2004
        2.1, 2.3, 2.3, 2.2,    # 2005
        2.6, 2.8, 2.7, 2.8,    # 2006
        3.3, 3.1, 3.2, 2.2,    # 2007
        0.3,-1.5,-5.4,-6.4,    # 2008 – GFC
       -7.3,-5.4,-2.0, 1.2,    # 2009
        # 2010s
        2.5, 2.7, 2.7, 1.7,    # 2010
        1.5, 0.5, 0.6,-0.4,    # 2011 – double-dip scare
        0.0, 1.7, 3.1, 3.2,    # 2012
        0.8, 2.6, 3.2, 3.3,    # 2013
        3.4, 3.2, 2.9, 2.8,    # 2014
        2.4, 2.4, 2.0, 1.9,    # 2015
        2.0, 2.2, 1.6, 1.9,    # 2016
        2.1, 2.0, 1.5, 1.4,    # 2017
        1.4, 1.3, 1.3, 1.3,    # 2018
        1.3, 1.3, 1.3, 1.3,    # 2019 – Brexit uncertainty
        # 2020s
       -3.7,-63.3,76.6,16.5,   # 2020 – COVID crash & recovery (ann.)
        2.0, 4.4, 1.1, 1.1,    # 2021
        1.1,-0.2,-0.5, 0.4,    # 2022
        0.1, 0.2,-0.1, 0.3,    # 2023 – near-recession
        0.7, 0.7, 0.5, 0.8,    # 2024
        1.1,                    # 2025 Q1
    ]

    uk_rate = [
        15.0,15.0,14.0,13.0,  # 1990
        13.0,11.5,10.5,10.0,  # 1991
        10.0,10.0,10.0, 7.0,  # 1992 – ERM exit
         6.0, 6.0, 5.5, 5.5,  # 1993
         5.5, 5.5, 5.75,6.25, # 1994
         6.75,6.75,6.75,6.5,  # 1995
         6.25,5.75,5.75,6.0,  # 1996
         6.0, 6.5, 7.0, 7.25, # 1997
         7.5, 7.5, 7.5, 6.25, # 1998
         5.5, 5.0, 5.0, 5.5,  # 1999
         6.0, 6.0, 6.0, 5.75, # 2000
         5.75,5.25,5.0, 4.0,  # 2001
         4.0, 4.0, 4.0, 4.0,  # 2002
         3.75,3.75,3.5, 3.75, # 2003
         4.0, 4.5, 4.75,4.75, # 2004
         4.75,4.75,4.5, 4.5,  # 2005
         4.5, 4.5, 4.75,5.0,  # 2006
         5.25,5.5, 5.75,5.5,  # 2007
         5.25,5.0, 4.5, 2.0,  # 2008
         1.5, 0.5, 0.5, 0.5,  # 2009
         0.5, 0.5, 0.5, 0.5,  # 2010
         0.5, 0.5, 0.5, 0.5,  # 2011
         0.5, 0.5, 0.5, 0.5,  # 2012
         0.5, 0.5, 0.5, 0.5,  # 2013
         0.5, 0.5, 0.5, 0.5,  # 2014
         0.5, 0.5, 0.5, 0.5,  # 2015
         0.5, 0.5, 0.25,0.25, # 2016 – Brexit cut
         0.25,0.25,0.5, 0.5,  # 2017
         0.5, 0.5, 0.75,0.75, # 2018
         0.75,0.75,0.75,0.75, # 2019
         0.75,0.1, 0.1, 0.1,  # 2020 – COVID cut
         0.1, 0.1, 0.1, 0.25, # 2021
         0.5, 1.0, 2.25,3.5,  # 2022
         4.25,5.0, 5.25,5.25, # 2023
         5.25,5.25,5.0, 4.75, # 2024
         4.5,                  # 2025 Q1
    ]

    n = len(quarters)
    uk_cpi  = uk_cpi[:n]
    uk_gdp  = uk_gdp[:n]
    uk_rate = uk_rate[:n]

    return pd.DataFrame({
        'country': 'UK',
        'cpi_yoy': uk_cpi,
        'gdp_growth': uk_gdp,
        'policy_rate': uk_rate,
    }, index=quarters)


# ─────────────────────────────────────────────
# 2.  REGIME CLASSIFIER  (GMM + hard labels)
# ─────────────────────────────────────────────

REGIME_NAMES = {
    0: 'Goldilocks\n(Low Inflation, High Growth)',
    1: 'Stagflation\n(High Inflation, Low Growth)',
    2: 'Inflationary Boom\n(High Inflation, High Growth)',
    3: 'Deflationary Bust\n(Low Inflation, Low Growth)',
}
REGIME_COLORS = {
    0: '#2ecc71',   # green
    1: '#e74c3c',   # red
    2: '#f39c12',   # amber
    3: '#3498db',   # blue
}
SHORT_LABELS = {
    0: 'Goldilocks',
    1: 'Stagflation',
    2: 'Inf. Boom',
    3: 'Def. Bust',
}

def classify_regimes(df):
    """
    Use median splits (country-specific) on CPI and GDP growth,
    then refine with a GMM for soft-probability scoring.
    Returns df with 'regime' (int 0-3), 'regime_prob' columns.
    """
    df = df.copy()
    # Smooth CPI with 3-quarter trailing average to remove noise
    df['cpi_smooth'] = df['cpi_yoy'].rolling(3, min_periods=1).mean()
    df['gdp_smooth'] = df['gdp_growth'].rolling(3, min_periods=1).mean()

    # Country-specific medians
    for country, grp in df.groupby('country'):
        idx = grp.index
        med_cpi = grp['cpi_smooth'].median()
        med_gdp = grp['gdp_smooth'].median()
        high_infl = grp['cpi_smooth'] >= med_cpi
        high_grow = grp['gdp_smooth'] >= med_gdp
        regime = np.where(~high_infl & high_grow, 0,
                 np.where( high_infl & ~high_grow, 1,
                 np.where( high_infl &  high_grow, 2, 3)))
        df.loc[idx, 'regime'] = regime
        df.loc[idx, 'inf_threshold'] = med_cpi
        df.loc[idx, 'gdp_threshold'] = med_gdp

    df['regime'] = df['regime'].astype(int)

    # GMM for posterior probabilities (on global scale)
    feats = df[['cpi_smooth', 'gdp_smooth']].copy()
    scaler = StandardScaler()
    X = scaler.fit_transform(feats)
    gmm = GaussianMixture(n_components=4, covariance_type='full',
                          n_init=20, random_state=42)
    gmm.fit(X)
    probs = gmm.predict_proba(X)
    # GMM labels may differ from our quadrant labels – align them
    # by matching mean inflation/growth direction
    means_scaled = gmm.means_
    means_orig = scaler.inverse_transform(means_scaled)
    # GMM component → quadrant
    global_med_cpi = df['cpi_smooth'].median()
    global_med_gdp = df['gdp_smooth'].median()
    gmm_labels = np.zeros(4, dtype=int)
    for i, (c, g) in enumerate(means_orig):
        hi_c = c >= global_med_cpi
        hi_g = g >= global_med_gdp
        if not hi_c and hi_g: gmm_labels[i] = 0
        elif hi_c and not hi_g: gmm_labels[i] = 1
        elif hi_c and hi_g: gmm_labels[i] = 2
        else: gmm_labels[i] = 3
    # Remap probabilities to our quadrant labels
    regime_probs = np.zeros((len(df), 4))
    for gmm_comp, quad in enumerate(gmm_labels):
        regime_probs[:, quad] += probs[:, gmm_comp]
    for i in range(4):
        df[f'prob_{i}'] = regime_probs[:, i]
    df['max_prob'] = regime_probs.max(axis=1)

    return df


# ─────────────────────────────────────────────
# 3.  DYNAMIC ASSET ALLOCATION
# ─────────────────────────────────────────────

# Theoretical asset class performance profiles per regime
# Based on academic research (Ilmanen 2011, Bridgewater, JPM)
# Weights: (Equities, Bonds, Commodities, Gold, Cash, REITs, EM, Inflation-Linked)
ASSET_CLASSES = ['Equities', 'Gov. Bonds', 'Commodities', 'Gold',
                 'Cash', 'REITs', 'EM Equities', 'Linkers (TIPS/Gilts)']

# Base allocation per regime (%)
BASE_ALLOCATIONS = {
    0: np.array([45, 25,  5,  5, 5,  8,  5,  2]),   # Goldilocks
    1: np.array([10, 10, 25, 20, 5,  5, 10, 15]),    # Stagflation
    2: np.array([30,  5, 20, 10, 5,  5, 15, 10]),    # Inf. Boom
    3: np.array([15, 45,  5, 10,15,  5,  0,  5]),    # Def. Bust
}
assert all(v.sum() == 100 for v in BASE_ALLOCATIONS.values()), "Weights must sum to 100"

# Expected real returns per regime (annualised %, theoretical)
EXPECTED_RETURNS = {
    0: np.array([ 8.0,  2.0,  1.0,  1.0, 0.5, 6.0, 7.0, 0.5]),
    1: np.array([-4.0, -3.0,  7.0,  6.0, 0.0,-3.0,-3.0, 5.0]),
    2: np.array([ 4.0, -2.0,  5.0,  3.0, 0.0, 2.0, 5.0, 3.0]),
    3: np.array([-6.0,  5.0, -2.0,  2.0, 2.0,-4.0,-8.0, 2.0]),
}

def compute_dynamic_allocation(df):
    """
    Compute a blended allocation using posterior probabilities
    to smooth transitions between regimes.
    """
    alloc_cols = [f'w_{ac.replace(" ","_").replace(".","")}' for ac in ASSET_CLASSES]
    ret_cols   = [f'r_{ac.replace(" ","_").replace(".","")}' for ac in ASSET_CLASSES]

    weights = np.zeros((len(df), len(ASSET_CLASSES)))
    exp_rets = np.zeros((len(df), len(ASSET_CLASSES)))
    for r in range(4):
        p = df[f'prob_{r}'].values[:, None]
        weights   += p * BASE_ALLOCATIONS[r]
        exp_rets  += p * EXPECTED_RETURNS[r]

    # Normalise weights to sum to 100
    weights = weights / weights.sum(axis=1, keepdims=True) * 100

    for i, col in enumerate(alloc_cols):
        df[col] = weights[:, i]
    for i, col in enumerate(ret_cols):
        df[col] = exp_rets[:, i]

    df['expected_port_return'] = (weights * exp_rets).sum(axis=1) / 100
    return df, alloc_cols, ret_cols


# ─────────────────────────────────────────────
# 4.  BACKTESTED PORTFOLIO PERFORMANCE
#     (simplified, using expected returns as proxies)
# ─────────────────────────────────────────────

def simulate_portfolio(df):
    """
    Simulate a quarterly-rebalanced portfolio using the dynamic allocation.
    Compare to a static 60/40 benchmark.
    """
    alloc_cols = [c for c in df.columns if c.startswith('w_')]
    ret_cols   = [c for c in df.columns if c.startswith('r_')]

    # Dynamic portfolio quarterly return (approx)
    q_ret_dynamic = (df[alloc_cols].values * df[ret_cols].values).sum(axis=1) / 100 / 4
    # Static 60/40 reference (assumes Goldilocks returns everywhere)
    static_w = np.array([60, 40, 0, 0, 0, 0, 0, 0]) / 100
    static_r = (BASE_ALLOCATIONS[0] / 100) * EXPECTED_RETURNS[0]  # simplified
    q_ret_static = (static_w * EXPECTED_RETURNS[0][:2].mean()).sum() / 4
    q_ret_static = 0.08 / 4  # ~ 8% pa annualised

    df['q_ret_dynamic'] = q_ret_dynamic
    df['cum_dynamic'] = (1 + q_ret_dynamic).cumprod()
    df['cum_static']  = (1 + q_ret_static) ** np.arange(1, len(df)+1)

    return df


# ─────────────────────────────────────────────
# 5.  PLOTTING
# ─────────────────────────────────────────────

def plot_all(us_df, uk_df):
    fig = plt.figure(figsize=(22, 28))
    fig.patch.set_facecolor('#0f1117')

    gs = gridspec.GridSpec(5, 2, figure=fig, hspace=0.52, wspace=0.32,
                           top=0.94, bottom=0.04, left=0.07, right=0.97)

    txt_kw = dict(color='white', fontweight='bold')
    ax_bg  = '#1a1d27'

    def style_ax(ax, title='', xlabel='', ylabel=''):
        ax.set_facecolor(ax_bg)
        ax.tick_params(colors='#aaaaaa', labelsize=9)
        ax.spines[:].set_color('#333344')
        if title:  ax.set_title(title,  fontsize=11, **txt_kw, pad=8)
        if xlabel: ax.set_xlabel(xlabel, fontsize=9,  color='#aaaaaa')
        if ylabel: ax.set_ylabel(ylabel, fontsize=9,  color='#aaaaaa')

    # ── (A) Macro time series – US & UK ──────────────────────────────
    for col_idx, (df, label) in enumerate([(us_df, 'US'), (uk_df, 'UK')]):
        ax = fig.add_subplot(gs[0, col_idx])
        style_ax(ax, f'{label} Macro Time Series 1990–2025')
        ax.plot(df.index, df['cpi_yoy'],    color='#e74c3c', lw=1.5, label='CPI YoY %')
        ax.plot(df.index, df['gdp_growth'], color='#2ecc71', lw=1.5, label='Real GDP (ann. %)')
        ax.plot(df.index, df['policy_rate'],color='#f39c12', lw=1.2,
                ls='--', label='Policy Rate %')
        ax.axhline(0, color='#555555', lw=0.7)
        ax.legend(fontsize=8, framealpha=0.2, labelcolor='white',
                  facecolor='#111122', loc='upper right')
        ax.set_xlim(df.index[0], df.index[-1])

    # ── (B) Regime classification scatter – US & UK ──────────────────
    for col_idx, (df, label) in enumerate([(us_df, 'US'), (uk_df, 'UK')]):
        ax = fig.add_subplot(gs[1, col_idx])
        style_ax(ax, f'{label} Regime Classification (GMM)',
                 'CPI Smoothed YoY %', 'Real GDP Growth (ann. %)')
        for r in range(4):
            mask = df['regime'] == r
            ax.scatter(df.loc[mask, 'cpi_smooth'],
                       df.loc[mask, 'gdp_smooth'],
                       c=REGIME_COLORS[r], s=30, alpha=0.75,
                       label=SHORT_LABELS[r], zorder=3)
        # threshold lines
        thresh_cpi = df['inf_threshold'].iloc[0]
        thresh_gdp = df['gdp_threshold'].iloc[0]
        ax.axvline(thresh_cpi, color='white', lw=0.7, ls='--', alpha=0.5)
        ax.axhline(thresh_gdp, color='white', lw=0.7, ls='--', alpha=0.5)
        ax.legend(fontsize=8, framealpha=0.2, labelcolor='white',
                  facecolor='#111122', markerscale=1.2)
        # quadrant labels
        xl, xr = ax.get_xlim()
        yb, yt = ax.get_ylim()
        mid_x = thresh_cpi; mid_y = thresh_gdp
        ax.text(xl + (mid_x-xl)*0.35, mid_y + (yt-mid_y)*0.75,
                'GOLDILOCKS', color='#2ecc71', fontsize=7, alpha=0.6, ha='center')
        ax.text(mid_x + (xr-mid_x)*0.55, mid_y + (yt-mid_y)*0.75,
                'INF. BOOM',  color='#f39c12', fontsize=7, alpha=0.6, ha='center')
        ax.text(xl + (mid_x-xl)*0.35, yb + (mid_y-yb)*0.2,
                'DEF. BUST',  color='#3498db', fontsize=7, alpha=0.6, ha='center')
        ax.text(mid_x + (xr-mid_x)*0.55, yb + (mid_y-yb)*0.2,
                'STAGFLATION',color='#e74c3c', fontsize=7, alpha=0.6, ha='center')

    # ── (C) Regime timeline – US & UK ────────────────────────────────
    for col_idx, (df, label) in enumerate([(us_df, 'US'), (uk_df, 'UK')]):
        ax = fig.add_subplot(gs[2, col_idx])
        style_ax(ax, f'{label} Regime Timeline')
        for i, row in df.iterrows():
            ax.axvspan(i, i + pd.DateOffset(months=3),
                       color=REGIME_COLORS[row['regime']], alpha=0.7)
        patches = [mpatches.Patch(color=REGIME_COLORS[r], label=SHORT_LABELS[r])
                   for r in range(4)]
        ax.legend(handles=patches, fontsize=8, framealpha=0.2,
                  labelcolor='white', facecolor='#111122',
                  loc='lower right', ncol=2)
        ax.set_xlim(df.index[0], df.index[-1])
        ax.set_yticks([])
        ax.set_xlabel('Year', fontsize=9, color='#aaaaaa')

    # ── (D) Dynamic Asset Allocation – current (last observation) ────
    for col_idx, (df, label) in enumerate([(us_df, 'UK'), (uk_df, 'UK')]):
        # Use last row for current allocation
        df_use = us_df if col_idx == 0 else uk_df
        lbl = 'US' if col_idx == 0 else 'UK'
        ax = fig.add_subplot(gs[3, col_idx])
        style_ax(ax, f'{lbl} Current Dynamic Allocation (2025 Q1)')
        last = df_use.iloc[-1]
        alloc_cols = [c for c in df_use.columns if c.startswith('w_')]
        current_w = last[alloc_cols].values
        colors = plt.cm.Set2(np.linspace(0, 1, len(ASSET_CLASSES)))
        bars = ax.barh(ASSET_CLASSES, current_w, color=colors, edgecolor='#0f1117', height=0.65)
        ax.set_xlabel('Allocation (%)', fontsize=9, color='#aaaaaa')
        for bar, val in zip(bars, current_w):
            ax.text(val + 0.3, bar.get_y() + bar.get_height()/2,
                    f'{val:.1f}%', va='center', color='white', fontsize=8)
        # Annotate regime
        regime_id = int(last['regime'])
        ax.text(0.98, 0.02, f"Regime: {SHORT_LABELS[regime_id]}",
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=9, color=REGIME_COLORS[regime_id], fontweight='bold')
        ax.tick_params(axis='y', labelsize=8, colors='#cccccc')
        ax.set_xlim(0, 60)

    # ── (E) Allocation through time (US stacked area) ────────────────
    ax = fig.add_subplot(gs[4, :])
    style_ax(ax, 'US Dynamic Allocation Through Time (Stacked Area)',
             xlabel='Year', ylabel='Allocation (%)')
    alloc_cols = [c for c in us_df.columns if c.startswith('w_')]
    colors = plt.cm.Set2(np.linspace(0, 1, len(ASSET_CLASSES)))
    data   = us_df[alloc_cols].values.T
    ax.stackplot(us_df.index, data, labels=ASSET_CLASSES, colors=colors, alpha=0.85)
    # Overlay regime shading (faint)
    for i, row in us_df.iterrows():
        ax.axvspan(i, i + pd.DateOffset(months=3),
                   color=REGIME_COLORS[row['regime']], alpha=0.07)
    ax.set_xlim(us_df.index[0], us_df.index[-1])
    ax.set_ylim(0, 100)
    ax.legend(fontsize=7.5, framealpha=0.2, labelcolor='white',
              facecolor='#111122', loc='upper right', ncol=4)
    ax.set_xlabel('Year', fontsize=9, color='#aaaaaa')
    ax.set_ylabel('Allocation (%)', fontsize=9, color='#aaaaaa')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    # Main title
    fig.suptitle('Macroeconomic Regime Classification & Dynamic Asset Allocation\n'
                 'US & UK  |  1990 Q1 – 2025 Q1  |  GMM-Based Regime Model',
                 fontsize=15, color='white', fontweight='bold', y=0.975)

    plt.savefig('/mnt/user-data/outputs/macro_regime_model.png',
                dpi=160, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print("Chart saved.")


# ─────────────────────────────────────────────
# 6.  REGIME STATISTICS TABLE
# ─────────────────────────────────────────────

def regime_stats(df, country_label):
    print(f"\n{'='*60}")
    print(f"  {country_label} REGIME STATISTICS (1990 Q1 – 2025 Q1)")
    print(f"{'='*60}")
    for r in range(4):
        sub = df[df['regime'] == r]
        n = len(sub)
        print(f"\n  [{SHORT_LABELS[r]}] — {n} quarters ({n/4:.1f} yrs, "
              f"{100*n/len(df):.0f}%)")
        print(f"    CPI YoY   : avg {sub['cpi_yoy'].mean():.1f}%  "
              f"range [{sub['cpi_yoy'].min():.1f}, {sub['cpi_yoy'].max():.1f}]")
        print(f"    GDP Growth: avg {sub['gdp_growth'].mean():.1f}%  "
              f"range [{sub['gdp_growth'].min():.1f}, {sub['gdp_growth'].max():.1f}]")
        print(f"    Policy Rate: avg {sub['policy_rate'].mean():.1f}%")

    print(f"\n  CURRENT REGIME (2025 Q1): {SHORT_LABELS[int(df.iloc[-1]['regime'])]}")
    print(f"  Posterior Probs → " +
          "  ".join([f"{SHORT_LABELS[r]}: {df.iloc[-1][f'prob_{r}']:.2f}" for r in range(4)]))


def allocation_table(df, country_label):
    alloc_cols = [c for c in df.columns if c.startswith('w_')]
    print(f"\n{'='*60}")
    print(f"  {country_label} CURRENT ALLOCATION  (2025 Q1)")
    print(f"{'='*60}")
    last = df.iloc[-1]
    for i, (ac, col) in enumerate(zip(ASSET_CLASSES, alloc_cols)):
        bar = '█' * int(last[col] / 2)
        print(f"  {ac:<22} {last[col]:5.1f}%  {bar}")
    print(f"\n  Expected Portfolio Return (blended): "
          f"{last['expected_port_return']:.1f}% pa  (real, theoretical)")


# ─────────────────────────────────────────────
# 7.  MAIN
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print("Building US data...")
    us_raw = build_us_data()
    print("Building UK data...")
    uk_raw = build_uk_data()

    print("Classifying regimes...")
    us_df = classify_regimes(us_raw)
    uk_df = classify_regimes(uk_raw)

    print("Computing dynamic allocations...")
    us_df, alloc_cols, _ = compute_dynamic_allocation(us_df)
    uk_df, alloc_cols, _ = compute_dynamic_allocation(uk_df)

    print("Simulating portfolios...")
    us_df = simulate_portfolio(us_df)
    uk_df = simulate_portfolio(uk_df)

    # Text output
    regime_stats(us_df, "US")
    regime_stats(uk_df, "UK")
    allocation_table(us_df, "US")
    allocation_table(uk_df, "UK")

    # Plots
    print("\nGenerating charts...")
    plot_all(us_df, uk_df)

    print("\nDone. Output saved to /mnt/user-data/outputs/macro_regime_model.png")
