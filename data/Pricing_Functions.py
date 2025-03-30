import pandas as pd 
import matplotlib.pyplot as plt
#import seaborn as sns
import re
import yfinance as yf
import numpy as np
from datetime import datetime
import time
from scipy.stats import norm
from scipy.stats import norminvgauss

# 1. Black-Scholes

def get_todays_price(ticker_symbol):
    """
    Fetches the most recent trading price for the given ticker.
    """
    ticker = yf.Ticker(ticker_symbol)
    try:
        data = ticker.history(period="1d", interval="1m")
        if data.empty:
            raise ValueError("No intraday data returned.")
        price = data["Close"].iloc[-1]
        return round(price, 2)
    except Exception as e:
        print(f"Error fetching price for {ticker_symbol}: {e}")
        return None
    

def fetch_full_option_chain(ticker_symbol, max_expirations=5, sleep_time=1):
    """
    Fetches all options (calls & puts) for a ticker across multiple expiration dates.
    """
    ticker = yf.Ticker(ticker_symbol)
    expirations = ticker.options[:max_expirations]
    all_options = []

    for expiry in expirations:
        try:
            chain = ticker.option_chain(expiry)
            calls = chain.calls.copy()
            puts = chain.puts.copy()

            # Label type
            calls["type"] = "call"
            puts["type"] = "put"

            # Add expiration column
            calls["expiration"] = expiry
            puts["expiration"] = expiry

            # Merge calls & puts
            combined = pd.concat([calls, puts], ignore_index=True)
            all_options.append(combined)

            print(f"Fetched {len(combined)} contracts for expiry {expiry}")
            time.sleep(sleep_time)

        except Exception as e:
            print(f"⚠️ Failed on {expiry}: {e}")
            continue

    return pd.concat(all_options, ignore_index=True) if all_options else pd.DataFrame()


def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    if T <= 0 or sigma <= 0:
        return max(0, S - K) if option_type == 'call' else max(0, K - S)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    

def calculate_bs_prices(df, S, r=0.045):
    df = df.copy()
    df["T"] = df["days_to_expiration"] / 365

    df["bs_price"] = df.apply(
        lambda row: black_scholes_price(
            S=S,
            K=row["strike"],
            T=row["T"],
            r=r,
            sigma=row["impliedVolatility"],
            option_type=row["type"]
        ),
        axis=1
    )

    df["pricing_error"] = df["lastPrice"] - df["bs_price"]
    return df


def enrich_option_data(df, underlying_price):
    """
    Enriches options DataFrame with moneyness & days-to-expiration.
    """
    df = df.copy()
    df["moneyness"] = df["strike"] / underlying_price
    df["current_date"] = pd.Timestamp.today().normalize()
    df["expiration"] = pd.to_datetime(df["expiration"])
    df["days_to_expiration"] = (df["expiration"] - df["current_date"]).dt.days
    df.drop(columns=["current_date"], inplace=True)
    return df

# 2. Lévy Process – Variance Gamma
def simulate_variance_gamma(S0, r, T, theta, sigma, nu, N=10000):
    dt = T
    gamma = np.random.gamma(dt / nu, nu, N)
    Z = np.random.normal(0, 1, N)
    ST = S0 * np.exp(r * T + theta * gamma + sigma * np.sqrt(gamma) * Z)
    return ST


def price_option_vg_mc(S0, K, T, r, theta, sigma, nu, option_type='call', N=10000):
    ST = simulate_variance_gamma(S0, r, T, theta, sigma, nu, N)
    if option_type == 'call':
        payoff = np.maximum(ST - K, 0)
    else:
        payoff = np.maximum(K - ST, 0)
    return np.exp(-r * T) * np.mean(payoff)


# 3. Generalized Hyperbolic (NIG)
def simulate_nig_returns(alpha, beta, delta, mu, N=10000):
    return norminvgauss.rvs(alpha, beta, loc=mu, scale=delta, size=N)


def price_option_nig_mc(S0, K, T, r, alpha, beta, delta, mu, option_type='call', N=10000):
    rvs = simulate_nig_returns(alpha, beta, delta, mu, N)
    ST = S0 * np.exp((r - 0.5 * np.var(rvs)) * T + rvs * np.sqrt(T))
    if option_type == 'call':
        payoff = np.maximum(ST - K, 0)
    else:
        payoff = np.maximum(K - ST, 0)
    return np.exp(-r * T) * np.mean(payoff)

def compare_models(row,bs_df):

    bs_df = bs_df[
    (bs_df["days_to_expiration"] > 0) &
    (bs_df["impliedVolatility"].notna()) &
    (bs_df["impliedVolatility"] > 0)
]

    K = row["strike"]
    T = row["days_to_expiration"] / 365
    option_type = row["type"]
    sigma = row["impliedVolatility"]

    # Validate inputs
    if pd.isna(K) or pd.isna(T) or pd.isna(sigma) or T <= 0 or sigma <= 0:
        return pd.Series([np.nan, np.nan])

    try:
        vg = price_option_vg_mc(S, K, T, r, theta, sigma, nu, option_type)
        nig = price_option_nig_mc(S, K, T, r, alpha, beta, delta, mu, option_type)
    except Exception as e:
        print(f"Error at K={K}, T={T}, sigma={sigma}: {e}")
        return pd.Series([np.nan, np.nan])

    return pd.Series([vg, nig])


"""
# # # IMPLEMENTATION # # #

# Choose your ticker
ticker_symbol = "AAPL"

# Step 1: Get underlying price
S = get_todays_price(ticker_symbol)

# Step 2: Fetch option chain
df = fetch_full_option_chain(ticker_symbol, max_expirations=5)

# Step 3: Enrich with derived metrics
enriched_df = enrich_option_data(df, underlying_price=S)

# Step 4: Apply Black-Scholes pricing
bs_df = calculate_bs_prices(enriched_df, S)

# Step 5: View results
bs_df[["contractSymbol", "type", "strike", "lastPrice", "bs_price", "pricing_error"]].head()


"""





# -------------------------------
# 1. Configuration
# -------------------------------
ticker_symbol = "AAPL"
r = 0.045  # Risk-free rate
theta, nu = 0.1, 0.2  # VG parameters
alpha, beta, delta, mu = 3, -0.1, 0.2, 0  # NIG parameters

# -------------------------------
# 2. Fetch Market Data
# -------------------------------
S = get_todays_price(ticker_symbol)
print(f"📈 Spot Price of {ticker_symbol}: ${S}")

df = fetch_full_option_chain(ticker_symbol, max_expirations=3)
df = enrich_option_data(df, underlying_price=S)
df = df[
    (df["days_to_expiration"] > 0) &
    (df["impliedVolatility"].notna()) &
    (df["impliedVolatility"] > 0)
]

# -------------------------------
# 3. Black-Scholes Baseline
# -------------------------------
bs_df = calculate_bs_prices(df, S, r)

# -------------------------------
# 4. Compare with VG & NIG
# -------------------------------
def compare_models(row):
    K = row["strike"]
    T = row["days_to_expiration"] / 365
    sigma = row["impliedVolatility"]
    option_type = row["type"]

    try:
        vg = price_option_vg_mc(S, K, T, r, theta, sigma, nu, option_type)
        nig = price_option_nig_mc(S, K, T, r, alpha, beta, delta, mu, option_type)
    except Exception as e:
        print(f"⚠️ Error on strike {K}: {e}")
        return pd.Series([np.nan, np.nan])

    return pd.Series([vg, nig])

# Apply models
bs_df[["vg_price", "nig_price"]] = bs_df.apply(compare_models, axis=1)

# -------------------------------
# 5. Pricing Errors
# -------------------------------
bs_df["vg_error"] = bs_df["lastPrice"] - bs_df["vg_price"]
bs_df["nig_error"] = bs_df["lastPrice"] - bs_df["nig_price"]

# -------------------------------
# 6. Display Results
# -------------------------------
print(bs_df[[
    "contractSymbol", "type", "strike", "lastPrice",
    "bs_price", "vg_price", "nig_price",
    "pricing_error", "vg_error", "nig_error"
]].dropna().head())


