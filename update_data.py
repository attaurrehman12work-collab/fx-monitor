# Writes data.txt for the monitor page. Run daily by GitHub Actions (or run it yourself).
import os, numpy as np, pandas as pd
import yfinance as yf
W = {"XAUUSD":"GC=F","XAGUSD":"SI=F","WTI":"CL=F","EURUSD":"EURUSD=X","GBPUSD":"GBPUSD=X",
     "AUDUSD":"AUDUSD=X","NZDUSD":"NZDUSD=X","USDJPY":"USDJPY=X","USDCAD":"USDCAD=X",
     "USDCHF":"USDCHF=X","SPX":"^GSPC","NDX":"^IXIC","BTCUSD":"BTC-USD",
     "AAPL":"AAPL","MSFT":"MSFT","NVDA":"NVDA","TSLA":"TSLA","AMZN":"AMZN","GOOGL":"GOOGL","META":"META"}
if os.path.exists("watchlist.txt"):  # optional: lines like ETHUSD=ETH-USD
    for s in open("watchlist.txt"):
        if "=" in s: a, b = s.strip().split("=", 1); W[a] = b
def bt(c, up, dn, p=1000.0):  # long above 100d high, short below 100d low, exit on 10d level
    pos, ent, eq, pk, dd, rs = 0, 0, p, p, 0, []
    for i in range(100, len(c)):
        x = c[i]
        if pos == 0:
            if x > up[i]: pos, ent = 1, x
            elif x < dn[i]: pos, ent = -1, x
        if pos == 1 and x < c[i-10:i].min(): rs.append(x/ent-1); eq *= x/ent; pos = 0
        elif pos == -1 and x > c[i-10:i].max(): rs.append(1-x/ent); eq *= 2-x/ent; pos = 0
        pk = max(pk, eq); dd = max(dd, 1-eq/pk)
    return eq, rs, dd
L, BT, CN, HS, SC = [], [], [], [], []
for n, t in W.items():
    try:
        d = yf.download(t, period="10y", progress=False, auto_adjust=True)
        if isinstance(d.columns, pd.MultiIndex): d.columns = d.columns.get_level_values(0)
        d = d.dropna()
        c, h, l, o = d.Close, d.High, d.Low, d.Open
        v = c.values
        up = c.shift(1).rolling(100).max().values; dn = c.shift(1).rolling(100).min().values
        dlt = c.diff(); g = dlt.clip(lower=0).ewm(alpha=1/14).mean(); s = (-dlt.clip(upper=0)).ewm(alpha=1/14).mean()
        rsi = 100 - 100/(1 + g/s)
        tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/14).mean()
        L.append(f"{n}|{d.index[-1].date()}|{v[-1]:.6g}|{c.iloc[-100:].max():.6g}|{c.iloc[-100:].min():.6g}|"
                 f"{c.iloc[-10:].max():.6g}|{c.iloc[-10:].min():.6g}|{rsi.iloc[-1]:.1f}|{atr.iloc[-1]:.6g}")
        eq, rs, dd = bt(v, up, dn)
        w = 100*sum(r > 0 for r in rs)/max(len(rs), 1)
        BT.append(f"BT|{n}|{len(rs)}|{w:.0f}|{dd*100:.0f}|{eq:.0f}")
        m = len(v)//2; a = bt(v[:m], up[:m], dn[:m]); b = bt(v[m:], up[m:], dn[m:])
        SC.append(f"SCAN|{t}|{a[0]:.0f}|{b[0]:.0f}|{len(rs)}")
        CN.append("CND|" + n + "|" + ";".join(f"{o.iloc[i]:.6g},{h.iloc[i]:.6g},{l.iloc[i]:.6g},{c.iloc[i]:.6g}" for i in (-3,-2,-1)))
        HS.append("HST|" + n + "|" + ",".join(f"{x:.6g}" for x in v[-100:]))
    except Exception as e:
        print("skip", n, e)
out = "\n".join(L + BT + SC + CN + HS)
print(out)
if L:  # only overwrite data.txt when at least one market downloaded
    open("data.txt", "w").write(out + "\n")
