import os, json, requests, pandas as pd, numpy as np
from datetime import date
S = requests.Session(); S.auth = ('API_KEY', os.environ['ICU_API_KEY'])
B = 'https://intervals.icu/api/v1/athlete/' + os.environ.get('ICU_ATHLETE_ID', '0')
today = pd.Timestamp(date.today())
def get(p, **q):
    r = S.get(B + p, params=q, timeout=60); r.raise_for_status(); return r.json()
acts = get('/activities', oldest=str((today - pd.Timedelta(days=400)).date()), newest=str(today.date()))
print(len(acts), 'activities. Fields available:', sorted(acts[0])[:80] if acts else None)  # names only, no values
g = lambda a, *k: next((a[x] for x in k if a.get(x) is not None), None)
df = pd.DataFrame([dict(d=pd.to_datetime(g(a, 'start_date_local')), type=a.get('type'), secs=g(a, 'moving_time') or 0,
    load=g(a, 'icu_training_load') or 0, nw=g(a, 'icu_weighted_avg_watts'), iff=g(a, 'icu_intensity'),
    ftp=g(a, 'icu_ftp'), wt=g(a, 'icu_weight')) for a in acts]).sort_values('d').reset_index(drop=True)
df['ftp'] = df.ftp.astype(float).ffill().bfill(); df['wt'] = df.wt.astype(float).ffill().bfill()
df['iff'] = df.iff.astype(float)
if df.iff.median() < 3: df['iff'] = df.iff * 100
df['iff'] = df.iff.fillna(df.nw.astype(float) / df.ftp * 100)
try:
    w = sorted(get('/wellness', oldest=str((today - pd.Timedelta(days=45)).date()), newest=str(today.date())), key=lambda x: str(x.get('id')))
    ws = [x['weight'] for x in w if x.get('weight')]
except Exception as e:
    ws = []; print('Wellness not available:', type(e).__name__)
ftp = float(df.ftp.dropna().iloc[-1]); wt = float(ws[-1] if ws else df.wt.dropna().iloc[-1])
daily = df.groupby(df.d.dt.normalize())['load'].sum()
daily = daily.reindex(pd.date_range(daily.index.min(), today), fill_value=0)
ctl = daily.ewm(alpha=1/42, adjust=False).mean(); atl = daily.ewm(alpha=1/7, adjust=False).mean()
fit = [[str(i.date()), round(ctl[i], 1), round(atl[i], 1)] for i in daily.index if i >= today - pd.Timedelta(days=182)]
cyc = df[df.type.isin(['VirtualRide', 'Ride'])].copy(); s = cyc.set_index('d')
wkh = (s['secs'].resample('W').sum() / 3600).tail(26); wkl = df.set_index('d')['load'].resample('W').sum().tail(26)
f = df.set_index('d')[['ftp', 'wt']].resample('W').last().ffill().bfill().tail(52)
w8 = df[df.d >= today - pd.Timedelta(days=56)].dropna(subset=['wt'])
slope = float(np.polyfit((w8.d - w8.d.iloc[0]).dt.days, w8.wt, 1)[0] * 7) if len(w8) > 3 else 0.0
r8 = cyc[(cyc.d >= today - pd.Timedelta(days=56)) & cyc.iff.notna()]; tt = max(r8.secs.sum(), 1)
sp = [round(100 * r8[m].secs.sum() / tt) for m in [r8.iff < 76, (r8.iff >= 76) & (r8.iff < 91), r8.iff >= 91]]
lg = cyc[cyc.secs >= 5400].set_index('d')['secs'].resample('MS').count().tail(12)
# ---- today's suggestion ----
hard = lambda x: len(x[x.iff >= 88])
h48 = hard(cyc[cyc.d >= today - pd.Timedelta(days=1)]); h7 = hard(cyc[cyc.d >= today - pd.Timedelta(days=6)])
tsb = float(ctl.iloc[-1] - atl.iloc[-1]); P = lambda a, b: f'{round(ftp*a)}-{round(ftp*b)} W'
longd = cyc[(cyc.secs >= 6600) & (cyc.d >= today - pd.Timedelta(days=6))]
if h48 >= 2 or tsb < -25: t, dt = 'Rest day or easy spin', f'30-60 min at {P(.5, .6)}, or take the day off.'
elif h48 == 1 or tsb < -10 or h7 >= 4:
    t, dt = ('Long steady ride', f'2-2.5 h at {P(.6, .7)}, conversational pace.') if longd.empty else ('Endurance ride', f'75-90 min at {P(.6, .7)}.')
elif tsb > 10: t, dt = 'VO2 max session', f'5 x 4 min at {P(1.12, 1.2)}, 4 min easy between, with a 15 min warm-up and cool-down.'
elif h7 >= 3: t, dt = 'Sweet spot', f'3 x 15 min at {P(.88, .93)}, 5 min easy between.'
else: t, dt = 'Threshold session', f'3 x 12 min at {P(.95, 1.0)}, 5 min easy between.'
why = f'Form {tsb:+.0f}. {h48} hard ride(s) in the last 2 days and {h7} in the last 7.'
data = dict(fit=fit, wk=[[str(i.date()), round(float(a), 1), round(float(b))] for i, a, b in zip(wkh.index, wkh.values, wkl.values)],
    ftpw=[[str(i.date()), float(a), float(b)] for i, (a, b) in f.iterrows()], lg=[[i.strftime('%b'), int(v)] for i, v in lg.items()],
    split=sp, slope=round(slope, 2), ftp=ftp, wt=wt, ctl=int(round(ctl.iloc[-1])), ctl4=int(round(ctl.iloc[-29])), tsb=int(round(tsb)),
    hrs4=round(float(wkh.tail(4).mean()), 1), longest=round(float(cyc[cyc.d >= today - pd.Timedelta(days=90)].secs.max()) / 3600, 1),
    asof=today.strftime('%d %b %Y'), sug=dict(title=t, detail=dt, why=why))
os.makedirs('docs', exist_ok=True)
open('docs/index.html', 'w').write(open('template.html').read().replace('__DATA__', json.dumps(data)))
print('Built docs/index.html. Suggestion:', t)
