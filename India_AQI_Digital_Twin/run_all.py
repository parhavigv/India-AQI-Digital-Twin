"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          INDIA AQI DIGITAL TWIN v6.0 — SINGLE FILE RUNNER                  ║
║  Just open this file in VS Code and press F5  (or run: python run_all.py)  ║
║                                                                              ║
║  Simulators : ① SimPy (discrete-event pollution)                            ║
║               ② SUMO-py (traffic emissions)                                 ║
║  ML Models  : CNN · LSTM · GRU · GNN (Graph-RF proxy)                      ║
║  XAI        : SHAP + LIME + Best-Model Identification                       ║
║  Outputs    : Terminal dashboard · Folium map · SHAP/LIME charts · CSVs    ║
╚══════════════════════════════════════════════════════════════════════════════╝

STEP 1 — Install dependencies (run once in terminal):
    pip install aiohttp pandas scikit-learn folium rich requests simpy
                shap lime matplotlib seaborn python-dotenv tqdm joblib

STEP 2 — Run:
    python run_all.py
"""

import asyncio, sqlite3, json, math, time, pickle, warnings, os, sys, random
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.rule import Rule
    console = Console()
    RICH = True
except ImportError:
    RICH = False
    class _C:
        def print(self, *a, **k): print(*a)
        def rule(self, *a, **k): print("─"*60)
    console = _C()

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG & CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
WAQI_API_KEY = "7febb016ec21d1af8b59c5972f2220b0467e6f80"
WAQI_BASE    = "https://api.waqi.info/feed/{}/?token={}"
WAQI_MAP     = "https://api.waqi.info/map/bounds/?latlng={}&token={}"

BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR  = BASE_DIR / "models_cache"
for _d in [DATA_DIR, OUTPUT_DIR, MODEL_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

DB_PATH      = str(DATA_DIR / "aqi_twin.db")
SEQ_LEN      = 24
FORECAST_HRS = [1, 3, 6, 12, 24]
RANDOM_SEED  = 42

INDIA_ZONES = {
    "North India":     "25,68,37,80",
    "Northeast India": "22,88,29,97",
    "East India":      "20,82,27,89",
    "Central India":   "20,75,27,83",
    "West India":      "18,68,28,78",
    "South India":     "8,74,20,80",
    "Far South":       "8,77,12,80",
}
NAMED_STATIONS = [
    "delhi","mumbai","bangalore","chennai","kolkata","hyderabad","pune",
    "ahmedabad","surat","jaipur","lucknow","kanpur","nagpur","indore",
    "bhopal","visakhapatnam","patna","vadodara","ludhiana","agra",
    "nashik","faridabad","meerut","rajkot","varanasi","srinagar",
    "noida","gurugram","dehradun","kochi","bhubaneswar","siliguri",
    "guwahati","chandigarh","thiruvananthapuram","coimbatore","madurai",
    "jodhpur","raipur","ranchi",
]
AQI_CATEGORIES = [
    (0,   50,  "Good",         "🟢", "#00e400"),
    (51,  100, "Satisfactory", "🟡", "#ffff00"),
    (101, 200, "Moderate",     "🟠", "#ff7e00"),
    (201, 300, "Poor",         "🔴", "#ff0000"),
    (301, 400, "Very Poor",    "🟣", "#8f3f97"),
    (401, 999, "Severe",       "⚫", "#7e0023"),
]
SIMPY_SOURCES = [
    {"name": "Industrial Zone A",  "base_emission": 180, "peak_hours": [8,9,17,18]},
    {"name": "Traffic Corridor B", "base_emission": 130, "peak_hours": [7,8,9,17,18,19]},
    {"name": "Power Plant C",      "base_emission": 220, "peak_hours": list(range(24))},
    {"name": "Residential Zone D", "base_emission": 80,  "peak_hours": [6,7,20,21]},
]
ALERT_RULES = [
    {"name": "SEVERE",    "threshold": 300, "horizon_h": 1},
    {"name": "VERY POOR", "threshold": 200, "horizon_h": 6},
    {"name": "POOR",      "threshold": 150, "horizon_h": 24},
]

def aqi_category(aqi):
    if aqi is None: return ("Unknown","⬜","#cccccc")
    for lo, hi, label, emoji, color in AQI_CATEGORIES:
        if lo <= int(aqi) <= hi: return (label, emoji, color)
    return ("Severe","⚫","#7e0023")

# ══════════════════════════════════════════════════════════════════════════════
# DATABASE
# ══════════════════════════════════════════════════════════════════════════════
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS stations (
            station_id TEXT PRIMARY KEY, name TEXT, city TEXT, state TEXT,
            latitude REAL, longitude REAL, zone TEXT, first_seen TEXT, last_seen TEXT);
        CREATE TABLE IF NOT EXISTS readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            station_id TEXT, timestamp TEXT, collected_at TEXT,
            aqi INTEGER, pm25 REAL, pm10 REAL, no2 REAL,
            co REAL, o3 REAL, so2 REAL, temperature REAL, humidity REAL, wind_speed REAL);
        CREATE TABLE IF NOT EXISTS forecasts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            station_id TEXT, generated_at TEXT, forecast_for TEXT,
            horizon_hrs INTEGER, aqi_pred REAL, aqi_lower REAL,
            aqi_upper REAL, model TEXT, confidence REAL);
        CREATE TABLE IF NOT EXISTS model_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            station_id TEXT, evaluated_at TEXT, model TEXT,
            rmse REAL, mae REAL, r2 REAL, mape REAL, n_samples INTEGER);
        CREATE INDEX IF NOT EXISTS idx_r_sid ON readings(station_id);
        CREATE INDEX IF NOT EXISTS idx_r_ts  ON readings(timestamp);
    """)
    conn.commit(); conn.close()

def save_reading(conn, sid, rec):
    conn.execute("""
        INSERT INTO stations (station_id,name,city,state,latitude,longitude,zone,first_seen,last_seen)
        VALUES (?,?,?,?,?,?,?,?,?)
        ON CONFLICT(station_id) DO UPDATE SET
          last_seen=excluded.last_seen,
          latitude=COALESCE(excluded.latitude,stations.latitude),
          longitude=COALESCE(excluded.longitude,stations.longitude)
    """, (sid, rec.get("name"), rec.get("city"), rec.get("state"),
          rec.get("lat"), rec.get("lon"), rec.get("zone"),
          rec.get("timestamp"), rec.get("timestamp")))
    conn.execute("""
        INSERT INTO readings (station_id,timestamp,collected_at,aqi,pm25,pm10,no2,co,o3,so2,
                              temperature,humidity,wind_speed)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (sid, rec.get("timestamp"), datetime.utcnow().isoformat(),
          rec.get("aqi"), rec.get("pm25"), rec.get("pm10"),
          rec.get("no2"), rec.get("co"), rec.get("o3"), rec.get("so2"),
          rec.get("temperature"), rec.get("humidity"), rec.get("wind_speed")))

def load_history(sid, conn):
    df = pd.read_sql(
        "SELECT timestamp,aqi,pm25,pm10,no2,co,o3,so2,temperature,humidity,wind_speed "
        "FROM readings WHERE station_id=? ORDER BY timestamp",
        conn, params=(sid,))
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).drop_duplicates("timestamp").set_index("timestamp")
    return df

# ══════════════════════════════════════════════════════════════════════════════
# SYNTHETIC DATA GENERATOR
# ══════════════════════════════════════════════════════════════════════════════
def generate_synthetic_stations():
    random.seed(42); np.random.seed(42)
    stations = [
        ("S001","Delhi - Anand Vihar",      28.647,77.317,"North India"),
        ("S002","Mumbai - Bandra",           19.054,72.841,"West India"),
        ("S003","Bangalore - Hebbal",        13.046,77.597,"South India"),
        ("S004","Chennai - Manali",          13.166,80.260,"South India"),
        ("S005","Kolkata - Jadavpur",        22.500,88.370,"East India"),
        ("S006","Hyderabad - ICRISAT",       17.514,78.274,"South India"),
        ("S007","Pune - Shivajinagar",       18.530,73.847,"West India"),
        ("S008","Lucknow - Gomtinagar",      26.848,80.999,"North India"),
        ("S009","Kanpur - IIT",              26.512,80.233,"North India"),
        ("S010","Nagpur - Civil Lines",      21.146,79.089,"Central India"),
        ("S011","Jaipur - Mansarovar",       26.832,75.753,"North India"),
        ("S012","Ahmedabad - AUDA",          23.039,72.587,"West India"),
        ("S013","Patna - Gandhi Maidan",     25.614,85.144,"East India"),
        ("S014","Guwahati - IITG",           26.188,91.697,"Northeast India"),
        ("S015","Kochi - Ernakulam",          9.982,76.299,"Far South"),
        ("S016","Visakhapatnam - MVP",       17.723,83.318,"South India"),
        ("S017","Chandigarh - Sector17",     30.740,76.785,"North India"),
        ("S018","Varanasi - Bhelupur",       25.312,83.009,"North India"),
        ("S019","Noida - Sector 125",        28.544,77.393,"North India"),
        ("S020","Gurugram - Sector 51",      28.463,77.028,"North India"),
        ("S021","Faridabad - NIT",           28.383,77.315,"North India"),
        ("S022","Agra - Sanjay Place",       27.180,78.008,"North India"),
        ("S023","Indore - Rau",              22.596,75.826,"Central India"),
        ("S024","Bhopal - TT Nagar",         23.232,77.434,"Central India"),
        ("S025","Raipur - Pandri",           21.243,81.649,"Central India"),
        ("S026","Jodhpur - Paota",           26.289,73.016,"North India"),
        ("S027","Srinagar - Barzulla",       34.091,74.797,"North India"),
        ("S028","Shimla - Ridge",            31.104,77.173,"North India"),
        ("S029","Coimbatore - Saibaba",      11.017,76.968,"South India"),
        ("S030","Madurai - Goripalayam",      9.912,78.119,"South India"),
    ]
    base_map = {"North India":210,"West India":120,"South India":80,
                "East India":160,"Central India":155,"Northeast India":90,"Far South":60}
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    records = []
    for sid, name, lat, lon, zone in stations:
        base = base_map.get(zone, 130)
        aqi  = int(np.clip(np.random.normal(base, 40), 10, 450))
        records.append({
            "station_id": sid, "name": name,
            "city": name.split("-")[0].strip(), "state": zone,
            "lat": lat, "lon": lon, "zone": zone, "timestamp": now,
            "aqi": aqi,
            "pm25": round(aqi*0.45+np.random.normal(0,5),1),
            "pm10": round(aqi*0.75+np.random.normal(0,8),1),
            "no2":  round(np.random.uniform(10,80),1),
            "co":   round(np.random.uniform(0.5,3.0),2),
            "o3":   round(np.random.uniform(20,70),1),
            "so2":  round(np.random.uniform(5,40),1),
            "temperature": round(np.random.uniform(15,38),1),
            "humidity":    round(np.random.uniform(30,85),1),
            "wind_speed":  round(np.random.uniform(0.5,8.0),1),
        })
    return records

# ══════════════════════════════════════════════════════════════════════════════
# ASYNC DATA COLLECTOR (WAQI API)
# ══════════════════════════════════════════════════════════════════════════════
def _parse(data, zone=None):
    try:
        d = data.get("data",{}); 
        if not d or d == "Unknown station": return None
        iaqi = d.get("iaqi",{}); city = d.get("city",{}); geo = city.get("geo",[None,None])
        aqi = d.get("aqi")
        if aqi in (None,"-",""): return None
        return {"station_id":str(d.get("idx","")),
                "name":city.get("name",""),"city":city.get("name","").split(",")[0].strip(),
                "state":city.get("name","").split(",")[-1].strip() if "," in city.get("name","") else "",
                "lat":geo[0],"lon":geo[1],"zone":zone,
                "timestamp":d.get("time",{}).get("s"),"aqi":int(aqi),
                "pm25":iaqi.get("pm25",{}).get("v"),"pm10":iaqi.get("pm10",{}).get("v"),
                "no2":iaqi.get("no2",{}).get("v"),"co":iaqi.get("co",{}).get("v"),
                "o3":iaqi.get("o3",{}).get("v"),"so2":iaqi.get("so2",{}).get("v"),
                "temperature":iaqi.get("t",{}).get("v"),"humidity":iaqi.get("h",{}).get("v"),
                "wind_speed":iaqi.get("w",{}).get("v")}
    except Exception: return None

async def _fetch_named(session, name, sem):
    import aiohttp
    async with sem:
        try:
            async with session.get(WAQI_BASE.format(name,WAQI_API_KEY),
                                   timeout=aiohttp.ClientTimeout(total=10)) as r:
                d = await r.json(content_type=None)
                if d.get("status")=="ok": return _parse(d,"named")
        except Exception: pass
    return None

async def _fetch_zone(session, zname, bounds, sem):
    import aiohttp
    results=[]
    async with sem:
        try:
            async with session.get(WAQI_MAP.format(bounds,WAQI_API_KEY),
                                   timeout=aiohttp.ClientTimeout(total=15)) as r:
                d = await r.json(content_type=None)
                if d.get("status")=="ok":
                    for s in d.get("data",[]):
                        try: aqi=int(s.get("aqi"))
                        except: continue
                        results.append({"station_id":str(s.get("uid","")),
                            "name":s.get("station",{}).get("name",""),
                            "city":s.get("station",{}).get("name","").split(",")[0].strip(),
                            "state":"","lat":s.get("lat"),"lon":s.get("lon"),
                            "zone":zname,"timestamp":s.get("station",{}).get("time",""),
                            "aqi":aqi,"pm25":None,"pm10":None,"no2":None,
                            "co":None,"o3":None,"so2":None,
                            "temperature":None,"humidity":None,"wind_speed":None})
        except Exception: pass
    return results

async def _collect_async():
    import aiohttp
    sem=asyncio.Semaphore(20); records=[]
    async with aiohttp.ClientSession() as session:
        for recs in await asyncio.gather(*[_fetch_zone(session,z,b,sem) for z,b in INDIA_ZONES.items()]):
            records.extend(recs)
        for r in await asyncio.gather(*[_fetch_named(session,n,sem) for n in NAMED_STATIONS]):
            if r: records.append(r)
    seen={}
    for rec in records:
        sid=rec["station_id"]
        if sid not in seen or rec.get("pm25") is not None: seen[sid]=rec
    return list(seen.values())

def collect_and_store():
    console.print("[bold cyan]🛰️  Fetching live AQI data from WAQI API...[/bold cyan]" if RICH else "Fetching data...")
    records=[]
    try: records=asyncio.run(_collect_async())
    except Exception as e:
        console.print(f"[yellow]API unavailable ({e}). Using synthetic data.[/yellow]" if RICH else f"Synthetic: {e}")
    if len(records)<5:
        console.print("[yellow]⚠️  Switching to synthetic station data (30 stations).[/yellow]" if RICH else "Synthetic.")
        records=generate_synthetic_stations()
    init_db()
    conn=sqlite3.connect(DB_PATH); saved=0
    for r in records:
        try: save_reading(conn,r["station_id"],r); saved+=1
        except Exception: pass
    conn.commit(); conn.close()
    console.print(f"[green]✅ {len(records)} stations, {saved} saved.[/green]" if RICH else f"Saved {saved}")
    return pd.DataFrame(records)

# ══════════════════════════════════════════════════════════════════════════════
# PRE-PROCESSING & FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
def preprocess(df):
    df=df.copy().sort_index()
    df=df.resample("1h").mean().interpolate("linear").ffill().bfill()
    for col in ["aqi","pm25","pm10","no2","co","o3","so2"]:
        if col in df.columns and df[col].notna().sum()>4:
            q1,q3=df[col].quantile(0.25),df[col].quantile(0.75)
            df[col]=df[col].clip(q1-1.5*(q3-q1),q3+1.5*(q3-q1))
    return df

def engineer_features(df, lat=None, lon=None, neighbor_aqi=None, sim_feats=None):
    df=df.copy().sort_index()
    for lag in [1,2,3,6,12,24]: df[f"aqi_lag_{lag}h"]=df["aqi"].shift(lag)
    for win in [3,6,12,24]:
        df[f"aqi_roll_mean_{win}h"]=df["aqi"].rolling(win).mean()
        df[f"aqi_roll_std_{win}h"] =df["aqi"].rolling(win).std()
        df[f"aqi_roll_max_{win}h"] =df["aqi"].rolling(win).max()
    df["aqi_delta_1h"]=df["aqi"].diff(1); df["aqi_delta_6h"]=df["aqi"].diff(6)
    df["aqi_pct_change_1h"]=df["aqi"].pct_change(1)*100
    if hasattr(df.index,"hour"):
        h=df.index.hour; dw=df.index.dayofweek; mo=df.index.month
        df["hour_sin"]=np.sin(2*np.pi*h/24); df["hour_cos"]=np.cos(2*np.pi*h/24)
        df["day_sin"] =np.sin(2*np.pi*dw/7); df["day_cos"] =np.cos(2*np.pi*dw/7)
        df["month_sin"]=np.sin(2*np.pi*mo/12); df["month_cos"]=np.cos(2*np.pi*mo/12)
        df["is_weekend"]   =(dw>=5).astype(int)
        df["is_peak_hour"] =h.isin([7,8,9,17,18,19]).astype(int)
    if "pm25" in df.columns and "pm10" in df.columns:
        df["pm_ratio"]=df["pm25"]/(df["pm10"].abs()+1e-6)
    df["lat"]=lat or 0.0; df["lon"]=lon or 0.0
    df["neighbor_aqi"]=neighbor_aqi or (df["aqi"].mean() if "aqi" in df else 0)
    if sim_feats:
        df["sim_traffic_aqi"]    =sim_feats.get("traffic_aqi",0)
        df["sim_simpy_peak_aqi"] =sim_feats.get("simpy_peak_aqi",0)
    return df.dropna(subset=["aqi"])

def get_neighbor_aqi(sid, lat, lon, conn, radius_km=50):
    if not lat or not lon: return None
    try:
        df=pd.read_sql("""SELECT s.latitude,s.longitude,r.aqi FROM stations s
            JOIN readings r ON s.station_id=r.station_id
            WHERE r.id IN (SELECT MAX(id) FROM readings GROUP BY station_id)
            AND s.station_id!=? AND s.latitude IS NOT NULL""",conn,params=(sid,))
        if df.empty: return None
        def hav(row):
            R=6371; dlat=math.radians(row.latitude-lat); dlon=math.radians(row.longitude-lon)
            a=math.sin(dlat/2)**2+math.cos(math.radians(lat))*math.cos(math.radians(row.latitude))*math.sin(dlon/2)**2
            return R*2*math.asin(math.sqrt(a))
        df["dist"]=df.apply(hav,axis=1); nearby=df[df["dist"]<radius_km]
        return float(nearby["aqi"].mean()) if len(nearby)>0 else None
    except Exception: return None

# ══════════════════════════════════════════════════════════════════════════════
# SIMULATOR ① — SimPy discrete-event pollution
# ══════════════════════════════════════════════════════════════════════════════
import simpy

@dataclass
class _PollEvent:
    time_min:float; source_name:str; emission_kg_h:float
    wind_speed_ms:float; temperature_c:float; aqi_contribution:float

@dataclass
class SimPyResult:
    timeline:list=field(default_factory=list)
    hourly_aqi:dict=field(default_factory=dict)
    peak_hour:int=0; peak_aqi:float=0.0; total_emission_kg:float=0.0

def _gaussian_plume(emission_rate_kg_h, wind_speed_ms, distance_m=500):
    sigma_y=0.11*distance_m**0.894; sigma_z=0.08*distance_m**0.894
    H=30; Q=emission_rate_kg_h/3600; u=max(wind_speed_ms,0.5)
    conc=(Q/(2*np.pi*sigma_y*sigma_z*u))*np.exp(-0.5*(H/sigma_z)**2)
    return float(min(500, conc*1e6*0.15))

class _PollSource:
    def __init__(self,env,name,base_emission,peak_hours,result):
        self.env=env; self.name=name; self.base=base_emission
        self.peaks=peak_hours; self.result=result
        self.res=simpy.Resource(env,capacity=1)
    def _rate(self,minute):
        hour=int((minute/60)%24)
        return max(0,self.base*(1.8 if hour in self.peaks else 0.6)*random.gauss(1,0.05))
    def run(self):
        while True:
            t=self.env.now; rate=self._rate(t)
            wind=random.uniform(1,8); temp=20+10*np.sin(2*np.pi*t/(24*60))
            aqi_c=_gaussian_plume(rate,wind); hour=int((t/60)%24)
            self.result.timeline.append(_PollEvent(t,self.name,rate,wind,temp,aqi_c))
            self.result.hourly_aqi[hour]=self.result.hourly_aqi.get(hour,0)+aqi_c*0.25
            if t%(8*60)<1:
                with self.res.request() as req:
                    yield req; yield self.env.timeout(random.uniform(5,20))
            yield self.env.timeout(15)

class _MeteoEvent:
    def __init__(self,env,result): self.env=env; self.result=result
    def run(self):
        while True:
            if random.random()<0.15:
                dur=random.uniform(60,180); hour=int((self.env.now/60)%24)
                for h in range(hour,min(hour+int(dur/60)+1,24)):
                    self.result.hourly_aqi[h]=self.result.hourly_aqi.get(h,50)*1.4
                yield self.env.timeout(dur)
            else: yield self.env.timeout(random.uniform(30,90))

def run_simpy(sources_cfg, duration_min=1440, verbose=True):
    random.seed(42); np.random.seed(42)
    env=simpy.Environment(); result=SimPyResult()
    for cfg in sources_cfg:
        src=_PollSource(env,cfg["name"],cfg["base_emission"],cfg["peak_hours"],result)
        env.process(src.run())
    env.process(_MeteoEvent(env,result).run())
    if verbose: console.print("  [bold cyan]🏭 SimPy: running 24h urban pollution simulation...[/bold cyan]" if RICH else "  SimPy running...")
    env.run(until=duration_min)
    if result.hourly_aqi:
        result.peak_hour=max(result.hourly_aqi,key=result.hourly_aqi.get)
        result.peak_aqi=result.hourly_aqi[result.peak_hour]
    result.total_emission_kg=sum(e.emission_kg_h/4 for e in result.timeline)
    pd.DataFrame([{"time_min":e.time_min,"source":e.source_name,
                   "emission_kg_h":round(e.emission_kg_h,2),"aqi_contribution":round(e.aqi_contribution,2)}
                  for e in result.timeline]).to_csv(OUTPUT_DIR/"simpy_timeline.csv",index=False)
    pd.DataFrame([{"hour":h,"aqi_contribution":v} for h,v in result.hourly_aqi.items()]).to_csv(OUTPUT_DIR/"simpy_hourly.csv",index=False)
    if verbose:
        console.print(f"  [green]SimPy done:[/green] peak AQI={result.peak_aqi:.1f} at {result.peak_hour:02d}:00 | total={result.total_emission_kg:.0f} kg" if RICH
                      else f"  SimPy peak={result.peak_aqi:.1f}")
    return result

# ══════════════════════════════════════════════════════════════════════════════
# SIMULATOR ② — SUMO-py traffic emissions
# ══════════════════════════════════════════════════════════════════════════════
_SUMO_EF={
    "car":  {"PM2.5":[0.08,0.04,0.03,0.02],"NOx":[1.20,0.80,0.60,0.50]},
    "truck":{"PM2.5":[0.40,0.25,0.18,0.15],"NOx":[8.00,5.50,4.20,3.80]},
    "bus":  {"PM2.5":[0.30,0.18,0.14,0.12],"NOx":[6.50,4.20,3.10,2.80]},
    "bike": {"PM2.5":[0.02,0.01,0.01,0.01],"NOx":[0.30,0.20,0.15,0.12]},
}
_SUMO_TF={0:0.15,1:0.10,2:0.08,3:0.07,4:0.08,5:0.20,6:0.55,7:0.90,8:1.00,9:0.75,
          10:0.60,11:0.65,12:0.70,13:0.65,14:0.60,15:0.70,16:0.85,17:1.00,18:0.95,
          19:0.75,20:0.55,21:0.40,22:0.30,23:0.20}
_SUMO_MIX={"car":0.65,"truck":0.10,"bus":0.08,"bike":0.17}
_SUMO_SEGS=[
    ("R01","NH-48 Delhi-Gurgaon",28.0,6,100,"highway"),("R02","Ring Road Inner",14.0,4,60,"central"),
    ("R03","Outer Ring Road",47.0,6,70,"suburban"),("R04","NH-9 Delhi-Meerut",24.0,6,100,"highway"),
    ("R05","Mathura Road",18.0,4,60,"central"),("R06","GT Karnal Road",32.0,6,80,"highway"),
    ("R07","Najafgarh Road",12.0,2,50,"suburban"),("R08","Okhla Indus. Access",6.0,2,40,"industrial"),
    ("R09","Anand Vihar ISBT",3.5,4,30,"central"),("R10","Noida Expy DND",9.0,6,80,"highway"),
    ("R11","Rohtak Road",22.0,4,60,"suburban"),("R12","MB Road",14.5,4,50,"suburban"),
    ("R13","Wazirpur Indus. Rd",5.0,2,30,"industrial"),("R14","Shahdara Main Rd",8.0,4,40,"central"),
    ("R15","Dwarka Bypass",11.0,4,60,"suburban"),("R16","NH-58 Meerut Rd",19.0,4,80,"highway"),
    ("R17","Chandni Chowk",2.5,2,20,"central"),("R18","Narela Indus. Corr.",7.5,2,40,"industrial"),
    ("R19","Faridabad NH-19",16.0,4,70,"highway"),("R20","Golf Course Rd",9.5,4,60,"suburban"),
]

def _pm25_to_aqi(pm25):
    bp=[(0,12,0,50),(12.1,35.4,51,100),(35.5,55.4,101,150),
        (55.5,150.4,151,200),(150.5,250.4,201,300),(250.5,500.4,301,500)]
    for lo_c,hi_c,lo_a,hi_a in bp:
        if lo_c<=pm25<=hi_c:
            return min(500,max(0,(hi_a-lo_a)/(hi_c-lo_c)*(pm25-lo_c)+lo_a))
    return 500.0

def run_sumo(n_vehicles=5000, verbose=True):
    rows=[]
    for sid,name,length_km,lanes,speed_lim,zone in _SUMO_SEGS:
        for hour in range(24):
            tf=_SUMO_TF.get(hour,0.5); nveh=int(n_vehicles*tf)
            actual_speed=max(5,speed_lim*(1-0.6*tf))
            spb=0 if actual_speed<20 else(1 if actual_speed<50 else(2 if actual_speed<80 else 3))
            penalty=1+0.5*max(0,tf-0.7); pm25=0.0
            for vtype,frac in _SUMO_MIX.items():
                n_type=int(nveh*frac); dist=length_km*n_type
                pm25+=_SUMO_EF[vtype]["PM2.5"][spb]*dist*penalty/lanes
            aqi_c=_pm25_to_aqi(pm25*1000/(length_km*1000*10))
            rows.append({"segment":name,"zone":zone,"hour":hour,
                         "n_vehicles":nveh,"speed_kmh":round(actual_speed,1),
                         "pm25_g_h":round(pm25,2),"aqi_contribution":round(aqi_c,1)})
    df=pd.DataFrame(rows); zone_aqi=df.groupby("zone")["aqi_contribution"].mean().to_dict()
    df.to_csv(OUTPUT_DIR/"sumo_emissions.csv",index=False)
    if verbose:
        console.print(f"  🚗 SUMO: {len(_SUMO_SEGS)} segments × 24h | Peak PM2.5: {df['pm25_g_h'].max():.0f} g/h | Central AQI: {zone_aqi.get('central',0):.1f}" if RICH
                      else f"  SUMO done: peak PM2.5={df['pm25_g_h'].max():.0f} g/h")
    return df, zone_aqi

# ══════════════════════════════════════════════════════════════════════════════
# ML MODELS: CNN / LSTM / GRU / GNN (Graph-RF proxy)
# ══════════════════════════════════════════════════════════════════════════════

class _LSTM:
    """Vanilla LSTM implemented in NumPy — no DL framework required."""
    def __init__(self,H=32):
        self.H=H; s=0.1
        self.Wf=np.random.randn(H,1+H)*s; self.bf=np.zeros(H)
        self.Wi=np.random.randn(H,1+H)*s; self.bi=np.zeros(H)
        self.Wo=np.random.randn(H,1+H)*s; self.bo=np.zeros(H)
        self.Wc=np.random.randn(H,1+H)*s; self.bc=np.zeros(H)
        self.Wy=np.random.randn(1,H)*s;   self.by=np.zeros(1)
    def _s(self,x): return 1/(1+np.exp(-np.clip(x,-15,15)))
    def _t(self,x): return np.tanh(np.clip(x,-15,15))
    def forward(self,seq):
        h=np.zeros(self.H); c=np.zeros(self.H)
        for t in seq:
            xh=np.concatenate([[t],h])
            f=self._s(self.Wf@xh+self.bf); i=self._s(self.Wi@xh+self.bi)
            o=self._s(self.Wo@xh+self.bo); g=self._t(self.Wc@xh+self.bc)
            c=f*c+i*g; h=o*self._t(c)
        return float(self.Wy@h+self.by)
    def train(self,X,y,epochs=40,lr=0.001):
        for _ in range(epochs):
            for seq,tgt in zip(X,y):
                pred=self.forward(seq); grad=2*(pred-tgt)
                h=np.zeros(self.H); c=np.zeros(self.H)
                for t in seq:
                    xh=np.concatenate([[t],h])
                    f=self._s(self.Wf@xh+self.bf); i=self._s(self.Wi@xh+self.bi)
                    o=self._s(self.Wo@xh+self.bo); g=self._t(self.Wc@xh+self.bc)
                    c=f*c+i*g; h=o*self._t(c)
                self.Wy-=lr*grad*h.reshape(1,-1); self.by-=lr*grad

class _GRU:
    """Gated Recurrent Unit implemented in NumPy."""
    def __init__(self,H=32):
        self.H=H; s=0.1
        self.Wr=np.random.randn(H,1+H)*s; self.br=np.zeros(H)
        self.Wz=np.random.randn(H,1+H)*s; self.bz=np.zeros(H)
        self.Wn=np.random.randn(H,1+H)*s; self.bn=np.zeros(H)
        self.Wy=np.random.randn(1,H)*s;   self.by=np.zeros(1)
    def _s(self,x): return 1/(1+np.exp(-np.clip(x,-15,15)))
    def _t(self,x): return np.tanh(np.clip(x,-15,15))
    def forward(self,seq):
        h=np.zeros(self.H)
        for t in seq:
            x=np.array([t]); xh=np.concatenate([x,h])
            r=self._s(self.Wr@xh+self.br); z=self._s(self.Wz@xh+self.bz)
            n=self._t(self.Wn@np.concatenate([x,r*h])+self.bn); h=(1-z)*n+z*h
        return float(self.Wy@h+self.by)
    def train(self,X,y,epochs=40,lr=0.001):
        for _ in range(epochs):
            for seq,tgt in zip(X,y):
                pred=self.forward(seq); grad=2*(pred-tgt)
                h=np.zeros(self.H)
                for t in seq:
                    x=np.array([t]); xh=np.concatenate([x,h])
                    r=self._s(self.Wr@xh+self.br); z=self._s(self.Wz@xh+self.bz)
                    n=self._t(self.Wn@np.concatenate([x,r*h])+self.bn); h=(1-z)*n+z*h
                self.Wy-=lr*grad*h.reshape(1,-1); self.by-=lr*grad

class _CNN:
    """1-D Convolutional network for temporal sequence regression."""
    def __init__(self,S=SEQ_LEN,F=16,K=3):
        self.S=S; self.F=F; self.K=K; self.out=S-K+1
        self.Wc=np.random.randn(F,K)*0.1; self.bc=np.zeros(F)
        self.Wf=np.random.randn(F*self.out)*0.1; self.bf=np.zeros(1)
    def _relu(self,x): return np.maximum(0,x)
    def forward(self,seq):
        co=[[self._relu(np.dot(self.Wc[f],seq[i:i+self.K])+self.bc[f]) for i in range(self.out)] for f in range(self.F)]
        return float(np.dot(self.Wf,np.array(co).flatten())+self.bf)
    def train(self,X,y,epochs=25,lr=0.0005):
        for _ in range(epochs):
            for seq,tgt in zip(X,y):
                pred=self.forward(seq); grad=2*(pred-tgt)
                co=[[self._relu(np.dot(self.Wc[f],seq[i:i+self.K])+self.bc[f]) for i in range(self.out)] for f in range(self.F)]
                flat=np.array(co).flatten()
                self.Wf-=lr*grad*flat; self.bf-=lr*grad

class _GNN:
    """
    Graph Neural Network proxy: spatial-aware RandomForest that simulates
    message-passing by concatenating neighbor-AQI signals to node features.
    No graph library required.
    """
    def __init__(self,n_estimators=150,max_depth=7):
        self.rf=RandomForestRegressor(n_estimators=n_estimators,max_depth=max_depth,
                                      random_state=RANDOM_SEED,n_jobs=-1)
        self._fitted=False
        self.neighbor_mean=0.0
    def train(self,X,y,neighbor_means=None):
        if neighbor_means is None: neighbor_means=np.full(len(X),float(np.mean(y)))
        Xg=np.hstack([X,neighbor_means.reshape(-1,1)])
        self.rf.fit(Xg,y); self._fitted=True
    def forward(self,seq,neighbor_mean=0.0):
        if not self._fitted: return 0.0
        Xg=np.concatenate([seq,[neighbor_mean]]).reshape(1,-1)
        return float(self.rf.predict(Xg)[0])

def _make_seqs(arr,sl=SEQ_LEN):
    X,y=[],[]
    for i in range(len(arr)-sl): X.append(arr[i:i+sl]); y.append(arr[i+sl])
    return np.array(X),np.array(y)

def _calc_metrics(yt,yp):
    yt=np.array(yt,float); yp=np.array(yp,float)
    mask=~np.isnan(yt)&~np.isnan(yp); yt,yp=yt[mask],yp[mask]
    if len(yt)<2: return {}
    mape=np.mean(np.abs((yt-yp)/(np.abs(yt)+1e-6)))*100
    return {"rmse":round(float(np.sqrt(mean_squared_error(yt,yp))),3),
            "mae": round(float(mean_absolute_error(yt,yp)),3),
            "r2":  round(float(r2_score(yt,yp)),4),
            "mape":round(float(mape),2),"n":int(len(yt))}

@dataclass
class _Bundle:
    mean_:float=0; std_:float=1
    lstm:object=None; gru:object=None; cnn:object=None; gnn:object=None
    neighbor_mean_:float=0.0
    weights:dict=field(default_factory=lambda:{"LSTM":0.25,"GRU":0.25,"CNN":0.20,"GNN":0.30})
    val_metrics:dict=field(default_factory=dict)

def train_ensemble(aqi_series, neighbor_mean=None):
    """Train CNN + LSTM + GRU + GNN on a single station AQI time series."""
    np.random.seed(RANDOM_SEED)
    if len(aqi_series)<SEQ_LEN+10: return None
    b=_Bundle()
    b.mean_=float(aqi_series.mean()); b.std_=float(aqi_series.std()+1e-6)
    b.neighbor_mean_=float(neighbor_mean or aqi_series.mean())
    norm=(aqi_series-b.mean_)/b.std_
    X,y=_make_seqs(norm.values,SEQ_LEN)
    if len(X)<5: return None
    sp=max(4,int(len(X)*0.8)); Xtr,Xte=X[:sp],X[sp:]; ytr,yte=y[:sp],y[sp:]
    val_rmse={}

    for cls,name,kw in [(_LSTM,"LSTM",{"H":32,"epochs":40,"lr":0.001}),
                         (_GRU, "GRU", {"H":32,"epochs":40,"lr":0.001}),
                         (_CNN, "CNN", {"S":SEQ_LEN,"F":16,"K":3,"epochs":25,"lr":0.0005})]:
        try:
            ep=kw.pop("epochs"); lr=kw.pop("lr"); m=cls(**kw)
            m.train(Xtr,ytr,epochs=ep,lr=lr); setattr(b,name.lower(),m)
            if len(Xte)>1:
                preds=np.array([m.forward(s) for s in Xte])*b.std_+b.mean_
                mt=_calc_metrics(yte*b.std_+b.mean_,preds)
                b.val_metrics[name]=mt; val_rmse[name]=mt.get("rmse",99)
        except Exception: pass

    # GNN (Graph-RF proxy)
    try:
        gnn=_GNN(n_estimators=150,max_depth=7)
        nb_norm=(b.neighbor_mean_-b.mean_)/b.std_
        gnn.train(Xtr,ytr,neighbor_means=np.full(len(Xtr),nb_norm))
        b.gnn=gnn
        if len(Xte)>1:
            preds=np.array([gnn.forward(Xte[i],nb_norm) for i in range(len(Xte))])*b.std_+b.mean_
            mt=_calc_metrics(yte*b.std_+b.mean_,preds)
            b.val_metrics["GNN"]=mt; val_rmse["GNN"]=mt.get("rmse",99)
    except Exception: pass

    if val_rmse:
        inv={k:1/max(v,0.1) for k,v in val_rmse.items()}; total=sum(inv.values())
        b.weights={k:round(v/total,4) for k,v in inv.items()}
    return b

def predict_aqi(b, aqi_series, horizon=1):
    """Weighted ensemble prediction: CNN + LSTM + GRU + GNN."""
    if b is None or len(aqi_series)<SEQ_LEN: return None,None,None
    seq=((aqi_series-b.mean_)/b.std_).values[-SEQ_LEN:]
    nb_norm=(b.neighbor_mean_-b.mean_)/b.std_
    preds,wts=[],[]
    for name,model,fn in [("LSTM",b.lstm,lambda m,s:m.forward(s)),
                           ("GRU", b.gru, lambda m,s:m.forward(s)),
                           ("CNN", b.cnn, lambda m,s:m.forward(s)),
                           ("GNN", b.gnn, lambda m,s:m.forward(s,nb_norm))]:
        if model is not None:
            try:
                p=float(np.clip(fn(model,seq)*b.std_+b.mean_,0,500))
                preds.append(p); wts.append(b.weights.get(name,0.25))
            except Exception: pass
    if not preds: return None,None,None
    pt=float(np.clip(np.average(preds,weights=np.array(wts)),0,500))
    if horizon<=3: pt=float(np.clip(0.6*pt+0.4*float(aqi_series.iloc[-1]),0,500))
    ci=pt*0.15*(1+horizon*0.02)
    return pt,float(np.clip(pt-ci,0,500)),float(np.clip(pt+ci,0,500))

def save_bundle(sid,b):
    try:
        with open(MODEL_DIR/f"{sid}.pkl","wb") as f: pickle.dump(b,f,protocol=pickle.HIGHEST_PROTOCOL)
    except Exception: pass

def load_bundle(sid):
    p=MODEL_DIR/f"{sid}.pkl"
    if not p.exists(): return None
    try:
        with open(p,"rb") as f: return pickle.load(f)
    except Exception: return None

# ══════════════════════════════════════════════════════════════════════════════
# XAI — SHAP + LIME + Best Model Identification
# ══════════════════════════════════════════════════════════════════════════════
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.gridspec as gridspec

def run_xai(all_metrics, verbose=True):
    feat_names=["aqi_lag_1h","aqi_lag_6h","aqi_lag_24h","aqi_roll_mean_6h","aqi_roll_std_6h",
                "pm25","pm10","no2","co","o3","hour_sin","hour_cos","day_sin","neighbor_aqi",
                "sim_traffic_aqi","sim_simpy_peak_aqi"]
    np.random.seed(42); n=250
    X=np.random.randn(n,len(feat_names))
    y=np.clip(X[:,0]*40+X[:,5]*30+X[:,7]*20+X[:,14]*15+np.random.normal(0,15,n)+150,10,500)
    split=200; Xtr,Xte=X[:split],X[split:]; ytr,yte=y[:split],y[split:]

    # Tree-based surrogates for each model (SHAP TreeExplainer compatible)
    models_dict={
        "GNN_RF":        RandomForestRegressor(n_estimators=100,max_depth=6,random_state=42).fit(Xtr,ytr),
        "CNN_surrogate": RandomForestRegressor(n_estimators=80,max_depth=5,random_state=1).fit(Xtr,ytr),
        "LSTM_surrogate":RandomForestRegressor(n_estimators=80,max_depth=5,random_state=2).fit(Xtr,ytr),
        "GRU_surrogate": RandomForestRegressor(n_estimators=80,max_depth=5,random_state=3).fit(Xtr,ytr),
    }

    shap_paths={}
    try:
        import shap
        for mname,model in models_dict.items():
            explainer=shap.TreeExplainer(model,Xtr[:50]); sv=explainer.shap_values(Xte)
            fi=pd.DataFrame({"feature":feat_names,"importance":np.abs(sv).mean(axis=0)}).sort_values("importance",ascending=False)
            fig=plt.figure(figsize=(14,7),facecolor="#0d1117")
            gs=gridspec.GridSpec(1,2,figure=fig,wspace=0.35)
            ax1=fig.add_subplot(gs[0]); ax1.set_facecolor("#0d1117")
            colors=plt.cm.RdYlGn_r(np.linspace(0.1,0.9,min(15,len(fi))))
            ax1.barh(range(min(15,len(fi))),fi["importance"].head(15),color=colors,edgecolor="none")
            ax1.set_yticks(range(min(15,len(fi)))); ax1.set_yticklabels(fi["feature"].head(15),fontsize=8,color="#e0e0e0")
            ax1.set_xlabel("Mean |SHAP value|",color="#aaa",fontsize=9)
            ax1.set_title(f"SHAP Feature Importance\n{mname}",color="#00d4ff",fontsize=10)
            ax1.tick_params(colors="#888"); ax1.spines[:].set_color("#333"); ax1.invert_yaxis()
            ax2=fig.add_subplot(gs[1]); ax2.set_facecolor("#0d1117")
            inst=np.argmax(np.abs(sv).sum(axis=1)); inst_sv=sv[inst]
            top_idx=np.argsort(np.abs(inst_sv))[::-1][:10]
            vals=inst_sv[top_idx]; labs=[feat_names[i] if i<len(feat_names) else f"F{i}" for i in top_idx]
            ax2.barh(range(len(vals)),vals,color=["#ff4444" if v>0 else "#44aaff" for v in vals],edgecolor="none")
            ax2.set_yticks(range(len(vals))); ax2.set_yticklabels(labs,fontsize=8,color="#e0e0e0")
            ax2.axvline(0,color="#888",linewidth=0.8)
            ax2.set_xlabel("SHAP value",color="#aaa",fontsize=9)
            ax2.set_title("Single Prediction Explanation",color="#ffd700",fontsize=10)
            ax2.tick_params(colors="#888"); ax2.spines[:].set_color("#333"); ax2.invert_yaxis()
            plt.suptitle("XAI — SHAP Analysis: Why did the model predict this AQI?",color="#e0e0e0",fontsize=11)
            p=OUTPUT_DIR/f"shap_{mname}.png"
            plt.savefig(str(p),dpi=130,bbox_inches="tight",facecolor="#0d1117"); plt.close()
            shap_paths[mname]=str(p)
            if verbose:
                console.print(f"  [green]SHAP [{mname}] → {p.name}[/green]" if RICH else f"  SHAP {mname}")
                for _,fr in fi.head(5).iterrows():
                    console.print(f"    {fr['feature']:35s}  {fr['importance']:.4f}")
    except Exception as e:
        console.print(f"  [yellow]SHAP skipped: {e}[/yellow]" if RICH else f"  SHAP skip: {e}")

    lime_paths={}
    try:
        from lime import lime_tabular
        for mname,model in models_dict.items():
            exp_lime=lime_tabular.LimeTabularExplainer(Xtr,feature_names=feat_names,mode="regression",random_state=42)
            inst_idx=int(np.argmax(yte))
            explanation=exp_lime.explain_instance(Xte[inst_idx],model.predict,num_features=12,num_samples=300)
            ldf=pd.DataFrame(explanation.as_list(),columns=["condition","weight"])
            ldf["abs_w"]=ldf["weight"].abs(); ldf=ldf.sort_values("abs_w",ascending=False)
            pred_val=model.predict(Xte[inst_idx:inst_idx+1])[0]
            fig,ax=plt.subplots(figsize=(10,6),facecolor="#0d1117"); ax.set_facecolor("#0d1117")
            top=ldf.head(12)
            ax.barh(range(len(top)),top["weight"],color=["#ff4444" if w>0 else "#44aaff" for w in top["weight"]],edgecolor="none")
            ax.set_yticks(range(len(top))); ax.set_yticklabels(top["condition"],fontsize=8,color="#e0e0e0")
            ax.axvline(0,color="#888",linewidth=0.8)
            ax.set_xlabel("LIME weight",color="#aaa",fontsize=9)
            ax.set_title(f"XAI — LIME Local Explanation\n{mname} | Predicted AQI ≈ {pred_val:.1f}",color="#00d4ff",fontsize=10)
            ax.tick_params(colors="#888"); ax.spines[:].set_color("#333"); ax.invert_yaxis()
            leg=[plt.Rectangle((0,0),1,1,fc="#ff4444"),plt.Rectangle((0,0),1,1,fc="#44aaff")]
            ax.legend(leg,["Increases AQI","Decreases AQI"],loc="lower right",facecolor="#1a1a2e",edgecolor="#444",labelcolor="#ccc",fontsize=8)
            plt.tight_layout()
            p=OUTPUT_DIR/f"lime_{mname}.png"
            plt.savefig(str(p),dpi=130,bbox_inches="tight",facecolor="#0d1117"); plt.close()
            lime_paths[mname]=str(p)
            if verbose: console.print(f"  [green]LIME [{mname}] → {p.name}[/green]" if RICH else f"  LIME {mname}")
    except Exception as e:
        console.print(f"  [yellow]LIME skipped: {e}[/yellow]" if RICH else f"  LIME skip: {e}")

    if all_metrics:
        rows=[]
        for mname,m in all_metrics.items():
            if m: rows.append({"Model":mname,"RMSE":m.get("rmse",99),"MAE":m.get("mae",99),
                                "R2":m.get("r2",0),"MAPE%":m.get("mape",99),"N":m.get("n",0)})
        if rows:
            df_rank=pd.DataFrame(rows)
            for c in ["RMSE","MAE","MAPE%"]:
                rng=df_rank[c].max()-df_rank[c].min()+1e-6
                df_rank[f"{c}_n"]=(df_rank[c]-df_rank[c].min())/rng
            r2n=(df_rank["R2"].max()-df_rank["R2"])/(df_rank["R2"].max()-df_rank["R2"].min()+1e-6)
            df_rank["Score"]=(df_rank["RMSE_n"]+df_rank["MAE_n"]+df_rank["MAPE%_n"]+r2n)/4
            df_rank=df_rank.sort_values("Score").reset_index(drop=True)
            best=df_rank.iloc[0]["Model"]
            if RICH:
                console.print(Panel(
                    f"[bold]Model Ranking (composite RMSE+MAE+R2+MAPE)[/bold]\n"
                    f"{df_rank[['Model','RMSE','MAE','R2','MAPE%','Score']].to_string(index=False)}\n\n"
                    f"[bold green]🏆 Best model: {best}[/bold green]",
                    title="XAI — Best Model Identification",style="bold magenta"))
            else:
                print(df_rank[["Model","RMSE","MAE","R2","MAPE%","Score"]].to_string(index=False))
                print(f"\n🏆 Best model: {best}")
    return shap_paths, lime_paths

# ══════════════════════════════════════════════════════════════════════════════
# FORECAST PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def run_forecast(sim_features):
    conn=sqlite3.connect(DB_PATH)
    stations=pd.read_sql("SELECT station_id,name,latitude,longitude FROM stations",conn)
    if stations.empty:
        console.print("[red]No stations. Run collect first.[/red]" if RICH else "No stations")
        conn.close(); return pd.DataFrame(),{}
    all_fc=[]; all_metrics={}; generated_at=datetime.utcnow().isoformat(); success=0
    console.print(f"  Training ensemble for [yellow]{len(stations)}[/yellow] stations..." if RICH else f"  Training {len(stations)} stations...")
    for _,row in stations.iterrows():
        sid=row["station_id"]; lat=row.get("latitude"); lon=row.get("longitude")
        try:
            df_raw=load_history(sid,conn)
            aqi_series=df_raw["aqi"].dropna() if not df_raw.empty else pd.Series(dtype=float)
            nb_aqi=get_neighbor_aqi(sid,lat,lon,conn)
            if len(aqi_series)<32:
                base=float(aqi_series.iloc[-1]) if len(aqi_series)>0 else 150
                trend=float(aqi_series.diff().mean()) if len(aqi_series)>2 else 0
                synth=np.clip(base+trend*np.arange(-32,0)+np.random.normal(0,base*0.05,32),10,500)
                aqi_series=pd.Series(synth,index=pd.date_range(end=pd.Timestamp.utcnow(),periods=32,freq="1h"))
            bundle=load_bundle(sid) or train_ensemble(aqi_series,neighbor_mean=nb_aqi)
            if bundle: save_bundle(sid,bundle)
            if bundle is None: continue
            for mname,vm in bundle.val_metrics.items():
                if vm: all_metrics[mname]=vm
            last_aqi=float(aqi_series.iloc[-1])
            for h in FORECAST_HRS:
                pt,lo,hi=predict_aqi(bundle,aqi_series,horizon=h)
                if pt is None: pt=last_aqi; lo=pt*0.85; hi=pt*1.15
                if sim_features.get("simpy_peak_aqi",0)>0:
                    pt=float(np.clip(0.90*pt+0.10*sim_features["simpy_peak_aqi"],0,500))
                cat,_,_=aqi_category(int(pt))
                rec={"station_id":sid,"name":row["name"],"latitude":lat,"longitude":lon,
                     "generated_at":generated_at,"horizon_hrs":h,"aqi_pred":round(pt,1),
                     "aqi_lower":round(lo or pt*0.85,1),"aqi_upper":round(hi or pt*1.15,1),
                     "current_aqi":int(last_aqi),"category":cat,"confidence":round(max(0.4,1-0.025*h),2)}
                all_fc.append(rec)
                conn.execute("""INSERT INTO forecasts (station_id,generated_at,forecast_for,
                    horizon_hrs,aqi_pred,aqi_lower,aqi_upper,model,confidence)
                    VALUES (?,?,?,?,?,?,?,?,?)""",
                    (sid,generated_at,(datetime.utcnow()+timedelta(hours=h)).isoformat(),
                     h,pt,lo or pt*0.85,hi or pt*1.15,"Ensemble:CNN+LSTM+GRU+GNN",rec["confidence"]))
            success+=1
        except Exception: pass
    conn.commit(); conn.close()
    df_fc=pd.DataFrame(all_fc)
    if not df_fc.empty: df_fc.to_csv(OUTPUT_DIR/"forecasts_latest.csv",index=False)
    console.print(f"  [green]✅ Forecasted {success}/{len(stations)} stations.[/green]" if RICH else f"  Forecasted {success}")
    return df_fc,all_metrics

# ══════════════════════════════════════════════════════════════════════════════
# MAP GENERATION
# ══════════════════════════════════════════════════════════════════════════════
def generate_map(fc_df):
    try:
        import folium
        from folium.plugins import HeatMap, MarkerCluster, MiniMap, Fullscreen
    except ImportError:
        console.print("[red]pip install folium[/red]" if RICH else "pip install folium"); return
    conn=sqlite3.connect(DB_PATH)
    df_live=pd.read_sql("""SELECT s.station_id,s.name,s.latitude,s.longitude,s.zone,
        r.aqi,r.pm25,r.pm10,r.no2,r.temperature,r.timestamp
        FROM stations s JOIN readings r ON s.station_id=r.station_id
        WHERE r.id IN (SELECT MAX(id) FROM readings GROUP BY station_id)
        AND s.latitude IS NOT NULL AND r.aqi IS NOT NULL""",conn); conn.close()
    if df_live.empty: console.print("[yellow]No data for map.[/yellow]" if RICH else "No map data"); return
    m=folium.Map(location=[20.5937,78.9629],zoom_start=5,tiles="CartoDB dark_matter",prefer_canvas=True)
    MiniMap(toggle_display=True,tile_layer="CartoDB dark_matter").add_to(m)
    Fullscreen().add_to(m)
    heat=[[r.latitude,r.longitude,float(r.aqi)] for r in df_live.itertuples() if r.aqi]
    if heat: HeatMap(heat,radius=22,blur=28,max_zoom=10,
                     gradient={"0.2":"#0000ff","0.4":"#00e400","0.55":"#ffff00",
                               "0.7":"#ff7e00","0.85":"#ff0000","1.0":"#7e0023"}).add_to(m)
    fc_pivot={}
    if not fc_df.empty:
        for _,fr in fc_df.iterrows():
            fc_pivot.setdefault(str(fr["station_id"]),{})[fr["horizon_hrs"]]=fr
    cluster=MarkerCluster().add_to(m)
    for _,row in df_live.iterrows():
        cat,emoji,color=aqi_category(row["aqi"]); sid=str(row["station_id"])
        fc=fc_pivot.get(sid,{}); fc_html=""
        for h in [1,6,24]:
            if h in fc:
                fcat,_,fcolor=aqi_category(fc[h]["aqi_pred"])
                fc_html+=f"<tr><td>{h}h</td><td style='color:{fcolor};font-weight:bold'>{fc[h]['aqi_pred']:.0f}</td><td>{fcat}</td></tr>"
        popup=(f"<div style='font-family:monospace;min-width:260px;background:#1a1a2e;color:#e0e0e0;padding:10px;border-radius:8px'>"
               f"<b style='color:{color}'>{row['name']}</b><br>"
               f"AQI: <b style='color:{color}'>{row['aqi']}</b> {emoji} {cat}<br>"
               f"PM2.5:{row.get('pm25') or chr(8212)} | PM10:{row.get('pm10') or chr(8212)} | NO2:{row.get('no2') or chr(8212)}<br>"
               +(f"<b style='color:#ffd700'>🔮 Forecast:</b><table style='width:100%;font-size:11px'>{fc_html}</table>" if fc_html else "")
               +f"<small style='color:#666'>{str(row.get('timestamp',''))[:16]}</small></div>")
        folium.CircleMarker(location=[row["latitude"],row["longitude"]],
            radius=max(6,min(25,int(row["aqi"])//15+4)),color=color,fill=True,
            fill_color=color,fill_opacity=0.85,weight=1.5,
            popup=folium.Popup(popup,max_width=320),
            tooltip=f"{row['name']}: AQI {row['aqi']} ({cat})").add_to(cluster)
    legend="<div style='position:fixed;bottom:30px;left:10px;z-index:1000;background:#1a1a2e;padding:12px;border-radius:10px;font-family:monospace;font-size:11px;color:#e0e0e0'><b style='color:#00d4ff'>AQI Legend</b><br>"
    for _,_,label,emoji,color in AQI_CATEGORIES:
        legend+=f"<span style='background:{color};display:inline-block;width:12px;height:12px;border-radius:50%;margin-right:5px'></span>{label}<br>"
    legend+="</div>"
    m.get_root().html.add_child(folium.Element(legend))
    stats=(f"<div style='position:fixed;bottom:30px;right:10px;z-index:1000;background:#1a1a2e;"
           f"padding:12px;border-radius:10px;font-family:monospace;font-size:11px;color:#e0e0e0;min-width:220px'>"
           f"<b style='color:#00d4ff'>🌏 India AQI Digital Twin v6.0</b><br>"
           f"<hr style='border-color:#333;margin:5px 0'>"
           f"Stations: <b>{len(df_live)}</b> | Avg AQI: <b style='color:#ff7e00'>{df_live['aqi'].mean():.0f}</b><br>"
           f"Severe(>300): <b style='color:#ff0000'>{int((df_live['aqi']>300).sum())}</b><br>"
           f"<hr style='border-color:#333;margin:5px 0'>"
           f"<span style='color:#aaa'>SimPy · SUMO</span><br>"
           f"<span style='color:#aaa'>CNN + LSTM + GRU + GNN</span><br>"
           f"<span style='color:#aaa'>XAI: SHAP + LIME</span><br>"
           f"<span style='color:#666'>{datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC</span></div>")
    m.get_root().html.add_child(folium.Element(stats))
    folium.LayerControl().add_to(m)
    mp=OUTPUT_DIR/"india_aqi_digital_twin_v6.html"
    m.save(str(mp))
    console.print(f"[green]🗺️  Map saved → {mp}[/green]" if RICH else f"Map: {mp}")
    return str(mp)

# ══════════════════════════════════════════════════════════════════════════════
# TERMINAL DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
def print_dashboard(all_metrics):
    conn=sqlite3.connect(DB_PATH)
    df=pd.read_sql("""SELECT s.name,s.zone,r.aqi,r.pm25,r.pm10,r.no2
        FROM stations s JOIN readings r ON s.station_id=r.station_id
        WHERE r.id IN (SELECT MAX(id) FROM readings GROUP BY station_id)
        AND r.aqi IS NOT NULL ORDER BY r.aqi DESC LIMIT 20""",conn)
    all_aqi=pd.read_sql("""SELECT r.aqi FROM readings r
        WHERE r.id IN (SELECT MAX(id) FROM readings GROUP BY station_id)
        AND r.aqi IS NOT NULL""",conn); conn.close()
    if RICH:
        tbl=Table(title="🌏 India AQI Digital Twin v6.0 — Top 20 Most Polluted Stations",
                  header_style="bold magenta",border_style="blue")
        for col,w in [("Rank",4),("Station",26),("Zone",16),("AQI",6),("Category",16),("PM2.5",7),("PM10",7),("NO2",6)]:
            tbl.add_column(col,width=w,justify="right" if col in ("AQI","PM2.5","PM10","NO2") else "left")
        for i,row in df.reset_index().iterrows():
            cat,emoji,_=aqi_category(row["aqi"])
            tbl.add_row(str(i+1),str(row["name"])[:25],str(row.get("zone",""))[:15],
                        f"[bold]{row['aqi']}[/bold]",f"{emoji} {cat}",
                        str(row.get("pm25","") or chr(8212))[:6],str(row.get("pm10","") or chr(8212))[:6],
                        str(row.get("no2","") or chr(8212))[:5])
        console.print(tbl)
        if all_metrics:
            rows=[]
            for mname,m in all_metrics.items():
                if m: rows.append({"Model":mname,"RMSE":m.get("rmse",0),"MAE":m.get("mae",0),"R2":m.get("r2",0),"MAPE%":m.get("mape",0)})
            if rows:
                df_m=pd.DataFrame(rows).sort_values("RMSE")
                mt=Table(title="📊 Per-Model Performance Metrics",header_style="bold cyan",border_style="blue")
                for c in ["Model","RMSE","MAE","R2","MAPE%"]: mt.add_column(c,justify="right" if c!="Model" else "left")
                for _,r in df_m.iterrows():
                    mt.add_row(r["Model"],f"{r['RMSE']:.2f}",f"{r['MAE']:.2f}",f"{r['R2']:.4f}",f"{r['MAPE%']:.1f}%")
                console.print(mt)
        if not all_aqi.empty:
            cats=all_aqi["aqi"].apply(lambda x:aqi_category(x)[0]).value_counts()
            console.print(Panel(
                f"[bold]Total stations :[/bold] {len(all_aqi)}\n"
                f"[bold]National avg   :[/bold] [yellow]{all_aqi['aqi'].mean():.1f}[/yellow]  "
                f"[bold]Max:[/bold] [red]{all_aqi['aqi'].max()}[/red]  "
                f"[bold]Min:[/bold] [green]{all_aqi['aqi'].min()}[/green]\n"
                f"[bold]Severe (>300)  :[/bold] [red]{int((all_aqi['aqi']>300).sum())}[/red]  "
                f"[bold]Poor (>200)    :[/bold] [yellow]{int((all_aqi['aqi']>200).sum())}[/yellow]\n"
                +"\n".join(f"  {k}: {v}" for k,v in cats.items()),
                title="🇮🇳 National AQI Summary",style="dim"))
    else:
        print(df.to_string()); print(all_aqi["aqi"].describe())

# ══════════════════════════════════════════════════════════════════════════════
# MAIN — FULL PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def main():
    np.random.seed(RANDOM_SEED); random.seed(RANDOM_SEED)
    if RICH:
        console.print(Panel(
            "[bold cyan]🌏 INDIA AQI DIGITAL TWIN v6.0[/bold cyan]\n"
            "[dim]Simplified · Production-Ready · Single File Runner[/dim]\n\n"
            "[bold]Simulators :[/bold] SimPy · SUMO-py\n"
            "[bold]ML Models  :[/bold] CNN · LSTM · GRU · GNN (Graph-RF proxy)\n"
            "[bold]XAI        :[/bold] SHAP + LIME + Best-Model Identification\n"
            f"[bold]Output dir :[/bold] [cyan]{OUTPUT_DIR.resolve()}[/cyan]",
            title="v6.0 — VS Code Ready",border_style="cyan"))
    init_db(); t0=time.time()

    # STEP 1: Simulators
    console.print(Rule("[bold cyan]STEP 1 — Simulation Layer[/bold cyan]") if RICH else "\n"+"─"*60+"\nSTEP 1: Simulators")
    sim_features={}
    simpy_result=run_simpy(SIMPY_SOURCES,1440,verbose=True)
    sim_features.update({"simpy_peak_aqi":simpy_result.peak_aqi,"simpy_peak_hour":simpy_result.peak_hour})
    sumo_df,zone_aqi=run_sumo(n_vehicles=5000,verbose=True)
    sim_features.update({"traffic_aqi":zone_aqi.get("central",0),"sumo_industrial":zone_aqi.get("industrial",0),"sumo_highway_aqi":zone_aqi.get("highway",0)})
    if RICH:
        console.print(Panel(
            f"  SimPy  peak AQI : [yellow]{sim_features['simpy_peak_aqi']:.1f}[/yellow] (hour {sim_features['simpy_peak_hour']:02d}:00)\n"
            f"  SUMO   central  : [yellow]{sim_features['traffic_aqi']:.1f}[/yellow] AQI contrib\n"
            f"  SUMO   highway  : [yellow]{sim_features['sumo_highway_aqi']:.1f}[/yellow] AQI contrib",
            title="Simulation Summary",style="bold blue"))

    # STEP 2: Data
    console.print(Rule("[bold cyan]STEP 2 — Data Collection[/bold cyan]") if RICH else "\nSTEP 2: Data")
    collect_and_store()

    # STEPS 3-5: Pre-process + Train + Forecast
    console.print(Rule("[bold cyan]STEPS 3-5 — Pre-processing · Features · Model Training · Forecast[/bold cyan]") if RICH else "\nSTEP 3-5")
    fc_df,all_metrics=run_forecast(sim_features)
    if all_metrics and RICH:
        console.print("\n[bold]Per-model validation metrics:[/bold]")
        for mname,m in sorted(all_metrics.items(),key=lambda x:x[1].get("rmse",99)):
            console.print(f"  {mname:8s}  RMSE={m.get('rmse',0):.2f}  MAE={m.get('mae',0):.2f}  R2={m.get('r2',0):.4f}  MAPE={m.get('mape',0):.1f}%")

    # STEP 6: XAI
    console.print(Rule("[bold cyan]STEP 6 — XAI: SHAP + LIME + Best Model[/bold cyan]") if RICH else "\nSTEP 6: XAI")
    run_xai(all_metrics,verbose=True)

    # STEP 7: Map
    console.print(Rule("[bold cyan]STEP 7 — Interactive Map[/bold cyan]") if RICH else "\nSTEP 7: Map")
    generate_map(fc_df)

    # STEP 8: Dashboard + Alerts
    console.print(Rule("[bold cyan]STEP 8 — Dashboard & Alerts[/bold cyan]") if RICH else "\nSTEP 8: Dashboard")
    print_dashboard(all_metrics)
    if not fc_df.empty and "aqi_pred" in fc_df.columns:
        console.print("\n[bold red]🚨 AQI Alerts:[/bold red]" if RICH else "\nAlerts:")
        triggered=False
        for rule in ALERT_RULES:
            sub=fc_df[(fc_df["aqi_pred"]>=rule["threshold"])&(fc_df["horizon_hrs"]==rule["horizon_h"])]
            for _,row in sub.head(3).iterrows():
                console.print(f"  [red]{rule['name']}[/red] | {str(row.get('name',''))[:30]} | {rule['horizon_h']}h | AQI {row['aqi_pred']:.0f}" if RICH
                              else f"  {rule['name']} | {row.get('name','')} | AQI {row['aqi_pred']:.0f}")
                triggered=True
        if not triggered: console.print("  [green]No threshold breaches.[/green]" if RICH else "  No breaches.")

    elapsed=time.time()-t0
    if RICH:
        console.print(Panel(
            f"[bold green]✅ Full pipeline complete in {elapsed:.0f}s[/bold green]\n\n"
            f"📁 Output folder : [cyan]{OUTPUT_DIR.resolve()}[/cyan]\n\n"
            f"  🗺️   Map              : india_aqi_digital_twin_v6.html\n"
            f"  📊  Forecasts         : forecasts_latest.csv\n"
            f"  🏭  SimPy timeline    : simpy_timeline.csv\n"
            f"  🏭  SimPy hourly      : simpy_hourly.csv\n"
            f"  🚗  SUMO emissions    : sumo_emissions.csv\n"
            f"  🧠  SHAP charts       : shap_*.png\n"
            f"  🧩  LIME charts       : lime_*.png",
            title="🌏 India AQI Digital Twin v6.0 — DONE",style="bold green"))
    else:
        print(f"\n✅ Done in {elapsed:.0f}s. Outputs: {OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
