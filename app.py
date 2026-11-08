from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" if (BASE / "data").is_dir() else BASE

st.set_page_config(page_title="Road Accident Intelligence", page_icon="🚦", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root { color-scheme: dark; --bg:#0b1020; --panel:#121a2b; --panel2:#172238; --line:rgba(148,163,184,.16); --muted:#91a0b8; --text:#edf3ff; --accent:#fb7185; --cyan:#38bdf8; }
.stApp { background:radial-gradient(ellipse at 12% 0%,rgba(56,189,248,.10),transparent 34%),radial-gradient(ellipse at 90% 8%,rgba(251,113,133,.09),transparent 30%),#0b1020; color:var(--text); font-family:'DM Sans',sans-serif; }
[data-testid="stHeader"] { background:rgba(11,16,32,.78); backdrop-filter:blur(14px); }
.block-container { padding-top:1.35rem; padding-bottom:3rem; max-width:1580px; }
h1,h2,h3,h4 { color:#f4f7ff !important; font-family:'Space Grotesk',sans-serif !important; letter-spacing:-.025em; }
p,label,.stMarkdown { color:#d4deef; }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#111a2c 0%,#0d1424 100%); border-right:1px solid var(--line); }
[data-testid="stSidebar"] > div:first-child { padding-top:1.3rem; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2 { font-size:1.1rem; }
[data-testid="stMetric"] { background:linear-gradient(145deg,rgba(25,39,62,.96),rgba(17,26,43,.96)); border:1px solid rgba(148,163,184,.18); border-radius:16px; padding:17px 18px; box-shadow:0 10px 30px rgba(0,0,0,.12); min-height:112px; }
[data-testid="stMetricLabel"] { color:#9fb0c9 !important; font-size:.72rem !important; font-weight:700 !important; letter-spacing:.09em; }
[data-testid="stMetricValue"] { color:#f5f8ff !important; font-family:'Space Grotesk',sans-serif; font-weight:700; }
[data-testid="stMetricDelta"] { font-size:.72rem; }
div[data-testid="stTabs"] { margin-top:1.2rem; }
div[data-testid="stTabs"] button { color:#aebbd0; font-weight:600; border-radius:10px 10px 0 0; padding:12px 13px; }
div[data-testid="stTabs"] button[aria-selected="true"] { color:#fff; border-bottom:3px solid #fb7185; background:rgba(251,113,133,.07); }
div[data-testid="stPlotlyChart"] { background:linear-gradient(145deg,rgba(20,30,49,.9),rgba(15,23,40,.9)); border:1px solid var(--line); border-radius:16px; padding:10px; }
[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:14px; overflow:hidden; }
[data-testid="stMarkdownContainer"] p,[data-testid="stMarkdownContainer"] li { color:#d7e0ef !important; }
[data-testid="stCaptionContainer"] { color:#93a3bc !important; }
hr { border-color:var(--line); }
.stButton button,.stDownloadButton button,[data-testid="stFormSubmitButton"] button { border-radius:10px; border:1px solid rgba(251,113,133,.38); background:linear-gradient(110deg,#f45f7a,#e94d75); color:white; font-weight:700; transition:all .2s ease; }
.stButton button:hover,.stDownloadButton button:hover,[data-testid="stFormSubmitButton"] button:hover { border-color:#fda4af; box-shadow:0 8px 24px rgba(244,95,122,.22); transform:translateY(-1px); color:white; }
[data-baseweb="select"] > div,[data-baseweb="input"] > div { background:#111b2d; border-color:rgba(148,163,184,.23); border-radius:10px; }
[data-testid="stExpander"] { border:1px solid var(--line); border-radius:12px; background:rgba(18,26,43,.55); }
.hero { position:relative; overflow:hidden; padding:26px 30px; border:1px solid rgba(148,163,184,.18); border-radius:22px; background:linear-gradient(112deg,rgba(25,42,67,.96),rgba(18,27,46,.93) 60%,rgba(53,31,53,.7)); box-shadow:0 18px 50px rgba(0,0,0,.18); margin-bottom:20px; }
.hero:after { content:' '; position:absolute; width:250px; height:250px; right:-65px; top:-120px; border-radius:50%; border:1px solid rgba(56,189,248,.24); box-shadow:0 0 0 25px rgba(56,189,248,.025),0 0 0 55px rgba(56,189,248,.02); }
.hero-kicker { color:#7dd3fc; text-transform:uppercase; letter-spacing:.16em; font-size:.72rem; font-weight:700; }
.hero-title { color:#f8fbff; font:700 clamp(1.7rem,3vw,2.5rem) 'Space Grotesk',sans-serif; margin:7px 0 5px; letter-spacing:-.04em; }
.hero-sub { color:#aebdd2; font-size:.94rem; max-width:760px; }
.status-pill { display:inline-block; border:1px solid rgba(74,222,128,.28); background:rgba(34,197,94,.10); color:#86efac; border-radius:999px; padding:5px 10px; font-size:.72rem; font-weight:700; margin-top:14px; }
.section-intro { color:#8fa1bc; font-size:.86rem; margin-top:-7px; margin-bottom:14px; }
.small-muted { color:#91a0b8; font-size:.82rem; }
@media(max-width:700px){ .block-container{padding-left:1rem;padding-right:1rem;} .hero{padding:21px 19px;} [data-testid="stMetric"]{min-height:96px;padding:12px;} }
</style>
""", unsafe_allow_html=True)

BG = "#111b2d"
PLOT_LAYOUT = dict(
    paper_bgcolor=BG, plot_bgcolor=BG,
    font=dict(family="DM Sans, Arial, sans-serif", color="#eaf1ff", size=14),
    margin=dict(l=24,r=20,t=42,b=42),
    xaxis=dict(gridcolor="#26344b",zerolinecolor="#26344b",tickfont=dict(color="#b8c6dc",size=12),title_font=dict(color="#eaf1ff",size=13)),
    yaxis=dict(gridcolor="#26344b",zerolinecolor="#26344b",tickfont=dict(color="#b8c6dc",size=12),title_font=dict(color="#eaf1ff",size=13)),
    legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(color="#dce7f8",size=13))
)
STATE, CITY, YEAR = "State Name", "City Name", "Year"
SEVERITY, WEATHER, ROAD = "Accident Severity", "Weather Conditions", "Road Type"
FEATURES = [WEATHER,ROAD,"Road Condition","Lighting Conditions","Traffic Control Presence","TimePeriod","Season",
            "Number of Vehicles Involved","Vehicle Type Involved","Speed Limit (km/h)","Driver Age",
            "Driver Gender","Driver License Status","Alcohol Involvement","Accident Location Details",STATE]
INDIAN_STATES = {"Andhra Pradesh","Arunachal Pradesh","Assam","Bihar","Chhattisgarh","Goa","Gujarat","Haryana","Himachal Pradesh","Jharkhand","Karnataka","Kerala","Madhya Pradesh","Maharashtra","Manipur","Meghalaya","Mizoram","Nagaland","Odisha","Punjab","Rajasthan","Sikkim","Tamil Nadu","Telangana","Tripura","Uttar Pradesh","Uttarakhand","West Bengal"}
INDIAN_UTS = {"Andaman and Nicobar Islands","Chandigarh","Dadra and Nagar Haveli and Daman and Diu","Delhi","Jammu and Kashmir","Ladakh","Lakshadweep","Puducherry"}

@st.cache_data
def load_csv(name):
    return pd.read_csv(DATA / name)

@st.cache_data
def load_map():
    return (DATA / "geographical_hotspot_map.html").read_text(encoding="utf-8")

def style_fig(fig, height=280):
    fig.update_layout(**PLOT_LAYOUT, height=height)
    fig.update_layout(title_font=dict(color="#f5f6fb",size=16))
    return fig

def counts(frame, col, label, limit=None):
    s=frame[col].fillna("Unknown").astype(str).value_counts()
    if limit: s=s.head(limit)
    return s.rename_axis(label).reset_index(name="Accidents")

def bar(frame,col,label,limit=10,horizontal=False):
    d=counts(frame,col,label,limit)
    if horizontal:
        d=d.sort_values("Accidents")
        fig=px.bar(d,x="Accidents",y=label,orientation="h",color="Accidents",color_continuous_scale=["#3949ab","#ff6b78"])
    else:
        fig=px.bar(d,x=label,y="Accidents",color="Accidents",color_continuous_scale=["#3949ab","#ff6b78"])
        fig.update_xaxes(tickangle=-25)
    fig.update_coloraxes(showscale=False)
    return style_fig(fig)

# All project outputs are read from the bundled data directory. No visitor upload is needed.
required = ["cleaned_data.csv","dbscan_labels.csv","hier_labels.csv","clustering_comparison.csv","algorithm_comparison.csv","association_rules.csv","geographical_hotspot_map.html"]
missing_files = [f for f in required if not (DATA/f).exists()]
if missing_files:
    st.error("Project data files are missing from the deployment package: " + ", ".join(missing_files))
    st.stop()

df=load_csv("cleaned_data.csv")
dbscan=load_csv("dbscan_labels.csv")
hier=load_csv("hier_labels.csv")
cluster_comparison=load_csv("clustering_comparison.csv")
algorithm=load_csv("algorithm_comparison.csv")
rules=load_csv("association_rules.csv")

if "row_id" not in df:
    df["row_id"]=np.arange(len(df))
if STATE in df:
    df[STATE]=df[STATE].fillna("Unknown").astype(str).str.strip()

st.markdown('''<div class="hero"><div class="hero-kicker">India · Road safety analytics</div><div class="hero-title">🚦 Road Accident Intelligence</div><div class="hero-sub">Explore accident patterns, identify regional hotspots, compare machine-learning outputs and understand severity trends through one interactive workspace.</div><div class="status-pill">● PROJECT DATA CONNECTED · BUILT-IN DATASET</div></div>''',unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🚦 Control centre")
    st.caption("Your project dataset and analysis outputs are built in. No visitor uploads required.")
    st.markdown("<div class='small-muted'>LIVE FILTERS · ALL SECTIONS</div>",unsafe_allow_html=True)
    state_options=sorted(df[STATE].dropna().astype(str).unique()) if STATE in df else []
    selected_states=st.multiselect("States",state_options,placeholder="All states")
    year_options=sorted(pd.to_numeric(df[YEAR],errors="coerce").dropna().astype(int).unique()) if YEAR in df else []
    selected_years=st.multiselect("Years",year_options,placeholder="All years")
    severity_options=sorted(df[SEVERITY].dropna().astype(str).unique()) if SEVERITY in df else []
    selected_severity=st.multiselect("Severity",severity_options,default=severity_options,placeholder="Choose severity")
    city_options=sorted(df[CITY].dropna().astype(str).unique()) if CITY in df else []
    selected_cities=st.multiselect("Cities",city_options,placeholder="All cities")
    st.markdown("---")
    st.markdown("### About")
    st.caption("Interactive academic dashboard using the supplied road accident project files.")

view=df.copy()
if selected_states: view=view[view[STATE].astype(str).isin(selected_states)]
if selected_years: view=view[pd.to_numeric(view[YEAR],errors="coerce").isin(selected_years)]
if selected_severity and SEVERITY in view: view=view[view[SEVERITY].astype(str).isin(selected_severity)]
if selected_cities and CITY in view: view=view[view[CITY].astype(str).isin(selected_cities)]

k1,k2,k3,k4,k5=st.columns(5)
k1.metric("RECORDED ACCIDENTS",f"{len(view):,}")
regions=set(view[STATE].dropna().astype(str)) if STATE in view else set()
k2.metric("STATES COVERED",f"{len(regions & INDIAN_STATES)}/28")
k3.metric("UTs COVERED",f"{len(regions & INDIAN_UTS)}/8")
k4.metric("CITIES COVERED",f"{view[CITY].nunique():,}" if CITY in view else "N/A")
k5.metric("SEVERITY CLASSES",f"{view[SEVERITY].nunique():,}" if SEVERITY in view else "N/A")
st.markdown("---")

tabs=st.tabs(["◈ Overview","⌖ Hotspots","◎ Geographical Analysis","✦ Severity Prediction","⠿ Clustering","⇄ Apriori vs FP-Growth","▤ Dataset"])

with tabs[0]:
    st.markdown("### Accident patterns at a glance")
    st.markdown("<div class='section-intro'>A consolidated view of severity, location and contributing conditions. All visuals respond to the global filters.</div>",unsafe_allow_html=True)
    left,right=st.columns([1.05,1.4])
    with left:
        st.markdown("### Severity distribution")
        if SEVERITY in view:
            pie=counts(view,SEVERITY,"Severity")
            fig=px.pie(pie,names="Severity",values="Accidents",hole=.58,color_discrete_sequence=["#ff6b78","#f5b84b","#6c8cff","#5dd6c0","#ad8cff"])
            fig.update_traces(textposition="inside",textinfo="percent",textfont=dict(color="#101116",size=14))
            st.plotly_chart(style_fig(fig,300),use_container_width=True)
    with right:
        st.markdown("### Recorded accidents by state")
        if STATE in view: st.plotly_chart(bar(view,STATE,STATE,10,True),use_container_width=True)
    a,b,c=st.columns(3)
    with a:
        st.markdown("#### Road type")
        if ROAD in view: st.plotly_chart(bar(view,ROAD,"Road type",8),use_container_width=True)
    with b:
        st.markdown("#### Weather")
        if WEATHER in view: st.plotly_chart(bar(view,WEATHER,"Weather",8),use_container_width=True)
    with c:
        st.markdown("#### Time period")
        tcol="TimePeriod" if "TimePeriod" in view else "Time of Day"
        if tcol in view: st.plotly_chart(bar(view,tcol,"Time period",8),use_container_width=True)
    a,b=st.columns(2)
    with a:
        st.markdown("#### Seasonal distribution")
        if "Season" in view: st.plotly_chart(bar(view,"Season","Season",8),use_container_width=True)
    with b:
        st.markdown("#### Severity by time period")
        tcol="TimePeriod" if "TimePeriod" in view else "Time of Day"
        if tcol in view and SEVERITY in view:
            cross=view.groupby([tcol,SEVERITY],dropna=False).size().reset_index(name="Accidents")
            st.plotly_chart(style_fig(px.bar(cross,x=tcol,y="Accidents",color=SEVERITY,barmode="group",color_discrete_sequence=["#ff6b78","#6c8cff","#f5b84b","#5dd6c0"]),300),use_container_width=True)
    if "Day of Week" in view:
        st.markdown("#### Accidents by day of week")
        day_order=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        day_counts=view["Day of Week"].fillna("Unknown").astype(str).value_counts().rename_axis("Day").reset_index(name="Accidents")
        day_counts["_order"]=day_counts["Day"].map({d:i for i,d in enumerate(day_order)}).fillna(99)
        day_counts=day_counts.sort_values("_order")
        fig=px.bar(day_counts,x="Day",y="Accidents",color="Accidents",color_continuous_scale=["#2563eb","#38bdf8","#fb7185"])
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(style_fig(fig,260),use_container_width=True)

with tabs[1]:
    st.markdown("### State & city hotspot intelligence")
    st.markdown("<div class='section-intro'>Rankings are based on recorded rows in the supplied dataset, not population-adjusted risk.</div>",unsafe_allow_html=True)
    if STATE in view:
        state_counts=view.groupby(STATE,dropna=False).size().reset_index(name="Accident_Count").sort_values("Accident_Count",ascending=False)
        l,r=st.columns([1.2,1])
        with l: st.plotly_chart(bar(view,STATE,STATE,15,True),use_container_width=True)
        with r:
            st.markdown("#### Top state totals")
            st.dataframe(state_counts.head(15),use_container_width=True,hide_index=True)
        st.download_button("Download state counts CSV",state_counts.to_csv(index=False).encode(),"state_hotspots.csv","text/csv")
    if STATE in view and CITY in view:
        city_frame=view[view[CITY].fillna("Unknown").astype(str).str.lower()!="unknown"]
        city_counts=city_frame.groupby([STATE,CITY],dropna=False).size().reset_index(name="Accident_Count").sort_values("Accident_Count",ascending=False)
        st.markdown("### Top state-city combinations")
        st.dataframe(city_counts.head(25),use_container_width=True,hide_index=True)
        st.caption("Counts represent records in the supplied dataset; they are not population-adjusted risk estimates.")

with tabs[2]:
    st.markdown("### Geographical hotspot explorer")
    st.markdown("<div class='section-intro'>Explore the supplied Folium map with its built-in zoom controls and selectable map layers.</div>",unsafe_allow_html=True)
    components.html(load_map(),height=720,scrolling=True)

with tabs[3]:
    st.markdown("### Decision Tree severity prediction")
    st.markdown("<div class='section-intro'>Enter scenario attributes to see the model output and class-score distribution. This is an academic demonstration, not a real-world risk assessment.</div>",unsafe_allow_html=True)
    missing=[c for c in FEATURES+[SEVERITY] if c not in df.columns]
    if missing:
        st.error("The bundled dataset is missing model columns: "+", ".join(missing))
    else:
        @st.cache_resource
        def train_model():
            X=df[FEATURES].copy()
            y=df[SEVERITY].astype(str)
            cat=[c for c in FEATURES if X[c].dtype=="object" or c in [STATE,WEATHER,ROAD,"Road Condition","Lighting Conditions","Traffic Control Presence","TimePeriod","Season","Vehicle Type Involved","Driver Gender","Driver License Status","Alcohol Involvement","Accident Location Details"]]
            num=[c for c in FEATURES if c not in cat]
            for c in cat: X[c]=X[c].fillna("Unknown").astype(str)
            for c in num:
                X[c]=pd.to_numeric(X[c],errors="coerce")
                X[c]=X[c].fillna(X[c].median())
            prep=ColumnTransformer([("cat",OneHotEncoder(handle_unknown="ignore"),cat),("num","passthrough",num)])
            pipe=Pipeline([("prep",prep),("tree",DecisionTreeClassifier(max_depth=6,class_weight="balanced",random_state=42))])
            Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
            pipe.fit(Xtr,ytr)
            return pipe,accuracy_score(yte,pipe.predict(Xte))
        try:
            model,acc=train_model()
            x,y=st.columns(2)
            x.metric("Held-out test accuracy",f"{acc:.1%}")
            y.metric("Model","Decision Tree · max depth 6")
            st.caption("The model is trained from the bundled dataset. Predictions are educational outputs, not safety-critical advice.")
            formvals={}
            with st.form("predict_form"):
                left,right=st.columns(2)
                for i,feat in enumerate(FEATURES):
                    with (left if i%2==0 else right):
                        categorical=feat in [STATE,WEATHER,ROAD,"Road Condition","Lighting Conditions","Traffic Control Presence","TimePeriod","Season","Vehicle Type Involved","Driver Gender","Driver License Status","Alcohol Involvement","Accident Location Details"]
                        if categorical:
                            opts=sorted(df[feat].fillna("Unknown").astype(str).unique().tolist())
                            if "Unknown" not in opts: opts.append("Unknown")
                            formvals[feat]=st.selectbox(feat,opts,key="predict_"+feat)
                        else:
                            vals=pd.to_numeric(df[feat],errors="coerce").dropna()
                            formvals[feat]=st.number_input(feat,value=float(vals.median()) if len(vals) else 0.0,step=1.0,key="predict_"+feat)
                submitted=st.form_submit_button("Predict severity",type="primary")
            if submitted:
                row=pd.DataFrame([formvals],columns=FEATURES)
                prediction=model.predict(row)[0]
                st.success(f"Predicted accident severity: {prediction}")
                probs=model.predict_proba(row)[0]
                prob_df=pd.DataFrame({"Severity":model.named_steps["tree"].classes_,"Model score":probs}).sort_values("Model score",ascending=False)
                fig=px.bar(prob_df,x="Severity",y="Model score",color="Model score",color_continuous_scale=["#3949ab","#ff6b78"])
                fig.update_coloraxes(showscale=False)
                st.plotly_chart(style_fig(fig,260),use_container_width=True)
        except Exception as e:
            st.error(f"Model training failed: {e}")

with tabs[4]:
    st.markdown("### Clustering intelligence")
    st.markdown("<div class='section-intro'>Compare the saved clustering outputs and inspect the distribution of assigned labels.</div>",unsafe_allow_html=True)
    st.caption(f"DBSCAN labels: {len(dbscan):,} saved records · Hierarchical labels: {len(hier):,} saved records. The hierarchical output contains a subset of the full dataset.")
    st.dataframe(cluster_comparison,use_container_width=True,hide_index=True)
    if "Method" in cluster_comparison and "Silhouette score" in cluster_comparison:
        st.plotly_chart(style_fig(px.bar(cluster_comparison,x="Method",y="Silhouette score",color="Method",color_discrete_sequence=["#6c8cff","#ff6b78"]),280),use_container_width=True)
    c1,c2=st.columns(2)
    with c1:
        if "cluster" in dbscan:
            db_counts=dbscan["cluster"].value_counts().rename_axis("DBSCAN cluster").reset_index(name="Records")
            st.markdown("#### DBSCAN cluster sizes")
            st.plotly_chart(style_fig(px.bar(db_counts,x="DBSCAN cluster",y="Records",color="Records",color_continuous_scale=["#3949ab","#ff6b78"]),260),use_container_width=True)
            st.dataframe(db_counts,use_container_width=True,hide_index=True)
    with c2:
        if "hier_cluster" in hier:
            hier_counts=hier["hier_cluster"].value_counts().rename_axis("Hierarchical cluster").reset_index(name="Records")
            st.markdown("#### Hierarchical cluster sizes")
            st.plotly_chart(style_fig(px.bar(hier_counts,x="Hierarchical cluster",y="Records",color="Records",color_continuous_scale=["#3949ab","#ff6b78"]),260),use_container_width=True)
            st.dataframe(hier_counts,use_container_width=True,hide_index=True)
    with st.expander("View bundled cluster labels"):
        st.markdown("**DBSCAN labels**")
        st.dataframe(dbscan,use_container_width=True,hide_index=True)
        st.markdown("**Hierarchical labels**")
        st.dataframe(hier,use_container_width=True,hide_index=True)

with tabs[5]:
    st.markdown("### Association mining · Apriori vs FP-Growth")
    st.markdown("<div class='section-intro'>Compare saved algorithm results and explore the strongest association rules.</div>",unsafe_allow_html=True)
    st.dataframe(algorithm,use_container_width=True,hide_index=True)
    if "Algorithm" in algorithm and "Runtime (seconds)" in algorithm:
        st.plotly_chart(style_fig(px.bar(algorithm,x="Algorithm",y="Runtime (seconds)",color="Algorithm",color_discrete_sequence=["#6c8cff","#ff6b78"]),280),use_container_width=True)
    st.markdown("#### Saved association rules")
    if "lift" in rules: rules=rules.sort_values("lift",ascending=False)
    if not rules.empty:
        max_rules=min(74,len(rules))
        rule_limit=st.slider("Number of top rules to display",min_value=5,max_value=max(5,max_rules),value=min(20,max_rules),step=1)
        shown_rules=rules.head(rule_limit)
        st.dataframe(shown_rules,use_container_width=True,hide_index=True)
        st.download_button("↓ Download displayed rules",shown_rules.to_csv(index=False).encode(),"association_rules_top.csv","text/csv")
    st.caption("Support, confidence and lift describe co-occurrence patterns; they do not establish causation.")

with tabs[6]:
    st.markdown("### Dataset explorer")
    st.markdown("<div class='section-intro'>Search and inspect the project records currently selected by the global filters.</div>",unsafe_allow_html=True)
    st.caption(f"{len(view):,} rows after filters · {len(view.columns)} columns")
    search_text=st.text_input("Search across visible records",placeholder="Type a state, city, road type, weather condition…")
    table_view=view
    if search_text.strip():
        mask=table_view.astype(str).apply(lambda col: col.str.contains(search_text.strip(),case=False,na=False)).any(axis=1)
        table_view=table_view[mask]
    st.caption(f"Showing {len(table_view):,} matching records")
    st.dataframe(table_view,use_container_width=True,hide_index=True,height=480)
    st.download_button("↓ Download filtered records CSV",table_view.to_csv(index=False).encode(),"filtered_accident_data.csv","text/csv")

st.markdown("---")
st.markdown('<div class="small-muted">ROAD ACCIDENT INTELLIGENCE · Academic project dashboard · Bundled project data · Model predictions are experimental and not for emergency or safety-critical decisions.</div>',unsafe_allow_html=True)
