import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import pickle
import re
import ast
import heapq
import os
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from apify_client import ApifyClient
from dotenv import load_dotenv

# Load env variables if .env exists
load_dotenv()

# App configuration
st.set_page_config(
    page_title="Mitigate Filter Bubble — MBDA*",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling for a premium dark theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Header styling */
    .app-title {
        background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.8rem;
        margin-bottom: 0.2rem;
    }
    .app-subtitle {
        color: inherit;
        opacity: 0.85;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        font-weight: 500;
    }
    
    /* Card design */
    .custom-card {
        background: rgba(30, 41, 59, 0.85);
        backdrop-filter: blur(10px);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
        color: #f1f5f9 !important;
    }
    
    .custom-card p, .custom-card li, .custom-card ul, .custom-card ol {
        color: #e2e8f0 !important;
    }
    
    .card-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #f8fafc !important;
        margin-bottom: 1rem;
        border-bottom: 1px solid #334155;
        padding-bottom: 0.5rem;
    }
    
    /* Metrics grid */
    .metric-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
        background: rgba(15, 23, 42, 0.75);
        border-radius: 8px;
        padding: 1rem;
        border: 1px solid #334155;
    }
    
    .metric-num {
        font-size: 2.2rem;
        font-weight: 700;
        color: #38bdf8 !important;
        line-height: 1;
        margin-bottom: 0.5rem;
    }
    
    .metric-lbl {
        font-size: 0.8rem;
        font-weight: 500;
        color: #cbd5e1 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Step list styling */
    .step-box {
        background: #1e293b;
        border-left: 4px solid #818cf8;
        padding: 1rem;
        margin-bottom: 0.8rem;
        border-radius: 0 8px 8px 0;
        border: 1px solid #334155;
        border-left-width: 4px;
        transition: transform 0.2s ease;
    }
    
    .step-box p {
        color: #f1f5f9 !important;
    }
    
    .step-box:hover {
        transform: translateX(4px);
    }
    
    .step-index {
        font-weight: 700;
        color: #818cf8;
        font-size: 1rem;
    }
    
    .step-name {
        font-weight: 600;
        color: #f1f5f9;
        font-size: 1.1rem;
    }
    
    .step-cluster {
        display: inline-block;
        padding: 0.1rem 0.5rem;
        border-radius: 4px;
        font-size: 0.7rem;
        font-weight: 600;
        background-color: #3b82f6;
        color: white;
        margin-left: 0.5rem;
    }
    
    .step-tweet {
        color: #cbd5e1;
        font-style: italic;
        font-size: 0.9rem;
        margin-top: 0.5rem;
        border-top: 1px dashed #475569;
        padding-top: 0.5rem;
    }
    
</style>
""", unsafe_allow_html=True)

# Helper function to safely parse nested columns
def safe_parse(val):
    if isinstance(val, dict):
        return val
    if pd.isna(val):
        return {}
    try:
        return ast.literal_eval(str(val))
    except:
        return {}

# Normalization logic matching pipeline
def normalize_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

# Caching loaders
@st.cache_resource
def load_graph_data():
    try:
        with open("models/graph.pkl", "rb") as f:
            G = pickle.load(f)
        with open("models/mbda_final.pkl", "rb") as f:
            final_data = pickle.load(f)
        
        G_lcc = final_data["G_lcc"]
        partition_sub = final_data["partition_sub"]
        precomputed_results = final_data["results"]
        return G, G_lcc, partition_sub, precomputed_results
    except Exception as e:
        st.error(f"Gagal memuat file model (models/graph.pkl / models/mbda_final.pkl): {e}")
        return None, None, None, None

@st.cache_data
def load_tfidf_data():
    try:
        df = pd.read_csv("data/dataTwitter.csv")
        df.drop_duplicates(subset="id", inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        df["author_parsed"] = df["author"].apply(safe_parse)
        df["entities_parsed"] = df["entities"].apply(safe_parse)
        df["username"] = df["author_parsed"].apply(lambda x: x.get("userName", ""))
        df["display_name"] = df["author_parsed"].apply(lambda x: x.get("name", ""))
        df["hashtags_list"] = df["entities_parsed"].apply(
            lambda x: [h.get("text", "").lower() for h in x.get("hashtags", [])])
        df["mentions_list"] = df["entities_parsed"].apply(
            lambda x: [m.get("screen_name", "") for m in x.get("user_mentions", [])])
        
        df = df[df["lang"].isin(["in", "en"])]
        df = df[df["viewCount"] > 0]
        df.reset_index(drop=True, inplace=True)
        
        df["text_clean"] = df["text"].apply(normalize_text)
        
        user_texts = (df.groupby("username")["text_clean"]
                        .apply(lambda x: " ".join(x)).reset_index())
        user_texts.columns = ["username", "combined_text"]
        
        vectorizer = TfidfVectorizer(max_features=500, min_df=2, ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform(user_texts["combined_text"])
        
        neutral_text = ("kebijakan pemerintah indonesia masyarakat ekonomi sosial "
                        "program pembangunan kesejahteraan rakyat")
        neutral_vec = vectorizer.transform([neutral_text])
        similarities = cosine_similarity(tfidf_matrix, neutral_vec).flatten()
        
        user_texts["h_value"] = 1 - similarities
        h_dict = dict(zip(user_texts["username"], user_texts["h_value"]))
        
        return vectorizer, tfidf_matrix, user_texts, h_dict, neutral_vec, df
    except Exception as e:
        st.error(f"Gagal memuat dataset CSV (data/dataTwitter.csv): {e}")
        return None, None, None, None, None, None

# Precompute/cache spring layout positions for stable rendering
@st.cache_data
def get_base_layout(_G_lcc):
    return nx.spring_layout(_G_lcc, k=0.18, seed=42)

# Dynamic position resolver for new user
def get_layout_with_new_user(G, base_positions, new_username):
    pos = base_positions.copy()
    if new_username in G and new_username not in pos:
        neighbors = list(G.neighbors(new_username))
        if neighbors:
            xs = [pos[nb][0] for nb in neighbors if nb in pos]
            ys = [pos[nb][1] for nb in neighbors if nb in pos]
            if xs and ys:
                pos[new_username] = np.array([np.mean(xs) + 0.05, np.mean(ys) + 0.05])
            else:
                pos[new_username] = np.array([0.0, 0.0])
        else:
            pos[new_username] = np.array([0.0, 0.0])
    return pos

# Helper to build profiles for Louvain clusters
@st.cache_data
def get_cluster_profiles(_df, partition_sub):
    df_with_cluster = _df.copy()
    df_with_cluster["cluster"] = df_with_cluster["username"].map(partition_sub)
    
    profiles = {}
    for cluster_id in sorted(list(set(partition_sub.values()))):
        cluster_tweets = df_with_cluster[df_with_cluster["cluster"] == cluster_id]
        if cluster_tweets.empty:
            continue
            
        all_hashtags = [h for hs in cluster_tweets["hashtags_list"] for h in hs]
        top_hashtags = [f"#{h}" for h, _ in Counter(all_hashtags).most_common(5)]
        
        all_mentions = [m for ms in cluster_tweets["mentions_list"] for m in ms]
        top_mentions = [f"@{m}" for m, _ in Counter(all_mentions).most_common(5)]
        
        typical_tweets = cluster_tweets.sort_values(by="retweetCount", ascending=False).head(3)["text"].tolist()
        
        profiles[cluster_id] = {
            "top_hashtags": ", ".join(top_hashtags) if top_hashtags else "None",
            "top_mentions": ", ".join(top_mentions) if top_mentions else "None",
            "sample_tweets": typical_tweets
        }
    return profiles

# Apify Actor call
def scrape_twitter_user(username, api_token, max_items=25):
    client = ApifyClient(api_token)
    clean_username = username.strip().replace("@", "")
    
    # Configure run parameters
    run_input = {
        "filter:blue_verified": False,
        "filter:consumer_video": False,
        "filter:has_engagement": False,
        "filter:hashtags": False,
        "filter:images": False,
        "filter:links": False,
        "filter:media": False,
        "filter:mentions": False,
        "filter:native_video": False,
        "filter:nativeretweets": False,
        "filter:news": False,
        "filter:pro_video": False,
        "filter:quote": False,
        "filter:replies": False,
        "filter:safe": False,
        "filter:spaces": False,
        "filter:twimg": False,
        "filter:videos": False,
        "filter:vine": False,
        "include:nativeretweets": False,
        "lang": "in",
        "maxItems": max_items,
        "from": clean_username,
        "queryType": "Latest"
    }
    
    try:
        run = client.actor("CJdippxWmn9uRfooo").call(run_input=run_input)
        dataset_id = run.default_dataset_id
        items = list(client.dataset(dataset_id).iterate_items())
        return items, None
    except Exception as e:
        return None, str(e)

# Dynamic insertion function
def integrate_new_user(username, scraped_tweets, G_lcc, partition_sub, vectorizer, tfidf_matrix, user_texts, h_dict, neutral_vec):
    clean_username = username.strip().replace("@", "")
    
    cleaned_tweets = [normalize_text(t.get("text", "")) for t in scraped_tweets if t.get("text")]
    combined_text = " ".join(cleaned_tweets)
    
    if not combined_text.strip():
        combined_text = "kebijakan prabowo"  # Minimum fallback to avoid zero TF-IDF
        
    new_user_vec = vectorizer.transform([combined_text])
    similarity_neutral = cosine_similarity(new_user_vec, neutral_vec).flatten()[0]
    h_val = 1 - similarity_neutral
    
    G_local = G_lcc.copy()
    partition_local = partition_sub.copy()
    
    display_name = clean_username
    if scraped_tweets and scraped_tweets[0].get("author", {}).get("name"):
        display_name = scraped_tweets[0]["author"]["name"]
        
    G_local.add_node(clean_username, 
                     node_type="user", 
                     tweet_count=len(scraped_tweets),
                     total_rt=sum(int(t.get("retweetCount", 0)) for t in scraped_tweets),
                     total_reply=sum(int(t.get("replyCount", 0)) for t in scraped_tweets),
                     display_name=display_name,
                     h_value=h_val)
                     
    # Extract mentions and connect
    mentions_found = []
    for t in scraped_tweets:
        entities = t.get("entities", {})
        if isinstance(entities, str):
            entities = safe_parse(entities)
        for m in entities.get("user_mentions", []):
            screen_name = m.get("screen_name")
            if screen_name and screen_name in G_local.nodes:
                engagement = int(t.get("retweetCount", 0)) + int(t.get("replyCount", 0))
                mentions_found.append((screen_name, engagement))
                
    connected = False
    if mentions_found:
        for tgt, engagement in mentions_found:
            w = engagement + 1
            if G_local.has_edge(clean_username, tgt):
                G_local[clean_username][tgt]["weight"] += w
            else:
                G_local.add_edge(clean_username, tgt, weight=w, edge_type="mention")
        connected = "mention"
        
        # Cluster resolution
        mention_counts = Counter([m[0] for m in mentions_found])
        most_mentioned = mention_counts.most_common(1)[0][0]
        assigned_cluster = partition_local.get(most_mentioned, -1)
        partition_local[clean_username] = assigned_cluster
    else:
        # Isolated node bridging using content similarity
        sims = cosine_similarity(tfidf_matrix, new_user_vec).flatten()
        best_sim = -1.0
        best_user = None
        
        for idx, row in user_texts.iterrows():
            uname = row["username"]
            if uname in G_lcc.nodes and uname != clean_username:
                if sims[idx] > best_sim:
                    best_sim = sims[idx]
                    best_user = uname
                    
        if best_user:
            virtual_weight = int(best_sim * 10) + 1
            G_local.add_edge(clean_username, best_user, weight=virtual_weight, edge_type="content_bridge")
            assigned_cluster = partition_local.get(best_user, -1)
            partition_local[clean_username] = assigned_cluster
            connected = f"content_bridge (linked to @{best_user})"
        else:
            partition_local[clean_username] = 0
            
    return G_local, partition_local, clean_username, new_user_vec, connected

# MBDA* execution
def run_mbda_star(G, source, goal, user_texts, tfidf_matrix, vectorizer, new_username=None, new_user_vec=None, max_iter=5000):
    if source not in G or goal not in G:
        return None, float("inf"), 0, []
        
    h_g_dict = {}
    try:
        if source == new_username and new_user_vec is not None:
            src_vec = new_user_vec
        else:
            src_idx = user_texts[user_texts["username"] == source].index[0]
            src_vec = tfidf_matrix[src_idx]
            
        sims_to_src = cosine_similarity(tfidf_matrix, src_vec).flatten()
        h_g_dict = dict(zip(user_texts["username"], 1 - sims_to_src))
        
        if new_username and new_user_vec is not None:
            h_g_dict[new_username] = 0.0
    except Exception as e:
        h_g_dict = {}
        
    def h_s(n): return G.nodes[n].get("h_value", 1.0)
    def h_g(n): return h_g_dict.get(n, 1.0)
    
    def cost(u, v):
        edge_data = G[u][v]
        return max(0.1, 1.0 / (edge_data.get("weight", 1) + 1))
        
    f_source = 0.5 * (h_s(source) - h_g(source))
    f_goal = 0.5 * (h_g(goal) - h_s(goal))
    
    open_f = [(f_source, 0.0, source, [source])]
    open_b = [(f_goal,   0.0, goal,   [goal])]
    cf, cb = {}, {}
    best = {"cost": float("inf"), "path": None}
    exp = [0]
    trace = []
    
    def step(oq, ct, co, fwd):
        if not oq:
            return
        fe, g, cur, path = heapq.heappop(oq)
        exp[0] += 1
        
        # Prepare frontier (top 5 elements in the queue)
        frontier_snapshot = [(item[0], item[1], item[2]) for item in sorted(oq)[:5]]
        
        step_info = {
            "step_num": exp[0],
            "direction": "Forward" if fwd else "Backward",
            "node_popped": cur,
            "f_val": fe,
            "g_val": g,
            "h_s_val": h_s(cur),
            "h_g_val": h_g(cur),
            "path_so_far": list(path),
            "neighbors_evaluated": [],
            "collision_detected": False,
            "collision_node": None,
            "collision_path": None,
            "collision_cost": None,
            "frontier": frontier_snapshot,
            "status": "expanded"
        }
        
        if cur in ct:
            step_info["status"] = "skipped"
            trace.append(step_info)
            return
            
        ct[cur] = (g, path)
        
        if cur in co:
            g2, p2 = co[cur]
            total = g + g2
            step_info["collision_detected"] = True
            step_info["collision_node"] = cur
            step_info["collision_cost"] = total
            if fwd:
                step_info["collision_path"] = path + list(reversed(p2))[1:]
            else:
                step_info["collision_path"] = p2 + list(reversed(path))[1:]
                
            if total < best["cost"]:
                best["cost"] = total
                best["path"] = step_info["collision_path"]
                
        for nb in G.neighbors(cur):
            w = G[cur][nb].get("weight", 1)
            c_edge = cost(cur, nb)
            gn = g + c_edge
            
            if fwd:
                fn = gn + 0.5 * (h_s(nb) - h_g(nb))
            else:
                fn = gn + 0.5 * (h_g(nb) - h_s(nb))
                
            if nb in ct:
                step_info["neighbors_evaluated"].append({
                    "name": nb,
                    "status": "visited",
                    "weight": w,
                    "edge_cost": c_edge,
                    "g_new": gn,
                    "h_s": h_s(nb),
                    "h_g": h_g(nb),
                    "f_new": fn
                })
                continue
                
            step_info["neighbors_evaluated"].append({
                "name": nb,
                "status": "pushed",
                "weight": w,
                "edge_cost": c_edge,
                "g_new": gn,
                "h_s": h_s(nb),
                "h_g": h_g(nb),
                "f_new": fn
            })
            heapq.heappush(oq, (fn, gn, nb, path+[nb]))
            
        trace.append(step_info)
                
    for _ in range(max_iter):
        step(open_f, cf, cb, True)
        step(open_b, cb, cf, False)
        if best["path"]:
            break
        if not open_f and not open_b:
            break
            
    return best["path"], best["cost"], exp[0], trace

# Render network plot
def render_graph_path(G, partition_sub, path, source, goal, base_positions, new_username=None):
    import matplotlib.pyplot as plt
    
    pos = get_layout_with_new_user(G, base_positions, new_username)
    
    fig, ax = plt.subplots(figsize=(12, 9), facecolor='#0f172a')
    ax.set_facecolor('#0f172a')
    
    unique_clusters = sorted(list(set(partition_sub.values())))
    color_palette = plt.colormaps["tab20"]
    cluster_color_map = {cluster: color_palette(i % 20) for i, cluster in enumerate(unique_clusters)}
    
    node_colors = []
    for node in G.nodes():
        cluster = partition_sub.get(node, -1)
        if cluster != -1:
            node_colors.append(cluster_color_map[cluster])
        else:
            node_colors.append((0.4, 0.4, 0.4, 0.4))
            
    # Draw background graph
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=25, alpha=0.3, edgecolors="none", ax=ax)
    nx.draw_networkx_edges(G, pos, width=0.15, edge_color="#334155", alpha=0.2, ax=ax)
    
    # Highlight path
    if path and len(path) >= 2:
        path_edges = list(zip(path[:-1], path[1:]))
        
        # Thick glowing path edges
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, width=3.5, edge_color="#f59e0b", alpha=0.9, ax=ax)
        
        # Path nodes
        nx.draw_networkx_nodes(G, pos, nodelist=path, node_color="#f59e0b", node_size=120, edgecolors="#ffffff", linewidths=1.2, ax=ax)
        
        # Label path elements
        label_pos = {node: np.array([coord[0], coord[1] + 0.025]) for node, coord in pos.items() if node in path}
        
        for node, coord in label_pos.items():
            is_endpoint = (node == source or node == goal)
            bg = '#ffffff' if is_endpoint else '#1e293b'
            tc = '#0f172a' if is_endpoint else '#f8fafc'
            fw = 'bold' if is_endpoint else 'normal'
            
            ax.text(coord[0], coord[1], f"@{node}" if node != new_username else f"@{node} (ANDA)", 
                    fontsize=8, fontweight=fw, color=tc,
                    horizontalalignment='center',
                    bbox=dict(facecolor=bg, alpha=0.95, edgecolor='#475569', boxstyle='round,pad=0.2'))
                    
    # Highlight source & goal uniquely
    if source in pos:
        nx.draw_networkx_nodes(G, pos, nodelist=[source], node_color="#fbbf24", node_shape="^", node_size=250, edgecolors="#000000", linewidths=1.5, ax=ax)
    if goal in pos:
        nx.draw_networkx_nodes(G, pos, nodelist=[goal], node_color="#22d3ee", node_shape="s", node_size=220, edgecolors="#000000", linewidths=1.5, ax=ax)
        
    ax.axis("off")
    plt.tight_layout()
    return fig

# Load initial data
G_raw, G_lcc, partition_sub, precomputed_results = load_graph_data()
vectorizer, tfidf_matrix, user_texts, h_dict, neutral_vec, df_raw = load_tfidf_data()
base_positions = get_base_layout(G_lcc) if G_lcc is not None else None
cluster_profiles = get_cluster_profiles(df_raw, partition_sub) if df_raw is not None and partition_sub is not None else None

# Header Title
st.markdown('<div class="app-title">Dashboard Mitigasi Filter Bubble</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">Visualisasi jalur optimasi jembatan sosial menggunakan algoritma Modified Bidirectional A* (MBDA*)</div>', unsafe_allow_html=True)

# Sidebar setup
st.sidebar.title("🛠️ Konfigurasi")

# Option selection
mode = st.sidebar.radio("Pilih Mode:", ["Mode Demo (Data Tersedia)", "Scraper Twitter Live (Apify)"])

# Shared parameters for new dynamic runs
dynamic_running = False
scraped_tweets = None
new_username = None
new_user_vec = None
connected_type = None

# Local copies of network state for dynamic calculations
active_G = G_lcc
active_partition = partition_sub

if mode == "Scraper Twitter Live (Apify)":
    st.sidebar.markdown("### Pengaturan Apify")
    default_token = os.getenv("APIFY_API_TOKEN", "")
    api_token = st.sidebar.text_input("Token API Apify", value=default_token, type="password", help="Dapatkan dari console.apify.com")
    
    st.sidebar.markdown("### Target Scrape")
    username_input = st.sidebar.text_input("Username Twitter/X", placeholder="misal: jokowi")
    tweets_count = st.sidebar.slider("Jumlah tweet yang di-scrape", 10, 100, 25)
    
    if st.sidebar.button("Ambil & Masukkan Akun"):
        if not api_token:
            st.sidebar.error("Error: Token API Apify diperlukan!")
        elif not username_input:
            st.sidebar.error("Error: Username Twitter diperlukan!")
        else:
            with st.spinner(f"Mengambil tweet terbaru @{username_input} via Apify (memakan waktu 30-45 detik)..."):
                items, err = scrape_twitter_user(username_input, api_token, tweets_count)
                if err:
                    st.error(f"Gagal melakukan scraping: {err}")
                elif not items:
                    st.warning(f"Tidak ada tweet ditemukan untuk @{username_input}. Pastikan akun bersifat publik dan memiliki postingan.")
                else:
                    st.sidebar.success(f"Berhasil mengambil {len(items)} tweet!")
                    st.session_state["scraped_tweets"] = items
                    st.session_state["target_username"] = username_input
                    
    # Maintain state on re-renders
    if "scraped_tweets" in st.session_state and "target_username" in st.session_state:
        scraped_tweets = st.session_state["scraped_tweets"]
        new_username = st.session_state["target_username"]
        
        with st.spinner("Menyelaraskan akun ke dalam graf jaringan..."):
            active_G, active_partition, new_username, new_user_vec, connected_type = integrate_new_user(
                new_username, scraped_tweets, G_lcc, partition_sub, 
                vectorizer, tfidf_matrix, user_texts, h_dict, neutral_vec
            )
            dynamic_running = True

else:
    # Demo Mode Setup
    st.sidebar.markdown("### Pilih Akun Demo")
    demo_accounts = {
        "________dyah (Cluster 2 - Bubble MBG)": "________dyah",
        "Deka_Ajaa (Cluster 2 - Pro-Prabowo)": "Deka_Ajaa",
        "karirfess (Cluster 0 - Karirfess/Pencari Kerja)": "karirfess",
        "DaudJTP (Cluster 1 - Netral/Media)": "DaudJTP"
    }
    
    # Add other LCC nodes as custom choice
    custom_choice = st.sidebar.checkbox("Pilih akun kustom dari database jaringan")
    if custom_choice:
        node_list = sorted(list(G_lcc.nodes()))
        selected_username = st.sidebar.selectbox("Pilih Akun dari LCC", node_list)
    else:
        choice = st.sidebar.selectbox("Pilih Akun Demo", list(demo_accounts.keys()))
        selected_username = demo_accounts[choice]
        
    st.session_state["demo_user"] = selected_username
    new_username = selected_username

# UI Logic Layout
if new_username:
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "👤 Profil Akun", 
        "🗺️ Pencari Rute MBDA*", 
        "🕸️ Visualisasi Graf",
        "ℹ️ Penjelasan Algoritma"
    ])
    
    # Tab 1: Account Profiling
    with tab1:
        st.markdown(f"## Profil Akun: `@{new_username}`")
        
        # Display meta values
        col1, col2, col3 = st.columns(3)
        
        cluster_id = active_partition.get(new_username, -1)
        
        with col1:
            st.markdown(f"""
            <div class="custom-card">
                <div class="card-title">Kelompok Komunitas (Cluster)</div>
                <div class="metric-num">#{cluster_id if cluster_id != -1 else 'Terisolasi'}</div>
                <div class="metric-lbl">ID Grup Louvain</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            # Calculate Content Bias
            if dynamic_running and new_username == st.session_state.get("target_username"):
                h_val = active_G.nodes[new_username]["h_value"]
            else:
                h_val = h_dict.get(new_username, 1.0)
                
            bias_percent = int((1 - h_val) * 100)
            
            st.markdown(f"""
            <div class="custom-card">
                <div class="card-title">Kemiripan Konten dengan Profil Netral</div>
                <div class="metric-num">{bias_percent}%</div>
                <div class="metric-lbl">Kemiripan TF-IDF</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            if dynamic_running:
                tweets_len = len(scraped_tweets)
            else:
                tweets_len = len(df_raw[df_raw["username"] == new_username]) if df_raw is not None else 0
                
            st.markdown(f"""
            <div class="custom-card">
                <div class="card-title">Volume Analisis</div>
                <div class="metric-num">{tweets_len}</div>
                <div class="metric-lbl">Tweet Dianalisis</div>
            </div>
            """, unsafe_allow_html=True)
            
        # Cluster explanation
        if cluster_id in cluster_profiles:
            st.markdown(f"""
            <div class="custom-card">
                <div class="card-title">Profil Cluster (Konteks Filter Bubble)</div>
                <p><strong>Hashtag Teratas di Cluster Ini:</strong> <span style="color:#f0abfc; font-weight:600;">{cluster_profiles[cluster_id]['top_hashtags']}</span></p>
                <p><strong>Akun yang Paling Sering Disebut (Mention):</strong> <span style="color:#38bdf8; font-weight:600;">{cluster_profiles[cluster_id]['top_mentions']}</span></p>
                <p><strong>Topik/Argumen Umum:</strong></p>
                <ul>
                    {"".join([f"<li><i>{t}</i></li>" for t in cluster_profiles[cluster_id]['sample_tweets'][:2]])}
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
        # Scraped tweets view
        st.subheader("Tweet Terbaru yang Dianalisis")
        if dynamic_running and scraped_tweets:
            for t in scraped_tweets[:10]:
                created = t.get("createdAt", "")
                text = t.get("text", "")
                st.markdown(f"""
                <div class="step-box" style="border-left-color: #ec4899;">
                    <span style="color: #94a3b8; font-size: 0.8rem;">{created}</span>
                    <p style="margin: 0.3rem 0 0 0; color: #f1f5f9;">{text}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            user_raw = df_raw[df_raw["username"] == new_username] if df_raw is not None else pd.DataFrame()
            if not user_raw.empty:
                for _, row in user_raw.head(10).iterrows():
                    st.markdown(f"""
                    <div class="step-box" style="border-left-color: #ec4899;">
                        <span style="color: #94a3b8; font-size: 0.8rem;">{row['createdAt']}</span>
                        <p style="margin: 0.3rem 0 0 0; color: #f1f5f9;">{row['text']}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("Tidak ada konten teks yang tersedia di database lokal untuk akun ini.")

    # Tab 2: MBDA* Pathfinder
    with tab2:
        st.markdown("## 🗺️ Pencari Rute Jembatan (Mitigasi Filter Bubble)")
        st.write("MBDA* menghitung jalur melalui jaringan mention, memilih node yang secara bertahap menghubungkan profil awal ke akun target. Tujuannya adalah memandu pengguna dari cluster politik/sosial mereka saat ini menuju informasi netral atau perspektif lain tanpa kejutan konten secara drastis.")
        
        # Path Goal Selector
        st.markdown("### Pilih Node Target / Tujuan")
        goals = {
            "kompascom (Cluster 1 — Media Netral/Informatif)": "kompascom",
            "jokowi (Cluster 2 — Pro-Pemerintah)": "jokowi",
            "Fahrihamzah (Cluster 5 — Kritis/Oposisi)": "Fahrihamzah"
        }
        
        goal_labels = list(goals.keys())
        selected_goal_label = st.selectbox("Pilih Akun Target:", goal_labels)
        goal_username = goals[selected_goal_label]
        
        # Run MBDA* Button
        if st.button("Jalankan Pencarian Rute MBDA*"):
            with st.spinner("Mencari jalur optimal melalui graf sosial..."):
                path, cost_val, explored, trace = run_mbda_star(
                    active_G, new_username, goal_username, 
                    user_texts, tfidf_matrix, vectorizer, 
                    new_username=new_username if dynamic_running else None, 
                    new_user_vec=new_user_vec
                )
                
                if path:
                    # Save results in session state
                    st.session_state["path_results"] = {
                        "path": path,
                        "cost": cost_val,
                        "explored": explored,
                        "trace": trace,
                        "source": new_username,
                        "goal": goal_username
                    }
                    st.success("Jalur berhasil ditemukan!")
                else:
                    st.error("Jalur tidak ditemukan. Node awal dan target berada pada komponen graf yang benar-benar terpisah.")
                    
        # Render Path Results
        if "path_results" in st.session_state and st.session_state["path_results"]["source"] == new_username and st.session_state["path_results"]["goal"] == goal_username:
            res = st.session_state["path_results"]
            path = res["path"]
            cost_val = res["cost"]
            explored = res["explored"]
            trace = res.get("trace", [])
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Langkah Jembatan (Node)", len(path))
            with col2:
                st.metric("Total Biaya Heuristik", f"{cost_val:.4f}")
            with col3:
                # Diversity score: unique clusters visited / total nodes in path
                path_cl = [active_partition.get(n, -1) for n in path]
                diversity = len(set(c for c in path_cl if c != -1)) / len(path)
                st.metric("Indeks Keragaman Jalur", f"{diversity:.3f}")
            
            # --- NEW STEP-BY-STEP ALGORITHM PROGRESSION TRACE ---
            st.write("---")
            st.subheader("🕵️ MBDA* Search Progression Trace")
            st.write("Sebelum mendapatkan rekomendasi akhir, algoritma MBDA* melakukan pencarian dua arah (bidirectional A*) dengan mengevaluasi antrean secara paralel dari sisi Source (Forward) dan sisi Goal (Backward). Gunakan slider di bawah ini untuk melihat detail node yang di-pop dan dievaluasi di setiap langkah:")
            
            if trace:
                step_idx = st.slider("Pilih Langkah (Langkah Pencarian):", 1, len(trace), 1, 
                                     help="Geser slider untuk melihat proses pencarian node dari queue dan perhitungan heuristiknya.")
                
                selected_step = trace[step_idx - 1]
                direction = selected_step["direction"]
                node_popped = selected_step["node_popped"]
                status = selected_step["status"]
                f_val = selected_step["f_val"]
                g_val = selected_step["g_val"]
                h_s_val = selected_step["h_s_val"]
                h_g_val = selected_step["h_g_val"]
                path_so_far = selected_step["path_so_far"]
                neighbors_eval = selected_step["neighbors_evaluated"]
                collision = selected_step["collision_detected"]
                
                dir_color = "#3b82f6" if direction == "Forward" else "#ec4899"
                dir_text = "Antrean Forward (Source ➔ Goal)" if direction == "Forward" else "Antrean Backward (Goal ➔ Source)"
                
                col_left, col_right = st.columns([1, 2])
                
                with col_left:
                    st.markdown(f"""
                    <div class="custom-card" style="border-left: 5px solid {dir_color}; margin-top: 10px;">
                        <div style="font-size: 0.8rem; text-transform: uppercase; color: #94a3b8; font-weight:600;">{dir_text}</div>
                        <div style="font-size: 1.5rem; font-weight:700; color: #ffffff; margin: 5px 0;">@{node_popped}</div>
                        <div style="font-size: 0.9rem; color: #cbd5e1;">Status: <span style="font-weight:600; color: {'#10b981' if status == 'expanded' else '#f59e0b'};">{status.upper()}</span></div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if node_popped.lower() == "jokowi":
                        st.info("💡 **Jembatan Lintas Bubble:** Akun `@jokowi` adalah akun hub utama yang mengumpulkan beragam cluster. Interaksinya tidak hanya datang dari pihak yang pro (pendukung), tetapi juga pihak kontra (oposisi) maupun pihak netral terhadap pemerintah. Hal ini membuatnya sering bertindak sebagai jembatan lintasan terpendek antar bubble.")
                    elif len(neighbors_eval) > 8:
                        st.info(f"💡 **Akun Hub:** Akun `@{node_popped}` bertindak sebagai node jembatan lokal karena memiliki banyak interaksi sebutan (mention) dengan {len(neighbors_eval)} akun lainnya di dataset.")
                
                with col_right:
                    st.markdown(f"""
                    <div class="custom-card" style="margin-top: 10px;">
                        <div class="card-title" style="font-size: 1.0rem; margin-bottom: 5px; padding-bottom: 3px;">Detail Evaluasi Heuristik untuk @{node_popped}</div>
                        <div style="font-size: 0.8rem; color: #94a3b8; font-style: italic; margin-bottom: 10px;">Formula: f(n) = g(n) + 0.5 * (h_s(n) - h_g(n)) [Fwd] / g(n) + 0.5 * (h_g(n) - h_s(n)) [Bwd]</div>
                        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; text-align: center;">
                            <div style="background: rgba(15,23,42,0.5); padding: 8px; border-radius: 6px; border: 1px solid #334155;">
                                <div style="font-size: 1.2rem; font-weight:700; color: #38bdf8;">{f_val:.4f}</div>
                                <div style="font-size: 0.7rem; color: #94a3b8;">Prioritas f(n)</div>
                            </div>
                            <div style="background: rgba(15,23,42,0.5); padding: 8px; border-radius: 6px; border: 1px solid #334155;">
                                <div style="font-size: 1.2rem; font-weight:700; color: #818cf8;">{g_val:.4f}</div>
                                <div style="font-size: 0.7rem; color: #94a3b8;">Jarak Sosial g(n)</div>
                            </div>
                            <div style="background: rgba(15,23,42,0.5); padding: 8px; border-radius: 6px; border: 1px solid #334155;">
                                <div style="font-size: 1.2rem; font-weight:700; color: #a855f7;">{h_s_val:.4f}</div>
                                <div style="font-size: 0.7rem; color: #94a3b8;">Netral h_s(n)</div>
                            </div>
                            <div style="background: rgba(15,23,42,0.5); padding: 8px; border-radius: 6px; border: 1px solid #334155;">
                                <div style="font-size: 1.2rem; font-weight:700; color: #e11d48;">{h_g_val:.4f}</div>
                                <div style="font-size: 0.7rem; color: #94a3b8;">Asal h_g(n)</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown(f"**Jalur parsial aktif dari pencarian:** `{' ➔ '.join(['@'+n for n in path_so_far])}`")
                
                if collision:
                    col_node = selected_step["collision_node"]
                    col_cost = selected_step["collision_cost"]
                    col_path = selected_step["collision_path"]
                    st.markdown(f"""
                    <div style="background: rgba(16, 185, 129, 0.15); border: 1.5px solid #10b981; border-radius: 8px; padding: 15px; margin: 15px 0;">
                        <h4 style="color: #10b981; margin-top: 0; font-size:1.1rem;">🤝 Pertemuan Antrean (Collision) di @{col_node}!</h4>
                        <p style="margin: 5px 0; font-size: 0.9rem;">
                            Pencarian arah Forward dan Backward telah bertemu di node <b>@{col_node}</b>. Jalur jembatan lengkap terbentuk dengan total biaya heuristik <b>{col_cost:.4f}</b>.
                        </p>
                        <p style="margin: 5px 0; font-size: 0.85rem; font-style: italic; color: #10b981;">
                            Jalur terbentuk: {' ➔ '.join(['@'+n for n in col_path])}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                if neighbors_eval:
                    st.write(f"**Tetangga yang dievaluasi dari `@{node_popped}` pada langkah ini ({len(neighbors_eval)} akun):**")
                    nb_rows = []
                    for entry in neighbors_eval:
                        nb_rows.append({
                            "Akun Tetangga": f"@{entry['name']}",
                            "Bobot Interaksi": entry["weight"],
                            "Biaya Sisi g(u,v)": round(entry["edge_cost"], 4),
                            "Akumulasi g(n)": round(entry["g_new"], 4),
                            "Jarak Netral h_s(n)": round(entry["h_s"], 4),
                            "Jarak Source h_g(n)": round(entry["h_g"], 4),
                            "Prioritas f(n)": round(entry["f_new"], 4),
                            "Status Aksi": "Sudah Dikunjungi (Closed)" if entry["status"] == "visited" else "Dimasukkan ke Antrean (Pushed)"
                        })
                    df_nb = pd.DataFrame(nb_rows)
                    st.dataframe(df_nb, use_container_width=True)
                else:
                    st.info("Tidak ada tetangga yang dievaluasi pada langkah ini (semua tetangga sudah berada di Closed Set).")
                    
                frontier = selected_step["frontier"]
                if frontier:
                    st.write("**5 Antrean teratas berikutnya (Frontier):**")
                    frontier_items = []
                    for fn, gn, name in frontier:
                        frontier_items.append(f"`@{name}` (f={fn:.4f}, g={gn:.4f})")
                    st.markdown("  |  ".join(frontier_items))
                    
            with st.expander("📝 Lihat Log Lengkap MBDA* (Text Trace)"):
                log_lines = []
                for s in trace:
                    log_lines.append(
                        f"**[Langkah {s['step_num']}] Antrean {s['direction']}**\n"
                        f"  * Pop Node: `@{s['node_popped']}` (Status: {s['status'].upper()})\n"
                        f"  * Metrik: f={s['f_val']:.4f}, g={s['g_val']:.4f}, h_s={s['h_s_val']:.4f}, h_g={s['h_g_val']:.4f}\n"
                    )
                    if s["collision_detected"]:
                        log_lines.append(f"  * **🤝 PERTEMUAN ANTRIAN di `@{s['collision_node']}`! Biaya Jalur: {s['collision_cost']:.4f}**\n")
                    if s["neighbors_evaluated"]:
                        log_lines.append(f"  * Evaluasi {len(s['neighbors_evaluated'])} akun tetangga.\n")
                    log_lines.append("---")
                st.markdown("\n".join(log_lines))
            
            st.write("---")
            # Sequential Path display
            st.subheader("💡 Rekomendasi Akun Jembatan (Stepping Stones)")
            st.write("Kami merekomendasikan pengguna untuk mengikuti atau berinteraksi dengan akun-akun berikut secara bertahap untuk mendiversifikasi lini masa secara aman:")
            
            for idx, node in enumerate(path):
                node_cluster = active_partition.get(node, -1)
                
                # Retrieve one top tweet for context
                top_tweet = ""
                if node == new_username and dynamic_running and scraped_tweets:
                    top_tweet = scraped_tweets[0].get("text", "")
                else:
                    node_tweets = df_raw[df_raw["username"] == node] if df_raw is not None else pd.DataFrame()
                    if not node_tweets.empty:
                        top_tweet = node_tweets.sort_values(by="retweetCount", ascending=False).iloc[0]["text"]
                        
                # Check endpoints
                badge = ""
                if node == new_username:
                    badge = '<span class="step-cluster" style="background-color: #fbbf24; color: #000;">START</span>'
                elif node == goal_username:
                    badge = '<span class="step-cluster" style="background-color: #22d3ee; color: #000;">GOAL</span>'
                else:
                    badge = f'<span class="step-cluster">CLUSTER #{node_cluster}</span>'
                    
                st.markdown(f"""
                <div class="step-box">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span class="step-index">Langkah {idx + 1}</span>
                        <span style="font-weight: 700;">@{node}</span>
                        <div>{badge}</div>
                    </div>
                    {f'<div class="step-tweet"><b>Contoh Postingan Representatif:</b> "{top_tweet}"</div>' if top_tweet else ''}
                </div>
                """, unsafe_allow_html=True)
                
                # Special Hub Explanation Badge for Jokowi
                if node.lower() == "jokowi":
                    st.markdown("""
                    <div style="background: rgba(59, 130, 246, 0.1); border-left: 4px solid #3b82f6; border-radius: 0 8px 8px 0; padding: 0.8rem; margin: -0.5rem 0 0.8rem 0; font-size: 0.88rem; color: #cbd5e1; border: 1px solid #334155; border-left-width: 4px;">
                        💡 <b>Penjelasan Hub Jembatan:</b> Akun @jokowi sering muncul sebagai jembatan dalam pencarian MBDA* karena akun ini adalah akun yang cukup memenuhi untuk mengumpulkan beragam cluster di sini. Hal ini terjadi karena interaksi dan sebutannya (mentions) tidak hanya datang dari pihak yang pro pemerintah saja, melainkan juga dari pihak kontra (oposisi) maupun pihak netral. Karakteristik ini membuat @jokowi memiliki bobot mention yang sangat kuat dari berbagai penjuru cluster, sehingga bertindak sebagai jembatan lintas bubble untuk menyeimbangkan bias informasi.
                    </div>
                    """, unsafe_allow_html=True)
                
    # Tab 3: Graph Visualization
    with tab3:
        st.markdown("## 🕸️ Peta Jaringan Mention")
        st.write("Tata letak jaringan mention interaktif dari Komponen Terhubung Terbesar (LCC). Jalur yang ditemukan oleh algoritma MBDA* disorot dengan warna emas.")
        
        if "path_results" in st.session_state and st.session_state["path_results"]["source"] == new_username:
            res = st.session_state["path_results"]
            path = res["path"]
            goal = res["goal"]
            
            with st.spinner("Membuat plot visualisasi jaringan..."):
                fig = render_graph_path(
                    active_G, active_partition, path, 
                    new_username, goal, base_positions, 
                    new_username=new_username if dynamic_running else None
                )
                st.pyplot(fig)
        else:
            # Draw standard LCC graph highlighting nothing
            with st.spinner("Membuat plot dasar jaringan..."):
                fig = render_graph_path(active_G, active_partition, [], new_username, "", base_positions)
                st.pyplot(fig)
                st.caption("Jalankan pencari rute di Tab 2 untuk menyorot jalur pada graf.")

    # Tab 4: Algorithm Explanation
    with tab4:
        st.markdown("## 📖 Formulasi Akademik & Cara Kerja MBDA*")
        st.markdown("""
        ### **Konsep Dasar Mitigasi Filter Bubble**
        Algoritma **Modified Bidirectional A* (MBDA*)** merupakan metode pencarian rute dalam jaringan sosial (*mention graph*) yang dirancang khusus untuk memitigasi fenomena **Filter Bubble** (gelembung penyaringan opini). Secara akademis, filter bubble menyebabkan polarisasi informasi karena pengguna hanya terpapar pada konten yang sesuai dengan preferensi mereka. 
        
        MBDA* memecahkan masalah ini dengan cara tidak hanya mencari jalur terpendek secara relasi sosial (*mention*), melainkan juga memandu pengguna secara bertahap (*stepping stones*) menuju perspektif netral atau berimbang dengan memanfaatkan analisis kesamaan konten berbasis **TF-IDF (Term Frequency-Inverse Document Frequency)**.

        ---

        ### 🧪 **Fungsi Evaluasi Dua Arah (Bidirectional)**
        Pencarian dilakukan secara simultan dari dua ujung antrean:
        1. **Pencarian Maju (Forward):** Bergerak dari **Source ($S$)**—akun pengguna yang bias—menuju target.
        2. **Pencarian Mundur (Backward):** Bergerak dari **Goal ($G$)**—akun penyedia konten netral—menuju pengguna.
        
        Prioritas eksplorasi node ditentukan oleh fungsi evaluasi dinamis $f(n) = g(n) + h(n)$ berikut:
        
        * **Arah Forward (Source $\\rightarrow$ Goal):**
          $$f_s(n) = g(S,n) + \\frac{1}{2} [ h_s(n) - h_g(n) ]$$
          
        * **Arah Backward (Goal $\\rightarrow$ Source):**
          $$f_g(n) = g(G,n) + \\frac{1}{2} [ h_g(n) - h_s(n) ]$$
          
        ---
        
        ### 📐 **Penjelasan Komponen Matematis & Intuisi Akademik**
        
        1. **Metrik Jarak Sosial ($g(n)$) — Aspek Konektivitas Jejaring:**
           Mengukur kedekatan relasi komunikasi antar-akun dalam graf. Semakin sering akun $u$ menyebut (*mention*) atau membalas tweet akun $v$, semakin kuat ikatan sosial mereka. Secara matematis, biaya (*cost*) interaksi dirumuskan berbanding terbalik dengan bobot hubungan:
           $$cost(u, v) = \\max\\left(0.1, \\frac{1.0}{\\text{weight}(u,v) + 1}\\right)$$
           *Intuisinya:* Rute rekomendasi diutamakan melewati akun-akun yang terhubung secara riil di dalam jaringan agar transisi informasi terasa natural bagi pengguna.
           
        2. **Jarak Konten ke Profil Awal ($h_g(n)$) — Penolak Polarisasi:**
           Mengukur tingkat ketidakmiripan konten tweet akun $n$ terhadap konten historis pengguna awal ($S$) menggunakan *Cosine Distance*:
           $$h_g(n) = 1 - \\text{CosineSimilarity}(T_n, T_S)$$
           
        3. **Jarak Konten ke Profil Netral ($h_s(n)$) — Penarik Netralitas:**
           Mengukur tingkat ketidakmiripan konten tweet akun $n$ terhadap profil teks acuan netral ($G$):
           $$h_s(n) = 1 - \\text{CosineSimilarity}(T_n, T_{\\text{neutral}})$$
        
        ---
        
        ### 🛡️ **Bagaimana Heuristik MBDA* Memecah Filter Bubble?**
        Inti inovasi MBDA* terletak pada kombinasi perbedaan heuristik $[h_s(n) - h_g(n)]$ pada pencarian maju:
        * **Mendorong Keluar dari Bubble ($h_g(n)$ membesar):** Algoritma memberikan penalti pada jalur yang hanya berputar-putar di sekitar opini lama pengguna (ketika kontennya mirip dengan $S$, nilai $h_g(n)$ kecil, sehingga $f_s(n)$ menjadi lebih besar/mahal).
        * **Menarik ke Arah Netralitas ($h_s(n)$ mengecil):** Algoritma memprioritaskan akun-akun yang kontennya mulai bergeser mendekati topik netral atau berimbang (nilai $h_s(n)$ mengecil, membuat biaya $f_s(n)$ lebih murah).
        
        **Kesimpulan Akademis:** Melalui formulasi ini, algoritma menghasilkan rangkaian akun jembatan yang membawa pengguna keluar dari polarisasi informasi secara perlahan (gradual), meminimalkan risiko kejutan konten (*content shock*) yang sering kali menyebabkan penolakan informasi baru oleh pengguna (*backfire effect*).
        """)
else:
    st.info("👈 Silakan masukkan username Twitter/X di panel samping (sidebar) dan jalankan analisis, atau pilih Akun Demo untuk mulai bereksplorasi.")
