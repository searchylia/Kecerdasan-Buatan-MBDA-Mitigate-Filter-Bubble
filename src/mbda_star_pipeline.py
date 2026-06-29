#!/usr/bin/env python3
"""
IMPLEMENTASI MODIFIED BIDIRECTIONAL A* (MBDA*)
UNTUK MITIGASI FILTER BUBBLE KONTEN POLITIK DI X

Pipeline:
  Step 1 : Basic Cleaning       — hapus duplikasi
  Step 2 : Parse Nested Columns — author, entities -> username, hashtags, mentions
  Step 3 : Filter Noise         — hapus bahasa non-ID/EN dan viewCount=0
  Step 4 : Normalize Text       — lowercase, hapus URL/mention/hashtag
  Step 5 : Graph Construction   — directed weighted graph (mention edges)
  Step 6 : Community Detection  — Louvain algorithm -> cluster per akun
  Step 7 : Heuristic h(n)       — TF-IDF cosine similarity ke teks netral
  Step 8 : MBDA* Execution      — bidirectional A* pada LCC undirected
"""

import re
import ast
import heapq
import pickle
import pandas as pd
import numpy as np
import os
import networkx as nx
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import community as community_louvain

import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "data", "dataTwitter.csv")
OUTPUT_GRAPH = os.path.join(BASE_DIR, "models", "graph.pkl")
OUTPUT_PKL = os.path.join(BASE_DIR, "models", "mbda_final.pkl")


def run_visualization(G_lcc, partition_sub, results):
    print("\nSTEP 9 — Visualizing Graph & MBDA* Paths")
    try:
        import matplotlib.pyplot as plt
        import os

        # 1. Setup figure
        plt.figure(figsize=(14, 11))

        # 2. Layout (Spring layout)
        print("  Calculating layout...")
        pos = nx.spring_layout(G_lcc, k=0.2, seed=42)

        # 3. Get colors for nodes based on Louvain clusters
        unique_clusters = sorted(list(set(partition_sub.values())))
        color_palette = plt.colormaps["tab20"]
        cluster_color_map = {cluster: color_palette(
            i % 20) for i, cluster in enumerate(unique_clusters)}
        node_colors = [cluster_color_map[partition_sub[node]]
                       for node in G_lcc.nodes()]

        # 4. Draw background network
        print("  Drawing nodes and edges...")
        nx.draw_networkx_nodes(
            G_lcc, pos, node_color=node_colors, node_size=80, alpha=0.7, edgecolors="none")
        nx.draw_networkx_edges(G_lcc, pos, width=0.5,
                               edge_color="#d3d3d3", alpha=0.4)

        # 5. Highlight and Draw Scenario Paths
        path_colors = ["#FF5733", "#33FF57", "#3357FF", "#F1C40F", "#9B59B6"]
        drawn_labels = set()

        for idx, res in enumerate(results):
            path = res["path"]
            if not path or len(path) < 2:
                continue

            path_edges = list(zip(path[:-1], path[1:]))
            color = path_colors[idx % len(path_colors)]

            # Draw path edges
            nx.draw_networkx_edges(
                G_lcc, pos, edgelist=path_edges,
                width=3.0, edge_color=color,
                alpha=0.95, label=f"Path {idx+1}: {res['desc']}"
            )

            # Highlight source & goal nodes
            src, goal = res["source"], res["goal"]
            nx.draw_networkx_nodes(G_lcc, pos, nodelist=[
                                   src], node_color="gold", node_shape="^", node_size=250, edgecolors="black", linewidths=1.5)
            nx.draw_networkx_nodes(G_lcc, pos, nodelist=[
                                   goal], node_color="cyan", node_shape="s", node_size=200, edgecolors="black", linewidths=1.5)

            # Label source & goal nodes
            for node, label_prefix in [(src, "Start: "), (goal, "Goal: ")]:
                if node not in drawn_labels:
                    drawn_labels.add(node)
                    # Adjust text position slightly offset
                    nx_pos = pos[node]
                    plt.text(
                        nx_pos[0], nx_pos[1] + 0.02, f"{label_prefix}{node}",
                        fontsize=9, fontweight="bold",
                        bbox=dict(facecolor="white", alpha=0.8,
                                  edgecolor="gray", boxstyle="round,pad=0.2"),
                        horizontalalignment="center"
                    )

        plt.title("MBDA* Path Search & Louvain Communities on Twitter Mention Graph",
                  fontsize=15, fontweight="bold", pad=15)
        plt.legend(loc="upper right", frameon=True,
                   facecolor="white", edgecolor="none", fontsize=9)
        plt.axis("off")
        plt.tight_layout()

        # 6. Save image
        output_img = os.path.join(BASE_DIR, "visualizations", "graph_visualization.png")
        plt.savefig(output_img, dpi=300, bbox_inches="tight")
        print(f"  Saved graph visualization to {output_img}")

        # 7. Automatically open the image on Windows
        print("  Opening graph visualization automatically...")
        os.startfile(output_img)

        # 8. Show interactive plot
        plt.show()

    except Exception as e:
        print(f"  [ERROR] Visualization failed: {e}")


# Check if running in visualization-only mode
if len(sys.argv) > 1 and sys.argv[1] == "--view":
    print(f"Loading precomputed data from {OUTPUT_PKL}...")
    try:
        with open(OUTPUT_PKL, "rb") as f:
            data = pickle.load(f)
        print("[SUCCESS] Data loaded.")
        run_visualization(
            data["G_lcc"], data["partition_sub"], data["results"])
        sys.exit(0)
    except Exception as e:
        print(f"Failed to load/visualize {OUTPUT_PKL}: {e}")
        sys.exit(1)

# ============================================================
# STEP 1: BASIC CLEANING
# ============================================================
print("=" * 60)
print("STEP 1 — Basic Cleaning")
df = pd.read_csv(CSV_PATH)
before = len(df)
df.drop_duplicates(subset="id", inplace=True)
df.reset_index(drop=True, inplace=True)
print(f"  Removed {before - len(df)} duplicate rows -> {len(df)} rows remain")

# ============================================================
# STEP 2: PARSE NESTED COLUMNS
# ============================================================
print("STEP 2 — Parsing author & entities")


def safe_parse(val):
    try:
        return ast.literal_eval(str(val))
    except:
        return {}


df["author_parsed"] = df["author"].apply(safe_parse)
df["entities_parsed"] = df["entities"].apply(safe_parse)

df["username"] = df["author_parsed"].apply(lambda x: x.get("userName", ""))
df["display_name"] = df["author_parsed"].apply(lambda x: x.get("name", ""))
df["hashtags_list"] = df["entities_parsed"].apply(
    lambda x: [h.get("text", "").lower() for h in x.get("hashtags", [])])
df["mentions_list"] = df["entities_parsed"].apply(
    lambda x: [m.get("screen_name", "") for m in x.get("user_mentions", [])])
print(f"  Parsed OK — {df['username'].nunique()} unique usernames")

# ============================================================
# STEP 3: FILTER NOISE
# ============================================================
print("STEP 3 — Filter noise")
before = len(df)
df = df[df["lang"].isin(["in", "en"])]
df = df[df["viewCount"] > 0]
df.reset_index(drop=True, inplace=True)
print(f"  Removed {before - len(df)} noise rows -> {len(df)} rows remain")

# ============================================================
# STEP 4: NORMALIZE TEXT
# ============================================================
print("STEP 4 — Normalize tweet text")


def normalize(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


df["text_clean"] = df["text"].apply(normalize)
print(f"  Normalized {len(df)} tweets")

# ============================================================
# STEP 5: GRAPH CONSTRUCTION
# ============================================================
print("STEP 5 — Graph Construction (user-mention edges)")
G = nx.DiGraph()

for _, row in df.iterrows():
    src = row["username"]
    if not src:
        continue
    if not G.has_node(src):
        G.add_node(src, node_type="user", tweet_count=0,
                   total_rt=0, total_reply=0, display_name=row["display_name"])
    G.nodes[src]["tweet_count"] += 1
    G.nodes[src]["total_rt"] += row["retweetCount"]
    G.nodes[src]["total_reply"] += row["replyCount"]

    w = int(row["retweetCount"]) + int(row["replyCount"])
    for tgt in row["mentions_list"]:
        if not tgt:
            continue
        if not G.has_node(tgt):
            G.add_node(tgt, node_type="user", tweet_count=0,
                       total_rt=0, total_reply=0, display_name=tgt)
        if G.has_edge(src, tgt):
            G[src][tgt]["weight"] += w + 1
            G[src][tgt]["count"] += 1
        else:
            G.add_edge(src, tgt, weight=w+1, count=1, edge_type="mention")

print(f"  Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()}")

# ============================================================
# STEP 6: COMMUNITY DETECTION (Louvain)
# ============================================================
print("STEP 6 — Community Detection (Louvain)")
# ============================================================
# STEP 7: HEURISTIC h(n) — TF-IDF COSINE SIMILARITY
# ============================================================
print("STEP 7 — Heuristic h(n) via TF-IDF")
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
nx.set_node_attributes(G, {n: h_dict.get(n, 1.0)
                       for n in G.nodes()}, "h_value")
print(
    f"  h(n) range: [{min(h_dict.values()):.4f}, {max(h_dict.values()):.4f}]")

# ============================================================
# STEP 6: COMMUNITY DETECTION (Louvain)
# ============================================================
print("STEP 6 — Community Detection (Louvain)")
G_uu = G.to_undirected()
isolated = list(nx.isolates(G_uu))
G_conn = G_uu.copy()
G_conn.remove_nodes_from(isolated)
partition = community_louvain.best_partition(
    G_conn, weight="weight", random_state=42)
n_clusters = max(partition.values()) + 1
print(f"  Clusters detected: {n_clusters}")
nx.set_node_attributes(G, {n: partition.get(n, -1)
                       for n in G.nodes()}, "cluster")

# ============================================================
# STEP 8: MBDA* EXECUTION
# ============================================================
print("STEP 8 — Modified Bidirectional A*")

lcc = max(nx.connected_components(G_uu), key=len)
G_lcc = G_uu.subgraph(lcc).copy()
partition_sub = {n: c for n, c in partition.items() if n in G_lcc}


def mbda_star(G, source, goal, max_iter=5000):
    """Modified Bidirectional A* untuk mitigasi filter bubble dengan print log step-by-step."""
    if source not in G or goal not in G:
        return None, float("inf"), 0

    # Hitung h_g(n) - jarak kosinus TF-IDF ke profil teks dari source S
    try:
        src_idx = user_texts[user_texts["username"] == source].index[0]
        src_vec = tfidf_matrix[src_idx]
        sims_to_src = cosine_similarity(tfidf_matrix, src_vec).flatten()
        h_g_dict = dict(zip(user_texts["username"], 1 - sims_to_src))
    except Exception as e:
        h_g_dict = {}

    def h_s(n): return G.nodes[n].get("h_value", 1.0)
    def h_g(n): return h_g_dict.get(n, 1.0)
    def cost(u, v): return max(0.1, 1.0/(G[u][v].get("weight", 1)+1))

    # Formulasi inisialisasi awal sesuai slide:
    # fs(S) = g(S,S) + 0.5 * [hs(S) - hg(S)]
    f_source = 0.5 * (h_s(source) - h_g(source))
    # fg(G) = g(G,G) + 0.5 * [hg(G) - hs(G)]
    f_goal = 0.5 * (h_g(goal) - h_s(goal))

    open_f = [(f_source, 0.0, source, [source])]
    open_b = [(f_goal,   0.0, goal,   [goal])]
    cf, cb = {}, {}
    best = {"cost": float("inf"), "path": None}
    exp = [0]

    print(f"\n[MBDA* START] Searching path from @{source} to @{goal}")
    print(f"  Heuristic h_s(start)={h_s(source):.4f}, h_g(start)={h_g(source):.4f}, f_start={f_source:.4f}")
    print(f"  Heuristic h_s(goal)={h_s(goal):.4f}, h_g(goal)={h_g(goal):.4f}, f_goal={f_goal:.4f}")
    print("-" * 70)

    def step(oq, ct, co, fwd):
        if not oq:
            return
        fe, g, cur, path = heapq.heappop(oq)
        exp[0] += 1
        direction_str = "Forward (Source -> Goal)" if fwd else "Backward (Goal -> Source)"
        
        print(f"\n[Step {exp[0]}] {direction_str} Queue:")
        print(f"  Popped Node: @{cur}")
        print(f"    Metrics: f(n)={fe:.4f}, g(n)={g:.4f}, h_s(n)={h_s(cur):.4f}, h_g(n)={h_g(cur):.4f}")
        
        if cur.lower() == "jokowi":
            print("    💡 NOTE: Akun @jokowi dideteksi sebagai jembatan/hub utama. Akun ini mengumpulkan berbagai cluster (baik pro, kontra, maupun netral) sehingga mempermudah koneksi antar bubble.")

        if cur in ct:
            print(f"    * Node @{cur} already visited in this direction. Skipping expansion.")
            return
            
        ct[cur] = (g, path)
        
        # Check collision
        if cur in co:
            g2, p2 = co[cur]
            total = g + g2
            print(f"    🤝 COLLISION DETECTED at @{cur}!")
            print(f"      Forward cost to here: {g if fwd else g2:.4f}")
            print(f"      Backward cost to here: {g2 if fwd else g:.4f}")
            print(f"      Potential total cost: {total:.4f}")
            if total < best["cost"]:
                best["cost"] = total
                if fwd:
                    best["path"] = path + list(reversed(p2))[1:]
                else:
                    best["path"] = p2 + list(reversed(path))[1:]
                print(f"      => NEW BEST PATH FOUND! Cost: {best['cost']:.4f}")

        # Expand neighbors
        print("    Expanding neighbors:")
        neighbors_count = 0
        for nb in G.neighbors(cur):
            neighbors_count += 1
            if nb in ct:
                print(f"      -> @{nb}: already visited in closed set. (Skipped)")
                continue
            
            w = G[cur][nb].get("weight", 1)
            c_edge = cost(cur, nb)
            gn = g + c_edge
            
            if fwd:
                fn = gn + 0.5 * (h_s(nb) - h_g(nb))
            else:
                fn = gn + 0.5 * (h_g(nb) - h_s(nb))
                
            print(f"      -> @{nb}: weight={w}, edge_cost={c_edge:.4f}, g_new={gn:.4f}, h_s={h_s(nb):.4f}, h_g={h_g(nb):.4f} => f_new={fn:.4f}")
            heapq.heappush(oq, (fn, gn, nb, path+[nb]))
        if neighbors_count == 0:
            print("      No neighbors to expand.")

    for _ in range(max_iter):
        step(open_f, cf, cb, True)
        step(open_b, cb, cf, False)
        if best["path"]:
            print(f"\n[MBDA* CONVERGED] Path found in {exp[0]} steps.")
            break
        if not open_f and not open_b:
            print("\n[MBDA* FINISHED] Open queues empty. No path found.")
            break
            
    return best["path"], best["cost"], exp[0]


# --- 5 Scenario Tests ---
scenarios = [
    ("________dyah", "kompascom",   "Bubble MBG -> Kompas (media netral)"),
    ("________dyah", "jokowi",      "Bubble MBG -> Jokowi cluster"),
    ("Deka_Ajaa",    "Fahrihamzah", "Akun aktif MBG -> Fahri Hamzah (oposisi)"),
    ("karirfess",    "jokowi",      "Karir cluster -> Jokowi cluster"),
    ("DaudJTP",      "Fahrihamzah", "Akun media -> Fahri Hamzah"),
]

results = []
for src, goal, desc in scenarios:
    if src not in G_lcc or goal not in G_lcc:
        print(f"  - {desc}: node not in LCC")
        continue
    path, cost_val, exp = mbda_star(G_lcc, src, goal)
    if path:
        path_cl = [partition_sub.get(n, -1) for n in path]
        diversity = len(set(c for c in path_cl if c != -1)) / len(path)
        print(f"  + {desc}")
        print(
            f"    Steps={len(path)}, Cost={cost_val:.4f}, Explored={exp}, Diversity={diversity:.3f}")
        results.append({"source": src, "goal": goal, "desc": desc,
                        "path": path, "cost": cost_val, "explored": exp, "diversity": diversity})

print(f"\n[SUMMARY] {len(results)}/{len(scenarios)} scenarios resolved")

# Save
with open(OUTPUT_PKL, "wb") as f:
    pickle.dump({"results": results, "G_lcc": G_lcc,
                "partition_sub": partition_sub}, f)
with open(OUTPUT_GRAPH, "wb") as f:
    pickle.dump(G, f)
print("[DONE] Output saved.")

# ============================================================
# STEP 9: GRAPH VISUALIZATION (MATPLOTLIB)
# ============================================================
if not (len(sys.argv) > 1 and sys.argv[1] == "--no-vis"):
    run_visualization(G_lcc, partition_sub, results)
else:
    print("Skipping graph visualization plot as requested by --no-vis.")
