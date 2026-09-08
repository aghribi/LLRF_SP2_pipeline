# pip install pyvis
# python pipeline_diagram_pyvis_source.py
# Then open pipeline_interactif.html in a browser (pyvis loads vis-network.js from a
# CDN, so this needs to be opened directly, not viewed inside a sandboxed preview).
#
# All node positions are fixed by hand (physics disabled) and every non-axis-aligned
# connection is routed through invisible waypoint nodes so every drawn segment is
# purely horizontal or purely vertical -- vis-network has no native orthogonal/
# Manhattan edge routing, so this is the standard workaround: a "bend" is a real
# (but invisible) node, and the arrowhead is only enabled on the final segment
# into a real, visible node.
#
# Update, same day 2026-09-05: Precursor's Isolation Forest/LOF/Mahalanobis/PCA
# reconstruction/DBSCAN/Ensemble and Explainability's LIME/Saliency were "not
# yet built" earlier the same day -- implemented and verified against V6 a
# few hours later, so both nodes are merged into one "done" node each rather
# than staying split (the TODO color/dashed style is no longer used anywhere).

from pyvis.network import Network

net = Network(directed=True, height="850px", width="100%", bgcolor="#FBF6EC", font_color="#2C2013")

DONE = "#F4E3C8"      # implemented, verified in code
ADDED = "#DCEDEA"     # added here, missing from the original diagram
STAGE = "#F5CBA7"     # data source / terminal node

BOX_FONT = {"size": 14, "face": "arial", "multi": False, "align": "center"}
WIDTH = {"maximum": 300}


def add_box(node_id, label, x, y, color, border="#8A4A22", dashed=False, shape="box"):
    props = {}
    if dashed:
        props["shapeProperties"] = {"borderDashes": [6, 4]}
    net.add_node(
        node_id, label=label, shape=shape, x=x, y=y, color={"background": color, "border": border},
        font=BOX_FONT, widthConstraint=WIDTH if shape == "box" else None,
        physics=False, fixed={"x": True, "y": True}, **props,
    )


def add_waypoint(node_id, x, y):
    net.add_node(
        node_id, label="", shape="dot", size=1, x=x, y=y,
        color={"background": "rgba(0,0,0,0)", "border": "rgba(0,0,0,0)"},
        physics=False, fixed={"x": True, "y": True},
    )


def add_edge(a, b, arrow=True, dashed=False, label=None, color="#8A4A22"):
    net.add_edge(
        a, b, arrows="to" if arrow else "",
        smooth=False, dashes=dashed, color=color,
        label=label, font={"size": 12, "color": "#5B5142", "strokeWidth": 0} if label else None,
    )


# ---- real, visible nodes ----
add_box("data", "Waveforms\nraw data", 0, 0, STAGE, shape="database")
add_box("prep", "Pre-processing\nNormalisation, filtering, segmentation", 420, 0, DONE)
add_box("feat", "Feature engineering\nPhysics-based, statistical, rate-of-change", 860, 0, DONE)

add_box("anomaly", "Anomaly scoring (unsupervised)\nIsolation Forest, LOF, DBSCAN, Autoencoder, VAE", 1340, -320, DONE)
add_box("binary", "Binary classification (supervised)\nLogReg, RF, XGBoost, SVM, DNN, CNN, Ensemble", 1340, 320, DONE)

add_box("dec1", "Anomalous?", 1820, 0, "#DCEDEA", border="#0F6E63", shape="diamond")

add_box("precursor",
        "Precursor analysis\nRandom Forest, LSTM Classifier, LSTM Autoencoder,\n"
        "Isolation Forest, LOF, Mahalanobis distance,\nPCA reconstruction error, DBSCAN, Ensemble",
        2320, -320, DONE)

net.add_node("junction", label="", shape="dot", size=8, x=3360, y=0,
              color={"background": "#8A4A22", "border": "#8A4A22"}, physics=False, fixed={"x": True, "y": True})

add_box("multilabel", "Multi-label classification (supervised)\nBinary Relevance, Classifier Chain, Label Powerset,\nCNN/Transformer/CNN-LSTM/MLP encoders, Random Forest", 3860, -180, DONE)
add_box("clustering", "Fault clustering (unsupervised)\nK-Means, DBSCAN, Hierarchical, Gaussian Mixture", 3860, 320, DONE)

add_box("rootcause", "Root cause ID\nRandom Forest -- added, missing from original diagram", 4420, -180, ADDED, border="#0F6E63")

add_box("subclass", "Subclass discovery\nK-Means, Hierarchical, Gaussian Mixture, Spectral, HDBSCAN", 4960, 80, DONE)

add_box("explain", "Explainability\nSHAP, LIME, Saliency", 5480, 80, DONE)

add_box("artifacts", "Artifacts /\noutputs", 6500, 0, STAGE)

# ---- invisible waypoints (one per right-angle bend) ----
add_waypoint("wp_fork1", 1340, 0)        # feat forks up/down to anomaly/binary
add_waypoint("wp_a2d", 1820, -320)       # anomaly across to dec1's column
add_waypoint("wp_b2d", 1820, 320)        # binary across to dec1's column
add_waypoint("wp_dec_no", 1820, -320)    # dec1 up to precursor row (No)
add_waypoint("wp_p2j", 2320, 0)          # precursor down to junction row
add_waypoint("wp_clear_top", 6500, -320)  # precursor bypass, far right then down
add_waypoint("wp_fork2", 3860, 0)        # junction forks up/down to multilabel/clustering
add_waypoint("wp_r2s", 4960, -180)       # rootcause down to subclass row
add_waypoint("wp_c2s", 4960, 320)        # clustering up to subclass row
add_waypoint("wp_e2a", 5480, 0)          # explain up to spine row, into artifacts

# ---- edges: every segment purely horizontal or vertical; arrow only on the ----
# ---- final hop into a real, visible node                                  ----
add_edge("data", "prep")
add_edge("prep", "feat")

add_edge("feat", "wp_fork1", arrow=False)
add_edge("wp_fork1", "anomaly")
add_edge("wp_fork1", "binary")

add_edge("anomaly", "wp_a2d", arrow=False)
add_edge("wp_a2d", "dec1")
add_edge("binary", "wp_b2d", arrow=False)
add_edge("wp_b2d", "dec1")

add_edge("dec1", "wp_dec_no", arrow=False, label="No")
add_edge("wp_dec_no", "precursor")
add_edge("dec1", "junction", label="Yes")

add_edge("precursor", "wp_p2j", arrow=False, label="signature found")
add_edge("wp_p2j", "junction")

add_edge("precursor", "wp_clear_top", arrow=False, dashed=True, label="clear")
add_edge("wp_clear_top", "artifacts", dashed=True)

add_edge("junction", "wp_fork2", arrow=False)
add_edge("wp_fork2", "multilabel")
add_edge("wp_fork2", "clustering")

add_edge("multilabel", "rootcause")

add_edge("rootcause", "wp_r2s", arrow=False)
add_edge("wp_r2s", "subclass")
add_edge("clustering", "wp_c2s", arrow=False)
add_edge("wp_c2s", "subclass")

add_edge("subclass", "explain")

add_edge("explain", "wp_e2a", arrow=False)
add_edge("wp_e2a", "artifacts")

net.set_options("""
var options = {
  "physics": { "enabled": false },
  "interaction": { "dragNodes": true, "zoomView": true, "hover": false },
  "edges": { "smooth": false }
}
""")

net.show("pipeline_interactif.html", notebook=False)
