import cobra
import networkx as nx
import matplotlib.pyplot as plt

# 1) Load the model
model = cobra.io.load_model("textbook")

# 2) Define a list of reaction IDs you want to focus on
#    Replace these IDs with whichever reactions interest you.
selected_rxns = [
    "EX_glc__D_e",      # Glucose exchange
    "GLCpts",           # Glucose PTS transport
    "PGI",              # Phosphoglucose isomerase
    "PFK",              # Phosphofructokinase
    "FBA",              # Fructose-bisphosphate aldolase
    "TPI",              # Triose-phosphate isomerase
    "GAPD",             # Glyceraldehyde-3-phosphate dehydrogenase
    "PGK",              # Phosphoglycerate kinase
    "PYK",              # Pyruvate kinase
    "EX_co2_e",         # CO₂ exchange
    "Biomass_Ecoli_core"  # Biomass reaction
]

# 3) Build a bipartite subgraph: reactions ↔ metabolites
G_sub = nx.DiGraph()
met_nodes = set()
rxn_nodes = set()

for rxn_id in selected_rxns:
    if rxn_id not in model.reactions:
        continue
    rxn = model.reactions.get_by_id(rxn_id)
    rxn_nodes.add(rxn_id)
    G_sub.add_node(rxn_id, bipartite="reaction")

    # For every metabolite in this reaction, add metabolite nodes and edges
    for met, coeff in rxn.metabolites.items():
        met_id = met.id
        met_nodes.add(met_id)
        G_sub.add_node(met_id, bipartite="metabolite")

        # If coeff < 0, metabolite is a reactant: met --> rxn
        # If coeff > 0, metabolite is a product: rxn --> met
        if coeff < 0:
            G_sub.add_edge(met_id, rxn_id)
        else:
            G_sub.add_edge(rxn_id, met_id)

# 4) Decide positions: place metabolites on x=0, reactions on x=1
pos = {}
for i, met_id in enumerate(sorted(met_nodes)):
    pos[met_id] = (0, i)

for j, rxn_id in enumerate(sorted(rxn_nodes)):
    pos[rxn_id] = (1, j)

# 5) Draw the subgraph
plt.figure(figsize=(8, 6))

# Draw metabolite nodes (blue circles)
nx.draw_networkx_nodes(
    G_sub, pos,
    nodelist=list(met_nodes),
    node_color="skyblue",
    node_shape="o",
    node_size=300,
    label="Metabolites"
)

# Draw reaction nodes (red squares)
nx.draw_networkx_nodes(
    G_sub, pos,
    nodelist=list(rxn_nodes),
    node_color="salmon",
    node_shape="s",
    node_size=300,
    label="Reactions"
)

# Draw edges
nx.draw_networkx_edges(
    G_sub, pos,
    arrowstyle="-|>",
    arrowsize=10,
    edge_color="gray",
    width=1
)

# Label every node
labels = {n: n for n in G_sub.nodes()}
nx.draw_networkx_labels(G_sub, pos, labels, font_size=8)

plt.title("Subnetwork: Selected Reactions & Their Metabolites")
plt.axis("off")
plt.legend(scatterpoints=1, fontsize=10)
plt.tight_layout()
plt.show()
