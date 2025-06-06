from cobra.io import load_model

# Load the simplified E. coli metabolic model
model = load_model("textbook")

print(f"Model: {model.id}")
print(model.summary())
print(f"Reactions: {len(model.reactions)}") # biochemical transformations
print(f"Metabolites: {len(model.metabolites)}") # chemical compounds involved
print(f"Genes: {len(model.genes)}") # genes associated with enzymes that catalyze reactions

# Run Flux Balance Analysis (FBA)
default_solution = model.optimize()

# Default objective function is biomass production reaction
print(f"\nMax growth rate: {default_solution.objective_value:.4f}") # units: 1/hour

print("\nTop flux-carrying reactions:")
print(default_solution.fluxes.sort_values(ascending=False).head(5))

'''
# Print all reaction IDs
print("\nAll reaction IDs:")
for reaction in model.reactions:
    print(reaction.id)
'''

# Key fluxes (consumes glucose and oxygen -> produces biomass)
reaction_ids = ['EX_glc__D_e', 'EX_o2_e', 'EX_nh4_e', 'EX_pi_e', 'Biomass_Ecoli_core']

# Inspect lower and upper bounds
print('\nLower and upper bounds for glucose and oxygen:')
print(f'EX_glc__D_e:\t{model.reactions.get_by_id("EX_glc__D_e").bounds}')
print(f'EX_o2_e:\t{model.reactions.get_by_id("EX_o2_e").bounds}')

# Inspect reaction descriptions
print('\nReaction descriptions:')
for rxn_id in reaction_ids:
    if rxn_id in model.reactions:
        rxn = model.reactions.get_by_id(rxn_id)
        print(f"{rxn.id} — {rxn.name}")
        print(f"  Equation: {rxn.reaction}")
        #print(f"  Subsystem: {rxn.subsystem}")

print("\nKey reaction fluxes:")
for rxn_id in reaction_ids:
    if rxn_id in model.reactions:
        rxn = model.reactions.get_by_id(rxn_id)
        print(f"{rxn.id}: {default_solution.fluxes[rxn.id]:.4f}")
    else:
        print(f"Reaction {rxn_id} not found in the model.")

# Test with less glucose and oxygen uptake
model.reactions.get_by_id("EX_glc__D_e").lower_bound = -1
less_glucose_solution = model.optimize()

model.reactions.get_by_id("EX_glc__D_e").lower_bound = -10 # back to default
model.reactions.get_by_id("EX_o2_e").lower_bound = -1
less_oxygen_solution = model.optimize()

print(f"\nDefault growth rate: {default_solution.objective_value:.4f}")
print(f"Growth rate with less glucose: {less_glucose_solution.objective_value:.4f}")
print(f"Growth rate with less oxygen: {less_oxygen_solution.objective_value:.4f}")

'''
# Set the objective to lactate production
model.objective = "EX_lac__D_e"
print('\nObjective set to lactate production')

solution = model.optimize()

print(f"\nMax lactate production: {solution.objective_value:.4f}\n")
'''