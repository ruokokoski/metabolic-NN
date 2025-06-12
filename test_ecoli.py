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

print("\nExchange reactions:")
exchange_count = 0
for rxn in model.reactions:
    if rxn.id.startswith('EX'):
        exchange_count += 1
        print(f"Reaction ID:   {rxn.id}")
        print(f"  Name:        {rxn.name}")
        print(f"  Bounds:      {rxn.bounds}")          # (lower_bound, upper_bound)
        print(f"  Equation:    {rxn.reaction}")
        # print(f"  Subsystem:   {rxn.subsystem}")
        # genes = [g.id for g in rxn.genes]
        # print(f"  Genes:       {genes}")
        print(f"  Flux:        {default_solution.fluxes[rxn.id]:.4f}")
        print("-" * 60)
print(f"Number of exchange reactions: {exchange_count}")

# Key fluxes (consumes glucose and oxygen -> produces biomass + byproducts)
reaction_ids = [
    'EX_glc__D_e',     # D-Glucose uptake
    'EX_o2_e',         # O2 uptake
    'EX_nh4_e',        # Ammonia uptake
    'EX_pi_e',         # Phosphate uptake
    'EX_h_e',          # Proton exchange
    'EX_co2_e',        # CO2 secretion
    'EX_h2o_e',        # H2O secretion
    'Biomass_Ecoli_core'  # Biomass production (growth)
]

print("\nKey reaction fluxes:")
for rxn_id in reaction_ids:
    rxn = model.reactions.get_by_id(rxn_id)
    print(f"{rxn.id}: {default_solution.fluxes[rxn.id]:.4f}")

# Define sugars to test
sugar_ids = [
    'glc__D_e',  # Glucose
    'fru_e',     # Fructose
    'lac__D_e',  # D-Lactate
    'pyr_e',     # Pyruvate (intermediate, not a sugar but a carbon source)
    'ac_e',      # Acetate
    'akg_e',     # Alpha-ketoglutarate
    'succ_e',    # Succinate
    'fum_e',     # Fumarate
    'mal__L_e'   # L-Malate
]
sugar_ex_ids = [f'EX_{sugar}' for sugar in sugar_ids]

print("\n=== Testing growth on different sugars ===")
for sugar_ex in sugar_ex_ids:
    if sugar_ex not in model.reactions:
        print(f"{sugar_ex} not in model, skipping...")
        continue

    for ex in sugar_ex_ids:
        if ex in model.reactions:
            model.reactions.get_by_id(ex).lower_bound = 0

    model.reactions.get_by_id(sugar_ex).lower_bound = -10

    sol = model.optimize()
    print(f"\nSugar: {sugar_ex}")
    print(f"  {sugar_ex} flux: {sol.fluxes[sugar_ex]:.4f}")
    print(f"  Biomass flux: {sol.fluxes['Biomass_Ecoli_core']:.4f}")

'''
# Set the objective to lactate production
model.objective = "EX_lac__D_e"
print('\nObjective set to lactate production')

solution = model.optimize()

print(f"\nMax lactate production: {solution.objective_value:.4f}\n")
'''