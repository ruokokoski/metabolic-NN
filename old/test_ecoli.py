from cobra.io import load_model, read_sbml_model
import warnings
warnings.filterwarnings("ignore", message="Solver status is 'infeasible'")

# Load the simplified E. coli metabolic model
model = load_model("textbook")
#model = read_sbml_model("./models/e_coli_core.xml") # essentially the same as textbook

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

# Disable oxygen uptake
oxygen = model.reactions.get_by_id('EX_o2_e')
oxygen_lb_backup = oxygen.lower_bound
oxygen.lower_bound = 0.0

# Re-run FBA with no oxygen
anaerobic_solution = model.optimize()

print("\n=== Anaerobic growth test (oxygen uptake disabled) ===")
if anaerobic_solution.status == 'optimal':
    print(f"Biomass without oxygen: {anaerobic_solution.objective_value:.4f}")
    print("\nTop flux-carrying reactions under anaerobic conditions:")
    print(anaerobic_solution.fluxes.sort_values(ascending=False).head(5))
else:
    print("Optimization failed: no feasible solution without oxygen.")

# Restore original oxygen bound
oxygen.lower_bound = oxygen_lb_backup

# Define carbon sources to test
carbon_sources = [
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
carbon_exchanges = [f'EX_{met}' for met in carbon_sources]

# Enable all carbon source uptakes first
for ex in carbon_exchanges:
    if ex in model.reactions:
        model.reactions.get_by_id(ex).lower_bound = -10

# Test if any single carbon source is truly essential
non_redundant = []
threshold = 1e-9
for ex in carbon_exchanges:
    rxn = model.reactions.get_by_id(ex)
    original_lb = rxn.lower_bound
    rxn.lower_bound = 0

    sol = model.optimize()
    if sol.status != 'optimal' or sol.objective_value < threshold:
        non_redundant.append(ex)

    rxn.lower_bound = original_lb  # restore

print("\nCarbon sources that are truly essential:")
print(non_redundant)

# Remove carbon sources
uptake_exchanges = [
    rxn for rxn in model.exchanges
    if rxn.lower_bound < 0 and rxn.id not in carbon_exchanges
]
print("\nDefault exchange reactions that allow uptake (excluding carbon sources):")
print([rxn.id for rxn in uptake_exchanges])

# Find which of other exhange reactions are essential
essential = []
for rxn in uptake_exchanges:
    # save original lower bound
    lb_orig = rxn.lower_bound

    # disable uptake
    rxn.lower_bound = 0.0

    sol = model.optimize()
    failed = (sol.status != 'optimal')
    very_low = (sol.objective_value is not None and sol.objective_value < threshold)

    if failed or very_low:
        essential.append(rxn.id)

    # restore original lower bound
    rxn.lower_bound = lb_orig

print("\nMinimal essential composition (excluding carbon source):")
for rxn_id in essential:
    rxn = model.reactions.get_by_id(rxn_id)
    print(f"  {rxn.id}: {rxn.name}")
print(f"\nTotal essential uptake reactions: {len(essential)}")

print("\n=== Testing growth on different sugars ===")
for sugar_ex in carbon_exchanges:
    if sugar_ex not in model.reactions:
        print(f"{sugar_ex} not in model, skipping...")
        continue

    for ex in carbon_exchanges:
        if ex in model.reactions:
            model.reactions.get_by_id(ex).lower_bound = 0

    model.reactions.get_by_id(sugar_ex).lower_bound = -10

    sol = model.optimize()
    print(f"\nSugar: {sugar_ex}")
    print(f"  {sugar_ex} flux: {sol.fluxes[sugar_ex]:.4f}")
    print(f"  Biomass flux: {sol.fluxes['Biomass_Ecoli_core']:.4f}")

'''
# Print all reactions
print("\nAll reactions:")
for reaction in model.reactions:
    print(f"ID: {reaction.id}, name: {reaction.name}")


# Set the objective to lactate production
model.objective = "EX_lac__D_e"
print('\nObjective set to lactate production')

solution = model.optimize()

print(f"\nMax lactate production: {solution.objective_value:.4f}\n")
'''