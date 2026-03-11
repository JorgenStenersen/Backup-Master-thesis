import src.tree as tree

time_str = "2025-10-09 21:00:00+00:00"
n = 3

# Build two bundles with the same seed — every node ID should match exactly
tree1 = tree.build_scenario_tree(time_str, n, seed=42)
tree2 = tree.build_scenario_tree(time_str, n, seed=42)

ids_1 = set(tree1["nodes"].keys())
ids_2 = set(tree2["nodes"].keys())

print(f"Tree 1 nodes: {len(ids_1)}")
print(f"Tree 2 nodes: {len(ids_2)}")
print(f"Intersection: {len(ids_1 & ids_2)}")
print(f"Same-seed match: {ids_1 == ids_2}")  # should be True

# Now build with a different seed — some nodes may overlap if they share values
tree3 = tree.build_scenario_tree(time_str, n, seed=100)
ids_3 = set(tree3["nodes"].keys())

overlap = ids_1 & ids_3
print(f"\nTree 3 nodes: {len(ids_3)}")
print(f"Overlap with tree 1: {len(overlap)}")
print(f"Shared node IDs: {overlap if overlap else 'none'}")

# Print a few example IDs to see the format
for name in list(ids_1)[:5]:
    node = tree1["nodes"][name]
    print(f"  stage={node.stage}  id={name}")