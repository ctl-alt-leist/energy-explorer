# Energy Explorer Project

## Installation and Setup

## The Structure of the Data

The dataset comprises approximately 150,000 entries representing energy storage systems across California. Each entry
includes key information to analyze the geographical and temporal distribution of energy storage capacity across various
sectors and fuel types. The main columns include:

- **Approval Date**: The date when each energy storage project received approval, allowing for time-series analysis of
  storage adoption rates.

- **Capacity [kW]**: The rated capacity added in each entry, representing the instantaneous contribution of the project.

- **Cumulative Capacity (kW)**: The cumulative sum of capacities up to each date for tracking the total integrated
  storage capacity.

- **Fuel Types**: A list of one or more fuel types for each entry, indicating the energy sources associated with the
  storage system. Up to four fuel types can be included per entry, enabling hybrid systems (e.g., Solar + Battery).

The fuel types in this dataset encompass a variety of sources, including renewable options like Solar, Wind, and Biogas,
as well as conventional sources like Natural Gas and Diesel Fuel. This mix allows analysis of hybrid configurations and
their contributions to overall capacity.

Two heatmaps illustrate the mean capacities associated with pairs of fuel types:

1. **Exclusive Fuel Type Pairs**:
   - This chart (Figure 1) shows the mean capacities for pairs of fuel types that appear exclusively together, without
     any additional types. This visual helps highlight combinations of fuel types that are strictly co-located and
     provides insight into specific hybrid configurations.
   - ![Exclusive Fuel Type Pairs](figures/fuel_pairs_exclusive_heatmap.png)

2. **Inclusive Fuel Type Pairs**:
   - This chart (Figure 2) includes mean capacities for pairs of fuel types but allows for additional types in each
     entry. While this shows a broader view of how fuel types are paired, it does not isolate exclusive combinations,
     which means we cannot determine which additional types may be present for a given pair.
   - ![Inclusive Fuel Type Pairs](figures/fuel_pairs_inclusive_heatmap.png)

Both figures have been scaled to 0.7 of their original size to fit the document layout better. These visuals provide
insight into typical capacity sizes associated with different fuel combinations, supporting analysis of trends in hybrid
energy storage systems.
