# Data Dictionary

Dataset: 4934 recipes total (3200 labelled training rows, 1734 unlabelled test rows).
Data itself is not included in this repository.

| Column | Type | Description |
|---|---|---|
| `y` | categorical (train only) | Cuisine label<br>`1` = American, `2` = Italian.<br>Not present in the test rows. |
| `salt`<br>…<br>`pepper`<br> (40 columns) | continuous, ≥ 0 | TF–IDF relevance score of the named ingredient in the recipe.<br>`0` means the ingredient is absent from the recipe.<br>Non-zero values combine how often the ingredient appears in the recipe (term frequency) with how rare it is across the full recipe collection (inverse document frequency). |

## The 40 ingredient columns
salt, sugar, water, cereals, oil, flour, fruits, milk, seeds, onion, garlic, chocolate, yeast, egg, vinegar, tomato, cream, rice, corn, cheese, butter, red.meat, white.meat, potato, honey, paprika, mustard, turmeric, ginger, parsley, wine, chili, mushrooms, bread, pasta, fish, seafood, veggies, legumes, pepper

## Notes
- The feature matrix is sparse: about 81.6% of ingredient cells are zero across the dataset.
- Some columns are broad categories rather than single ingredients (e.g. `cheese`, `red.meat`, `white.meat`, `veggies` bundle several specific foods together) - see the report's Discussion section for how this affects interpretation.
