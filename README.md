# paper_properties
Code for the Paper

All the procedures are implemented in the src_2_eng.py Python file. This was executed in a Python 3.11 environment, with all the dependencies that should be installed via pip specified in the requirements.txt file.

## Input requirements

This was executed with a Madrid dataset of properties. The required columns in the dataframe file are:

- "id": unique identifier of the posting.

- "operation": either "buy" or "rent". This code is specifically constructed for "buy" postings.

- "size": property size in squared meters.

- "bedrooms": number of bedrooms (used for clustering purposes).

- "bathrooms": number of bathrooms (used for clustering purposes).

- "lift": categorical variable to specify presence of a lift.

- "garage": categorical variable to specify presence of a garage.

- "storage": categorical variable to specify presence of a storage.

- 
  "bedrooms",
        "bathrooms",
        "lift",
        "garage",
        "storage",
        "terrace",
        "air_conditioning",
        "swimming_pool",
        "garden",
        "sports",
        # "status",
        # "new_construction",
        "rating_leads",
        "rating_visits",
        "floor_int",



We also need an ESRI Shapefile of the city neighbours.

Finally, the code should be adapted in the first lines to include neighboring cities of the main hub, in this case Madrid, including economic data for each neighborhood as needed for the income analysis.





