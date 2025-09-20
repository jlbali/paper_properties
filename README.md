# paper_properties
Code for the Paper

All the procedures are implemented in the src_2_eng.py Python file. This was executed in a Python 3.11 environment, with all the dependencies that should be installed via pip specified in the requirements.txt file.

Each function takes as an input some paths to a file, that may be an excel file, an ESRI Shapefile or a Python pickle, depending the case. As an output, usually a pickle file is generated that should be an input for another function.

An example of how to run this procedure in our paper situation in Madrid is provided in the Jupyter Notebook Ciclo-English.ipynb. 

## Input requirements for properties posting

This was executed with a Madrid dataset of properties. The required columns in the dataframe file are:

- "id": unique identifier of the posting.

- "operation": either "buy" or "rent". This code is specifically constructed for "buy" postings.

- "size": property size in squared meters.

- "bedrooms": number of bedrooms (used for clustering purposes).

- "bathrooms": number of bathrooms (used for clustering purposes).

- "lift": categorical variable to specify presence of a lift.

- "garage": categorical variable to specify presence of a garage.

- "storage": categorical variable to specify presence of a storage.

- "terrace": categorical variable to specify presence of a terrace.

- "air_conditioning": categorical variable to specify presence of an air conditioner.

- "swimming_pool": categorical variable to specify presence of a swimming pool.

- "garden": categorical variable to specify presence of a garden.

- "floor_int": floor number.

Any changes regarding clustering feature should be accordingly adjusted in the construct_clusters function.


We also need an ESRI Shapefile of the city neighbours.

Finally, the code should be adapted in the first lines to include neighboring cities of the main hub, in this case Madrid, including economic data for each neighborhood as needed for the income analysis.

## Income Handling.

The function income_handling should be adapted to the specific demographic excel file provided for the city that should be analyzed. Informationg regarding income for persons and/or families should be provided.


