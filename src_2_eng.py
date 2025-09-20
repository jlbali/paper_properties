import pandas as pd
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle

## Helpers, common variables and functions.

neighboring_cities_Madrid = [
    "Alcobendas",
    "Alcorcón",
    "Coslada",
    "Fuenlabrada",
    "Getafe",
    "Leganés",
    "Majadahonda",
    "Madrid",
    "Las Rozas",
    "Móstoles",
    "Paracuellos de Jarama",
    "Parla",
    "Pozuelo de Alarcón",
    "Rivas-Vaciamadrid",
    "San Fernando de Henares",
    "San Sebastián de los Reyes",
    "Torrejón de Ardoz",
    "Tres Cantos",
    "Villanueva de la Cañada",
    "Villaviciosa de Odón"
]    

neighboring_income = {
    "Alcobendas": 19394,            # según elEconomista
    "Alcorcón": 17004,             # INE (renta bruta media 2021)
    "Coslada": 16329,              # INE
    "Fuenlabrada": 13896,          # INE
    "Getafe": 17279,              # INE
    "Leganés": 15658,             # INE
    "Madrid": 22587,              # INE
    "Majadahonda": 21248,         # elEconomista (renta neta per cápita)
    "Móstoles": 14875,            # INE
    "Paracuellos de Jarama": 14000,  # valor aproximado para municipio pequeño (entre 10k y 14k)
    "Parla": 11965,               # INE
    "Pozuelo de Alarcón": 27167,   # elEconomista
    "Rivas-Vaciamadrid": 20386,    # INE
    "San Fernando de Henares": 13000,  # valor aproximado para municipio de tamaño medio
    "San Sebastián de los Reyes": 20876,  # INE
    "Torrejón de Ardoz": 15313,    # INE
    "Tres Cantos": 26657,         # INE
    "Villanueva de la Cañada": 13000,  # valor aproximado (municipio pequeño)
    "Villaviciosa de Odón": 19257  # elEconomista
}


from shapely.geometry import Point, Polygon

def _get_neighborhood(municipality, lat, lon, barrios_dict, extents, polygons):
    if municipality != "Madrid":
        return municipality
    # Pertenencia a barrio. Si no cae en ninguno se descarta.
    point = Point(lon, lat)
    for barrio in barrios_dict.keys():
        extent = extents[barrio]
        polygon = polygons[barrio]
        if lon < extent["min_lon"] or lon > extent["max_lon"] or lat < extent["min_lat"] or lat > extent["max_lat"]:
            continue
        if polygon.contains(point):
            return barrio
    return None 


def _haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
    return c * r    


def normalize(x, minimun, maximun):
    if x < minimun:
        return minimun
    elif x > maximun:
        return maximun
    else:
        return (x - minimun)/(maximun - minimun)




def supply_side_V6(NO, SC, betaO, sigmaO, x_bar, x_max, ere, kappa, eta, T=100):
    """
    Translation of the 'supply_side' function originally written in MATLAB.
    
    Parameters:
    -----------
    NO      : int   -> Number of steps (equivalent to N in the MATLAB code).
    SC      : float -> Parameter SC.
    betaO   : float -> Parameter beta.
    sigmaO  : float -> Parameter sigma.
    x_bar   : float -> Minimum value of the region (Klo in MATLAB).
    x_max   : float -> Maximum value of the region (Kmax in MATLAB).
    ere     : float -> Parameter ere.
    kappa   : float -> Parameter kappa.
    eta     : float -> Parameter eta.

    Returns:
    --------
    F       : numpy.ndarray -> Equivalent to final 'r' (or updated 'R').
    K       : numpy.ndarray -> Capital grid (K).
    Kprime  : numpy.ndarray -> Policy function (optimal K' for each K).
    """

    # -------------------------------------------------------------------------
    # 1. Initialization and parameter definitions
    # -------------------------------------------------------------------------
    N = NO
    beta = betaO
    sigma = sigmaO
    
    # Klo and Kmax in MATLAB
    Klo = x_bar
    Kmax = x_max
    
    # Grid step for K
    step = (Kmax - Klo) / N
    
    # Vector K (equivalent to K = Klo:step:Kmax in MATLAB)
    # To include the endpoint Kmax, adjust np.arange with a small epsilon.
    # Another option is np.linspace: np.linspace(Klo, Kmax, N+1).
    K = np.arange(Klo, Kmax + 0.5*step, step)
    
    # Auxiliary arrays
    R = np.zeros(K.shape)   # Equivalent to R(1, size(K,2)) in MATLAB
    r = np.zeros(K.shape)   # Will be used to iterate at each step
    m = np.zeros(K.shape, dtype=int)  # To store indices of the maximum
    Kprime = np.zeros(K.shape)
    
    #T = 100        # Maximum number of iterations
    toler = 0.001  # Tolerance
    D = np.zeros(T)
    
    # alpha = (kappa-1)/Klo in MATLAB
    alpha = (kappa - 1) / Klo
    
    # -------------------------------------------------------------------------
    # 2. Initial computation of R with a first "while" (in MATLAB)
    #    MATLAB: while i <= size(K,2) -> for i in range(len(K))
    # -------------------------------------------------------------------------
    for i in range(len(K)):
        K1max = K[i]
        # Build the vector K1 = Klo:step:K1max
        # To avoid floating-point issues, adjust if K1max < Klo
        if K1max < Klo:
            K1 = np.array([Klo])  # If K1max is smaller than Klo, force a single point
        else:
            K1 = np.arange(Klo, K1max + 0.5*step, step)
        
        # S will have the same size as K1 (or the minimal dimension with K)
        S = np.zeros(len(K1))
        
        # Now fill S with the given formula
        for j in range(len(K1)):
            # CC = alpha*eta*K(i)^(-kappa) [in the original code, it's called CC in the S part]
            CC = alpha * eta * (K[i] ** (-kappa))
            # The expression added to S:
            # (ere*(x_bar-(K1(j)+SC))^(1-sigma))/(1-sigma)
            # + (eta*K1(j)*(1 - CC*(eta*K1(j))^(1+kappa)/(1+kappa))) * (beta/(1-beta))
            # + beta*(eta*(K1(j))^(1-sigma)/(1-sigma)) * (CC*K(i)^(1+kappa)/(1+kappa))
            #
            # Note: Ensure (x_bar - (K1(j)+SC)) does not become negative if sigma < 1.
            # In MATLAB, NaNs may appear if this is negative.
            
            # Term 1
            term1 = (ere * (x_bar - (K1[j] + SC)) ** (1 - sigma)) / (1 - sigma)
            # Term 2
            term2 = eta * K1[j] * (
                1 - CC * (eta * K1[j]) ** (1 + kappa) / (1 + kappa)
            )
            term2 *= (beta / (1 - beta))
            # Term 3
            term3 = beta * (
                eta * (K1[j] ** (1 - sigma)) / (1 - sigma)
            ) * (CC * (K[i] ** (1 + kappa)) / (1 + kappa))
            
            # S[j] = sum of term1, term2, term3
            S[j] = term1 + term2 + term3
        
        # R(i) = max(S)
        R[i] = np.max(S)

    # print(R)
    
    # -------------------------------------------------------------------------
    # 3. Main iterations (loop h = 1 to T in MATLAB)
    # -------------------------------------------------------------------------
    for h in range(T):
        for i in range(len(K)):
            K1max = K[i]
            if K1max < Klo:
                K1 = np.array([Klo])
            else:
                K1 = np.arange(Klo, K1max + 0.5*step, step)
            
            s = np.zeros(len(K1))
            
            for j in range(len(K1)):
                # CC = alpha*K(i)^(-kappa)
                CC = alpha * (K[i] ** (-kappa))
                
                # s(j) = ...
                term1 = (ere * (x_bar - (K1[j] + SC)) ** (1 - sigma)) / (1 - sigma)
                term2 = eta * K1[j] * (
                    1 - CC * (K1[j] ** (1 + kappa)) / (1 + kappa)
                )
                term2 *= (beta / (1 - beta))
                term3 = beta * R[j] * (
                    CC * (K[j] ** (1 + kappa)) / (1 + kappa)
                )
                
                s[j] = term1 + term2 + term3
            
            # r(i) = max(s) and m(i) = argmax(s)
            r[i] = np.max(s)
            m[i] = np.argmax(s)
        
        # diff = (R - r) / (1 + R)
        diff = (R - r) / (1 + R)
        
        # Convergence check using the maximum absolute value
        if np.max(np.abs(diff)) <= toler:
            # If all points have converged within tolerance, break the loop
            break
        else:
            R = r.copy()
        
        D[h] = np.max(diff)
    
    # -------------------------------------------------------------------------
    # 4. Output: F, K, Kprime
    #    F = r
    #    Kprime(i) = K(m(i))
    # -------------------------------------------------------------------------
    F = r.copy()
    for i in range(len(K)):
        Kprime[i] = K[m[i]]  # Optimal K' for each K[i]
    
    return F, K, Kprime



def get_normalized_supply(cluster, clusters_stats, sigma=5.0, points=100, iteraciones=100):

    MIN_VALUE = 0.01
    MAX_EUROS = clusters_stats[cluster]["x_max"]
    
    num_points= points
    # capital_max_value = 12000 # Pick a "reasonable" maximum.
    # capital_min_value = clusters_stats[cluster]["x_min"]
    capital_min_value = MIN_VALUE
    capital_max_value = 1.0
    
    kappa_param = clusters_stats[cluster]["kappa"]
    beta_param = 0.99
    ere_param = 0.05
    eta_param = 1  # Fixed. # CONSIDER REMOVING IT.
    sigma_param = sigma
    fixed_cost = 0.0
    _, k_arr, k_prima = supply_side_V6(NO=num_points, SC=0.1, betaO=beta_param, 
                                       sigmaO=sigma_param, x_bar=capital_min_value, 
                                       x_max=capital_max_value, ere=ere_param, kappa=kappa_param, eta=eta_param,
                                       T=iteraciones)
    def supply(x):
        if x < clusters_stats[cluster]["x_min"]:
            return np.nan
        if x > MAX_EUROS:
            x_norm = 1.0
        x_norm = MIN_VALUE + (x - clusters_stats[cluster]["x_min"])/(MAX_EUROS - clusters_stats[cluster]["x_min"])        
        close_index = np.abs(k_arr - x_norm).argmin()
        # Use that index to find the corresponding value in k_prima
        normalized_price = k_prima[close_index]

        price = (normalized_price - MIN_VALUE)*(MAX_EUROS - clusters_stats[cluster]["x_min"]) + clusters_stats[cluster]["x_min"]
        return price
        
    return supply




def get_normalized_demand(
    clusters,
    neighborhoods,
    cluster_neighborhood_dict,
    incomes,
    normalized_thetas,
    cluster_sizes,
    min_price,
    max_price,
    gamma=0.5,
    size_factor_h=1.0,
    size_factor_income=1.0,
    debug=False,
):
    # Theta should be computed with normalized values...
    # min_theta = 0.0  # min(thetas)
    # max_theta = max(thetas)
    min_income = 0.0  # min([incomes[nbhd] for nbhd in incomes.keys()])
    max_income = max([incomes[nbhd] for nbhd in incomes.keys()])
    min_size = 0.0  # min(cluster_sizes)
    max_size = max(cluster_sizes)

    # normalized_thetas_local = thetas  # [normalize(theta, min_theta, max_theta) for theta in thetas]
    normalized_thetas_local = normalized_thetas
    # normalized_sizes = [normalize(size, min_size, max_size) for size in cluster_sizes]
    normalized_sizes = cluster_sizes

    normalized_incomes = {}
    for nbhd in incomes.keys():
        normalized_incomes[nbhd] = normalize(incomes[nbhd], min_income, max_income)

    # A_factors: translation of "ahorcados" (scaling terms per theta)
    A_factors = [
        (gamma ** ((1.0 / gamma) * (1.0 - 1.0 / gamma))) * theta ** (1.0 / gamma + 1.0)
        for theta in normalized_thetas_local
    ]

    # A_factor_single = (gamma**((1.0/gamma)*(1.0 - 1.0/gamma)))*slope**(1.0/gamma + 1.0)
    epsilon_1 = (1.0 / gamma) * (gamma - 2.0 + 1.0 / gamma)
    epsilon_2 = (1.0 / gamma) * (1.0 / gamma - 1.0)
    epsilon_3 = 1.0 / gamma
    if debug:
        # print("Epsilon 1", epsilon_1)
        # print("Epsilon 2", epsilon_2)
        # print("Epsilon 3", epsilon_3)
        # print("A_factors", A_factors)
        pass

    income_factor = []
    for cl in clusters:
        income_factor.append(0.0)
        for nbhd in neighborhoods:
            income_factor[cl] += cluster_neighborhood_dict[(cl, nbhd)] * (
                (normalized_incomes[nbhd] * size_factor_income) ** epsilon_1
            )
    # print(income_factor)

    def demand(x, cl):
        # print("Evaluating", x)
        x_norm = normalize(x, 0.0, max_price)  # normalize(x, min_price, max_price)
        # print("Normalized", x_norm)
        # print(f"x {x} x_norm {x_norm} min_price {min_price} max_price {max_price}")
        h = (normalized_sizes[cl] * size_factor_h) ** epsilon_2
        # print("H", h)
        # print(f"Income factor {income_factor[cl]} H {h} x {x} x_norm {x_norm} Epsilon_3 {epsilon_3} A_factor {A_factors[cl]}")
        # print(f"x_norm**epsilon_3 {x_norm**epsilon_3}")
        # print(f"income_factor[cl]*h*(x_norm**epsilon_3) {income_factor[cl]*h*(x_norm**epsilon_3)}")
        value = income_factor[cl] * h * (x_norm ** epsilon_3)
        # print(f"value*A_factors[cl] {value*A_factors[cl]}")
        return value * A_factors[cl]

    return demand


### STEP 1 --- Loading of data


def initial_load(input_path="./data/raw/FotocasaINE.dta", output_path="./data/raw/data_buy.pkl"):
    df = pd.read_stata(input_path)
    df_buy = df[df["operation"]=="buy"]
    df_buy.to_pickle(output_path)


### STEP 2 --- Handling of neighborhood.

def handle_neighborhood(input_path="./data/raw/data_buy.pkl", shp_path='./data/Madrid/Barrios.shp', output_path="./data/paper/data_madrid_barrios.pkl"):
    df = pd.read_pickle(input_path)
    print("Records", len(df))
    df_neighboring = df[df.municipality.isin(neighboring_cities_Madrid)]

    import geopandas as gpd
    from shapely.geometry import mapping
    import json
    from pyproj import Transformer
    
    # Load the SHP file using geopandas
    gdf = gpd.read_file(shp_path)
    
    # Create a transformer to convert from UTM to WGS84
    # Assume coordinates are in UTM zone 30N (EPSG:25830)
    transformer = Transformer.from_crs("epsg:25830", "epsg:4326", always_xy=True)
    
    # Create a dictionary with neighborhood name and coordinates
    neighborhoods_dict = {}
    
    for _, row in gdf.iterrows():
        neighborhood_name = row['NOMBRE']  # Adjust if your shapefile uses a different field name
        geometry_obj = row['geometry']
        
        # Check whether the geometry is a Polygon or MultiPolygon
        if geometry_obj.geom_type == 'Polygon':
            coords = list(geometry_obj.exterior.coords)
            # Convert coordinates to longitude and latitude (WGS84)
            coords = [transformer.transform(x, y) for x, y in coords]
            neighborhoods_dict[neighborhood_name] = coords
        elif geometry_obj.geom_type == 'MultiPolygon':
            coords = []
            for polygon in geometry_obj:
                polygon_coords = list(polygon.exterior.coords)
                # Convert coordinates to longitude and latitude (WGS84)
                polygon_coords = [transformer.transform(x, y) for x, y in polygon_coords]
                coords.extend(polygon_coords)
            neighborhoods_dict[neighborhood_name] = coords

    extents = {}
    
    for neighborhood in neighborhoods_dict.keys():
        coords = neighborhoods_dict[neighborhood]
        min_lon = np.min([coord[0] for coord in coords])
        max_lon = np.max([coord[0] for coord in coords])
        min_lat = np.min([coord[1] for coord in coords])
        max_lat = np.max([coord[1] for coord in coords])
        extents[neighborhood] = {
            "min_lon": min_lon,
            "min_lat": min_lat,
            "max_lon": max_lon,
            "max_lat": max_lat,
        }
    
    from shapely.geometry import Point, Polygon
    
    polygons = {}
    
    for neighborhood in neighborhoods_dict.keys():
        coords = neighborhoods_dict[neighborhood]
        polygons[neighborhood] = Polygon(coords)
    
    df = df_neighboring

    df["neighborhood"] = df[["municipality", "latitude", "longitude"]].apply(
        lambda x: _get_neighborhood(x[0], x[1], x[2], neighborhoods_dict, extents, polygons),
        axis=1
    )
    df.to_pickle(output_path)




## STEP 4 - CLUSTERING

def construct_clusters(
    input_file="./data/paper/data_madrid_google.pkl",
    demo_file="./data/Madrid/ConsolidadoDemográfico.xlsx",
    output_file="./data/paper/data_madrid_cluster.pkl",
):
    import pandas as pd
    df = pd.read_pickle(input_file)
    df_global = df.copy()
    print("Records", len(df))
    
    # Path to the Excel file
    file_path = demo_file
    
    # Load the Excel file
    xls = pd.ExcelFile(file_path)
    
    # Show sheet names
    print(xls.sheet_names)
    
    # Assume data are in the first sheet
    df_raw = xls.parse(xls.sheet_names[0])
    
    # Initialize a list to store processed rows
    processed_data = []
    
    # Iterate over DataFrame columns to extract the needed values
    for i in range(0, len(df_raw.columns), 2):
        neighborhood = df_raw.iloc[0, i]
        avg_household_income = df_raw.iloc[53, i + 1]
        avg_person_income = df_raw.iloc[54, i + 1]
        
        # Keep column names as in source (Spanish) since they come from the Excel schema
        processed_data.append({
            "Barrio": neighborhood,
            "Ingreso Medio por Hogar": avg_household_income,
            "Ingreso Medio por Persona": avg_person_income,
        })
    
    # Build a DataFrame with processed income data
    df_income = pd.DataFrame(processed_data)
    
    # Add neighboring municipalities around Madrid
    import copy
    incomes_dict = copy.copy(neighboring_income)
    
    for i in range(len(df_income)):
        row_neigh = df_income.iloc[i]
        incomes_dict[row_neigh["Barrio"]] = row_neigh["Ingreso Medio por Persona"]
    
    neighborhoods_list = []
    income_per_person_list = []
    for neigh in incomes_dict.keys():
        neighborhoods_list.append(neigh)
        income_per_person_list.append(incomes_dict[neigh])
    
    df_income = pd.DataFrame({
        "Barrio": neighborhoods_list,
        "Ingreso Medio por Persona": income_per_person_list,
    })
    
    # Note: the right-hand columns are kept in Spanish to match the Excel source.
    # If your main DataFrame uses an English column ("neighborhood"), update left_on accordingly.
    df = df.merge(df_income, left_on="barrio", right_on="Barrio", how="left")

    features = [
        "size",
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
    ]

    def encode_floor(floor_value):
        floor_mapping = {
            "1º": 1,
            "2º": 2,
            "3º": 3,
            "4º": 4,
            "5º": 5,
            "6º": 6,
            "7º": 7,
            "8º": 8,
            "9º": 9,
            "10º": 10,
            "11º": 11,
            "12º": 12,
            "13º": 13,
            "14º": 14,
            "15º": 15,
            "A partir del 15º": 16,  # Adjust if you need a specific value
            "Bajos": 0,              # Ground floor
            "Entresuelo": 0.5,       # Mezzanine between ground and first floor
            "Principal": 0.5,        # Similar to mezzanine; adjust if preferred
            "Sótano": -1,            # Basement
            "Subsótano": -2,         # Sub-basement
            "Otro": 0,               # Use None or another value for unknown categories
            "": 0,                   # Missing values
        }
        # Return None if not found in the mapping
        return floor_mapping.get(floor_value, None)

    df_dedup = df.drop_duplicates(subset="id", keep="last")
    df_dedup["floor_int"] = df_dedup.floor.apply(encode_floor)
    dg = df_dedup[features]

    from sklearn.preprocessing import OneHotEncoder

    # Identify categorical columns
    categorical_features = [
        feat for feat in features
        if ("object" in str(dg[feat].dtype)) or ("cate" in str(dg[feat].dtype))
    ]
    print(categorical_features)
    
    # Apply One-Hot Encoding
    encoder = OneHotEncoder(drop="first", sparse_output=False)
    X_categorical = encoder.fit_transform(dg[categorical_features])
    print(X_categorical.shape)
    
    # Convert to DataFrame to combine with numeric data
    X_categorical_df = pd.DataFrame(
        X_categorical,
        columns=encoder.get_feature_names_out(categorical_features),
    )
    print(X_categorical_df.shape)
    
    # Combine encoded categorical data with remaining numeric data
    X_numeric = dg.drop(columns=categorical_features)
    print(X_numeric.shape)
    # X_combined = pd.concat([X_numeric, X_categorical_df], axis=1)
    X_combined = X_numeric  # Keep as in original logic

    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt
    
    # Impute missing values
    imputer = SimpleImputer(strategy="median")  # You can change to 'most_frequent', etc.
    X_imputed = imputer.fit_transform(X_combined)
    
    # Standardize the imputed data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    # Dimensionality reduction with PCA
    pca = PCA(n_components=0.95)  # Keep 95% of the variance
    X_pca = pca.fit_transform(X_scaled)
    print(len(X_pca))
    
    # KMeans clustering
    kmeans = KMeans(n_clusters=5, random_state=42)
    clusters = kmeans.fit_predict(X_pca)
    
    # Add cluster result to the original (deduplicated) DataFrame
    df_dedup["Cluster"] = clusters

    # Median price per cluster
    cluster_median_price = df_dedup.groupby("Cluster")["price"].median()
    
    # Sort clusters by median price and re-label them 0..4
    sorted_clusters = cluster_median_price.sort_values().index
    new_cluster_labels = {old_label: new_label for new_label, old_label in enumerate(sorted_clusters)}
    df_dedup["Cluster"] = df_dedup["Cluster"].map(new_cluster_labels)

    cluster_ids = sorted(list(df_dedup.Cluster.unique()))
    prices_per_cluster = [df_dedup[df_dedup.Cluster == c].price for c in cluster_ids]
    plt.boxplot(prices_per_cluster, labels=cluster_ids, showfliers=False)
    plt.xlabel("Cluster")
    plt.ylabel("Price (euros)")
    plt.show()

    df = df.merge(df_dedup[["id", "Cluster"]], on="id", how="left")

    df.to_pickle(output_file)


## STEP 5 - SUPPLY SIDE CALIBRATION

def supply_side_calib(
    input_file="./data/paper/data_madrid_cluster.pkl",
    output_file="./data/paper/data_Madrid_probs_propietario.pkl",
    cluster_stats_file="./data/paper/clusters_stats.pkl",
):

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    from importlib import reload
    import powerlaw
    from scipy.stats import lognorm
    from tqdm import tqdm

    target = "price_m2"
    # target = "price"
    df = pd.read_pickle(input_file)
    # Remove duplicates and switch to properties for sale.
    if False:  # True:
        df = df.drop_duplicates(subset="id", keep="last")
    
    df = df.dropna(subset="Cluster")
    clusters = df.Cluster.unique().tolist()
    clusters.sort()
    data = df[target]

    # Count number of appearances per listing id (spells in months)
    df_counts = df.groupby("id")["fecha"].count().reset_index()

    # Std of target per id and merge with counts
    df_stats = df.groupby("id")[target].std().reset_index()
    df_stats = df_counts.merge(df_stats, on="id", how="left")
    df_stats = df_stats.rename(columns={
        "fecha": "count",
        "price_m2": "std_price_m2",
    })
    
    total_rows = len(df_stats)
    
    counts_list = sorted(list(df_stats["count"].unique()))
    frequency = {}
    for count_val in counts_list:
        frequency[count_val] = len(df_stats[df_stats.count == count_val]) / total_rows

    # LaTeX table for empirical spell distribution by cluster
    text = """
    \\begin{table}[h!] 
    \\label{spells}
    \\begin{tabular}{|c|c|c|c|c|c|}
    \\hline
    \\textbf{Spell (in months) / Cluster} & \\textbf{0} & \\textbf{1} & \\textbf{2} & \\textbf{3} & \\textbf{4} \\\\ \\hline
    
    """
    
    for freq in range(1, 6):
        if freq < 5:
            text += "\\textbf{" + str(freq) + "} &"
        else:
            text += "\\textbf{5 or more} &"
    
        for cluster in clusters:
            df_stats_c = df[df.Cluster == cluster].groupby("id")[target].std().reset_index()
            ids_c = set(df[df.Cluster == cluster]["id"].unique())
            df_stats_c = df_counts[df_counts["id"].isin(ids_c)].merge(df_stats_c, on="id", how="left")
            df_stats_c = df_stats_c.rename(columns={
                "fecha": "count",
                "price_m2": "std_price_m2",
            })
            
            total_rows_c = len(df_stats_c)
            counts_list_c = sorted(list(df_stats_c["count"].unique()))
            frequency_c = {}
            for count_val in counts_list_c:
                if count_val < 5:
                    frequency_c[count_val] = len(df_stats_c[df_stats_c.count == count_val]) / total_rows_c
                else:
                    frequency_c[count_val] = len(df_stats_c[df_stats_c.count >= count_val]) / total_rows_c
            if cluster != 4:
                text += str(round(frequency_c[freq], 2)) + " &"
            else:
                text += str(round(frequency_c[freq], 2)) + "\\\\ \\hline \n"
    text += """
    \\end{tabular}
    \\caption{Empirical distribution of spells across clusters.}
    \\label{table:cal_1}
    \\end{table}
    """
    print(text)

    ## SUPPLY

    target = "price_m2"
    # target = "price"
    df = pd.read_pickle(input_file)
    # Remove duplicates and keep last observation per id
    if True:
        df = df.drop_duplicates(subset="id", keep="last")
    
    df = df.dropna(subset="Cluster")
    clusters = df.Cluster.unique().tolist()
    clusters.sort()
    data = df[target]

    # Boxplots by cluster (price per m2)
    data_list = []
    for cluster in clusters:
        prices_m2 = df[df.Cluster == cluster].price_m2
        data_list.append(prices_m2)
        print(f"cluster-", np.percentile(prices_m2, 95.0))
        
    plt.boxplot(data_list, showfliers=False)
    plt.show()

    # Boxplots by cluster (total price)
    data_list = []
    for cluster in clusters:
        prices_total = df[df.Cluster == cluster].price
        data_list.append(prices_total)
        print(f"cluster-", np.percentile(prices_total, 95.0))
        
    plt.boxplot(data_list, showfliers=False)
    plt.show()

    def generate_power_law_data(x_min, alpha, size, seed=42):
        """
        Generate random data from a power-law distribution.
        
        Parameters:
        - x_min: the minimum value (x_min) of the distribution.
        - alpha: the power-law exponent.
        - size: number of data points to generate.
        
        Returns:
        - A list/array of generated data.
        """
        np.random.seed(seed)
        u = np.random.uniform(0, 1, size)  # Uniform draws
        data = x_min * (1 - u) ** (-1 / (alpha - 1))  # Inverse CDF transform
        return data
    
    def power_law_mean_std(x_min, alpha):
        """
        Compute mean and standard deviation of a power-law distribution.
    
        Parameters:
        - x_min: minimum value of the distribution (x_min).
        - alpha: power-law exponent.
    
        Returns:
        - mean: the mean value (or np.inf if undefined).
        - std_dev: the standard deviation (or np.inf if undefined).
        """
        # Mean
        if alpha <= 2:
            mean_val = np.inf  # Mean undefined
        else:
            mean_val = (alpha * x_min) / (alpha - 1)
        
        # Variance
        if alpha <= 3:
            var_val = np.inf  # Variance undefined
        else:
            var_val = (alpha * x_min**2) / ((alpha - 1)**2 * (alpha - 2))
        
        # Standard deviation
        std_dev = np.sqrt(var_val) if var_val != np.inf else np.inf
        
        return mean_val, std_dev
    
    def calibrate(
        df_in,
        disc_kappa=10,
        disc_x_min=10,
        min_kappa=2.0,
        max_kappa=4.0,
        min_x_min=1000.0,
        max_x_min=5000.0,
    ):
        # FREQUENCY CALCULATION AND SUBSEQUENT STEPS
        df_in = df_in.dropna(subset="Cluster")
        clusters_in = df_in.Cluster.unique().tolist()
        clusters_in.sort()
        data_in = df_in[target]
        df_counts_in = df_in.groupby("id")["fecha"].count().reset_index()
        
        df_stats_in = df_in.groupby("id")[target].std().reset_index()
        df_stats_in = df_counts_in.merge(df_stats_in, on="id", how="left")
        df_stats_in = df_stats_in.rename(columns={
            "fecha": "count",
            "price_m2": "std_price_m2",
        })
        
        total_rows_in = len(df_stats_in)
        counts_list_in = sorted(list(df_stats_in["count"].unique()))
        frequency_in = {}
        for count_val in counts_list_in:
            frequency_in[count_val] = len(df_stats_in[df_stats_in.count == count_val]) / total_rows_in
        
        # Parameter grids
        kappas = np.linspace(start=min_kappa, stop=max_kappa, num=disc_kappa)
        x_mins = np.linspace(start=min_x_min, stop=max_x_min, num=disc_x_min)
        
        error_table = np.zeros((len(kappas), len(x_mins)))
        freqs_keys = list(frequency_in.keys())
        probabilities = np.array(list(frequency_in.values()))
        probabilities = probabilities / probabilities.sum()
    
        mean_prices_m2 = df_in["price_m2"].mean()
        std_prices_m2 = df_in["price_m2"].std()
    
        import tqdm
        
        best_kappa = None
        best_x_min = None
        best_error = None
        best_mean = None
        best_std = None
    
        MAX_EUROS = np.percentile(df_in["price_m2"].values, 95.0)
        print(f"MAX EUROS {MAX_EUROS}")
        
        for i in tqdm.tqdm(range(len(kappas))):
            for j in range(len(x_mins)):
                kappa_val = kappas[i]
                x_min_val = x_mins[j]
                # 1) Sample 100 prices from a power-law with kappa and x_min.
                # 2) Draw a spell count for each price from the empirical frequency table.
                # 3) If spell == 1, price stays as is.
                # 4) If spell > 1, iterate through the supply side (k -> k') on a grid.
                # 5) Map each simulated price to the closest grid point.
        
                prices = generate_power_law_data(x_min_val, kappa_val, 100, seed=42)
                spells = np.random.choice(freqs_keys, size=100, p=probabilities)
    
                MIN_VALUE = 0.01
        
                num_points = 100
                capital_min_value = MIN_VALUE
                capital_max_value = 1.0
        
                kappa_param = kappa_val
                beta_param = 0.99
                ere_param = 0.05
                eta_param = 1  # Fixed. # CONSIDER REMOVING IT.
                sigma_param = 5.0
                fixed_cost = 0.0
                iterations = 100
                _, k_arr, k_prime = supply_side_V6(
                    NO=num_points,
                    SC=0.1,
                    betaO=beta_param,
                    sigmaO=sigma_param,
                    x_bar=capital_min_value,
                    x_max=capital_max_value,
                    ere=ere_param,
                    kappa=kappa_param,
                    eta=eta_param,
                    T=iterations,
                )
    
                def supply(price_x):
                    if price_x == np.nan:
                        return np.nan
                    if price_x < x_min_val:
                        return np.nan
                    if price_x > MAX_EUROS:
                        x_norm = 1.0
                    x_norm = MIN_VALUE + (price_x - x_min_val) / (MAX_EUROS - x_min_val)
                    nearest_idx = np.abs(k_arr - x_norm).argmin()
                    # Use that index to find the corresponding value in k_prime
                    normalized_price = k_prime[nearest_idx]
            
                    price_out = (normalized_price - MIN_VALUE) * (12000 - x_min_val) + x_min_val
                    return price_out
                
                # Convert each simulated price by iterating through k -> k' (supply side),
                # assigning the closest grid value each period (when spell > 1).
                simulated_series = []
                for k in range(len(prices)):
                    spell = spells[k]
                    if spell == 1:
                        simulated_series.append(prices[k])
                    else:
                        price_tmp = prices[k]
                        for _ in range(spell - 1):
                            price_tmp = supply(price_tmp)
                            simulated_series.append(price_tmp)
                
                mean_val = np.mean(simulated_series)
                std_val = np.std(simulated_series)
                # mean_theoretical, std_theoretical = power_law_mean_std(alpha, kappa)
                # error = (mean_val - mean_theoretical)**2 + (std_val - std_theoretical)**2
                error = (mean_val - mean_prices_m2) ** 2 + (std_val - std_prices_m2) ** 2
                error_table[i, j] = error
                if best_error is None or error < best_error:
                    best_error = error
                    best_kappa = kappa_val
                    best_x_min = x_min_val
                    best_mean = mean_val
                    best_std = std_val
                # print("Kappa", kappa_val, "x_min", x_min_val, "Mean", mean_val, "STD", std_val, "ERROR", error)
                            
        return best_error, best_kappa, best_x_min, mean_prices_m2, best_mean, std_prices_m2, best_std

    # Calibrate per cluster
    clusters_stats = []
    for cluster in clusters:
        print(f"CLUSTER {cluster}")
        df_cluster = df[df.Cluster == cluster]
        MAX_EUROS = np.percentile(df_cluster["price_m2"].values, 95.0)
        error, kappa_opt, x_min_opt, mean_prices_m2, best_mean, std_prices_m2, best_std = calibrate(df_cluster)
        cluster_stats = {
            "kappa": kappa_opt,
            "x_min": x_min_opt,
            "x_max": MAX_EUROS,
            "true_mean": mean_prices_m2,
            "best_mean": best_mean,
            "true_std": std_prices_m2,
            "best_std": best_std,
        }
        print(cluster_stats)
        clusters_stats.append(cluster_stats)

    # Persist calibrated cluster stats
    import pickle
    f = open(cluster_stats_file, "wb")
    pickle.dump(clusters_stats, f)
    f.close()    

    ## LaTeX tables for calibrated parameters
    
    text = """
    \\begin{table}[h!]
    \\begin{tabular}{|c|c|c|c|c|c|}
    \\hline
    \\textbf{Parameter / Cluster} & \\textbf{0} & \\textbf{1} & \\textbf{2} & \\textbf{3} & \\textbf{4} \\\\ \\hline
    
    """
    
    text += "\\textbf{$\\kappa$} & "
    for cluster in clusters:
        if cluster < 4:
            text += str(round(clusters_stats[cluster]["kappa"], 2)) + " & "
        else:
            text += str(round(clusters_stats[cluster]["kappa"], 2)) + " \\\\ \\hline \n"
    
    text += "\\textbf{$\\underline{p}$} & "
    for cluster in clusters:
        if cluster < 4:
            text += str(round(clusters_stats[cluster]["x_min"], 2)) + " & "
        else:
            text += str(round(clusters_stats[cluster]["x_min"], 2)) + " \\\\ \\hline \n"
    
    text += "\\textbf{$\\overline{p}$} & "
    for cluster in clusters:
        if cluster < 4:
            text += str(round(clusters_stats[cluster]["x_max"], 2)) + " & "
        else:
            text += str(round(clusters_stats[cluster]["x_max"], 2)) + " \\\\ \\hline \n"
    
    text += """
    \\end{tabular}
    \\caption{Calibrated parameters. Supply side}
    \\label{table:cal_2}
    \\end{table}
    """
    print(text)

    text = """
    \\begin{table}[h!]
    \\begin{tabular}{|c|c|c|c|c|c|}
    \\hline
    \\textbf{Parameter / Cluster} & \\textbf{0} & \\textbf{1} & \\textbf{2} & \\textbf{3} & \\textbf{4} \\\\ \\hline
    
    """
    
    text += "\\textbf{Empirical Mean} & "
    for cluster in clusters:
        if cluster < 4:
            text += str(round(clusters_stats[cluster]["true_mean"], 2)) + " & "
        else:
            text += str(round(clusters_stats[cluster]["true_mean"], 2)) + " \\\\ \\hline \n"
    
    text += "\\textbf{Simulated Mean} & "
    for cluster in clusters:
        if cluster < 4:
            text += str(round(clusters_stats[cluster]["best_mean"], 2)) + " & "
        else:
            text += str(round(clusters_stats[cluster]["best_mean"], 2)) + " \\\\ \\hline \n"
    
    text += "\\textbf{Empirical SD} & "
    for cluster in clusters:
        if cluster < 4:
            text += str(round(clusters_stats[cluster]["true_std"], 2)) + " & "
        else:
            text += str(round(clusters_stats[cluster]["true_std"], 2)) + " \\\\ \\hline \n"
    
    text += "\\textbf{Simulated SD} & "
    for cluster in clusters:
        if cluster < 4:
            text += str(round(clusters_stats[cluster]["best_std"], 2)) + " & "
        else:
            text += str(round(clusters_stats[cluster]["best_std"], 2)) + " \\\\ \\hline \n"
    
    text += """
    \\end{tabular}
    \\caption{Calibrated parameters. Supply side}
    \\label{table:cal_2}
    \\end{table}
    """
    print(text)

    ## SUPPLY SIDE -- PERCEIVED PROBABILITY
    
    def compute_probability_simulated(x_min, x_max, alpha, x1, x2, iterations=10000, seed=42):
        if x_max is None:
            samples = generate_power_law_data(x_min, alpha, iterations, seed)
            successes = len([val for val in samples if (val >= x1) and (val <= x2)])
            return successes / iterations
        else:
            samples = generate_power_law_data(x_min, alpha, iterations, seed)
            denominator = len([val for val in samples if (val >= x_min) and (val <= x_max)])
            numerator = len([val for val in samples if (val >= x_min) and (val <= x_max) and (val >= x1) and (val <= x2)])
            return numerator / denominator    

    target = "price_m2"
    # target = "price"
    df = pd.read_pickle(input_file)
    # Remove duplicates and keep first observation per id
    df = df.drop_duplicates(subset="id", keep="first")
    
    df = df.dropna(subset="Cluster")
    clusters = df.Cluster.unique().tolist()
    clusters.sort()
    data = df[target]

    num_columns = 10
    
    from tqdm import tqdm
    
    all_probs = []
    ids_by_cluster = []
    df_list_per_cluster = []
    
    # clusters = [4]
    
    for cluster in clusters:
        df_cluster = df[df.Cluster == cluster]
        supply_fn = get_normalized_supply(cluster, clusters_stats, sigma=5.0, points=100, iteraciones=100)
        x_min = clusters_stats[cluster]["x_min"]
        x_max = clusters_stats[cluster]["x_max"]
        kappa_val = clusters_stats[cluster]["kappa"]
        assert x_min <= x_max
        cluster_ids = []
        ids_by_cluster.append(cluster_ids)
        cluster_probs = []
        all_probs.append(cluster_probs)
        for i in tqdm(range(len(df_cluster))):
            property_row = df_cluster.iloc[i]
            cluster_ids.append(property_row["id"])
            price_m2 = property_row["price_m2"]
            property_probs = []
            cluster_probs.append(property_probs)
            ceiling_price = price_m2
            floor_price = supply_fn(ceiling_price)
            prob = 1.0
            for j in range(num_columns):
                # assert floor_price <= ceiling_price
                # assert ceiling_price >= x_min
                if floor_price == np.nan or ceiling_price < x_min:
                    property_probs.append(None)
                    continue
                if prob == 0.0 or floor_price >= ceiling_price:
                    property_probs.append(0.0)
                    continue
                prob = compute_probability_simulated(
                    x_min=x_min, x_max=ceiling_price, alpha=kappa_val, x1=floor_price, x2=ceiling_price
                )
                property_probs.append(prob)
                ceiling_price = floor_price
                floor_price = supply_fn(floor_price)
        df_cluster_probs = pd.DataFrame({
            "id": cluster_ids,
        })    
        for j in range(num_columns):
            probs_col = [p[j] for p in cluster_probs]
            df_cluster_probs[f"prob_owner_{j}"] = probs_col
        df_list_per_cluster.append(df_cluster_probs)
    
    df_probs = pd.concat(df_list_per_cluster)
    df = df.merge(df_probs, on="id", how="left")

    # Diagnostics per cluster
    for cluster in clusters:
        df_c = df[df.Cluster == cluster]
        plt.boxplot(df_c["prob_owner_0"].dropna())
        plt.title(str(cluster))
        plt.show()
        
        plt.hist(df_c["prob_owner_0"].values)
        plt.title(str(cluster))
        plt.show()
    
        plt.hist(df_c["price_m2"].values)
        plt.title(str(cluster))
        plt.show()
    
        supply_fn = get_normalized_supply(cluster, clusters_stats, sigma=5.0, points=100, iteraciones=100)
        X = np.linspace(start=clusters_stats[cluster]["x_min"], stop=10000, num=1000)
        Y = [supply_fn(x) for x in X]
        plt.plot(X, Y)
        plt.plot([0, 10000], [0, 10000], color="r")
        plt.title(str(cluster))
        plt.show()    
        
        print(f"Cluster {cluster} mean: {df_c['prob_owner_0'].mean()}")
        print(f"Cluster {cluster} Prop NAN: {df_c['prob_owner_0'].isna().sum()/len(df_c)}")    
                
    df.to_pickle(output_file)



## STEP PRE-6 - INCOME HANDLING


def income_handling(
    input_file="./data/paper/data_madrid_cluster.pkl",
    demo_file="./data/Madrid/ConsolidadoDemográfico.xlsx",
    income_file="./data/paper/ingresos.pkl",
    thetas_file="./data/paper/thetas.pkl",
):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    from importlib import reload
    import powerlaw
    from scipy.stats import lognorm
    from tqdm import tqdm

    df = pd.read_pickle(input_file)
    # Remove duplicates and keep properties for sale (last record per id).
    if True:
        df = df.drop_duplicates(subset="id", keep="last")

    import pandas as pd
    
    # Path to the Excel file
    file_path = demo_file
    
    # Load the Excel file
    xls = pd.ExcelFile(file_path)
    
    # Show sheet names
    print(xls.sheet_names)
    
    # Assume the data are in the first sheet
    df_raw = xls.parse(xls.sheet_names[0])
    
    # Initialize a list to store processed rows
    processed_data = []
    
    # Iterate over the DataFrame columns to extract the needed values
    for i in range(0, len(df_raw.columns), 2):
        neighborhood_name = df_raw.iloc[0, i]
        avg_household_income = df_raw.iloc[53, i + 1]
        avg_person_income = df_raw.iloc[54, i + 1]
        
        processed_data.append({
            "Barrio": neighborhood_name,                 # keep Excel schema keys
            "Ingreso Medio por Hogar": avg_household_income,
            "Ingreso Medio por Persona": avg_person_income,
        })
    
    # Build a DataFrame with the processed data
    df_income = pd.DataFrame(processed_data)
    
    # Add neighboring municipalities around Madrid
    import copy
    incomes_dict = copy.copy(neighboring_income)
    
    for i in range(len(df_income)):
        row_neigh = df_income.iloc[i]
        incomes_dict[row_neigh["Barrio"] ] = row_neigh["Ingreso Medio por Persona"]

    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Assume DataFrame 'df' includes at least 'barrio' and 'price_m2'
    # And that 'incomes_dict' maps neighborhood -> average income per person
    
    # 1) Compute average price per neighborhood
    df_grouped = df.groupby("barrio")["price_m2"].mean().reset_index()
    df_grouped.rename(columns={"price_m2": "mean_price"}, inplace=True)
    
    # 2) Add income information using the incomes_dict
    df_grouped["income"] = df_grouped["barrio"].map(incomes_dict)
    
    # Optionally remove neighborhoods without income data
    df_grouped = df_grouped.dropna(subset=["income"])
    
    # 3) Scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(df_grouped["income"], df_grouped["mean_price"], alpha=0.7)
    plt.xlabel("Average income per person (€)")
    plt.ylabel("Average price per m² (€)")
    plt.title("Relationship between average income and average price per m²")
    plt.grid(True)
    
    # 4) Linear fit using np.polyfit
    slope, intercept = np.polyfit(df_grouped["income"], df_grouped["mean_price"], 1)
    
    # Generate fitted line for plotting
    x_vals = np.linspace(df_grouped["income"].min(), df_grouped["income"].max(), 100)
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, label=f"y = {slope:.2f}x + {intercept:.2f}")
    plt.legend()
    
    plt.show()
    
    # 5) Print fit coefficients
    print("Slope:", slope)
    print("Intercept:", intercept)

    thetas = []
    for cluster in sorted(df.Cluster.unique()):
        df_cluster = df[df.Cluster == cluster]
        # 1) Compute average price per neighborhood
        df_grouped = df_cluster.groupby("barrio")["price_m2"].mean().reset_index()
        df_grouped.rename(columns={"price_m2": "mean_price"}, inplace=True)
        
        # 2) Add income info via incomes_dict
        df_grouped["income"] = df_grouped["barrio"].map(incomes_dict)
        
        # Optionally drop neighborhoods without income data
        df_grouped = df_grouped.dropna(subset=["income"])
        
        # 4) Linear fit using np.polyfit (theta = slope)
        slope, intercept = np.polyfit(df_grouped["income"], df_grouped["mean_price"], 1)
        thetas.append(slope)
    print(thetas)

    import pickle
    f = open(income_file, "wb")
    pickle.dump(incomes_dict, f)
    f.close()
    
    f = open(thetas_file, "wb")
    pickle.dump(thetas, f)
    f.close()



## STEP 6 - DEMAND SIDE





def demand_side_calib(
    input_file="./data/paper/data_Madrid_probs_propietario.pkl",
    demo_file="./data/Madrid/ConsolidadoDemográfico.xlsx",
    input_file_cluster="./data/paper/data_madrid_cluster.pkl",
    income_file="./data/paper/ingresos.pkl",
    thetas_file="./data/paper/thetas.pkl",
    cluster_stats_file="./data/paper/clusters_stats.pkl",
    output_file="./data/paper/data_madrid_probs_v2.pkl",
    report_dir="./data/paper/",
):

    import numpy as np
    import matplotlib.pyplot as plt
    import os
    from importlib import reload    

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    from importlib import reload
    import powerlaw
    from scipy.stats import lognorm
    from tqdm import tqdm    

    df = pd.read_pickle(input_file)

    df_probs = df[["id", "prob_propietario_0", "Cluster"]]

    clusters = sorted(list(df.Cluster.unique()))

    import pickle
    f = open(cluster_stats_file, "rb")
    clusters_stats = pickle.load(f)
    f.close()
    supply_functions = [get_normalized_supply(cluster, clusters_stats) for cluster in clusters]    
    
    df = pd.read_pickle(input_file_cluster)
    df = df.sort_values(by="fecha")
    # df = df.drop_duplicates(subset="id", keep="first")
    df = df.drop_duplicates(subset="id", keep="last")
    
    max_date = df.fecha.max()
    df = df[df.fecha < max_date]

    for cluster in clusters:
        print("Cluster", cluster, "Price M2 10%", df[df.Cluster == cluster]["price_m2"].quantile(q=0.1))
        print("Cluster", cluster, "Price M2 50%", df[df.Cluster == cluster]["price_m2"].quantile(q=0.5))
        print("Cluster", cluster, "Price 50%", df[df.Cluster == cluster]["price"].quantile(q=0.5))
        print("Cluster", cluster, "Mean", df[df.Cluster == cluster]["price_m2"].mean())    

    df = df.merge(df_probs[["id", "prob_propietario_0"]], on="id", how="left")

    import pickle
    f = open(income_file, "rb")
    incomes = pickle.load(f)
    f.close()
    
    f = open(thetas_file, "rb")
    thetas = pickle.load(f)
    f.close()    
    
    clusters = sorted(list(df.Cluster.unique()))
    neighborhoods = list(df.barrio.unique())
    
    # neighborhoods = [nb for nb in neighborhoods if not nb is None or nb != 'Villaverde Alto - Casco Histórico de Villaverde']
    # neighborhoods.append('Villaverde Alto, C.H. Villaverde')
    if None in neighborhoods:
        neighborhoods.remove(None)
    if 'Villaverde Alto - Casco Histórico de Villaverde' in neighborhoods:
        neighborhoods.remove('Villaverde Alto - Casco Histórico de Villaverde')
    neighborhoods.append('Villaverde Alto, C.H. Villaverde')
    
    min_income = 0.0
    max_income = max([incomes[nb] for nb in incomes.keys()])
    
    incomes_normalized = {}
    for nb in incomes.keys():
        incomes_normalized = normalize(incomes[nb], min_income, max_income)
    
    print(None in neighborhoods)
    
    neighborhoods_list = []
    incomes_list = []
    for nb in incomes.keys():
        neighborhoods_list.append(nb)
        incomes_list.append(incomes[nb])
    df_incomes = pd.DataFrame({
        "barrio": neighborhoods_list,
        "ingreso": incomes_list,
    })
    
    # Merge neighborhood incomes
    df = df.merge(df_incomes, on="barrio", how="left")
    
    # Average income per cluster (simple mean over rows in cluster)
    cluster_incomes = []
    for cluster in clusters:
        income_val = df[df.Cluster == cluster].ingreso.mean()
        cluster_incomes.append(income_val)

    # Characteristic size per cluster (median)
    sizes_cluster = []
    for cluster in clusters:
        print("Cluster", cluster)
        series_size = df[df.Cluster == cluster]["size"]
        print("Mean", series_size.mean())
        print("Median", series_size.median())
        size_val = series_size.median()
        plt.hist(series_size)
        plt.show()
        plt.boxplot(series_size)
        plt.show()
        sizes_cluster.append(size_val)    
        
    # 1) Average price per neighborhood
    df_grouped = df.groupby('barrio')['price_m2'].mean().reset_index()
    df_grouped.rename(columns={'price_m2': 'mean_price'}, inplace=True)
    
    # 2) Add income info using 'incomes' dict
    df_grouped['income'] = df_grouped['barrio'].map(incomes)
    
    # Optionally drop neighborhoods without income data
    df_grouped = df_grouped.dropna(subset=['income'])
    
    # 4) Linear regression via np.polyfit
    slope, intercept = np.polyfit(df_grouped['income'], df_grouped['mean_price'], 1)
    
    # Cluster–neighborhood weights (share of listings in cluster that belong to neighborhood)
    cluster_neighborhood_dict = {}
    for cluster in clusters:
        total_n = len(df[df.Cluster == cluster])
        for nb in neighborhoods:
            n = len(df[(df.Cluster == cluster) & (df.barrio == nb)])
            cluster_neighborhood_dict[(cluster, nb)] = n / total_n

    # Theta should NOT work with normalized prices here...
    normalized_thetas = []
    for cluster in sorted(df.Cluster.unique()):
        df_cluster = df[df.Cluster == cluster]
    
        min_price = 0.0  # df["price_m2"].quantile(q=0.1)
        max_price = min(max(list(df_cluster["price_m2"].values)), 10000)
        df_cluster["price_norm"] = df_cluster["price_m2"]
        
        # 1) Average price per neighborhood
        df_grouped = df_cluster.groupby('barrio')['price_norm'].mean().reset_index()
        df_grouped.rename(columns={'price_norm': 'mean_price'}, inplace=True)
        
        # 2) Add income via 'incomes' dict
        df_grouped['income'] = df_grouped['barrio'].map(incomes)
        
        # Optionally remove neighborhoods without income
        df_grouped = df_grouped.dropna(subset=['income'])
        
        # 4) Linear regression via np.polyfit
        slope, intercept = np.polyfit(df_grouped['income'], df_grouped['mean_price'], 1)
        normalized_thetas.append(slope)

    ## MODE 7
    
    import scipy
    
    gammas = np.linspace(start=0.2, stop=0.45, num=100)
    demand_factor = np.power(10, np.linspace(start=-20, stop=20, num=200))
    
    # factor_multiplicativo_ingreso = 4.0
    # factor_multiplicativo_ingreso = 20.0
    # factor_multiplicativo_ingreso = 1.0
    income_multiplicative_factor = None  # If None, search space capped at 1.0 (income is normalized)
    # disc = 1000
    disc = 100
    
    dgs = []
    import scipy.stats as stats
    
    np.seterr(over='raise')
    gammas_cluster = []
    hs_cluster = []
    income_factors_cluster = []
    x_mins_demand_cluster = []
    demand_factors_cluster = []
    income_percentiles_cluster = []
    errors_cluster = []
    errors_1_cluster = []
    errors_2_cluster = []
    investor_medians_cluster = []
    incomes_cluster_copy = []
    min_price_cluster = []
    sales_shares_cluster = []
    
    min_income = 0.0  # min([incomes[nb] for nb in incomes.keys()])
    max_income = max([incomes[nb] for nb in incomes.keys()])
    
    # normalized_thetas_use = thetas  # [normalize(theta, min_theta, max_theta) ...]
    normalized_thetas_use = normalized_thetas
    # sizes_normalized = [normalize(size, min_size, max_size) for size in sizes_cluster]
    sizes_normalized = sizes_cluster
    incomes_normalized = {}
    
    for nb in incomes.keys():
        incomes_normalized[nb] = normalize(incomes[nb], min_income, max_income)
    
    # incomes_normalized = incomes
    
    k_limit = 10
    
    for cluster in clusters:
        print("Cluster", cluster)
    
        # Income mixture (cluster-level mean from neighborhood shares, using normalized income)
        cluster_income_mix = 0.0
        for nb in neighborhoods:
            cluster_income_mix += cluster_neighborhood_dict[(cluster, nb)] * (incomes_normalized[nb])
        print(f"Cluster Income {cluster_income_mix}")
    
        # Second moment of the mixture
        second_moment = 0.0
        for nb in neighborhoods:
            second_moment += cluster_neighborhood_dict[(cluster, nb)] * ((incomes_normalized[nb])**2)
        print(f"Second moment {second_moment}")
    
        cluster_variance = second_moment - cluster_income_mix**2
    
        # Log-normal parameters (simplified/assumed)
        mu = np.log(cluster_income_mix)
        sigma = 1.0
        
        print(f"mu (cluster income) {mu}")
        print(f"sigma (cluster income) {sigma}")
        
        kappa = clusters_stats[cluster]["kappa"]
        df_cluster = df[(df.Cluster == cluster)]
        print("Cluster 10th percentile", df_cluster.price_m2.quantile(q=0.1))
        print("Power-law activation (supply)", clusters_stats[cluster]["x_min"])
        # x_min = max(df_cluster.price_m2.quantile(q=0.1), clusters_stats[cluster]["x_min"])
        x_min = df_cluster.price_m2.quantile(q=0.1)  # Allow evaluating below supply-side power-law
        df_cluster = df_cluster[df_cluster.price_m2 >= x_min]
        print("Min price:", x_min)
        x_mins_demand_cluster.append(x_min)
        x_max = df_cluster.price_m2.quantile(q=0.95)
        X = np.linspace(start=0.01, stop=x_max, num=100)
    
        best_h = None
        best_gamma = None
        best_income = None
        best_cross = None
        best_error = None
        best_error_1 = None
        best_error_2 = None
        best_supply = None
        best_demand_factor = None
        best_income_factor = None
        best_income_percentile = None
        bext_x = None
        best_surplus = None
        best_demand = None
        best_supply = None
        best_index = None
        best_investor = None
        best_income_val = None
        min_price = None
        best_sales_share = None
        theta = normalized_thetas[cluster]
        solutions = []
        owner_median = df_cluster["prob_propietario_0"].median()
        print("Owner median", owner_median)
        for gamma in tqdm(gammas):
            A_factor = (gamma**((1.0/gamma)*(1.0 - 1.0/gamma))) * theta**(1.0/gamma + 1.0)
            epsilon_1 = (1.0/gamma) * (gamma - 2.0 + 1.0/gamma)
            epsilon_2 = (1.0/gamma) * (1.0/gamma - 1.0)
            epsilon_3 = 1.0 / gamma
            # Force A_factor = 1.0 (as in original)
            A_factor = 1.0
            
            for factor in demand_factor:
    
                income_factor = factor**(1.0/epsilon_1)
                        
                supply_f = supply_functions[cluster]        
                try:
                    demand = get_normalized_demand(
                        clusters, neighborhoods, cluster_neighborhood_dict, incomes,
                        normalized_thetas, sizes_cluster, x_min, x_max, gamma, debug=False
                    )
                except Exception as e:
                    print(e)
                    raise e
                    print(f"Gamma: {gamma}, H {h}, FI {income_factor}")
                    continue
                # Surplus curve
                Y_surplus = []
                Y_demand = []
                Y_supply = []
                new_X = []
                completed = True
                for x in X:
                    y_supply = supply_f(x)
                    if np.isnan(y_supply):
                        # y_supply = x
                        y_supply = clusters_stats[cluster]["x_min"]
                    try:
                        demanda_val = demand(x, cluster)
                        y_demand = factor * demanda_val
                        y_surplus = y_demand - y_supply
                        new_X.append(x)
                        Y_demand.append(y_demand)
                        Y_surplus.append(y_surplus)
                        Y_supply.append(y_supply)        
                    except Exception as e:
                        print("***********OVERFLOW*******")
                        print(f"Gamma: {gamma}, H {h}, Demand factor {factor} X {x} Demand {demanda_val}")
                        completed = False
                        break
                if not completed:
                    continue
                
                if len(Y_surplus) == 0:
                    continue
                                 
                # Detect sign change (crossing)
                went_negative = False
                negative_count = 0
                went_positive = False
                positive_count = 0
                found = False
                
                for idx in range(len(new_X)):
                    y = Y_surplus[idx]
                    if y < 0.0:
                        went_negative = True
                        negative_count += 1
                        continue
                    if y >= 0.0:
                        went_positive = True
                        positive_count += 1
                        if went_positive and went_negative:
                            found = True
                            break
                if not found:
                    continue
                idx = np.abs(np.array(Y_surplus) - 0.0).argmin()
                cross = new_X[idx]
                error = np.abs(cross - x_min)  # force near origin
                if error / x_min < 0.02:
                    solutions.append({
                        "Gamma": gamma, 
                        "Factor": factor,
                        "Error": error,
                        "Error_rel": error/x_min,
                        "Cruce": cross,
                    })
                error_1 = error / x_min
                if error_1 > 0.05:
                    continue
    
                ## PROBABILITIES SECTION
                # Center of ln-income mixture (using normalized incomes and income_factor)
                ln_income_center = 0.0
                for nb in neighborhoods:
                    ln_income_center += cluster_neighborhood_dict[(cluster, nb)] * ((incomes_normalized[nb] * income_factor) ** epsilon_1)
                property_income_center = ln_income_center
    
                h = sizes_cluster[cluster] ** epsilon_2
                probs = []
                prices_arr = df_cluster["price_m2"].values
                sales = 0
                cut_values = []
                for i in range(len(df_cluster)):
                    price_m2 = prices_arr[i]
                    price_norm = normalize(price_m2, 0.0, x_max)
                    supply_val = supply_functions[cluster](price_m2)  # Supply normalizes internally
                    def surplus(local_income):
                        demand_loc = ((local_income * income_factor) ** epsilon_1) * h * (price_norm ** epsilon_3) * A_factor
                        return demand_loc - supply_val
                    if surplus(cluster_income_mix) < 0.0 or np.isnan(supply_val):
                        probs.append(None)
                        continue                
                    sales += 1
                    if min_price is None or price_m2 < min_price:
                        min_price = price_m2
                    if income_multiplicative_factor is None:
                        grid = np.linspace(start=0.01, stop=0.99, num=disc)
                        percentiles = scipy.stats.norm.ppf(grid, loc=mu, scale=sigma)
                        income_grid = np.exp(percentiles)
                        values = [surplus(val) for val in income_grid]
                        idx = np.abs(np.array(values) - 0.0).argmin()
                        income_cut = grid[idx]
                        threshold = np.log(income_cut)
                        probability = 1 - stats.norm.cdf(threshold, loc=mu, scale=sigma) 
                        probs.append(probability)
                        value = np.log(income_cut)
                        cut_values.append(value)
                    else:
                        cap = income_multiplicative_factor * cluster_income_mix
                        grid = np.linspace(start=1e-6, stop=cap, num=disc)
                        values = [surplus(val) for val in grid]
                        idx = np.abs(np.array(values) - 0.0).argmin()
                        income_cut = grid[idx]
                        threshold = np.log(income_cut)
                        probability = 1 - stats.norm.cdf(threshold, loc=mu, scale=sigma) 
                        probs.append(probability)
                        value = np.log(income_cut)
                        cut_values.append(value)
                probs = [p for p in probs if not p is None]
                investor_median = np.median(probs)
                
                mu = np.log(cluster_income_mix)
                median_salary = np.median([val for val in cut_values if not val is None])
                median_income_cut = median_salary
                income_percentile = stats.norm.cdf(median_salary, loc=mu, scale=sigma) 
            
                error_2 = np.abs(investor_median - owner_median)
                total_error = error_1 + error_2
                
                if best_error is None or total_error < best_error:
                    best_gamma = gamma
                    best_error = total_error
                    best_cross = cross
                    best_x = new_X
                    best_surplus = Y_surplus
                    best_demand_factor = factor
                    best_income_factor = income_factor
                    best_income_percentile = income_percentile
                    best_supply = Y_supply
                    best_demand = Y_demand
                    best_min = cross
                    best_error_1 = error_1
                    best_error_2 = error_2
                    best_investor = investor_median
                    best_income_val = median_income_cut
                    best_sales_share = sales / len(df_cluster)

        plt.plot(best_x, best_surplus, color="blue", label="Surplus")
        plt.plot(best_x, best_supply, color="green", label="Supply")
        plt.plot(best_x, best_demand, color="red", label="Demand")
        plt.title(f"Cluster {cluster} Gamma {best_gamma}")
        plt.legend()
        plt.show()
        print("Cross", best_min)
        print("Error", best_error)
        print("Error 1", best_error_1)
        print("Error 2", best_error_2)
        print("Investor median:", best_investor)
        print("Minimum acquisition price:", min_price)
        print("Sales share:", best_sales_share)
        print("Income percentile", best_income_percentile)
        # print("Income", best_income_val)
        gammas_cluster.append(best_gamma)
        hs_cluster.append(best_h)
        demand_factors_cluster.append(best_demand_factor)
        income_factors_cluster.append(best_income_factor)
        income_percentiles_cluster.append(best_income_percentile)
        errors_cluster.append(best_error)
        errors_1_cluster.append(best_error_1)
        errors_2_cluster.append(best_error_2)
        incomes_cluster_copy.append(median_income_cut)
        min_price_cluster.append(min_price)
        sales_shares_cluster.append(best_sales_share)

        # Recompute per-listing investor probabilities at the calibrated params
        sales = 0
        gamma = best_gamma
        epsilon_1 = (1.0/gamma) * (gamma - 2.0 + 1.0/gamma)
        epsilon_2 = (1.0/gamma) * (1.0/gamma - 1.0)
        epsilon_3 = 1.0 / gamma
        A_factor = 1.0
        h = sizes_cluster[cluster] ** epsilon_2
        probs_k_cluster = []
        for i in tqdm(range(len(df_cluster))):
            listing = df_cluster.iloc[i]
            probs_k = []
            prices_k = []
            price_m2 = listing["price_m2"]
            price_norm = normalize(price_m2, 0.0, x_max)
            supply_f = supply_functions[cluster]
            supply_val = supply_f(price_m2)
            probs_k = []
            probs_k_cluster.append(probs_k)
            for t in range(k_limit):
                def surplus(local_income):
                    demand_loc = ((local_income * best_income_factor) ** epsilon_1) * h * (price_norm ** epsilon_3) * A_factor
                    return demand_loc - supply_val
    
                s_val = surplus(cluster_income_mix)
                if s_val < 0.0 or np.isnan(supply_val):
                    probs_k.append(None)
                    continue
                elif t == 0 and s_val >= 0.0:
                    sales += 1
                
                if income_multiplicative_factor is None:
                    grid = np.linspace(start=0.01, stop=0.99, num=disc)
                    percentiles = scipy.stats.norm.ppf(grid, loc=mu, scale=sigma)
                    income_grid = np.exp(percentiles)
                    values = [surplus(val) for val in income_grid]
                    idx = np.abs(np.array(values) - 0.0).argmin()
                    income_cut = grid[idx]
                    threshold = np.log(income_cut)
                    probability = 1 - stats.norm.cdf(threshold, loc=mu, scale=sigma) 
                    probs.append(probability)
                    value = np.log(income_cut)
                    cut_values.append(value)
                else:
                    cap = income_multiplicative_factor * cluster_income_mix
                    grid = np.linspace(start=1e-6, stop=cap, num=disc)
                    values = [surplus(val) for val in grid]
                    idx = np.abs(np.array(values) - 0.0).argmin()
                    income_cut = grid[idx]
                    threshold = np.log(income_cut)
                    probability = 1 - stats.norm.cdf(threshold, loc=mu, scale=sigma) 
                    probs.append(probability)
                    value = np.log(income_cut)
                    cut_values.append(value)
    
                probs_k.append(probability)  # Success probability of selling at stage k.
                supply_val = supply_f(supply_val)
        print("Sales proportion", float(sales) / len(df_cluster)) 
        for t in range(k_limit):
            df_cluster[f"prob_investor_{t}"] = [probs_k_cluster[prop_idx][t] for prop_idx in range(len(probs_k_cluster))]    
    
        # Diagnostics vs owners
        med_owner = df_cluster['prob_propietario_0'].median()
        med_inv = df_cluster['prob_investor_0'].median()
        
        plt.figure()
        plt.scatter(df_cluster['prob_propietario_0'], df_cluster['prob_investor_0'])
        plt.axvline(med_owner, linestyle='--')
        plt.axhline(med_inv, linestyle='--')
        plt.xlabel('Probability Owner')
        plt.ylabel('Probability Investor')
        plt.title(f'Scatter plot {cluster} of Probability Owner vs Probability Investor, with lines for the median values')
        plt.show()
        print("Minimum Investor prob", df_cluster['prob_investor_0'].min())
    
        dgs.append(df_cluster)

    dh = pd.concat(dgs)        
    dh.to_pickle(output_file)            

    for t in range(10):
        print(f"NaN in column {t}: {dh[f'prob_investor_{t}'].isna().mean()}")

    plt.boxplot(dh.prob_investor_0.dropna())

    incomes_cluster_by_neighborhood = []
    for cluster in clusters:
        print("Cluster", cluster)
    
        cluster_income = 0.0
        for nb in neighborhoods:
            cluster_income += cluster_neighborhood_dict[(cluster, nb)] * ((incomes[nb]))
        print(f"Cluster Income {cluster_income}")
        incomes_cluster_by_neighborhood.append(cluster_income)

    text = """
    \\begin{table}[h!]
    \\begin{tabular}{|c|c|c|c|c|c|}
    \\hline
    \\textbf{Parameter / Cluster} & \\textbf{0} & \\textbf{1} & \\textbf{2} & \\textbf{3} & \\textbf{4} \\\\ \\hline
    
    """

    obj = {}
    obj["ingresos_cluster_barriales"] = incomes_cluster_by_neighborhood
    obj["percentiles_ingreso_cluster"] = income_percentiles_cluster
    obj["sizes_cluster"] = sizes_cluster
    obj["min_price_cluster"] = min_price_cluster
    obj["thetas_cluster"] = thetas
    obj["errores_1_cluster"] = errors_1_cluster
    obj["errores_2_cluster"] = errors_2_cluster
    obj["percentiles_ingreso_cluster"] = income_percentiles_cluster
    obj["gammas_cluster"] = gammas_cluster
    obj["thetas"] = thetas

    import pickle
    f = open(report_dir + "/obj_demanda.pkl", "wb")
    pickle.dump(obj, f)
    f.close()
    
    text += "\\textbf{$\\omega(j,b)|_l$} & "
    for cluster in clusters:
        if cluster < 4:
            text += str(round(incomes_cluster_by_neighborhood[cluster], 2)) + " & "
        else:
            text += str(round(incomes_cluster_by_neighborhood[cluster], 2)) + " \\\\ \\hline \n"
    
    text += "Income Percentile & "
    for cluster in clusters:
        if cluster < 4:
            text += str(round(income_percentiles_cluster[cluster], 2)) + " & "
        else:
            text += str(round(income_percentiles_cluster[cluster], 2)) + " \\\\ \\hline \n"
    
    text += "\\textbf{$h(l)$} & "
    for cluster in clusters:
        if cluster < 4:
            text += str(round(sizes_cluster[cluster], 2)) + " & "
        else:
            text += str(round(sizes_cluster[cluster], 2)) + " \\\\ \\hline \n"
    
    text += "$\\underline{p}_l^{se}$ & "
    for cluster in clusters:
        if cluster < 4:
            text += str(round(min_price_cluster[cluster], 2)) + " & "
        else:
            text += str(round(min_price_cluster[cluster], 2)) + " \\\\ \\hline \n"
    
    text += "\\textbf{$\\theta^l$} & "
    for cluster in clusters:
        if cluster < 4:
            text += str(round(thetas[cluster], 2)) + " & "
        else:
            text += str(round(thetas[cluster], 2)) + " \\\\ \\hline \n"
    
    text += "\\textbf{$\\gamma^l$} & "
    for cluster in clusters:
        if cluster < 4:
            text += str(round(gammas_cluster[cluster], 2)) + " & "
        else:
            text += str(round(gammas_cluster[cluster], 2)) + " \\\\ \\hline \n"
    
    text += """
    \\end{tabular}
    \\caption{Calibrated parameters. Demand side and Stationary Equilibrium}
    \\label{table:cal_3}
    \\end{table}
    
    """
    print(text)    

    text = """
    \\begin{table}[h!]
    \\begin{tabular}{|c|c|c|c|c|c|}
    \\hline
    \\textbf{Parameter / Cluster} & \\textbf{0} & \\textbf{1} & \\textbf{2} & \\textbf{3} & \\textbf{4} \\\\ \\hline
    
    """
    
    text += "ERROR 1 & "
    for cluster in clusters:
        if cluster < 4:
            text += str(round(errors_1_cluster[cluster], 2)) + " & "
        else:
            text += str(round(errors_1_cluster[cluster], 2)) + " \\\\ \\hline \n"
    
    text += "$|m(f^l)-m(\\bar{F}^l)|$ & "
    for cluster in clusters:
        if cluster < 4:
            text += str(round(errors_2_cluster[cluster], 2)) + " & "
        else:
            text += str(round(errors_2_cluster[cluster], 2)) + " \\\\ \\hline \n"
    
    text += "Income Percentile & "
    for cluster in clusters:
        if cluster < 4:
            text += str(round(income_percentiles_cluster[cluster], 2)) + " & "
        else:
            text += str(round(income_percentiles_cluster[cluster], 2)) + " \\\\ \\hline \n"
    
    text += "\\textbf{$h(l)$} & "
    for cluster in clusters:
        if cluster < 4:
            text += str(round(sizes_cluster[cluster], 2)) + " & "
        else:
            text += str(round(sizes_cluster[cluster], 2)) + " \\\\ \\hline \n"
    
    text += "$\\underline{p}_l^{se}$ & "
    for cluster in clusters:
        if cluster < 4:
            text += str(round(min_price_cluster[cluster], 2)) + " & "
        else:
            text += str(round(min_price_cluster[cluster], 2)) + " \\\\ \\hline \n"
    
    text += "\\textbf{$\\theta^l$} & "
    for cluster in clusters:
        if cluster < 4:
            text += str(round(thetas[cluster], 2)) + " & "
        else:
            text += str(round(thetas[cluster], 2)) + " \\\\ \\hline \n"
    
    text += "\\textbf{$\\gamma^l$} & "
    for cluster in clusters:
        if cluster < 4:
            text += str(round(gammas_cluster[cluster], 2)) + " & "
        else:
            text += str(round(gammas_cluster[cluster], 2)) + " \\\\ \\hline \n"
    
    text += """
    \\end{tabular}
    \\caption{Calibrated parameters. Demand side and Stationary Equilibrium}
    \\label{table:cal_3}
    \\end{table}
    
    """
    
    print(text)









## STEP 7 - INVESTOR PROBLEM








def investor_problem(
    input_file="./data/paper/data_madrid_probs_v2.pkl",
    prob_file="./data/paper/data_Madrid_probs_propietario.pkl",
    cluster_stats_file="./data/paper/clusters_stats.pkl",
    output_file="./data/paper/data_ROI.pkl",
):

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    from importlib import reload
    import powerlaw
    from scipy.stats import lognorm
    from tqdm import tqdm

    # Load per-listing investor probabilities (and other data)
    df_calc = pd.read_pickle(input_file)
    clusters = sorted(list(df_calc.Cluster.unique()))
    
    # Load calibrated supply-side stats and build supply functions per cluster
    import pickle
    f = open(cluster_stats_file, "rb")
    clusters_stats = pickle.load(f)
    f.close()
    supply_functions = [get_normalized_supply(cluster, clusters_stats) for cluster in clusters]
    # clusters_stats

    # Load owner-side probabilities (per period k)
    df_owner = pd.read_pickle(prob_file)
    k_limit = 10
    selected_cols = [f"prob_propietario_{k}" for k in range(1, k_limit)] + ["id"]
    df_calc = df_calc.merge(df_owner[selected_cols], how="left", on="id")

    # Quick NA rate check (kept as an expression as in original)
    df_calc.prob_investor_0.isna().mean()

    # Keep rows with an initial investor probability
    df_calc = df_calc[df_calc.prob_investor_0.isna() == False]
    len(df_calc)

    # Parameters
    # r = 0.05
    # r = 0.01
    zeta = -4
    # zeta = 1.0
    beta = 0.99
    pS = 0.03
    # r = (1/beta)-1
    r = 0.0120
    
    print(r)
    
    k_limit = 10
    # k_limit = 1
    
    VI, VO, buyable_flags = [], [], np.zeros(len(df_calc))
    
    # Iterate listings to compute investor value (VI) and owner value (VO)
    for i in tqdm(range(len(df_calc))):
        listing = df_calc.iloc[i]
        cluster = listing["Cluster"]
        price_m2 = listing["price_m2"]  # ✅ PRICE PER m2
        # size = listing["size"]
        size = 1.0
        supply = supply_functions[cluster]
        
        # ---- VI (investor) ----
        sum_vi, prob_failure_vi, price_m2_t = 0.0, 1.0, price_m2
        price_t = price_m2_t * size
        if np.isnan(listing["prob_investor_0"]):
            VI.append(None); VO.append(None)
            continue
        for k in range(k_limit):
            f_k = listing[f"prob_investor_{k}"]
            if f_k is None:
                f_k = 0.0
            # income_vi = ((1+r)/r) * f_k * prob_failure_vi * price_t * (1+r)**(-k)
            income_vi = f_k * prob_failure_vi * price_t * ((1 + r) / r) * (1 + r) ** (-k)
            cost_vi = r * prob_failure_vi * price_t * (1 + r) ** (-k)
            sum_vi += income_vi - cost_vi
            prob_failure_vi *= (1 - f_k)
            price_m2_t = supply(price_m2_t) if not np.isnan(supply(price_m2_t)) else price_m2_t
            price_t = price_m2_t * size
        VI.append(sum_vi)
    
        # print("VI", sum_vi)    
    
        # ---- VO (owner) ----
        sum_vo, prob_failure_vo, price_m2_t = 0.0, 1.0, price_m2
        price_t = price_m2_t * size
        for k in range(k_limit):
            F_k = listing[f"prob_propietario_{k}"]
            if F_k is None:
                F_k = 0.0
            income_vo = F_k * prob_failure_vo * price_t * (beta / (1 - beta)) * beta**k
    
            ceiling = clusters_stats[cluster]["x_min"]
            # floor = supply(price_m2_t)
            floor = price_m2_t
            # Holding cost with curvature zeta
            holding_cost = (1 / zeta) * (size * ceiling - size * floor) ** zeta
            cost_vo = r * prob_failure_vo * holding_cost * beta**k
            """
            print("k", k)
            print("Income vo", income_vo)
            print("floor", floor)
            print("ceiling", ceiling)
            print("holding_cost", holding_cost)
            print("cost_vo", cost_vo)
            """
            # if np.isnan(cost_vo) or np.isinf(cost_vo):
            #     cost_vo = 0.0
    
            # print("vo", income_vo - cost_vo)
    
            if np.isnan(cost_vo) or np.isinf(cost_vo):
                sum_vo += 0.0
            else:
                sum_vo += income_vo - cost_vo   
                
            # sum_vo += income_vo - cost_vo
            prob_failure_vo *= (1 - F_k)
            price_m2_t = supply(price_m2_t) if not np.isnan(supply(price_m2_t)) else price_m2_t
            price_t = price_m2_t * size
        VO.append(sum_vo)
    
        # Buy decision flag (internal var translated; output column name preserved)
        buyable_flags[i] = 1.0 if VI[-1] > VO[-1] else 0.0
    
        # 1/0
    
    df_calc["VI"] = VI
    df_calc["VO"] = VO
    df_calc["comprable"] = buyable_flags  # keep output column name in Spanish
    df_calc.comprable.mean()
    
    # ROI computation
    def compute_ROI(x):
        if np.isnan(x["VI"]) or np.isnan(x["VO"]):
            return np.nan
        else:
            return (x["VI"] - x["VO"]) / x["VO"]
    
    df_calc["ROI"] = df_calc[["VI", "VO"]].apply(compute_ROI, axis=1)
    df_calc[["VI", "VO", "ROI"]]

    # Sort by ROI (not persisted separately; kept for parity with original)
    df_sorted = df_calc.sort_values(by="ROI", ascending=False)

    intrinsic = [
        "id",
        "fecha",
        "operation",
        "datasource_name",
        "property_type",
        "subtype",
        "municipality",
        "municipality_code5",
        "municipality_code5num",
        "comaut",
        "comaut_code",
        "province",
        "province_code",
        "district",
        "neighborhood",
        "title",
        "postal_code",
        "postal_codenum",
        "latitude_x",
        "longitude_x",
        "price",
        "lprice",
        "price_m2",
        "lprice_m2",
        "size",
        "lsize",
        "floor",
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
        "status",
        "new_construction",
        "rating_leads",
        "rating_visits",
    ]

    # Filtered DataFrame (not used later; kept to mirror original code)
    df_top = df_sorted[df_sorted.price_m2 <= 10000]

    # Persist results
    df_calc.to_pickle(output_file)









## STEP 8 - PLOTS








def neighborhood_plots(input_file="./data/paper/data_ROI.pkl"):
    
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    df = pd.read_pickle(input_file)
    df.ROI.min(), df.ROI.max()
    df = df[df.comprable == 1.0]
    df.ROI.min(), df.ROI.max()
    
    import geopandas as gpd
    from shapely.geometry import mapping
    import json
    from pyproj import Transformer
    
    # Load the SHP file using geopandas
    shp_path = './data/Madrid/Barrios.shp'  # Replace with your SHP file path
    gdf = gpd.read_file(shp_path)
    
    # Create a transformer to convert from UTM to WGS84
    # Assume coordinates are in UTM zone 30N (EPSG:25830)
    transformer = Transformer.from_crs("epsg:25830", "epsg:4326", always_xy=True)
    
    # Create a dictionary with neighborhood name and coordinates
    neighborhoods_dict = {}
    
    for _, row in gdf.iterrows():
        neighborhood_name = row['NOMBRE']  # Adjust according to the neighborhood-name field in your file
        geometry_obj = row['geometry']
        
        # Check whether geometry is Polygon or MultiPolygon
        if geometry_obj.geom_type == 'Polygon':
            coords = list(geometry_obj.exterior.coords)
            # Convert coordinates to longitude and latitude
            coords = [transformer.transform(x, y) for x, y in coords]
            neighborhoods_dict[neighborhood_name] = coords
        elif geometry_obj.geom_type == 'MultiPolygon':
            coords = []
            for polygon in geometry_obj:
                polygon_coords = list(polygon.exterior.coords)
                # Convert coordinates to longitude and latitude
                polygon_coords = [transformer.transform(x, y) for x, y in polygon_coords]
                coords.extend(polygon_coords)
            neighborhoods_dict[neighborhood_name] = coords
    
    import numpy as np
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    import plotly.graph_objects as go
    
    def rgba_to_plotly_string(rgba):
        """Convert a (r, g, b, a) tuple (values 0–1) into a 'rgba(R,G,B,A)' string."""
        r, g, b, a = rgba
        return f"rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, {a})"
    
    def plot_neighborhood_with_ranking(
            df, value_label, path, neighborhoods_dict,
            municipality="Madrid", min_count=0,
            fill_opacity=0.5, font_size=12):
        """
        Draw polygons with go.Scattermapbox using RGBA fillcolor (for transparency)
        and then overlay the ranking as text.

        Parameters
        ----------
        df : pd.DataFrame
            Minimum columns: 'barrio', 'municipality' and value_label.
        value_label : str
            Column whose means are colored.
        path : str
            Output HTML path.
        neighborhoods_dict : dict[str, list[tuple[lon, lat]]]
            Dictionary {neighborhood: [(lon, lat), …]} with the vertices.
        municipality : str
        min_count : int
        fill_opacity : float
            0.0 = fully transparent, 1.0 = fully opaque.
        font_size : int
            Ranking text size.
        """
        # 1) Filter, group and compute ranking
        dg = (df.loc[(df["municipality"] == municipality)]
                .groupby("barrio")
                .agg(
                    scoring=(value_label, "mean"),
                    count=(value_label, "size")
                )
                .reset_index())
        print(f"Number of neighborhoods for {value_label} BEFORE FILTER: {len(dg)}")
        print(dg.sort_values("count"))
        dg = dg[dg["count"] >= min_count].sort_values("scoring", ascending=False)
        dg["ranking"] = np.arange(1, len(dg) + 1)
    
        print(f"Number of neighborhoods for {value_label} AFTER FILTER: {len(dg)}")
        
        # 2) Prepare color mapping with transparency
        min_val, max_val = dg["scoring"].min(), dg["scoring"].max()
        norm = mcolors.Normalize(vmin=min_val, vmax=max_val)
        cmap = cm.get_cmap("viridis")
        scalar_map = cm.ScalarMappable(norm=norm, cmap=cmap)
        
        # 3) Create figure and add polygons
        fig = go.Figure()
        for _, row in dg.iterrows():
            neighborhood = row["barrio"]
            if neighborhood not in neighborhoods_dict:
                continue
            value, count, rank = row["scoring"], row["count"], row["ranking"]
            coords = neighborhoods_dict[neighborhood]
            lons, lats = zip(*coords)
            
            # RGBA color with desired transparency level
            r, g, b, _ = scalar_map.to_rgba(value)
            color = rgba_to_plotly_string((r, g, b, fill_opacity))
            
            fig.add_trace(go.Scattermapbox(
                fill="toself",
                lon=lons, lat=lats,
                marker={"size": 0},
                fillcolor=color,
                line={"color": "black", "width": 1},
                hoverinfo="text",
                hovertext=(
                    f"<b>{neighborhood}</b><br>"
                    f"{value_label}: {value:.2f}<br>"
                    f"N: {count}"
                ),
                showlegend=False
            ))
            
            # Centroid for ranking label
            centroid_lon = np.mean(lons)
            centroid_lat = np.mean(lats)
            fig.add_trace(go.Scattermapbox(
                lon=[centroid_lon], lat=[centroid_lat],
                mode="text",
                text=[str(rank)],
                textfont={"size": font_size, "color": "black"},
                hoverinfo="none",
                showlegend=False
            ))
        
        # 4) (Optional) Color scale as an invisible trace for the colorbar
        tick_vals = np.linspace(min_val, max_val, 5)
        tick_text = [f"{v:.2f}" for v in tick_vals]
        fig.add_trace(go.Scattermapbox(
            lon=[None], lat=[None], mode="markers",
            marker=dict(
                showscale=True,
                colorscale="Viridis",
                cmin=min_val, cmax=max_val,
                color=[min_val],
                colorbar=dict(
                    title=value_label,
                    tickvals=tick_vals,
                    ticktext=tick_text,
                    ticks="outside"
                )
            ),
            hoverinfo="none",
            showlegend=False
        ))
        
        # 5) Final layout
        fig.update_layout(
            mapbox_style="carto-positron",
            mapbox=dict(center=dict(lat=40.4168, lon=-3.7038), zoom=11),
            margin=dict(t=40, r=0, l=0, b=0),
            title=value_label
        )
        
        # 6) Save and return
        fig.write_html(path, include_plotlyjs="cdn")
        return fig
    
    import numpy as np
    
    df_investor = pd.read_pickle("./data/paper/data_madrid_probs.pkl")
    print("Listings in df_investor", len(df_investor))
    df_investor = df_investor[(df_investor.municipality == "Madrid") & (df_investor.barrio != None)]
    print("Listings in df_investor after first filter", len(df_investor))
    neighborhood_corrs = []
    new_neighborhoods = []
    
    neighborhoods = list(df_investor.barrio.unique())
    print("Number of neighborhoods with probs", len(df_investor.barrio.unique()))
    # print(neighborhoods)
    for neighborhood in neighborhoods:
        dg = df_investor[df_investor.barrio == neighborhood]
        dg = dg[["prob_investor_0", "price_m2"]].dropna()
        if len(dg) > 0:
            corr = np.corrcoef(dg["prob_investor_0"], dg["price_m2"])[0, 1]
            # if np.isnan(corr):
            #     corr = 0.0
            if not np.isnan(corr):
                new_neighborhoods.append(neighborhood)
                neighborhood_corrs.append(corr)
    
    df_corr = pd.DataFrame({
        "barrio": new_neighborhoods,
        "Corr_prob_investor_price": neighborhood_corrs
    })
    
    print(df_corr)
    
    df_investor = df_investor.merge(df_corr, on="barrio", how="inner")
    print(df_investor[["id", "barrio", "Corr_prob_investor_price"]])
    
    print("******Number of neighborhoods********", len(df_investor.barrio.unique()))
    
    # 1/0
    
    print(len(df_filt))
    
    plot_neighborhood_with_ranking(
        df_investor,
        value_label="prob_investor_0",
        path="./data/reportes/paper/mapas/madrid_B11.html",
        neighborhoods_dict=neighborhoods_dict,
        min_count=5,
        fill_opacity=0.5,
        font_size=12
    )
    
    plot_neighborhood_with_ranking(
        df_investor,
        value_label="Corr_prob_investor_price",
        path="./data/reportes/paper/mapas/madrid_B12.html",
        neighborhoods_dict=neighborhoods_dict,
        min_count=5,
        fill_opacity=0.5,
        font_size=12
    )
    
    plot_neighborhood_with_ranking(
        df_investor,
        value_label="price_m2",
        path="./data/reportes/paper/mapas/madrid_B13.html",
        neighborhoods_dict=neighborhoods_dict,
        min_count=5,
        fill_opacity=0.5,
        font_size=12
    )
    
    df_filtered = df[df.ROI.isna() == False]
    df_filtered = df_filtered[(df_filtered.municipality == "Madrid") & (df_filtered.barrio != None)]
    
    plot_neighborhood_with_ranking(
        df_filtered,
        value_label="ROI",
        path="./data/reportes/paper/mapas/madrid_B21.html",
        neighborhoods_dict=neighborhoods_dict,
        min_count=5,
        fill_opacity=0.5,
        font_size=12
    )
    
    """
    fig = plot_neighborhood_with_sidebar(
        df_filtered,
        value_label="ROI",
        path="./data/reportes/paper/madrid_ranking_tabla.html",
        neighborhoods_dict=neighborhoods_dict,
        min_count=5,
        fill_opacity=0.5,
        font_size=12
    )
    """
    
    df_investor = pd.read_pickle("./data/paper/data_madrid_probs.pkl")
    df_investor = df_investor[df_investor.municipality == "Madrid"]
    neighborhood_corrs = []
    neighborhoods = list(df_investor.barrio.unique())
    print("Number of neighborhoods with probs", len(df_investor.barrio.unique()))
    # print(neighborhoods)
    for neighborhood in neighborhoods:
        dg = df_investor[df_investor.barrio == neighborhood]
        dg = dg[["prob_investor_0", "price_m2"]].dropna()
        corr = np.corrcoef(dg["prob_investor_0"], dg["price_m2"])[0, 1]
        if np.isnan(corr):
            corr = 0.0
        neighborhood_corrs.append(corr)
    
    df_corr = pd.DataFrame({
        "barrio": neighborhoods,
        "Corr_prob_investor_price": neighborhood_corrs
    })

    df = pd.read_pickle(input_file)

    import numpy as np
    import plotly.graph_objects as go
    
    import numpy as np
    import plotly.graph_objects as go
    
    def plot_topN_roi_map(
        df, lat_col, lon_col, roi_col, path,
        marker_symbol="cross",
        marker_color="red",
        marker_size=12,
        bubble_scale=2.5,
        bubble_opacity=0.25,
        map_zoom=13,
        N=10,
    ):
        """
        Map with a “bubble” (semi-transparent circle) and a cross on top
        for the top-N properties by ROI.
        """
        # 1) Top-N by ROI
        df_top = df.nlargest(N, roi_col)
    
        # 2) Map center
        center_lat = df_top[lat_col].mean()
        center_lon = df_top[lon_col].mean()
    
        fig = go.Figure()
    
        # 3) Bubble layer – large semi-transparent circle
        fig.add_trace(go.Scattermapbox(
            lat=df_top[lat_col],
            lon=df_top[lon_col],
            mode="markers",
            marker=dict(
                symbol="circle",
                size=marker_size * bubble_scale,
                color=marker_color,
                opacity=bubble_opacity,
                allowoverlap=True
            ),
            hoverinfo="skip",
            showlegend=False
        ))
    
        # 4) Symbol + text layer
        fig.add_trace(go.Scattermapbox(
            lat=df_top[lat_col],
            lon=df_top[lon_col],
            mode="markers+text",
            marker=dict(
                symbol=marker_symbol,
                size=marker_size,
                color=marker_color
            ),
            text=df_top[roi_col].round(2).astype(str),
            textposition="top center",
            hovertemplate=(
                f"{roi_col}: " + "%{text}<br>" +
                "lat: " + "%{lat:.5f}<br>" +
                "lon: " + "%{lon:.5f}<extra></extra>"
            ),
            showlegend=False
        ))
    
        # 5) Layout and export
        fig.update_layout(
            mapbox_style="carto-positron",
            mapbox=dict(center=dict(lat=center_lat, lon=center_lon),
                        zoom=map_zoom),
            margin=dict(t=0, b=0, l=0, r=0),
            title=f"Top {N} properties by {roi_col}"
        )
    
        fig.write_html(path, include_plotlyjs="cdn")
        return fig

    df_filtered = df[df.ROI.isna() == False]
    df_filtered = df_filtered[(df_filtered.municipality == "Madrid") & (df_filtered.barrio != None)]
    
    fig = plot_topN_roi_map(
        df=df_filtered,
        lat_col="latitude_x",
        lon_col="longitude_x",
        roi_col="ROI",
        N=10,
        path="./data/reportes/paper/mapas/madrid_B22_10.html",
        marker_symbol="cross",    # or "marker", "circle", "star", etc.
        marker_color="blue",
        marker_size=16,
        map_zoom=13
    )
    
    fig = plot_topN_roi_map(
        df=df_filtered,
        lat_col="latitude_x",
        lon_col="longitude_x",
        roi_col="ROI",
        N=20,
        path="./data/reportes/paper/mapas/madrid_B22_20.html",
        marker_symbol="cross",    # or "marker", "circle", "star", etc.
        marker_color="blue",
        marker_size=16,
        map_zoom=13
    )
    
    fig = plot_topN_roi_map(
        df=df_filtered,
        lat_col="latitude_x",
        lon_col="longitude_x",
        roi_col="ROI",
        N=30,
        path="./data/reportes/paper/mapas/madrid_B22_30.html",
        marker_symbol="cross",    # or "marker", "circle", "star", etc.
        marker_color="blue",
        marker_size=16,
        map_zoom=13
    )
    

