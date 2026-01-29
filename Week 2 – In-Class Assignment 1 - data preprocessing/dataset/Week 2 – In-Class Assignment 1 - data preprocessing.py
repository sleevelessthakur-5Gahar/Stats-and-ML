#Please note that for each line of code, the explanation comment is inserted below the said line of code. (ALWAYS)
#What shall be achieved:-
#Data set --> CLEANING --> NOISE HANDLED --> OUTLIER HANDLING --> TRANSFORMED --> SCALING
#############################################################################
#STEP 1:-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#Telling pandas to ignore the commentes rows in the dataset also I used the following line of code because I was initially working on pycharm:-
data=pd.read_csv("PS_2026.01.18_00.01.05.csv",comment="#",low_memory=False)
#low_memory=False is used for avoiding mixed-type warnings
#Reading the csv file that I downloaded from NASA.
print("Output for data.shape :- ")
print(data.shape)
print()
print()
# This is an attribute, not a function (so you don't use parentheses ()).
# It returns a tuple representing the dimensions of the DataFrame.
# What it shows: (number_of_rows, number_of_columns).
# Best for: Quick size checks.
# If you perform a "merge" or "drop" operation, checking the .shape before and after is the fastest way to see if you lost or gained data unexpectedly.
print("Output for data.head(10) :-")
print(data.head(10))
print()
print()
print("Output for data.tail(20) :- ")
print(data.tail(20))
# This returns the first n rows of the DataFrame (the default is 5).
# What it shows: An actual "snapshot" of your raw data.Best for: Getting a visual sense of the data.
# It helps you see if the headers are aligned correctly, if there are weird characters in the strings,
# or if the data formatting looks consistent.Note: You can also use df.tail() to see the last few rows which I have.
print()
print()
print("Output for data.info() :- ")
print(data.info())
# This method provides a high-level summary of the DataFrame's structure. It is the best tool for checking if your data loaded correctly.
# What it shows: * The total number of rows (entries) and columns.
# The name of every column.
# The number of non-null (not missing) values in each column.
# The Data Type (e.g., int64, float64, object for strings).
# How much memory (RAM) the DataFrame is using.
# Best for: Identifying missing values and checking if column types are correct (e.g., ensuring a "Date" column isn't being read as a "String"):-
print()
print()
print("Output for data.describe() :- ")
print(data.describe())
# This generates descriptive statistics for the numerical columns in your dataset.
# What it shows: * Count: Number of non-empty values.
# Mean: The average value.
# Std: Standard deviation (how spread out the data is).
# Min/Max: The smallest and largest values.
# 25%, 50%, 75%: Percentiles (the 50% mark is the Median).
# Best for: Detecting outliers. For example, if you are looking at "Planet Radius" and the max is 1,000
# times larger than the mean, you likely have an outlier or a data entry error.
print()
print()
print()
print()
columns_to_keep = [
    # Star identifiers & position
    "hostname", "ra", "dec",

    # Stellar properties
    "st_spectype", "st_mass", "st_rad", "st_teff", "st_lum",
    "st_logg", "st_met", "st_age", "sy_dist",

    # Planet identifiers
    "pl_name",

    # Planet physical properties
    "pl_rade", "pl_masse", "pl_dens", "pl_eqt", "pl_insol",
    "pl_bmasse", "pl_orbeccen", "pl_orbincl", "pl_ratror",
    "pl_trandep", "pl_trandur",

    # Orbital properties
    "pl_orbper", "pl_orbsmax", "pl_orbtper", "pl_orblper",
    "pl_ratdor", "pl_imppar", "pl_tranmid", "pl_orbeccenlim",

    # Discovery & observation metadata
    "discoverymethod", "disc_year", "disc_facility",
    "disc_telescope", "disc_instrument",
    "rv_flag", "tran_flag", "ttv_flag"
]
data = data[columns_to_keep]
#This is where we delete the unnecessary columns and keep only what we need (40 columns)
# data[---] = select specific columns
# columns_to_keep = the 40 columns we care about
# data =  overwrite the dataframe with the reduced version
print("Remaining columns:", data.shape[1])

newNames = {
    # Star identifiers & position
    "hostname": "host_star_name",
    "ra": "right_ascension_deg",
    "dec": "declination_deg",

    # Stellar properties
    "st_spectype": "star_spectral_type",
    "st_mass": "star_mass_solar",
    "st_rad": "star_radius_solar",
    "st_teff": "star_temperature_K",
    "st_lum": "star_luminosity_solar",
    "st_logg": "star_surface_gravity_log",
    "st_met": "star_metallicity",
    "st_age": "star_age_gyr",
    "sy_dist": "star_distance_parsec",

    # Planet identifiers
    "pl_name": "planet_name",

    # Planet physical properties
    "pl_rade": "planet_radius_earth",
    "pl_masse": "planet_mass_earth",
    "pl_dens": "planet_density",
    "pl_eqt": "equilibrium_temperature_K",
    "pl_insol": "stellar_irradiance_earth_units",
    "pl_bmasse": "best_planet_mass_earth",
    "pl_orbeccen": "orbital_eccentricity",
    "pl_orbincl": "orbital_inclination_deg",
    "pl_ratror": "planet_star_radius_ratio",
    "pl_trandep": "transit_depth",
    "pl_trandur": "transit_duration_hours",

    # Orbital properties
    "pl_orbper": "orbital_period_days",
    "pl_orbsmax": "semi_major_axis_AU",
    "pl_orbtper": "time_of_periastron_days",
    "pl_orblper": "longitude_of_periastron_deg",
    "pl_ratdor": "distance_to_star_radius_ratio",
    "pl_imppar": "impact_parameter",
    "pl_tranmid": "transit_midpoint_time",
    "pl_orbeccenlim": "orbital_eccentricity_limit_flag",

    # Discovery & observation metadata
    "discoverymethod": "discovery_method",
    "disc_year": "discovery_year",
    "disc_facility": "discovery_facility",
    "disc_telescope": "discovery_telescope",
    "disc_instrument": "discovery_instrument",
    "rv_flag": "radial_velocity_used",
    "tran_flag": "transit_method_used",
    "ttv_flag": "transit_timing_variation_used"
}
#Pretty self explanatory that we are renaming all of the columns which will help the stakeholders and me
#Me specifically during the feature engineering phase.
#We have made a dictionary in which:-
#Keys are old NASA column names
#Values are clean scientific names
data=data.rename(columns=newNames)
#Applying the renaming.
#This step doesn't have to happen now but I did it just in case.
print()
print()
print()
print("Column names old and new in the dataset:- ")
print(data.columns.tolist())
#The aforementioned line of code will display each column as a list.
#I personally wanted for it to be like this because of renaming that I did.
print()
print()
print()
print("Number of missing values in each column:- ")
pd.set_option('display.max_rows', None)
#The aforementioned line of code will set display rule to none.
#Meanning everything will be displayed.
print(data.isnull().sum())
#What data.isNull() do:-
#This creates a new table (DataFrame) of the same size as df.
# Every value becomes:
# True = if the original value is missing
# False = if the original value exists
#What .sum() does:-
#In computer:-
#True = 1
# False = 0
# So .sum():
# Adds up all True values column by column
# Which results in:
# Number of missing values per column
pd.reset_option('display.max_rows')
#The aforementioned line of code will reset the rule to normal so that in future when we try displaying a DataFrame,
# everything is not displayed.
###################################################################################
#We are now all set for:-
# Missing value handling 1
# Noise simulation 2
# Outlier detection 3
# Feature engineering 4
# Encoding & scaling 5
###################################################################################
###################################################################################
###################################################################################
###################################################################################
# STEP 1.1 Missing value handling :-
# We now have:-
# Renamed columns (good descriptive names like orbital_period_days)
# Printed missing values per column
# A very large real-world dataset (40k rows, 289 columns)
####################################
#We will use Median for numeric values because of outliers in the NASA dataset since we have planets/stars which can weigh up to
#Million times the mass of our sun.
#We will use Mode for Categorical values because we can't perform maths on words.
#Missing Value Handling (Safe Approach):-
numerical_columns = [
    "right_ascension_deg",
    "declination_deg",
    "star_mass_solar",
    "star_radius_solar",
    "star_temperature_K",
    "star_age_gyr",
    "star_distance_parsec",
    "planet_radius_earth",
    "orbital_period_days",
    "semi_major_axis_AU",
    "discovery_year" ]
#These are numeric columns where using median won’t break
#ASTROPHYSICS principles too badly in general statistics.
categorical_columns = [
    "host_star_name",
    "planet_name",
    "star_spectral_type",
    "discovery_method",
    "discovery_facility",
    "discovery_telescope",
    "discovery_instrument",
    "radial_velocity_used",
    "transit_method_used",
    "transit_timing_variation_used"
]
#These columns are categorical,
# so replacing missing values with the mode (most common value) is safe.
# Fill numerical columns with median
physics_calculable_cols = [
    "star_luminosity_solar",
    "planet_density",
    "equilibrium_temperature_K",
    "stellar_irradiance_earth_units",
    "star_surface_gravity_log",
    "planet_mass_earth",
    "best_planet_mass_earth",
    "orbital_eccentricity",
    "orbital_inclination_deg",
    "planet_star_radius_ratio",
    "transit_depth",
    "transit_duration_hours",
    "time_of_periastron_days",
    "longitude_of_periastron_deg",
    "distance_to_star_radius_ratio",
    "impact_parameter",
    "transit_midpoint_time",
    "orbital_eccentricity_limit_flag",
]
#The aforementioned are calculable columns which we will calculate later during feature engineering.
for i in numerical_columns:
    median=data[i].median()
    data[i]=data[i].fillna(median)

for i in categorical_columns:
    mode=data[i].mode()[0]
    data[i]=data[i].fillna(mode)
for i in physics_calculable_cols:
    missingVals=data[i].isnull() #This will fetch all of the values which are null in the physics calculable col list.
    #True = missing
    #False = value exists
    newColumn=i+"_is_missing_values"
    data[newColumn]=missingVals
print()
print()
print()
print("Number of new missing vals:")
print(data.isnull().sum())
#After this print, we will be able to see that only physics calculable columns
#have missing values or NaN's
#which is fine.
#We have already filled in numerical and categorical columns which we could have.
####################################
####################################
# Step 2:-
#Visualize missing values in the dataset
print()
print()
print()
print()
print("A quick check of categorical and numerical column NaNs\n")
print("You will see that we have 0 for all which is the best thing\n")
safe_columns = numerical_columns + categorical_columns
print(data[safe_columns].isnull().sum())
print()
print()
print()
print()
print("\nA quick check for physics columns\n")
print("We are allowed to have NaN's here:\n")
print(data[physics_calculable_cols].isnull().sum())
print()
print()
print()
print()
print("Verifying missing-value flags for physics calculable columns (True=missing, False=present):\n")
flag_columns = []
# flag_columns = [] ========>> We start with an empty list to store column names.
# Loop through each physics-calculable column:-
for i in physics_calculable_cols: #for col in physics_calculable_cols: ====== Go through each column that we plan to calculate using physics formulas.
    flag_columns.append(i + "_is_missing_values")
    # col + "_is_missing_values" =======>>
    # Add a descriptive suffix to indicate this column tracks missing values.
    #.append() ==== Add the new name to our flag_columns list.
print(data[flag_columns].head(10))
#The output is directly related to line 263 which is:-
# missingVals=data[i].isnull()
#Which will fetch all of the values which are null in the physics calculable col list.
#True = missing
#False = value exists
# Display the first 10 rows of the missing-value flag columns
print("\n\n\n\nCount of True values in missing-value flags (number of missing entries):\n")
for i in flag_columns:
    # .sum() counts the True values (missing entries)
    print(i, ":", data[i].sum())
    #data[i].sum() :-
    # Since True = 1 and False = 0,
    #Basically .sum() here will keep adding 1
    # .sum() gives the number of missing values in that column.
#Step 1: Handling Missing Values
#  We have identified columns with missing values using data.isnull().sum().
#  We filled safe-to-fill numerical columns with the median.
#  We filled safe-to-fill categorical columns with the mode.
#  For physics-calculable columns, we left them as NaN and created
#  True/False missing-value flag columns.
#  We have verified that only physics-calculable columns have missing values left.
# So Step 1 is basically complete — all safe-to-fill columns are cleaned,
# and calculable columns are tracked with flags.
###########################################################################
###########################################################################
###########################################################################
###########################################################################
#Step 2: Noise Detection and Handling
# Select one numerical feature.
# Add artificial noise (small random variations) to this feature.
# Apply a simple noise-handling technique, such as:
# Moving average
# Smoothing by aggregation
# Compare the feature before and after noise handling.
# 📌 Goal: Understand the difference between raw and cleaned signals.
detector="star_mass_solar"
#For step 2 we need a numerical feature so we take the mass of suns in all of the universes
#Think of it like: if a star’s mass is 1.0 solar mass,
# real-world measurements might be slightly 1.01 or 0.99 = this is noise.
#######################################
#ADDITION OF RANDOM ARTIFICIAL REPEATABLE NOISE:-
np.random.seed(42)
#This ensures the random numbers are always the same every time we run the code.
# Otherwise, every run gives slightly different numbers.
#Why 42? Because Hitchhiker’s Guide to the Galaxy joke
# (“Answer to the Ultimate Question of Life, the Universe, and Everything”).
########################################
#CREATION OF NOISE:-
noiseBalance=0.05*data[detector].median()
#We do a 5% of median of detector in dataFrame data which is star_mass_solar
actualNoise=np.random.normal(loc=0,scale=noiseBalance,size=len(data))
#np.random.normal(loc=0, scale=noise_strength, size=len(data)) will  generate random numbers such that:-
#loc=0 ----> is the center of all.
#Scale ---> Defines the spread in our case can be 0 to 1.5 or 0 to -1.5 keep in mind that only 68% of the times this rule will be followed.
#Size beasically defines the amount of this has to be followed for.
########################################
#ADD THE NOISE TO THE ORIGINAL COLUMN AND MAKE A NEW COLUMN FOR REPRESENTATION:-
data[detector+"_Noisy_New"]=data[detector]+actualNoise #adds the noise we just created to the original column
#detector + "_Nosy_New" -> creates a new column name "star_mass_solar_Noisy_New"
########################################
#SMOOTHING THE NOISY DATA:-
gaps=5
data[detector+"_SmoothValues"]=data[detector+"_Noisy_New"].rolling(window=gaps,min_periods=1).mean()
#The aforementioned technique is used from Dr Jalali's github repo: - Link :- https://prnt.sc/coYZpo_1-931
#.rolling() → creates a "moving window" of 5 rows
# Example: if rows are [1,2,3,4,5,6,7] and window=3:
# First window = [1] → average = 1
# Second window = [1,2] → average = 1.5
# Third window = [1,2,3] → average = 2
# Fourth window = [2,3,4] → average = 3 and so on
#More Example:-
#Data: [10, 12, 15, 13, 14, 20, 18]
# Window size: 3
# Step 1: [10] -> mean = 10
# Step 2: [10,12] -> mean = 11
# Step 3: [10,12,15] -> mean = 12.33
# Step 4: [12,15,13] -> mean = 13.33
# Step 5: [15,13,14] -> mean = 14
# Step 6: [13,14,20] -> mean = 15.67
# Step 7: [14,20,18] -> mean = 17.33
#After this, the noisy spikes will be reduced.
########################################
#COMPARISON B/W ORIGINAL AND NOISY:-
# Print the first 10 rows of the original, noisy, and smoothed columns
print("\nStep 2 – Noise Detection and Handling:")
print(f"Feature selected: {detector}\n")
print(data[[detector, detector + "_Noisy_New", detector + "_SmoothValues"]].head(10))
###########################################################################
###########################################################################
###########################################################################
###########################################################################
#STEP 3:-
#Outlier detection and handling:-
#Outliers are extreme values in your dataset, very far from the majority of data points.
# We want to detect them and decide whether to remove, cap, or leave them.
#Most stars are around 1–2 solar masses, but for an example 23.5 is very extreme and real
#In our case, the extreme values are real stars, so we will cap unrealistic numbers only, instead of removing them.
# Z-Score Formula:- https://prnt.sc/e9ZM-TVc0sP7
#Where:- https://prnt.sc/LISPdPljB8Hm
#Example:-https://prnt.sc/5ZVBVsajjB5l
#Takeaways:- https://prnt.sc/uEN_PNVkPR1q
############################################
#Calculating Z Scores:-
from scipy.stats import zscore
data['star_mass_solar_zscore']=zscore(data['star_mass_solar'].fillna(data['star_mass_solar'].median()))
#zscore(--------) = calculates (X-mean)/std for every value/row
#.fillna(data['star_mass_solar'].median()) ---> temporarily fills missing values with median to avoid errors
#####################################################
#Capping rules because extreme values in the data set are real
#A star can weigh 23 timees that of our sun.
#So, Maximum allowed = mean + 3*standar Deviation
# Minimum allowed = mean – 3*standard deviation
# Calculating Limits:-
#Stars are not evenly distributed.
# So instead of mean/std, we use percentiles/quantiles in python.
# Percentile means:
# Bottom 1%
# Top 1%
upper_limit = data['star_mass_solar'].quantile(0.999)
lower_limit = data['star_mass_solar'].quantile(0.001)

#Copying:-
data['star_mass_solar_capped'] = data['star_mass_solar']
#Applying the capping using pandas:-
data['star_mass_solar_capped'] = data['star_mass_solar_capped'].clip(lower=lower_limit,upper=upper_limit)
#Counting the changes:-
num_high = (data['star_mass_solar'] > upper_limit).sum()
num_low  = (data['star_mass_solar'] < lower_limit).sum()
print("High-end values capped:", num_high)
print("Low-end values capped:", num_low)
#############################
#Using the log:-
data['star_mass_solar_log'] = np.log10(data['star_mass_solar'])
#Did this right now for future use when I start doing ML.

print(
    data[['star_mass_solar',
          'star_mass_solar_zscore',
          'star_mass_solar_capped',
          'star_mass_solar_log']].head(10)
)
print(data["star_mass_solar_capped"].describe())
#######################################################################
#######################################################################
#######################################################################
#######################################################################
#Step 4:- Feature engineering
# Step 4.1: Calculate stellar density (approximate)
data['star_density_solar'] = data['star_mass_solar_capped'] / ((4/3) * np.pi * (data['star_radius_solar']**3))
# Explanation:
# star_mass_solar_capped -> mass of star in solar masses (after removing outliers)
# star_radius_solar -> radius of star in solar radii
# (4/3)*pi*R^3 -> volume of a sphere
# Dividing mass by volume gives density
###########################################################
# Step 4.2: Calculate planet mass / star mass ratio
data['planet_star_mass_ratio'] = data['planet_mass_earth'] / (data['star_mass_solar_capped'] * 332946)
# Explanation:
# planet_mass_earth -> planet mass in Earth masses
# star_mass_solar_capped -> star mass in solar masses
# 1 solar mass = 332,946 Earth masses
# So we convert star mass to Earth masses to compute ratio
###########################################################
# Step 4.3: Log transform of stellar luminosity
data['star_luminosity_solar_log'] = np.log10(data['star_luminosity_solar'].replace(0, np.nan))
# Explanation:
# replace(0, np.nan) avoids log(0) which is undefined
# Log10 scales down very large luminosities
###########################################################
# Step 4.3a: Log transform of planet radius
data['planet_radius_earth_log'] = np.log10(data['planet_radius_earth'].replace(0, np.nan) + 1)
# Explanation:
# replace 0 with NaN + 1 to avoid log(0) which is undefined
# Log10 scales down very large planet radii for ML
###########################################################
# Step 4.4: Flag planets as rocky or gaseous based on density
data['planet_type'] = np.where(data['planet_density'] > 5, 'rocky',
                               np.where(data['planet_density'] < 1, 'gas_giant', 'intermediate'))
# Explanation:
# np.where(condition, value_if_true, value_if_false)
# First condition: density > 5 -> 'rocky'
# Else if density < 1 -> 'gas_giant'
# Else -> 'intermediate'
###########################################################
# Step 4.5: Convert orbital period from days to years
data['orbital_period_years'] = data['orbital_period_days'] / 365.25
# Explanation:
# orbital_period_days -> days
# Divide by 365.25 to convert to years
###########################################################
# Step 4.6: Flag planets in habitable zone
# Safely replace negative or zero luminosity with NaN before sqrt
data['hz_inner'] = np.sqrt(data['star_luminosity_solar'].replace(0, np.nan) / 1.1)
data['hz_outer'] = np.sqrt(data['star_luminosity_solar'].replace(0, np.nan) / 0.53)
data['in_habitable_zone'] = np.where((data['semi_major_axis_AU'] >= data['hz_inner']) &
                                     (data['semi_major_axis_AU'] <= data['hz_outer']), True, False)
# Explanation:
# hz_inner and hz_outer -> boundaries in AU
# semi_major_axis_AU -> planet distance in AU
# If planet distance is within boundaries -> True
print(data[[
    'star_mass_solar_capped',
    'star_mass_solar_log',
    'star_luminosity_solar',
    'hz_inner',
    'hz_outer',
    'semi_major_axis_AU',
    'in_habitable_zone',
    'planet_radius_earth',
    'planet_radius_earth_log'
]].head(10))
#######################################################################
#######################################################################
#######################################################################
#######################################################################
#STEP 5:- Scaling & Encoding
# We now have a cleaned and feature-engineered dataset
# Some features are numeric (continuous)
###########################################################
# Step 5.1: Scaling numeric columns
from sklearn.preprocessing import StandardScaler

# Select numeric columns for scaling
numeric_features = [
    "right_ascension_deg",
    "declination_deg",
    "star_mass_solar_capped",
    "star_radius_solar",
    "star_temperature_K",
    "star_age_gyr",
    "star_distance_parsec",
    "planet_radius_earth",
    "planet_mass_earth",
    "planet_density",
    "planet_star_mass_ratio",     # Derived numerical feature
    "planet_radius_earth_log",
    "star_luminosity_solar_log",
    "star_surface_gravity_log",
    "orbital_period_days",
    "orbital_period_years",
    "semi_major_axis_AU",
    "stellar_irradiance_earth_units",
    "equilibrium_temperature_K",
    "impact_parameter",
    "distance_to_star_radius_ratio",
    "transit_depth",
    "transit_duration_hours",
    "star_density_solar"          # Derived numerical feature
]
# Explanation:
# These are all numeric values which we want to scale for ML
# Scaling ensures features are on similar scale, avoids domination by large values

# Initialize StandardScaler
scaler = StandardScaler()
# Explanation:
# StandardScaler subtracts mean and divides by std
# Resulting distribution has mean=0, std=1

# Apply scaling and overwrite columns
data[numeric_features] = scaler.fit_transform(data[numeric_features])
# Explanation:
# fit_transform calculates mean/std from data
# and scales each column automatically
# Replaces original numeric values with scaled ones
###########################################################
# Step 5.2: Verify final ML-ready dataset
print("Shape of final ML-ready dataset:", data.shape)
print(data.head(5))
# Explanation:
# Check number of rows and columns
# Display first 5 rows to confirm everything is numeric and scaled



