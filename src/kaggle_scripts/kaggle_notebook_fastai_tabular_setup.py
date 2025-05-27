# %% [code]
# Second cell: Data merging and TabularLearner setup for predicting the 3-year emission trend

# Define an aggregation function to concatenate multiple row values for each company
def aggregate_concat(x):
    return '|'.join(x.dropna().astype(str).unique())

# Aggregate target data by concatenating values for each non-key column
target_grp = target_df.groupby('ISSUERID').agg(aggregate_concat).reset_index()

# Aggregate Program 1 data similarly
program1_grp = program1_df.groupby('ISSUERID').agg(aggregate_concat).reset_index()

# Aggregate Program 2 data similarly
program2_grp = program2_df.groupby('ISSUERID').agg(aggregate_concat).reset_index()

# Merge the aggregated data with the company emissions dataframe
full_df = company_emission_df.copy()

# Merge targets (using suffix '_target' for overlapping columns)
full_df = full_df.merge(target_grp, on='ISSUERID', how='left', suffixes=("", "_target"))

# Merge Program 1 data (using suffix '_prog1' for overlapping columns)
full_df = full_df.merge(program1_grp, on='ISSUERID', how='left', suffixes=("", "_prog1"))

# Merge Program 2 data (using suffix '_prog2' for overlapping columns)
full_df = full_df.merge(program2_grp, on='ISSUERID', how='left', suffixes=("", "_prog2"))

print('Merged dataframe shape:', full_df.shape)

# %% [code]
# Prepare data for fastai Tabular model setup

# Define the target variable column using the correct column name from the dataset
# Instead of the made-up "3Y_emission_trend", we use "Emission_Trend_3Y" as documented in the company emissions data dictionary.
y_name = 'Emission_Trend_3Y'

# Prepare a list of feature columns by excluding the target and the key
features = full_df.columns.tolist()
if y_name in features:
    features.remove(y_name)
if 'ISSUERID' in features:
    features.remove('ISSUERID')

# Automatically determine categorical and continuous features
cat_names = full_df[features].select_dtypes(include=['object']).columns.tolist()
cont_names = [col for col in features if col not in cat_names]

print('Categorical columns:', cat_names)
print('Continuous columns:', cont_names)

# Define fastai preprocessing steps
procs = [Categorify, FillMissing, Normalize]

# Create TabularPandas object (using 80-20 split for training-validation)
n = len(full_df)
split_idx = [list(range(0, int(n*0.8))), list(range(int(n*0.8), n))]

to = TabularPandas(full_df, procs=procs, cat_names=cat_names, cont_names=cont_names, y_names=y_name, splits=split_idx)

# Create dataloaders
dls = to.dataloaders(bs=64)

# Build a baseline TabularLearner; you can adjust network layers and hyperparameters as needed
learn = tabular_learner(dls, layers=[200, 100], metrics=rmse)

# Display model summary
print(learn.model) 