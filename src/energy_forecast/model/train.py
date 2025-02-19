import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
import polars as pl

from src.energy_forecast.dataset import Dataset

attributes = ["diff", "diff_t-1", 'hum_avg', 'hum_min', 'hum_max', 'tavg', 'tmin', 'tmax', 'prcp', 'snow', 'wdir',
              'wspd', 'wpgt',
              'pres', 'tsun', "holiday"]
attributes_ha = attributes + ["heated_area", "anzahlwhg"]

# Load the data
ds = Dataset()
ds.load_feat_data()
df = ds.df

# train test split
train_per = 0.8
gss = GroupShuffleSplit(n_splits=1, test_size=1 - train_per, random_state=42)
df = df.with_row_index()
for train_idx, test_idx in gss.split(df, groups=df["id"]):
    train_data = df.filter(pl.col("index").is_in(train_idx))
    test_data = df.filter(pl.col("index").is_in(test_idx))

# transform to pandas DataFrame input
X_train = train_data.to_pandas()[list(set(attributes) - {"diff"})]
y_train = train_data.to_pandas()["diff"]

X_test = test_data.to_pandas()[list(set(attributes) - {"diff"})]
y_test = test_data.to_pandas()["diff"]

# Define the model
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(len(attributes) - 1,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)  # perform regression
])

# Compile the model
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mae'])

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"Test MAE: {test_mae:.4f}")
