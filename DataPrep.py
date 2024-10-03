import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv('C:\Users\Shreyash\Documents\CTrain') 

# Selecting the columns for the input (features) and the target (Energy_Output)
features = dataset[['Step_Frequency (steps/sec)', 'Foot_Pressure (N)', 'Stride_Length (m)', 'User_Weight (kg)', 'Displacement_Force (N)']]
target = dataset['Energy_Output (mA)']

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
