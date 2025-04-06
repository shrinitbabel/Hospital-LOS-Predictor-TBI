import os
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def build_preprocessor(numerical_features, categorical_features):
    return ColumnTransformer(transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

def load_and_engineer_features(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Convert to datetime
    date_cols = [
        'Date of Admission', 'Date of Hospital Discharge', 'PEG ORDERS PLACED',
        'TRACH ORDERS PLACED', 'PEG DATE', 'TRACH DATE',
        'Date Decannulation', 'Date of oral intake'
    ]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')

    # Filter and feature engineering
    df = df[df['Hospital LOS Days'] >= 0]
    df['PEG_Duration'] = (df['Date of oral intake'] - df['PEG DATE']).dt.days
    df['PEG_Days_Difference'] = (df['PEG DATE'] - df['PEG ORDERS PLACED']).dt.days
    df['PEG_ORDER_to_ADMISSION'] = (df['PEG ORDERS PLACED'] - df['Date of Admission']).dt.days
    df['Trach_ORDER_to_ADMISSION'] = (df['TRACH ORDERS PLACED'] - df['Date of Admission']).dt.days
    df['Trach_Days_Difference'] = (df['TRACH DATE'] - df['TRACH ORDERS PLACED']).dt.days
    df['Trach_Duration'] = (df['Date Decannulation'] - df['TRACH DATE']).dt.days

    # Classify LOS
    df['Hospital_LOS_Category'] = pd.cut(df['Hospital LOS Days'], bins=[-float('inf'), 24, float('inf')], labels=[0, 1])
    return df
