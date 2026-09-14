# Step 1: Dataset

import pandas as pd
import numpy as np

messy_data = {
    'patient_name': ['Dela Cruz', 'Santos', 'Reyes', 'bautista', 'GARCIA ', ' Mendoza', 'Torres',
                     'Ramos', 'Aquino', 'Fernandez', 'delaCruz', 'Santos', 'Villanueva', 'Domingo',
                     'Castro', None, 'Reyes', 'Navarro', None, 'Pascual', 'Torres',
                     'Aquino', 'Bautista', 'Cruz', 'Mercado'],

    'sex': ['M','F','Male','female','M','F','m',
            'F','M','Female','F','M','F','M',
            'F','M','F','M','F','M','F',
            'M','F','M','F'],

    'barangay': ['San Roque','Bagumbayan','san roque','Poblacion','Bagumbayan ','San Jose','Poblacion',
                 'Sto. Nino','San Roque','Bagumbayan','Poblacion','San Jose','Sto Nino','San Roque',
                 'Bagumbayan','Poblacion','San Roque','San Jose','Bagumbayan','Poblacion','Sto. Nino',
                 'San Roque','San Jose','Bagumbayan','Poblacion'],

    'visit_date': ['2026-01-05','01/06/2026','2026-1-7','2026-01-08','08-01-2026','2026-01-09','2026-01-09',
                   '2026/01/10','2026-01-11','01-12-2026','2026-01-13','2026-01-13','2026-01-14','2026-01-15',
                   None,'2026-01-16','2026-01-17','2026-01-18','2026-01-19','2026-01-20','2026-01-20',
                   '2026-01-21','2026-01-22','2026-01-23','2026-01-24'],

    'age': [34, 5, 150, 27, np.nan, 62, 41,
            8, np.nan, 29, 34, 5, 71, 45,
            19, -3, 55, np.nan, 22, 38, 41,
            np.nan, 60, 34, 47],

    'weight_kg': ['65kg','18.5','58 kg','70.2','80','45.0kg', '150',
                  '20', '55.5', '62', '65kg', '18.5','68', '75',
                  np.nan, '-40', '90kg', '52', '30', '77', '68',
                  '58', '95', '65kg', '61'],

    'height_cm': [165, 110, 170, np.nan, 175, 150, 168,
                  128, 160, 172, 165, 110, 155, 180,
                  140, 169, 158, np.nan, 132, 162, 168,
                  173, 5.9, 165, 159],

    'bp_systolic': ['120','95','210','130','118','160','122',
                    '100','300','125','120','95','140','135',
                    '108','119','150','127','98','131','122',
                    '128','145','120','133'],

    'temperature_c': [36.5, 37.0, 39.8, 36.7, 36.6, 38.2, 36.9,
                      36.8, 42.5, 36.5, 36.5, 37.0, 37.1, 36.4,
                      36.9, 36.6, 40.1, 36.7, 36.8, 36.9, 36.5,
                      37.2, 39.0, 36.5, 36.6],

    'diagnosis': ['Hypertension','Common Cold','hypertension','Diabetes','COMMON COLD','Flu','Hypertension ',
                  'Asthma','Diabetes Type 2','common cold','Hypertension','Common Cold','Migraine','Flu',
                  'UTI','Hypertension','diabetes','Asthma','Common Cold','Flu','Hypertension',
                  'Diabetes','Migrain','Hypertension','UTI'],

    'follow_up_needed': ['Yes','No','yes','NO','Y','N','Yes',
                         'No','yes','No','Yes','No','No','Yes',
                         'No','Yes','yes','No','N','Yes','Yes',
                         'No','Y','Yes','No'],

    'senior_citizen': [0,0,1,0,1,1,0,
                       0,0,0,0,0,1,0,
                       0,0,1,0,0,0,0,
                       1,1,0,0]
}

df_messy = pd.DataFrame(messy_data)

missing_name = df_messy[df_messy['patient_name'].isnull()]
duplicate = df_messy[df_messy.duplicated(subset=['patient_name'], keep='first')]
missing_age = df_messy[df_messy['age'].isnull()]

problem_rows = pd.concat([missing_name, duplicate, missing_age])
print(problem_rows)

#Step 2: Check mising value

print('Missing values in each column:')
print(df_messy.isnull().sum())

#Step 3: REmove rows with missing value

df_clean = df_messy.dropna(subset=['patient_name', 'age'])
print(df_clean)

#Step 4: Remove duplicate rows

df_clean = df_clean.drop_duplicates(subset=['patient_name'])
print(df_clean)

#step 5: validate ranges

df_clean['visit_date'] = pd.to_datetime(df_clean['visit_date'], format='mixed', errors='coerce')
valid_dates = df_clean['visit_date'].dropna()
print("visit date range:", valid_dates.min().strftime('%Y-%m-%d'), "-", valid_dates.max().strftime('%Y-%m-%d'))

# For my Question 8, Validate the sex values into male and female values only
df_clean['sex'] = df_clean['sex'].str.strip().str.upper()

sex_mapping = {
    'm':'Male',
    'M': 'Male',
    'MALE': 'Male',
    'f': 'Female',
    'F': 'Female',
    'FEMALE': 'Female'
}

df_clean['sex'] = df_clean['sex'].map(sex_mapping)
print(sex_mapping)

#Step 6: Final Cleanup

print("Clean Data")
print(df_clean)

# for my question 6 computation
total_valid_patients = len(df_clean)
avg_age = df_clean['age'].mean()
most_common_dx = df_clean['diagnosis'].mode()[0]
follow_up_pct = (df_clean['follow_up_needed'] == 'Yes').mean() * 100

print("=== QUESTION 6 RESULTS ===")
print(f"Total Valid Patients: {total_valid_patients}")
print(f"Average Age: {avg_age:.1f} years old")
print(f"Most Common Diagnosis: {most_common_dx}")
print(f"Percentage Needing Follow-up: {follow_up_pct:.1f}%")

#Step 7: Data visualization
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize = (15, 10))

plt.subplot(2, 2, 1)
sns.histplot(df_clean['visit_date'], kde=True)
plt.title("Visit Date Distribution")

plt.subplot(2, 2, 2)
sns.histplot(df_clean['barangay'], kde=True)
plt.title("Barangay list")

plt.tight_layout()
plt.show()
