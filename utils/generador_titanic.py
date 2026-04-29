import seaborn as sns
import numpy as np

df_titanic = sns.load_dataset('titanic')


#Esto lo hago para comprobar mi sistema de detección
df_titanic['ID_Pasajero'] = np.arange(1, len(df_titanic) + 1)
df_personal = df_titanic[['ID_Pasajero', 'sex', 'age', 'who', 'embark_town']].copy()
df_personal['Planeta'] = 'Tierra'
df_viaje = df_titanic[['ID_Pasajero', 'pclass', 'fare', 'survived']].copy()
df_viaje.rename(columns={'ID_Pasajero': 'id_ticket'}, inplace=True)

# Guardamos
df_personal.to_csv("titanic_pasajeros.csv", index=False, sep=';')
df_viaje.to_csv("titanic_viaje.csv", index=False, sep=';')