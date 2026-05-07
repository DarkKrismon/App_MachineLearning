import seaborn as sns
import numpy as np

def generar_datos():
    """Genera los datasets del Titanic en memoria sin escribir en disco."""
    df_titanic = sns.load_dataset('titanic')

    df_titanic['ID_Pasajero'] = np.arange(1, len(df_titanic) + 1)
    df_personal = df_titanic[['ID_Pasajero', 'sex', 'age', 'who', 'embark_town']].copy()
    df_personal['Planeta'] = 'Tierra'
    df_viaje = df_titanic[['ID_Pasajero', 'pclass', 'fare', 'survived']].copy()
    df_viaje.rename(columns={'ID_Pasajero': 'id_ticket'}, inplace=True)

    return df_personal, df_viaje