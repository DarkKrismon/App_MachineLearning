import pandas as pd
import numpy as np


np.random.seed(42)

N_CLIENTES = 5000
N_COMPRAS = 4800

# TABLA 1:
# 1. IDs
ids_clientes = np.arange(1, N_CLIENTES + 1)

# 2. Nombres aleatorios
nombres_base = ['Ana', 'Luis', 'Carlos', 'Marta', 'Pedro', 'Sofia', 'Juan', 'Elena', 'Diego', 'Lucia']
nombres = np.random.choice(nombres_base, size=N_CLIENTES)

# 3. Edad entre 18 y 75 años con 5% de nulos
edades = np.random.randint(18, 75, size=N_CLIENTES).astype(float)
idx_nulos_edad = np.random.choice(N_CLIENTES, size=int(N_CLIENTES * 0.05), replace=False)
edades[idx_nulos_edad] = np.nan

# 4. País todos son de España
paises = ['España'] * N_CLIENTES

df_clientes = pd.DataFrame({
    'ID_Cliente': ids_clientes,
    'Nombre': nombres,
    'Edad': edades,
    'Pais': paises
})

# 5. 100 duplicados al final
duplicados = df_clientes.sample(100, random_state=42)
df_clientes = pd.concat([df_clientes, duplicados], ignore_index=True)



# TABLA 2:
# 1. id_usuario
ids_compras = np.arange(500, 500 + N_COMPRAS)

# 2. Total Gasto entre 20€ y 1500€ con 10% de nulos
gastos = np.round(np.random.uniform(20.0, 1500.0, size=N_COMPRAS), 2)
idx_nulos_gasto = np.random.choice(N_COMPRAS, size=int(N_COMPRAS * 0.10), replace=False)
gastos[idx_nulos_gasto] = np.nan

# 3. Variable objetivo
fugas = np.random.choice(['Si', 'No'], size=N_COMPRAS, p=[0.25, 0.75])

df_compras = pd.DataFrame({
    'id_usuario': ids_compras,
    'Total_Gasto': gastos,
    'Fuga': fugas
})

# Guardamos
df_clientes.to_csv("clientes_caos_masivo.csv", index=False, sep=';')
df_compras.to_csv("compras_caos_masivo.csv", index=False, sep=';')

print(f"¡Listo! Archivos generados:")
print(f"- clientes_caos_masivo.csv: {len(df_clientes)} filas.")
print(f"- compras_caos_masivo.csv: {len(df_compras)} filas.")