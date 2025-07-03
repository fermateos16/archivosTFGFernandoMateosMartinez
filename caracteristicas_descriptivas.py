import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

def tabla_descriptiva(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Calcula estadísticas descriptivas"""
    df_clean = df.dropna(subset=[metric, 'Combinación'])
    df_clean = df_clean[np.isfinite(df_clean[metric])]
    if df_clean.empty:
        return pd.DataFrame()
    stats = df_clean.groupby('Combinación')[metric].agg([
        ('N', 'count'),
        ('mean', 'mean'),
        ('median', 'median'),
        ('std', 'std'),
        ('min', 'min'),
        ('max', 'max'),
        ('Q1', lambda x: x.quantile(0.25)),
        ('Q3', lambda x: x.quantile(0.75))
    ]).round(4)
    stats['CV'] = np.where(
        stats['mean'] != 0, 
        stats['std'] / stats['mean'], 
        0
    ).round(4)
    return stats.sort_values('mean', ascending=False)

def cargar_datos_porcentaje(p: str):
    """Cargar archivos (todos los nombres posibles)"""
    formatos = [
        f'resultados{p}%.csv',
        f'resultados{p}pct.csv',
        f'resultados{p}.csv'
    ]
    for fmt in formatos:
        try:
            return pd.read_csv(fmt)
        except FileNotFoundError:
            continue
    raise FileNotFoundError(f"No se encontraron archivos para {p}%")

def calcular_top3(tablas: dict, metrica: str):
    """Calcula TOP 3  combinaciones (entre el TOP5 de cada porcentaje)"""
    tops = {}
    for p, tabla in tablas.items():
        tops[p] = set(tabla.head(5).index)
    
    elite = set.intersection(*tops.values())
    
    resultados = []
    for combinacion in elite:
        medias, cvs, posiciones = [], [], []
        for p, tabla in tablas.items():
            if combinacion in tabla.index:
                medias.append(tabla.loc[combinacion, 'mean'])
                cvs.append(tabla.loc[combinacion, 'CV'])
                posicion = list(tabla.index).index(combinacion) + 1
                posiciones.append(posicion)
        
        if len(medias) == 3:
            resultados.append({
                'combinacion': combinacion,
                'media_global': np.mean(medias),
                'cv_promedio': np.mean(cvs),
                'variabilidad': np.std(medias, ddof=1),
                'posicion_promedio': np.mean(posiciones)
            })
    
    resultados_ordenados = sorted(
        resultados, 
        key=lambda x: x['media_global'], 
        reverse=True
    )[:3]
    
    print("\n" + "=" * 100)
    print(f"TOP 3 - {metrica.upper()}")
    print("=" * 100)
    
    for i, res in enumerate(resultados_ordenados, 1):
        print(f"{i}. {res['combinacion']}")
        print(f"   Media global : {res['media_global']:.4f}")
        print(f"   CV medio     : {res['cv_promedio']:.4f}")
        print(f"   Variabilidad : {res['variabilidad']:.4f}")
        print(f"   Consistencia : Posición promedio #{res['posicion_promedio']:.1f}")

def analisis_por_metrica(metrica: str):
    """Realiza análisis completo para la métrica específica"""
    tablas = {}
    porcentajes = ['1', '5', '10']
    
    for p in porcentajes:
        try:
            df = cargar_datos_porcentaje(p)
            tabla = tabla_descriptiva(df, metrica)
            if not tabla.empty:
                print(f"\n{'=' * 100}")
                print(f"ESTADÍSTICAS DESCRIPTIVAS - {metrica.upper()} ({p}% DATOS)")
                print(f"{'=' * 100}")
                print(tabla)
                tablas[p] = tabla
        except Exception as e:
            print(f"Error procesando {p}% para {metrica}: {str(e)}")
    
    if len(tablas) == len(porcentajes):
        calcular_top3(tablas, metrica)


def main():
    metricas = [
        "Precision", "Recall", "Specificity", 
        "Accuracy", "F1Score", "AUC-ROC"
    ]
    
    print("=" * 100)
    print("ANÁLISIS COMPARATIVO DE TODAS LAS MÉTRICAS")
    print("=" * 100)
    
    # Ejecutar análisis para cada métrica
    for metrica in metricas:
        print(f"\n{'#' * 100}")
        print(f"INICIANDO ANÁLISIS PARA: {metrica.upper()}")
        print(f"{'#' * 100}")
        analisis_por_metrica(metrica)

if __name__ == "__main__":
    main()
