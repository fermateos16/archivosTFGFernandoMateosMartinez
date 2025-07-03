import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro, kruskal, f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import warnings

warnings.filterwarnings('ignore')

# Función para calcular epsilon cuadrado (tamaño del efecto para Kruskal-Wallis)
def calcular_epsilon_kruskal(data, groups, metric_name):
    """Calcula el estadístico epsilon cuadrado para Kruskal-Wallis"""
    grupos_unicos = data[groups].unique()
    k = len(grupos_unicos)
    n = len(data)
    grupos_datos = [data[data[groups] == g][metric_name].values for g in grupos_unicos]
    
    H, _ = kruskal(*grupos_datos)
    epsilon_sq = (H - k + 1) / (n - k)
    return max(epsilon_sq, 0)  # Evitar valores negativos

def verificar_normalidad(data, groups, metric_name, group_name):
    """Verifica la normalidad de los residuos para decidir entre ANOVA y Kruskal-Wallis"""
    print(f"\n=== VERIFICACIÓN DE NORMALIDAD PARA {metric_name} - {group_name} ===")
    
    # Calcular residuos
    group_means = data.groupby(groups)[metric_name].mean()
    residuos = data[metric_name] - data[groups].map(group_means)
    
    # Test de Shapiro-Wilk
    stat, p_value = shapiro(residuos)
    
    print(f"Shapiro-Wilk Test:")
    print(f"Estadístico: {stat:.4f}")
    print(f"p-value: {p_value:.4f}")
    
    alpha = 0.05
    if p_value > alpha:
        print(f"Los residuos son normales (p > {alpha})")
        print("→ Se puede usar ANOVA")
        print("""
        Si los residuos son normales, utilizaremos ANOVA
        """)
        return True
    else:
        print(f"Los residuos NO son normales (p ≤ {alpha})")
        print("→ Se debe usar Kruskal-Wallis")
        print("""
        Si los residuos NO son normales, debemos utilizar Kruskal-Wallis
        """)
        return False

def realizar_anova(data, groups, metric_name, group_name):
    """Realiza ANOVA y test post-hoc de Tukey"""
    print(f"\n=== ANOVA PARA {metric_name} - {group_name} ===")
    
    # Preparar datos para ANOVA
    grupos_unicos = data[groups].unique()
    grupos_datos = [data[data[groups] == grupo][metric_name].values for grupo in grupos_unicos]
    
    # ANOVA
    f_stat, p_value = f_oneway(*grupos_datos)
    
    # Calcular tamaño del efecto
    eta_sq = calcular_epsilon_kruskal(data, groups, metric_name)
    
    print(f"F-estadístico: {f_stat:.4f}")
    print(f"p-value: {p_value:.4f}")
    print(f"Tamaño del efecto (ε²): {eta_sq:.4f}")
    
    if p_value < 0.05:
        print("RESULTADO: Existen diferencias significativas entre grupos")
        
        # Test post-hoc de Tukey
        print("\n--- Test Post-hoc de Tukey ---")
        tukey_result = pairwise_tukeyhsd(data[metric_name], data[groups])
        print(tukey_result)
        
        return True, tukey_result, eta_sq
    else:
        print("RESULTADO: No hay diferencias significativas entre grupos")
 
        return False, None, eta_sq

def realizar_kruskal_wallis(data, groups, metric_name, group_name):
    """Realiza test de Kruskal-Wallis y comparaciones post-hoc"""
    print(f"\n=== KRUSKAL-WALLIS PARA {metric_name} - {group_name} ===")
    
    # Preparar datos
    grupos_unicos = data[groups].unique()
    grupos_datos = [data[data[groups] == grupo][metric_name].values for grupo in grupos_unicos]
    
    # Kruskal-Wallis
    h_stat, p_value = kruskal(*grupos_datos)
    
    # Calcular tamaño del efecto
    epsilon_sq = calcular_epsilon_kruskal(data, groups, metric_name)
    
    print(f"H-estadístico: {h_stat:.4f}")
    print(f"p-value: {p_value:.4f}")
    print(f"Tamaño del efecto (ε²): {epsilon_sq:.4f}")
    
    if p_value < 0.05:
        print("RESULTADO: Existen diferencias significativas entre grupos")
        
        # Comparaciones múltiples de Mann-Whitney con corrección de Bonferroni
        print("\n--- Comparaciones múltiples (Mann-Whitney con corrección Bonferroni) ---")
        n_comparisons = len(grupos_unicos) * (len(grupos_unicos) - 1) // 2
        alpha_corregido = 0.05 / n_comparisons
        
        comparaciones_significativas = []
        
        for i, grupo1 in enumerate(grupos_unicos):
            for j, grupo2 in enumerate(grupos_unicos[i+1:], i+1):
                data1 = data[data[groups] == grupo1][metric_name].values
                data2 = data[data[groups] == grupo2][metric_name].values
                
                u_stat, p_val = stats.mannwhitneyu(data1, data2, alternative='two-sided')
                
                significativo = "SÍ" if p_val < alpha_corregido else "No"
                if p_val < alpha_corregido:
                    comparaciones_significativas.append((grupo1, grupo2, p_val))
                
                print(f"{grupo1} vs {grupo2}: p-value = {p_val:.4f} (α = {alpha_corregido:.4f}) - {significativo}")
        
        return True, comparaciones_significativas, epsilon_sq
    else:
        print("RESULTADO: No hay diferencias significativas entre grupos")
        
        return False, None, epsilon_sq

def generar_visualizaciones(data, metric_name):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Boxplot por Combinación
    sns.boxplot(data=data, x="Combinación", y=metric_name, ax=axes[0])
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=90)
    axes[0].set_title(f"{metric_name} por Combinación de Características")
    
    # Boxplot por Arquitectura
    sns.boxplot(data=data, x="n_arc", y=metric_name, ax=axes[1])
    axes[1].set_title(f"{metric_name} por Arquitectura de Red")
    
    plt.tight_layout()
    plt.show()

def calcular_indice_dominancia(efecto_comb, efecto_arc):
    """Calcula el índice de dominancia entre combinación y arquitectura"""
    if efecto_arc == 0:
        return float('inf')  # EFECTO MUY GRANDE
    return efecto_comb / efecto_arc

def analizar_metrica(data, metric_name):
    """Función principal que analiza una métrica específica"""
    print(f"\n{'='*80}")
    print(f"ANÁLISIS ESTADÍSTICO PARA: {metric_name}")
    print(f"{'='*80}")
    
    if metric_name not in data.columns:
        print(f"ERROR: La métrica '{metric_name}' no existe en el dataset")
        return None
    
    generar_visualizaciones(data, metric_name)
   
    
    # Análisis por Combinación
    print(f"\n{'-'*60}")
    print(f"ANÁLISIS POR COMBINACIÓN")
    print(f"{'-'*60}")
    
    es_normal_comb = verificar_normalidad(data, "Combinación", metric_name, "Combinación")
    
    if es_normal_comb:
        hay_diff_comb, result_comb, efecto_comb = realizar_anova(data, "Combinación", metric_name, "Combinación")
    else:
        hay_diff_comb, result_comb, efecto_comb = realizar_kruskal_wallis(data, "Combinación", metric_name, "Combinación")
    
    # Análisis por Arquitectura
    print(f"\n{'-'*60}")
    print(f"ANÁLISIS POR ARQUITECTURA")
    print(f"{'-'*60}")
    
    es_normal_arc = verificar_normalidad(data, "n_arc", metric_name, "Arquitectura")
    
    if es_normal_arc:
        hay_diff_arc, result_arc, efecto_arc = realizar_anova(data, "n_arc", metric_name, "Arquitectura")
    else:
        hay_diff_arc, result_arc, efecto_arc = realizar_kruskal_wallis(data, "n_arc", metric_name, "Arquitectura")
    
    # Cálculo del índice de dominancia
    indice_dominancia = calcular_indice_dominancia(efecto_comb, efecto_arc)
    
    # Resumen de resultados
    print(f"\n{'-'*60}")
    print(f"RESUMEN PARA {metric_name}")
    print(f"{'-'*60}")
    print(f"Por Combinación: {'Diferencias significativas' if hay_diff_comb else 'Sin diferencias significativas'}")
    print(f"Tamaño efecto combinación: {efecto_comb:.4f}")
    print(f"\nPor Arquitectura: {'Diferencias significativas' if hay_diff_arc else 'Sin diferencias significativas'}")
    print(f"Tamaño efecto arquitectura: {efecto_arc:.4f}")
    
    # Manejo especial para valores infinitos en el resumen
    if np.isinf(indice_dominancia):
        print(f"\nÍndice de Dominancia (Comb/Arc): Inf")
    else:
        print(f"\nÍndice de Dominancia (Comb/Arc): {indice_dominancia:.2f}")
    
    # Interpretación del índice
    print("\nINTERPRETACIÓN DEL ÍNDICE DE DOMINANCIA:")
    if np.isinf(indice_dominancia):
        print("→ La combinación tiene DOMINANCIA ABSOLUTA (la arquitectura no muestra efecto detectable)")
    elif indice_dominancia > 3:
        print("→ La combinación tiene un efecto MUY SUPERIOR")
    elif indice_dominancia > 2:
        print("→ La combinación tiene un efecto SIGNIFICATIVAMENTE MAYOR")
    elif indice_dominancia > 1.5:
        print("→ La combinación tiene un efecto MODERADAMENTE MAYOR")
    elif indice_dominancia > 1:
        print("→ La combinación tiene un efecto LIGERAMENTE MAYOR")
    elif indice_dominancia == 1:
        print("→ Ambos factores tienen efectos EQUIVALENTES")
    else:
        print("→ La arquitectura tiene mayor impacto que la combinación")
    
    return {
        'metric': metric_name,
        'combinacion': {
            'normal': es_normal_comb,
            'significativo': hay_diff_comb,
            'efecto': efecto_comb,
            'resultado': result_comb
        },
        'arquitectura': {
            'normal': es_normal_arc,
            'significativo': hay_diff_arc,
            'efecto': efecto_arc,
            'resultado': result_arc
        },
        'indice_dominancia': indice_dominancia
    }

def main():
    """Función principal del análisis"""
    print("Cargando datos...")
    try:
        df = pd.read_csv("resultados10%.csv")
        print(f"Datos cargados exitosamente. Shape: {df.shape}")
        print(f"Columnas disponibles: {list(df.columns)}")
    except FileNotFoundError:
        print("ERROR: No se encontró el archivo 'resultados10%.csv'")
        return
    
    metricas = ["Precision", "Recall", "Specificity", "Accuracy", "F1Score", "AUC-ROC"]
    metricas_disponibles = [m for m in metricas if m in df.columns]
    
    if not metricas_disponibles:
        print("No se encontraron métricas válidas en el dataset")
        return
    
    resultados_totales = []
    
    for metrica in metricas_disponibles:
        resultado = analizar_metrica(df, metrica)
        if resultado:
            resultados_totales.append(resultado)
    
    
    # Resumen final comparativo
    print(f"\n{'='*80}")
    print(f"RESUMEN FINAL COMPARATIVO")
    print(f"{'='*80}")
    
    print("\n| Métrica     | Efecto Comb | Efecto Arc | Índice Dominancia | Interpretación          |")
    print("|-------------|-------------|------------|-------------------|-------------------------|")
    
    for res in resultados_totales:
        metrica = res['metric']
        efecto_comb = res['combinacion']['efecto']
        efecto_arc = res['arquitectura']['efecto']
        indice = res['indice_dominancia']
        
        # Determinar interpretación
        if np.isinf(indice):
            interpretacion = "DOMINANCIA ABSOLUTA"
        elif indice > 3:
            interpretacion = "Comb. MUY superior"
        elif indice > 2:
            interpretacion = "Comb. significativamente mayor"
        elif indice > 1.5:
            interpretacion = "Comb. moderadamente mayor"
        elif indice > 1:
            interpretacion = "Comb. ligeramente mayor"
        elif indice == 1:
            interpretacion = "Efectos equivalentes"
        else:
            interpretacion = "Arquitectura más influyente"
        
        # Formatear índice
        indice_str = "" if np.isinf(indice) else f"{indice:.2f}"
        
        print(f"| {metrica:11} | {efecto_comb:.4f}    | {efecto_arc:.4f}   | {indice_str:>17} | {interpretacion:23} |")

if __name__ == "__main__":
    main()
