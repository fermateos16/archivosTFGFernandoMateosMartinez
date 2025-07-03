# Imports
import os
import random
from datetime import datetime
from itertools import cycle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots
import matplotlib.backends.backend_pdf
import seaborn as sns
from pandas.plotting import parallel_coordinates
import librosa
from librosa import effects
import parselmouth
import parselmouth.praat
import joblib

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    auc,
    balanced_accuracy_score
)

# TensorFlow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D,
    MaxPooling1D,
    Flatten,
    Dense,
    Dropout,
    Reshape,
    Activation,
    Input
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


# ================= CONFIGURACIÓN =================
RUTA_RESULTADOS = "resultados_comparativos_elegir.csv"
RUTA_PDF = "reporte_completo_elegir.pdf" 
RUTA_MODELOS = "modelos/"
PORCENTAJE_AUDIOS = 5  # Porcentaje de audios

# Crear directorio si no existe
os.makedirs(RUTA_MODELOS, exist_ok=True)

# ================= COMBINACIONES DE PARÁMETROS =================
COMBINACIONES = {
    # Parámetros individuales
    "SOLO_MFCC": ['mfcc'],
    "SOLO_RMS": ['rms'],
    "SOLO_HNR": ['hnr'],
    "SOLO_FORMANTES": ['formantes'],
    "SOLO_SPECTRAL_CONTRAST": ['spectral_contrast'],
    "SOLO_ZERO_CROSSING": ['zero_crossing_rate'],
    
    # Combinaciones básicas
    "MFCCs+RMS": ['mfcc', 'rms'],
    "MFCCs+HNR": ['mfcc', 'hnr'],
    
    # Combinaciones más avanzadas
    "BASICO": ['mfcc', 'rms', 'hnr'],
    "CALIDAD_VOCAL": ['mfcc', 'formantes', 'hnr'],
    "ESPECTRAL_AVANZADO": ['mfcc', 'spectral_contrast', 'spectral_bandwidth'],
    
    # Todas las características
    "ANALISIS_COMPLETO": [
        'mfcc', 'rms', 'formantes', 
        'spectral_contrast', 'hnr',
        'delta_mfcc','zero_crossing_rate'
    ]
}

# ================= CONFIGURACIONES PREDEFINIDAS =================

CONFIGURACIONES_MODELOS = [
    # Configuraciones "rápidas" (1-5)
    {
        'conv_filters': 16,
        'dense_units': 16,
        'dropout_rate': 0.1,
        'learning_rate': 0.01,
        'max_epochs': 10,
        'early_stopping_patience': 2
    },
    {
        'conv_filters': 32,
        'dense_units': 32,
        'dropout_rate': 0.2,
        'learning_rate': 0.01,
        'max_epochs': 15,
        'early_stopping_patience': 3
    }
    ,
    {
        'conv_filters': 64,
        'dense_units': 32,
        'dropout_rate': 0.2,
        'learning_rate': 0.005,
        'max_epochs': 20,
        'early_stopping_patience': 4
    },
    {
        'conv_filters': 32,
        'dense_units': 64,
        'dropout_rate': 0.3,
        'learning_rate': 0.001,
        'max_epochs': 15,
        'early_stopping_patience': 3
    },
    {
        'conv_filters': 16,
        'dense_units': 128,
        'dropout_rate': 0.4,
        'learning_rate': 0.0005,
        'max_epochs': 25,
        'early_stopping_patience': 5
    },
    
    # Configuraciones "balanceadas" (6-10)
    {
        'conv_filters': 64,
        'dense_units': 64,
        'dropout_rate': 0.3,
        'learning_rate': 0.001,
        'max_epochs': 20,
        'early_stopping_patience': 5
    },
    {
        'conv_filters': 128,
        'dense_units': 64,
        'dropout_rate': 0.4,
        'learning_rate': 0.0005,
        'max_epochs': 25,
        'early_stopping_patience': 5
    },
    {
        'conv_filters': 64,
        'dense_units': 128,
        'dropout_rate': 0.4,
        'learning_rate': 0.0005,
        'max_epochs': 25,
        'early_stopping_patience': 5
    },
    {
        'conv_filters': 256,
        'dense_units': 64,
        'dropout_rate': 0.5,
        'learning_rate': 0.0001,
        'max_epochs': 30,
        'early_stopping_patience': 7
    },
    {
        'conv_filters': 64,
        'dense_units': 256,
        'dropout_rate': 0.5,
        'learning_rate': 0.0001,
        'max_epochs': 30,
        'early_stopping_patience': 7
    },
    
    # Configuraciones profundas (11-15)
    {
        'conv_filters': 128,
        'dense_units': 128,
        'dropout_rate': 0.5,
        'learning_rate': 0.0001,
        'max_epochs': 30,
        'early_stopping_patience': 7
    },
    {
        'conv_filters': 256,
        'dense_units': 256,
        'dropout_rate': 0.5,
        'learning_rate': 0.0001,
        'max_epochs': 40,
        'early_stopping_patience': 10
    },
    {
        'conv_filters': 512,
        'dense_units': 256,
        'dropout_rate': 0.6,
        'learning_rate': 0.00001,
        'max_epochs': 50,
        'early_stopping_patience': 15
    },
    {
        'conv_filters': 256,
        'dense_units': 512,
        'dropout_rate': 0.6,
        'learning_rate': 0.00001,
        'max_epochs': 50,
        'early_stopping_patience': 15
    },
    {
        'conv_filters': 512,
        'dense_units': 512,
        'dropout_rate': 0.6,
        'learning_rate': 0.00001,
        'max_epochs': 50,
        'early_stopping_patience': 15
    },
    
    # Configuraciones especializadas (16-20)
    {
        'conv_filters': 32,
        'dense_units': 512,
        'dropout_rate': 0.7,
        'learning_rate': 0.0001,
        'max_epochs': 40,
        'early_stopping_patience': 10
    },
    {
        'conv_filters': 512,
        'dense_units': 32,
        'dropout_rate': 0.7,
        'learning_rate': 0.0001,
        'max_epochs': 40,
        'early_stopping_patience': 10
    },
    {
        'conv_filters': 1024,
        'dense_units': 64,
        'dropout_rate': 0.8,
        'learning_rate': 0.00001,
        'max_epochs': 60,
        'early_stopping_patience': 20
    },
    {
        'conv_filters': 64,
        'dense_units': 1024,
        'dropout_rate': 0.8,
        'learning_rate': 0.00001,
        'max_epochs': 60,
        'early_stopping_patience': 20
    },
    {
        'conv_filters': 2048,
        'dense_units': 128,
        'dropout_rate': 0.9,
        'learning_rate': 0.000001,
        'max_epochs': 100,
        'early_stopping_patience': 30
    }
]


# ================= EXTRACCIÓN DE CARACTERÍSTICAS =================
def extraer_hnr(senal_audio):
    """Calcula la relación armónico-ruido (Harmonic-to-Noise Ratio)"""
    try:
        harmonic = effects.harmonic(y=senal_audio)
        return np.mean(harmonic)
    except:
        return 0.0

def cargar_audios_desde_carpeta(carpeta, etiqueta, parametros, porcentaje=PORCENTAJE_AUDIOS):
    start_time = datetime.now()  # <-- Añade esta línea
    
    caracteristicas = []
    etiquetas = []
    archivos = [f for f in os.listdir(carpeta) if f.endswith(('.wav', '.mp3'))]
    
    # Seleccionar el porcentaje de los audios
    n_audios = int(len(archivos) * porcentaje / 100)
    archivos = random.sample(archivos, n_audios)
    
    for archivo in archivos:
        ruta_audio = os.path.join(carpeta, archivo)
        try:
            senal_audio, sr = librosa.load(ruta_audio, sr=16000)
            
            if len(senal_audio) == 0:
                continue
                
            features = []
            
            # ---- MFCCs y derivadas ----
            if 'mfcc' in parametros:
                mfccs = librosa.feature.mfcc(y=senal_audio, sr=sr, n_mfcc=40)
                features.extend(np.mean(mfccs, axis=1))
            
            if 'delta_mfcc' in parametros:
                delta = librosa.feature.delta(mfccs)
                features.extend(np.mean(delta, axis=1))
            
            # ---- Características de calidad vocal ----
            if 'hnr' in parametros:
                features.append(extraer_hnr(senal_audio))

            # ---- Características espectrales ----
            if 'rms' in parametros:
                features.append(np.mean(librosa.feature.rms(y=senal_audio)))
            
            if 'spectral_contrast' in parametros:
                contrast = librosa.feature.spectral_contrast(y=senal_audio, sr=sr)
                features.extend(np.mean(contrast, axis=1))
            
            if 'spectral_bandwidth' in parametros:
                features.append(np.mean(librosa.feature.spectral_bandwidth(y=senal_audio, sr=sr)))
            
            if 'zero_crossing_rate' in parametros:
                features.append(np.mean(librosa.feature.zero_crossing_rate(y=senal_audio)))
            
            # ---- Formantes ----
            if 'formantes' in parametros and len(senal_audio) < 100000:
                sound = parselmouth.Sound(senal_audio, sr)
                formants = sound.to_formant_burg(max_number_of_formants=3)
                for i in range(1, 4):
                    f = formants.get_value_at_time(i, 0.5, "HERTZ")
                    features.append(0 if np.isnan(f) or np.isinf(f) else f)
            
            caracteristicas.append(features)
            etiquetas.append(etiqueta)
            
        except Exception as e:
            print(f"Error en {archivo}: {str(e)}")
            continue
    
    # Calcula el tiempo total de procesamiento
    processing_time = (datetime.now() - start_time).total_seconds() 
    
    # Ajustar longitud para igualar dimensiones
    max_len = max(len(x) for x in caracteristicas)
    caracteristicas = np.array([
        np.pad(x, (0, max_len - len(x)), 
        mode='constant', constant_values=0) 
        for x in caracteristicas
    ])
    
    return np.array(caracteristicas), np.array(etiquetas), processing_time

'''

Función para que el usuario pusiera sus propios parametros (quitado)

def obtener_parametros_modelo_interactivo():
    
    print("\n Configuración del Modelo CNN")
    print("(Dejar en blanco para usar los valores por defecto\n")
    
    parametros = {
        'conv_filters': 32,
        'dense_units': 32,
        'dropout_rate': 0.2,
        'learning_rate': 0.01,
        'max_epochs': 20,
        'early_stopping_patience': 3
    }
    
    try:
        # Filtros Conv1D
        print("- Filtros Conv1D: Número de "detectores" de características (valores típicos 16-128)")
        print("  Más filtros = más capacidad pero mayor riesgo de sobreajuste")
        parametros['conv_filters'] = int(input(f"  Número de filtros en capa Conv1D [{parametros['conv_filters']}]: ") or parametros['conv_filters'])
        
        # Unidades Dense
        print("\n- Unidades Dense: Capacidad de la capa completamente conectada (valores típicos 16-256)")
        print("  Más unidades = más capacidad de aprendizaje pero más lento")
        parametros['dense_units'] = int(input(f"  Unidades en capa Dense [{parametros['dense_units']}]: ") or parametros['dense_units'])
        
        # Dropout
        print("\n- Dropout: Regularización para evitar sobreajuste (rango 0-1)")
        parametros['dropout_rate'] = float(input(f"  Tasa de Dropout [{parametros['dropout_rate']}]: ") or parametros['dropout_rate'])
        
        # Learning Rate
        print("\n- Learning Rate: Tamaño de paso en optimización (valores típicos 0.001-0.1)")
        print("  Valores altos aprenden más rápido pero pueden no converger")
        print("  Valores bajos son estables pero más lentos")
        parametros['learning_rate'] = float(input(f"  Learning Rate [{parametros['learning_rate']}]: ") or parametros['learning_rate'])
        
        # Épocas
        print("\n- Épocas: Iteraciones completas sobre los datos (típico 10-50)")
        print("  Más épocas = más entrenamiento pero riesgo de sobreajuste")
        parametros['max_epochs'] = int(input(f"  Máximo de épocas [{parametros['max_epochs']}]: ") or parametros['max_epochs'])
        
        # Early Stopping
        print("\n- Early Stopping: Épocas sin mejora para detener entrenamiento (típico 3-10)")
        print("  Detiene el entrenamiento si no hay mejora ")
        parametros['early_stopping_patience'] = int(input(f"  Paciencia para Early Stopping [{parametros['early_stopping_patience']}]: ") or parametros['early_stopping_patience'])
        
    except ValueError as e:
        print(f"\n Ha habido un error: {str(e)}. Se usarán valores por defecto.")
    
    print("\n✅ Configuración final del modelo:")
    for k, v in parametros.items():
        print(f"- {k.replace('_', ' ').title()}: {v}")
    
    return parametros
    
    '''
# ================= MODELO CNN =================
def construir_modelo_cnn(forma_entrada, config):
    """Construye el modelo CNN con la configuración especificada"""
    # Creamos un modelo secuencial, que añade capas una tras otra
    modelo = Sequential()

    # Primera capa: la de entrada, que espera vectores del tamaño especificado
    modelo.add(Input(shape=forma_entrada))

    # Si los datos de entrada tienen más de 5 elementos (por ejemplo, 40 características),
    # se aplican capas convolucionales para aprender patrones locales
    if forma_entrada[0] > 5:
        # Cambiamos la forma para que sea compatible con la capa Conv1D (agregamos una dimensión extra)
        modelo.add(Reshape((forma_entrada[0], 1)))

        # Capa convolucional: detecta patrones en secuencias (como partes de una señal de audio)
        modelo.add(Conv1D(config['conv_filters'], kernel_size=3, padding='same', activation='relu'))

        # Capa de reducción de dimensión: toma el valor más importante de cada grupo de 2
        modelo.add(MaxPooling1D(pool_size=2))

        # Capa Dropout: desactiva aleatoriamente algunas neuronas para evitar que el modelo "memorice" los datos
        modelo.add(Dropout(config['dropout_rate']))

    # Aplanamos los datos para pasarlos a una capa densa (clásica)
    modelo.add(Flatten())

    # Capa oculta densa (neuronal): aprende combinaciones más complejas
    modelo.add(Dense(config['dense_units'], activation='relu'))

    # Capa de salida: una única neurona con activación sigmoide (valor entre 0 y 1)
    modelo.add(Dense(1, activation='sigmoid'))

    # Compilamos el modelo: especificamos el optimizador, la función de pérdida y la métrica
    modelo.compile(
        optimizer=Adam(learning_rate=config['learning_rate']),  # optimizador Adam con la tasa de aprendizaje elegida
        loss='binary_crossentropy',  # función de pérdida para clasificación binaria
        metrics=['accuracy'] 
    )

    #modelo creado
    return modelo

# ================= EVALUACIÓN =================
def evaluar_combinacion(parametros, nombre_combinacion, rutas, porcentaje, config_modelo):
    print(f"\n=== Evaluando combinación: {nombre_combinacion} ===")

    # Cargar datos reales y falsos midiendo el tiempo de procesamiento
    start_processing_time = datetime.now()
    X_real, y_real, real_processing_time = cargar_audios_desde_carpeta(rutas['train_real'], 1, parametros, porcentaje)
    X_falso, y_falso, fake_processing_time = cargar_audios_desde_carpeta(rutas['train_fake'], 0, parametros, porcentaje)
    total_processing_time = real_processing_time + fake_processing_time

    # Unir los dos conjuntos de datos
    X = np.vstack((X_real, X_falso))
    y = np.concatenate((y_real, y_falso))

    if len(X) < 10:
        print(f"⚠️ Saltando {nombre_combinacion} - muy pocos datos ({len(X)} muestras)")
        return None, None, None, None

    # Normalizamos los datos
    escalador = StandardScaler()
    X = escalador.fit_transform(X)

    # Dividimos los datos
    X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    # Construimos el modelo
    modelo = construir_modelo_cnn((X_entrenamiento.shape[1],), config_modelo)

    # Se definela parada temprana
    parada_temprana = EarlyStopping(
        monitor='val_loss',
        patience=config_modelo['early_stopping_patience'],
        restore_best_weights=True
    )

    # Medimos el tiempo de entrenamiento
    start_training_time = datetime.now()
    historial = modelo.fit(
        X_entrenamiento, y_entrenamiento,
        epochs=config_modelo['max_epochs'],
        batch_size=32,
        validation_split=0.2,
        callbacks=[parada_temprana],
        verbose=0
    )
    training_time = (datetime.now() - start_training_time).total_seconds()

    # Realizamos predicciones
    y_pred_proba = modelo.predict(X_prueba).flatten()
    y_prediccion = (y_pred_proba > 0.5).astype(int)

    # Calculamos métricas
    tn, fp, fn, tp = confusion_matrix(y_prueba, y_prediccion).ravel()
    precision, recall, _ = precision_recall_curve(y_prueba, y_pred_proba)
    fpr, tpr, _ = roc_curve(y_prueba, y_pred_proba)
    auc_pr = auc(recall, precision)
    auc_roc = roc_auc_score(y_prueba, y_pred_proba)

    # Recolectamos todas las métricas
    metricas = {
        'Combinación': nombre_combinacion,
        'Total Audios': len(X_prueba),
        'Verdaderos Positivos (VP)': tp,
        'Falsos Positivos (FP)': fp,
        'Verdaderos Negativos (VN)': tn,
        'Falsos Negativos (FN)': fn,
        'Accuracy': accuracy_score(y_prueba, y_prediccion),
        'Precision': precision_score(y_prueba, y_prediccion),
        'Recall': recall_score(y_prueba, y_prediccion),
        'F1-Score': f1_score(y_prueba, y_prediccion),
        'AUC-ROC': auc_roc,
        'AUC-PR': auc_pr,
        'Specificity': tn / (tn + fp),
        'Balanced Accuracy': balanced_accuracy_score(y_prueba, y_prediccion),
        'Características': ', '.join(parametros),
        'Épocas': len(historial.history['val_loss']),
        'Tiempo Procesamiento Audios (s)': total_processing_time, 
        'Tiempo Entrenamiento (s)': training_time, 
        'Tiempo Total (s)': total_processing_time + training_time 
    }

    # Guardamos el modelo y el escalador
    modelo.save(f"{RUTA_MODELOS}modelo_{nombre_combinacion.replace('+', 'plus').lower()}.keras")
    joblib.dump(escalador, f"{RUTA_MODELOS}escalador_{nombre_combinacion.replace('+', 'plus').lower()}.pkl")

    return metricas, historial.history, (fpr, tpr, auc_roc), (recall, precision, auc_pr)



def ejecutar_experimento():
    # Configurar porcentaje de audios a usar en el experimento (por defecto está en la variable global)
    global PORCENTAJE_AUDIOS, RUTA_RESULTADOS, RUTA_PDF

    try:
        # Cambiar el     porcentaje de audios a usar (entre 1 y 100)
        porcentaje = int(input(f"Ingrese porcentaje de audios a usar (1-100, actual={PORCENTAJE_AUDIOS}%): "))
        if 1 <= porcentaje <= 100:
            PORCENTAJE_AUDIOS = porcentaje  # Actualiza la variable global 
    except:
        # Si hay algún error, se mantiene el valor actual
        pass

    # Rutas a los conjuntos de entrenamiento reales y falsos
    rutas = {
        'train_real': "datos/entrenamiento/real",
        'train_fake': "datos/entrenamiento/falso"
    }

    # Crea los directorios
    os.makedirs("resultados", exist_ok=True)
    os.makedirs("reportes", exist_ok=True)
    os.makedirs(RUTA_MODELOS, exist_ok=True)

    # Ejecuta todas las configuraciones de modelo definidas
    for i, config_modelo in enumerate(CONFIGURACIONES_MODELOS):
        print(f"\n=== EJECUTANDO CONFIGURACIÓN {i+1}/{len(CONFIGURACIONES_MODELOS)} ===")
        print("Parámetros del modelo:")
        for k, v in config_modelo.items():
            print(f"- {k.replace('_', ' ').title()}: {v}")  

        # Genera un prefijo único para el archivo
        fecha_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        params_str = f"conv{config_modelo['conv_filters']}_dense{config_modelo['dense_units']}_lr{config_modelo['learning_rate']}"
        prefix = f"{fecha_str}_config{i+1}_{params_str}_audios{PORCENTAJE_AUDIOS}%_"

        # Define las rutas para guardar los resultados CSV y el PDF usando el prefijo
        RUTA_RESULTADOS = f"resultados/{prefix}comparativos.csv"
        RUTA_PDF = f"reportes/{prefix}reporte_completo.pdf"

        # Listas para almacenar los resultados de cada combinación de parámetros
        resultados = []
        historiales = []
        roc_curves = []
        pr_curves = []

        # Recorre cada combinación de características
        for nombre, parametros in COMBINACIONES.items():
            try:
                # Evalúa la combinación actual llamando a la función de evaluación
                metricas, historial, roc, pr = evaluar_combinacion(
                    parametros, nombre, rutas, PORCENTAJE_AUDIOS, config_modelo)
                
                # Las métricas válidas, se almacenan para su posterior análisis
                if metricas is not None:
                    resultados.append(metricas)
                    historiales.append((nombre, historial))
                    roc_curves.append(roc)
                    pr_curves.append(pr)
                    
                    print(f" {nombre}: F1={metricas['F1-Score']:.4f}, AUC-ROC={metricas['AUC-ROC']:.4f}")
            except Exception as e:
                # Si hay un error al evaluar una combinación, se muestra y continúa con la siguiente
                print(f" Error en {nombre}: {str(e)}")
                continue

        # Si se generaron resultados válidos, se guardan y se genera un reporte
        if resultados:
            # Guarda resultados en el archivo CSV
            df_resultados = pd.DataFrame(resultados)
            df_resultados.to_csv(RUTA_RESULTADOS, index=False)

        else:
            # Si no se obtuvo ningún resultado válido
            print("No se generaron resultados válidos para esta configuración")

if __name__ == "__main__":
    ejecutar_experimento()
