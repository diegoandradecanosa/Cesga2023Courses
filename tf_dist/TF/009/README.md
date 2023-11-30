# Tutorial: Population-Based Training


Este código implementa una Red Neuronal de Memoria (Memory Neural Network) para resolver preguntas sobre historias proporcionadas en el conjunto de datos bAbI. El bAbI dataset es un conjunto de datos diseñado para evaluar la capacidad de las máquinas para comprender y razonar sobre textos. La implementación se basa en Keras y utiliza Ray para la sintonización de hiperparámetros.
Requisitos previos:

Antes de ejecutar este código, asegúrate de tener instalados los siguientes paquetes y bibliotecas:

    TensorFlow
    Keras
    NumPy
    Ray
    FileLock (instalado con pip install filelock)

Cómo funciona el código:

El código se divide en varias secciones principales:
1. Importación de bibliotecas y módulos:

    Se importan las bibliotecas y módulos necesarios, como TensorFlow, Keras, NumPy, Ray, y otros.
    Se utiliza from __future__ import print_function para habilitar la impresión compatible con Python 3.

2. Funciones de procesamiento de texto:

    tokenize(sent): Esta función divide una oración en tokens, incluyendo la puntuación.
    parse_stories(lines, only_supporting=False): Procesa las historias en el formato bAbI y las convierte en una estructura de datos. Si only_supporting está configurado en True, solo se conservan las oraciones que respaldan la respuesta.

https://github.com/diegoandradecanosa/Cesga2023Courses/blob/a13c2fe11942cc63ea4578d2bf1fbdd03e175cbb/tf_dist/TF/009/pbt.py#L27-L65

3. Funciones para vectorizar historias:

    vectorize_stories(word_idx, story_maxlen, query_maxlen, data): Convierte las historias procesadas en vectores numéricos utilizando un índice de palabras (word_idx). También realiza el relleno de secuencias para que todas tengan la misma longitud.

https://github.com/diegoandradecanosa/Cesga2023Courses/blob/a13c2fe11942cc63ea4578d2bf1fbdd03e175cbb/tf_dist/TF/009/pbt.py#L89-L99

4. Función para leer los datos:

    read_data(finish_fast=False): Descarga y lee el conjunto de datos bAbI. Puedes establecer finish_fast en True para cargar solo un subconjunto de datos más pequeño.

https://github.com/diegoandradecanosa/Cesga2023Courses/blob/a13c2fe11942cc63ea4578d2bf1fbdd03e175cbb/tf_dist/TF/009/pbt.py#L102-L137

5. Clase MemNNModel:

Esta clase se utiliza para definir y entrenar el modelo de Red Neuronal de Memoria.

build_model: En este método, se crea la arquitectura del modelo utilizando capas de Embedding, LSTM y otras capas de Keras para procesar las historias y las preguntas.

https://github.com/diegoandradecanosa/Cesga2023Courses/blob/a13c2fe11942cc63ea4578d2bf1fbdd03e175cbb/tf_dist/TF/009/pbt.py#L140-L219

setup: Configura el modelo y los datos de entrenamiento y prueba.

https://github.com/diegoandradecanosa/Cesga2023Courses/blob/a13c2fe11942cc63ea4578d2bf1fbdd03e175cbb/tf_dist/TF/009/pbt.py#L221-L233

step: Realiza un paso de entrenamiento del modelo.

https://github.com/diegoandradecanosa/Cesga2023Courses/blob/a13c2fe11942cc63ea4578d2bf1fbdd03e175cbb/tf_dist/TF/009/pbt.py#L235-L248

save_checkpoint y load_checkpoint se utilizan para guardar y cargar el modelo durante la sintonización de hiperparámetros.

https://github.com/diegoandradecanosa/Cesga2023Courses/blob/a13c2fe11942cc63ea4578d2bf1fbdd03e175cbb/tf_dist/TF/009/pbt.py#L250-L258

6. Función main:

La función principal inicia una sesión de Ray y configura la sintonización de hiperparámetros utilizando el algoritmo de entrenamiento basado en población (PBT). Se define un conjunto de hiperparámetros que pueden ser sintonizados, como la tasa de abandono (dropout), la tasa de aprendizaje (lr), y otros.

https://github.com/diegoandradecanosa/Cesga2023Courses/blob/a13c2fe11942cc63ea4578d2bf1fbdd03e175cbb/tf_dist/TF/009/pbt.py#L275-L282

Se crea una instancia del tuner (MemNNModel) y se inicia el proceso de entrenamiento.

https://github.com/diegoandradecanosa/Cesga2023Courses/blob/a13c2fe11942cc63ea4578d2bf1fbdd03e175cbb/tf_dist/TF/009/pbt.py#L284-L309

# Resultados

Una vez ejecutado el proceso, los resultados se generar en un fichero con un nombre que sigue el formato. Además, se generan una serie de resultados auxiliares que se guardan en la carpeta $HOME/ray_results. Se proporcionan dos ejemplos de salida:

Ver: [jsimple.o5027335](https://github.com/diegoandradecanosa/Cesga2023Courses/blob/main/tf_dist/TF/009/jsimple.o5027335)