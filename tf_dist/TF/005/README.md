# Parameter Server Ejemplo Simple

Este es un ejemplo simple de uso de la estrategia Parameter Server. Para lanzar el código debemos seguir los pasos habituales:

```bash
ssh tusuario@ft3.cesga.es
cd $STORE/Cesga2023Courses/tf_dist/TF/005
sbatch sbatch_2nodes_ngpus.sh PSsample.py
```

Luego, examinamos las salidas

Recordemos que esta estrategia se 
basa en el uso de uno o varios nodos como servidores de parámetro (PS) y uno o varios nodos como trabajadores
(*workers*). También es posible nombrar un nodo adicional para el rol de coordinador (*coordinator*). El ejemplo se basa en utilizar
esta estrategia para hace una tarea muy sencilla, el incremento entre varios trabajadores de una variable *counter* siguiendo la estrategia Parameter Server.

El siguiente fragmento de código muestra la parte principal del código

https://github.com/diegoandradecanosa/Cesga2023Courses/blob/e2e12e163ea806e7868255b79b7e46d36aa15359/tf_dist/TF/005/PSsample.py#L90-L97

Esta parte se lanza a través de la siguiente llamada

https://github.com/diegoandradecanosa/Cesga2023Courses/blob/e2e12e163ea806e7868255b79b7e46d36aa15359/tf_dist/TF/005/PSsample.py#L100

La primera función *create_cluster* crea una especificación a través de un *SlurmClusterResolver* que recupera la información necesaria de la
reserva hecha a través de Slurm

https://github.com/diegoandradecanosa/Cesga2023Courses/blob/e2e12e163ea806e7868255b79b7e46d36aa15359/tf_dist/TF/005/PSsample.py#L45-L51

Además del clúster, la función *create_cluter* devuelve el nombre del trabajo y el índice de la tarea

https://github.com/diegoandradecanosa/Cesga2023Courses/blob/e2e12e163ea806e7868255b79b7e46d36aa15359/tf_dist/TF/005/PSsample.py#L56-L57

Las funciones *run_ps* y *run_worker* se utilizar para lanzar respectivamente el PS y los trabajadores (recuérdese el if en la función principal que lanza una función u otra en función del *job_name*)

En *run_ps* vemos la llamada al lanzamiento del trabajo a través de la función *tf.train.Server*

https://github.com/diegoandradecanosa/Cesga2023Courses/blob/e2e12e163ea806e7868255b79b7e46d36aa15359/tf_dist/TF/005/PSsample.py#L65-L69


En la función *run_worker*, existe una llamada similar. La función *run_worker* además contiene 
una llamada a la función *build_graph* para recuperar la computación que hay que hacer en ese trabajador concreto. Observemos la implementación de esa función.

https://github.com/diegoandradecanosa/Cesga2023Courses/blob/e2e12e163ea806e7868255b79b7e46d36aa15359/tf_dist/TF/005/PSsample.py#L14-L32

Volviendo a la función *run_worker*, el código que se ejecuta en cada trabajador se encuentra dentro del scope *tf.train.MonitoredTrainingSession*

https://github.com/diegoandradecanosa/Cesga2023Courses/blob/e2e12e163ea806e7868255b79b7e46d36aa15359/tf_dist/TF/005/PSsample.py#L83-L88


# Parameter Server con Keras

En este caso vamos a utilizar una estrategia PS para el entrenamiento de un entrenamiento escrito en Keras. Para ejecutar el código seguiremos los pasos habituales.

```bash
ssh tusuario@ft3.cesga.es
cd $STORE/Cesga2023Courses/tf_dist/TF/005
sbatch sbatch_2nodes_ngpus.sh PSKeras.py
```

Luego, examinamos las salidas.

Examinemos las partes relevantes del código. La función *create_in_process_cluter* genera los servidores necesarios a través del API de TF.

https://github.com/diegoandradecanosa/Cesga2023Courses/blob/7f8ce2a1abb0575591003e07fdf76e1b85a5aeb4/tf_dist/TF/005/PSKeras.py#L8-L42


Esta función se llama, en este caso, para crear 1 PS y 3 *workers*. Se crea el particionador de variables que distribuye los datos entre los PS, y luego se instancia
la estrategia PS.

https://github.com/diegoandradecanosa/Cesga2023Courses/blob/7f8ce2a1abb0575591003e07fdf76e1b85a5aeb4/tf_dist/TF/005/PSKeras.py#L48-L58

En esta parte del código se produce la carga del *dataset*

https://github.com/diegoandradecanosa/Cesga2023Courses/blob/7f8ce2a1abb0575591003e07fdf76e1b85a5aeb4/tf_dist/TF/005/PSKeras.py#L64-L69

A continuación, se construye y compila el modelo

https://github.com/diegoandradecanosa/Cesga2023Courses/blob/7f8ce2a1abb0575591003e07fdf76e1b85a5aeb4/tf_dist/TF/005/PSKeras.py#L71-L73

Finalmente, se llama a la función de entrenamiento con los callbacks correspondientes

https://github.com/diegoandradecanosa/Cesga2023Courses/blob/7f8ce2a1abb0575591003e07fdf76e1b85a5aeb4/tf_dist/TF/005/PSKeras.py#L80-L86

# Parameter Server con bucle propio

En este caso vamos a utilizar una estrategia PS para el entrenamiento de un entrenamiento escrito con un bucle propio. Para ejecutar el código seguiremos los pasos habituales.

```bash
ssh tusuario@ft3.cesga.es
cd $STORE/Cesga2023Courses/tf_dist/TF/005
sbatch sbatch_2nodes_ngpus.sh PSCustom.py
```

Luego, examinamos las salidas.

Respecto al código, este sigue conteniendo una definición de la función *create_in_process_cluster* similar al anterior ejemplo.
La definición del *partitioner* y la configuración de la *ParameterServerStrategy* ocurre de forma similar al caso anterior.

Las capas de preprocesamiento de los datos, se definen dentro de un entorno *with strategy.scope()*, así nos aseguramos que se crean en todos los trabajadores.
A continuación, vemos un extracto de código con la definición de las primeras.

https://github.com/diegoandradecanosa/Cesga2023Courses/blob/cbcb0655646346339cbea2e95c9015e058eee5f3/tf_dist/TF/005/PSCustom.py#L69-L76

La creación del *dataset* de entrenamiento se siguen produciendo en la función *dataset_fn*. La generación de algunos samples de ejemplo del *dataset* se produce en la función
*feature_and_label_gen* cuya definición vemos a continuación.

https://github.com/diegoandradecanosa/Cesga2023Courses/blob/cbcb0655646346339cbea2e95c9015e058eee5f3/tf_dist/TF/005/PSCustom.py#L97-L104

El modelo y otros objetos también se definen dentro de un entorno *with strategy.scope()* 

https://github.com/diegoandradecanosa/Cesga2023Courses/blob/cbcb0655646346339cbea2e95c9015e058eee5f3/tf_dist/TF/005/PSCustom.py#L120-L135

Dentro de un entorno similar se produce la distribución de estas variables del modelo entre los *PS* disponibles, con una estrategia de distribución
de tipo *round robin*.






