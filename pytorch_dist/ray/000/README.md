# Ejemplo de Ray y Ray Tune

Este ejemplo contiene dos códigos que ilustran:
- El lanzamiento de un trabajo básico de Ray ([simple-trainer.py](https://github.com/diegoandradecanosa/Cesga2023Courses/blob/main/pytorch_dist/ray/000/simple-trainer.py))
- El lanzamiento de un proceso de HPO a través de Ray Tune ([tune-sample.py](https://github.com/diegoandradecanosa/Cesga2023Courses/blob/main/pytorch_dist/ray/000/tune-sample.py))

En ambos casos el lanzamiento se realizar a través de sbatch:
```
sbatch raysimple.sbatch 
sbatch raytune.sbatch 
```
En ambos casos se generará una salida siguiendo el esquema de nombre *ray_dist_xxx.out* donde *xxx* es el id del trabajo en SLURM.

Si examinamos el script principal de lanzamiento (ray.sbatch) veremos que contiene la habitual reserva de recursos SBATCH en el preámbulo del fichero.
A continuación, configura el clúster ray lanzando en primer lugar el $head_node$ (a través del script [raymaster.sh](https://github.com/diegoandradecanosa/Cesga2023Courses/blob/main/pytorch_dist/ray/000/raymaster.sh)) 
y luego cada uno de los nodos *workers* (a través del script [rayworkers.sh](https://github.com/diegoandradecanosa/Cesga2023Courses/blob/main/pytorch_dist/ray/000/rayworkers.sh))

https://github.com/diegoandradecanosa/Cesga2023Courses/blob/0d95b66c498d7720fc392751043ef0ff2649c3af/pytorch_dist/ray/000/ray.sbatch#L40-L53

A continuación el script activa el entorno conda y lanza el script correspondiente directamente usando el intérprete de Python

https://github.com/diegoandradecanosa/Cesga2023Courses/blob/0d95b66c498d7720fc392751043ef0ff2649c3af/pytorch_dist/ray/000/ray.sbatch#L56-L60

# Script tune-sample.py

Este script contiene un ejemplo muy básico que ilustra el funcionamiento de Ray en el marco de una tarea extremedamente sencilla. Entre en cada nodo trabajador, 
recupera su dirección IP e imprime cuántas CPUs hay disponibles en cada trabajador.

El código empieza initcializando Ray con la dirección del *head_node* del clúster ray previamente lanzado, que recupera a través de una variable de entorno.

https://github.com/diegoandradecanosa/Cesga2023Courses/blob/0d95b66c498d7720fc392751043ef0ff2649c3af/pytorch_dist/ray/000/simple-trainer.py#L14

Este ejemplo hace uso del decorador *@ray.remote* para definir la función que se ejecutará en cada nodo trabajador

https://github.com/diegoandradecanosa/Cesga2023Courses/blob/0d95b66c498d7720fc392751043ef0ff2649c3af/pytorch_dist/ray/000/simple-trainer.py#L20-L23

El código que llama al *remote* está dentro de un bucle

https://github.com/diegoandradecanosa/Cesga2023Courses/blob/0d95b66c498d7720fc392751043ef0ff2649c3af/pytorch_dist/ray/000/simple-trainer.py#L28-L33

# Script tune-sample.py 

En este ejemplo vamos a demostrar las capacidades de Ray Tune para guiar un proceso de HPO. En el código principal del script podemos ver los elementos necesarios
para configurar un proceso de estas características.

Primero vemos la configuración del espacio de búsqueda.

https://github.com/diegoandradecanosa/Cesga2023Courses/blob/ce44446296074370c7921247ea3ed8981bbc283a/pytorch_dist/ray/000/tune-sample.py#L183-L188

Vemos que vamos a explorar distintos rangos de valores para los hiperparámetros $l1$,$l2$, $lr$ y *batch_size*.

El siguiente componente que debemos configurar es el planificador del proceso de búsqueda.

https://github.com/diegoandradecanosa/Cesga2023Courses/blob/ce44446296074370c7921247ea3ed8981bbc283a/pytorch_dist/ray/000/tune-sample.py#L189-L192

Todos estos elementos se integran en la creación de un objeto de la clase *Tuner*

https://github.com/diegoandradecanosa/Cesga2023Courses/blob/ce44446296074370c7921247ea3ed8981bbc283a/pytorch_dist/ray/000/tune-sample.py#L194-L206

Que recibe:
- Los parámetros de un modelo que es el que queremos entrenar *train_cifar* junto con los recursos disponibles.
- La configuración del proceso deo optimización indicando la métrica objetivo, el modo (minimización en el ejemplo), el planificador a utilizar y el número de samples
a utilizar.
- Finalmente, recibe la configuración generada anteriormente del espacio de búsqueda de los hiperparámetros a explorar.

El proceso se desencadena usando la función fit, y el mejor resultado se obtiene con la función *get_best_result*.

https://github.com/diegoandradecanosa/Cesga2023Courses/blob/ce44446296074370c7921247ea3ed8981bbc283a/pytorch_dist/ray/000/tune-sample.py#L207-L209

# Resultados

Una vez ejecutado el proceso, los resultados se generar en un fichero con un nombre que sigue el formato $ray_dist_xxx.out$.
Además, se generan una serie de resultados auxiliares que se guardan en la carpeta *$HOME/ray_results*.
Se proporcionan dos ejemplos de salida:
- Ejemplo de salida de tune-sample.py: [ray_dist_3438911.out](https://github.com/diegoandradecanosa/Cesga2023Courses/blob/main/pytorch_dist/ray/000/ray_dist_3438911.out)
- Ejemplo de salida de simple-trainer.py: [ray_dist_3439027.out](https://github.com/diegoandradecanosa/Cesga2023Courses/blob/main/pytorch_dist/ray/000/ray_dist_3439027.out) 

  
