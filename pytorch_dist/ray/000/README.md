# Ejemplo de Ray y Ray Tune

Este ejemplo contiene dos códigos que ilustran:
- El lanzamiento de un trabajo básico de Ray ([simple-trainer.py](https://github.com/diegoandradecanosa/Cesga2023Courses/blob/main/pytorch_dist/ray/000/simple-trainer.py))
- El lanzamiento de un proceso de HPO a través de Ray Tune ([tune-sample.py](https://github.com/diegoandradecanosa/Cesga2023Courses/blob/main/pytorch_dist/ray/000/tune-sample.py))

En ambos casos el lanzamiento se realizar a través del mismo script que se parametriza con el nombre del script Python a lanzar en cada caso. Es decir, el lanzamiento se haría
como sigue:
```
sbatch ray.sbatch simple-trainer.py
sbatch ray.sbatch tune-sample.py
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

