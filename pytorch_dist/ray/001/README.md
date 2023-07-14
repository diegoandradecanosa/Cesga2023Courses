# Ejemplo de uso de Ray Train

Este ejemplo ilustra el uso de la herramienta Ray Train, junto con Tune es la herramienta de Ray más relacionada con este curso.

Volvemos a proporcionar los mismos 3 scripts de lanzamiento que en el ejemplo anterior (el principal, y los del *head_node* y los *workers* respectivamente).

El lanzamiento se hace con el siguiente comando.
```
sbatch ray.sbatch ray-train-mnist.py
```
que van a generar un fichero con el nombre *ray_dist_xxx.out*. Se proporciona el fichero
[ray_dist_3458423.out](https://github.com/diegoandradecanosa/Cesga2023Courses/blob/main/pytorch_dist/ray/001/ray_dist_3458423.out) a modo de ejemplo.

# Script ray-train-mnist.py

Al principio del código se inicializa ray

https://github.com/diegoandradecanosa/Cesga2023Courses/blob/bb14e20a5e5b03ade5deb97a0173422010699392/pytorch_dist/ray/001/ray-train-mnist.py#L153-L159

El proceso de entrenamiento se define dentro de la función *train_fashion_mnist* donde se instancia un *TorchTrainer* usando una definición
de la función de entrenamiento a usar en cada trabajador, la configuración de los hiperparámetros de entrenamiento, y la configuración del escalado
del *trainer* que especifica los recursos utilizables. Una vez definido, se desencadena el entrenamiento usando la función *fit* del *trainer*.

https://github.com/diegoandradecanosa/Cesga2023Courses/blob/bb14e20a5e5b03ade5deb97a0173422010699392/pytorch_dist/ray/001/ray-train-mnist.py#L118-L123

La definición de lo que cada trabajador debe ejecutar para el entrenamiento se realiza dentro de la función *train_func*, donde

(1) Se preparan los cargadores de datos

https://github.com/diegoandradecanosa/Cesga2023Courses/blob/bb14e20a5e5b03ade5deb97a0173422010699392/pytorch_dist/ray/001/ray-train-mnist.py#L98-L102

(2) Se crea el modelo, se definen las funciones de pérdida y el optimizador

https://github.com/diegoandradecanosa/Cesga2023Courses/blob/bb14e20a5e5b03ade5deb97a0173422010699392/pytorch_dist/ray/001/ray-train-mnist.py#L105-L109

y (3) Se itera sobre el bucle de entrenamiento, donde se hace entrenamiento-validación-impresión de resultados durante varias epochs.

https://github.com/diegoandradecanosa/Cesga2023Courses/blob/bb14e20a5e5b03ade5deb97a0173422010699392/pytorch_dist/ray/001/ray-train-mnist.py#L111-L114

La función *train_epoch* es la que encapsula realmente el bucle de entrenamiento

https://github.com/diegoandradecanosa/Cesga2023Courses/blob/bb14e20a5e5b03ade5deb97a0173422010699392/pytorch_dist/ray/001/ray-train-mnist.py#L52-L67






