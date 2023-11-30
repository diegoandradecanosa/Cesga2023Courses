# Tune model on the MNIST dataset

En este ejemplo vamos a demostrar las capacidades de Ray Tune para guiar un proceso de HPO. En el código principal del script podemos ver los elementos necesarios para configurar un proceso de estas características.

Primero vemos la configuración del espacio de búsqueda.

https://github.com/diegoandradecanosa/Cesga2023Courses/blob/f50dd927aa2ca9a33994de7606712153bc32ab73/tf_dist/TF/008/tf_tune.py#L71-L75

Vemos que vamos a explorar distintos rangos de valores para los hiperparámetros ,$lr$, $momentum$ y $hidden layers$.

El siguiente componente que debemos configurar es el planificador del proceso de búsqueda.

https://github.com/diegoandradecanosa/Cesga2023Courses/blob/f50dd927aa2ca9a33994de7606712153bc32ab73/tf_dist/TF/008/tf_tune.py#L55-L57

Todos estos elementos se integran en la creación de un objeto de la clase Tuner

https://github.com/diegoandradecanosa/Cesga2023Courses/blob/f50dd927aa2ca9a33994de7606712153bc32ab73/tf_dist/TF/008/tf_tune.py#L59C18-L59C23-L77

Que recibe:

    Los parámetros de un modelo que es el que queremos entrenar train_mnist junto con los recursos disponibles.
    La configuración del proceso deo optimización indicando la métrica objetivo, el modo (minimización en el ejemplo), el planificador a utilizar y el número de samples a utilizar.
    Finalmente, recibe la configuración generada anteriormente del espacio de búsqueda de los hiperparámetros a explorar.

El proceso se desencadena usando la función fit, y el mejor resultado se obtiene con la función get_best_result.

# Resultados

Una vez ejecutado el proceso, los resultados se generar en un fichero con un nombre que sigue el formato. Además, se generan una serie de resultados auxiliares que se guardan en la carpeta $HOME/ray_results. Se proporcionan dos ejemplos de salida:

    Ver: [ray_dist_3439027.out](https://github.com/diegoandradecanosa/Cesga2023Courses/blob/main/tf_dist/TF/008/jsimple.o5027019)
