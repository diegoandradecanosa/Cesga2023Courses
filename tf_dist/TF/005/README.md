# Parameter Server Ejemplo Simple

Este es un ejemplo simple de uso de la estrategia Parameter Server. Recordemos que esta estrategia se 
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




