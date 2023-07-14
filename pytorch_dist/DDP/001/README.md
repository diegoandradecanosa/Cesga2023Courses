# Ejemplo DDP n-nodos n-gpus

Este ejemplo ilustra el uso de DDP cuando se hace una revista de n-nodos en un supercomputador gestionado a través del sistema de colas SLURM. 
En el que a su vez cada nodo dispone de varias GPUs (2 en este caso).

Primero, veamos los contenidos de los scripts que realizan la reserva de SLURM y el lanzamiento del script python de entrenamiento con esa reserva.

El script principal (ddpsrunSBATCH.sh) contiene un preámbulo en el que reservamos:
- 2 nodos
- 2 tareas por nodo
- 1 gpu a100 por tarea
- 64 G de RAM por nodo
- y 32 núcleos para cada tarea

https://github.com/diegoandradecanosa/Cesga2023Courses/blob/65eb3cbbd4a856f238f6baadfc1453052e531922/pytorch_dist/DDP/001/ddpsrunSBATCH.sh#L1-L9

Este script debe ser lanzado con el comando:
```
sbatch ddpsrunSBATCH.sh
```
La salida se recogerá en un fichero con el nombre *slurm-xxx.out* donde *xxx* es el identificador (número entero) del trabajo en SLURM. 
El fichero [slurm-3396279.out](https://github.com/diegoandradecanosa/Cesga2023Courses/blob/main/pytorch_dist/DDP/001/slurm-3396279.out)
contiene una salida de ejemplo.

Dentro del script de lanzamiento, se recuperan a través de variables de entorno de SLURM y se fijan a través de variables de entorno, valores que
serán relevante para la ejecución distribuida:
- Se establece el *MASTER_PORT* en base a un cálculo basado en el PID del trabajo SLURM
- Se recupera el *WORLD_SIZE* directamente de la variable de entorno *SLURM_NPROCS*
- Se recupera la *MASTER_ADDR* con un comando de SLURM

https://github.com/diegoandradecanosa/Cesga2023Courses/blob/65eb3cbbd4a856f238f6baadfc1453052e531922/pytorch_dist/DDP/001/ddpsrunSBATCH.sh#L11-L19

Finalmente, se lanza un script secundario, *mnist_classify_ddp.sh*, con srun.

https://github.com/diegoandradecanosa/Cesga2023Courses/blob/65eb3cbbd4a856f238f6baadfc1453052e531922/pytorch_dist/DDP/001/ddpsrunSBATCH.sh#L21

El script secundario carga el entorno conda y lanza el script de entrenamiento, pasando como parámetro el número de epochs

https://github.com/diegoandradecanosa/Cesga2023Courses/blob/65eb3cbbd4a856f238f6baadfc1453052e531922/pytorch_dist/DDP/001/mnist_classify_ddp.sh#L9-L10

## Script de entrenamiento

Cada trabajador desde el que se lanza el script de entrenamiento (se van a lanzar tantas copias como trabajadores tengamos, 2x2 en el ejemplo) tiene que ejecutar
el método *init_process_group* inicializando el backend y especificando su *rank*

https://github.com/diegoandradecanosa/Cesga2023Courses/blob/65eb3cbbd4a856f238f6baadfc1453052e531922/pytorch_dist/DDP/001/mnist_classify_ddp.py#L73-L77

Estos valores se obtiene a través de variables de entorno fijadas en el script de lanzamiento o por SLURM

https://github.com/diegoandradecanosa/Cesga2023Courses/blob/65eb3cbbd4a856f238f6baadfc1453052e531922/pytorch_dist/DDP/001/mnist_classify_ddp.py#L128-L131

Al ser un entorno en el que tenemos varios trabajadores, cada uno con una GPU, por nodo, necesitamos establecer qué GPU usará cada uno.

https://github.com/diegoandradecanosa/Cesga2023Courses/blob/65eb3cbbd4a856f238f6baadfc1453052e531922/pytorch_dist/DDP/001/mnist_classify_ddp.py#L140-L141

De nuevo, la carga eficiente de los datos se hace configurando un *DistributedSampler*

https://github.com/diegoandradecanosa/Cesga2023Courses/blob/65eb3cbbd4a856f238f6baadfc1453052e531922/pytorch_dist/DDP/001/mnist_classify_ddp.py#L144-L147

Se llama a DDP con el modelo Pytorch y especificando el dispositivo *device_id* que utilizará el trabajador actual.

https://github.com/diegoandradecanosa/Cesga2023Courses/blob/65eb3cbbd4a856f238f6baadfc1453052e531922/pytorch_dist/DDP/001/mnist_classify_ddp.py#L150-L150

El bucle de entrenamiento no tiene nada especial salvo que, como en el caso anterior, nos aseguramos de que cada batch vaya al dispositivo adecuado.

https://github.com/diegoandradecanosa/Cesga2023Courses/blob/65eb3cbbd4a856f238f6baadfc1453052e531922/pytorch_dist/DDP/001/mnist_classify_ddp.py#L39-L53

Al final del entrenamiento, es una buena práctica asegurarnos la destrucción del entorno distribuido

https://github.com/diegoandradecanosa/Cesga2023Courses/blob/65eb3cbbd4a856f238f6baadfc1453052e531922/pytorch_dist/DDP/001/mnist_classify_ddp.py#L162






