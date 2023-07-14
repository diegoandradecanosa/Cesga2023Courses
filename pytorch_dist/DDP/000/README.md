# DDP 101

Este directorio contiene un ejemplo básico de *DataParallelism* y *DistributedDataParallelism* en forma de [notebook](./pytorchDP101.ipynb)

Respecto al ejemplo de DataDistributedParallel (DDP), el más complejo. Debemos empezar inicializando el backend, en este caso **nccl**

https://github.com/diegoandradecanosa/Cesga2023Courses/blob/14187ad82941a502e051b8abba55b8e65ff1e226/pytorch_dist/DDP/000/ddp.py#L28

Luego, cogemos el **rank** del proceso actual y lo usamos para seleccionar el dispositivo que usaremos. Este ejemplo asume una reserva
de un 1 nodos - n dispositivos

https://github.com/diegoandradecanosa/Cesga2023Courses/blob/14187ad82941a502e051b8abba55b8e65ff1e226/pytorch_dist/DDP/000/ddp.py#L30-L34

A continuación, construimos el ddp_model a través del método DDP, al cual le pasamos:
- El modelo original de Pytorch
- Una lista con los ids de los dispositivos utilizados (global_rank)
- Una lista con los dispositivos de salida (global_rank)

  https://github.com/diegoandradecanosa/Cesga2023Courses/blob/14187ad82941a502e051b8abba55b8e65ff1e226/pytorch_dist/DDP/000/ddp.py#L40

  La carga de los datos se realiza a través de un **sampler** distribuido

  https://github.com/diegoandradecanosa/Cesga2023Courses/blob/14187ad82941a502e051b8abba55b8e65ff1e226/pytorch_dist/DDP/000/ddp.py#L42-L45
  

  El optimizador usa el *ddp_model* como parámetro

  https://github.com/diegoandradecanosa/Cesga2023Courses/blob/14187ad82941a502e051b8abba55b8e65ff1e226/pytorch_dist/DDP/000/ddp.py#L49

  El bucle de entrenamiento no tiene ninguna peculiaridad, salvo la asociación de los dastos con la GPU de Nvidia

  https://github.com/diegoandradecanosa/Cesga2023Courses/blob/14187ad82941a502e051b8abba55b8e65ff1e226/pytorch_dist/DDP/000/ddp.py#L51-L62
