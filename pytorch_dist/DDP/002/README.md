# Ejemplo Fully-Sharded Data Parallel

Este ejemplo ilustra el uso de una estrategia Fully-Sharded Data Parallel (FSDP) que está soportada por el soporte nativo de Pytorch para entrenamientos distribuidos.

Los scripts de lanzamiento [ddpsrunSBATCH.sh](https://github.com/diegoandradecanosa/Cesga2023Courses/blob/main/pytorch_dist/DDP/002/ddpsrunSBATCH.sh) y [trainfsdp.sh](https://github.com/diegoandradecanosa/Cesga2023Courses/blob/main/pytorch_dist/DDP/002/trainfsdp.sh)
son análogos a los del ejemplo DDP.

La ejecución se realiza ejecutando el comando
```
sbatch ddpsrunSBATCH.sh
```
y la salida se generaría en un fichero con el nombre *slurm-xxx.out* donde *xxx* es el identificador del trabajo en SLURM.
El fichero [slurm-3413292](https://github.com/diegoandradecanosa/Cesga2023Courses/blob/main/pytorch_dist/DDP/002/slurm-3413292.out) contiene un ejemplo de la salida generada durante la ejecución.

El script de entrenamiento es donde se concentran la diferencia más notable con el ejemplo DDP. Todo el código es análogo a aquel ejemplo, salvo que a la hora
de aplicar al modelo el entrenamiento distribuido, utilizadmos *FSDP()* en vez de *DDP()*.

https://github.com/diegoandradecanosa/Cesga2023Courses/blob/d4e3d41dfe21dfda44dbf3465a50b202966e38d3/pytorch_dist/DDP/002/fsdp.py#L143-L143



