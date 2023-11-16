# ClusterResolver

Este es un ejemplo simple que inspecciona los recursos reservados con Slurm a través del API correspondiente de TF.

Para ejecutar el ejemplo, nos conectamos al FT3 y lanzamos el script de ejecución

```bash
ssh tuusuario@ft3.cesga.es
sbatch sbatch_2nodes_ngpus.sh simple.py 
```

Los resultados se generarán en fichero con la forma jsimple*

Veamos cómo se realiza el lanzamiento. El script sbatch contiene la reserva de recursos.

https://github.com/diegoandradecanosa/Cesga2023Courses/blob/ba81c6a41c473c4e98227c1bf86588b553a36812/tf_dist/TF/002/sbatch_2nodes_ngpus.sh#L2-L10

Básicamente son 2 nodos y 2 GPUs por nodo. Una vez cargado el entorno mytf, tenemos que lanzar el script python a través de srun

https://github.com/diegoandradecanosa/Cesga2023Courses/blob/ba81c6a41c473c4e98227c1bf86588b553a36812/tf_dist/TF/002/sbatch_2nodes_ngpus.sh#L26

El código es extremadamente sencillo, simplemente descubre los recursos reservados a través de un SlurmClusterResolver

https://github.com/diegoandradecanosa/Cesga2023Courses/blob/ba81c6a41c473c4e98227c1bf86588b553a36812/tf_dist/TF/002/simple.py#L1-L5




