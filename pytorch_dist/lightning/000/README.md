# Ejemplo inicial de Lightning

Este código contiene un ejemplo básico de cómo lanzar un entrenamiento distribuido en Lightning en un entorno distribuido gestionado con SLURM.

El ejemplo se lanza ejecutando el comando

```
sbatch sampleLI.sbatch
```
y debería generar una salida similar a [slurm-3417863.out](https://github.com/diegoandradecanosa/Cesga2023Courses/blob/main/pytorch_dist/lightning/000/slurm-3417863.out)
pero con el nombre correspondiente al id del trabajo SLURM.

El script de entrenamiento es el fichero [sampleLI.py](https://github.com/diegoandradecanosa/Cesga2023Courses/blob/main/pytorch_dist/lightning/000/sampleLI.py)
