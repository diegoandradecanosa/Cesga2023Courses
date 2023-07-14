# Ejemplo inicial de Lightning

Este código contiene un ejemplo básico de cómo lanzar un entrenamiento distribuido en Lightning en un entorno distribuido gestionado con SLURM.

El ejemplo se lanza ejecutando el comando

```
sbatch sampleLI.sbatch
```
y debería generar una salida similar a [slurm-3417863.out](https://github.com/diegoandradecanosa/Cesga2023Courses/blob/main/pytorch_dist/lightning/000/slurm-3417863.out)
pero con el nombre correspondiente al id del trabajo SLURM.

El script de entrenamiento es el fichero [sampleLI.py](https://github.com/diegoandradecanosa/Cesga2023Courses/blob/main/pytorch_dist/lightning/000/sampleLI.py)

En el scrip de entrenamiento la definición del modelo se debe hacer dentro de un *LightningModule*

https://github.com/diegoandradecanosa/Cesga2023Courses/blob/0f8e4b878f9e9b5865b573d4783e5300e539a9cb/pytorch_dist/lightning/000/sampleLI.py#L10

a través del cual se proporciona una implementación para varios métodos:
- El constructor *__init__* que contiene la definición del modelo
- La implementación de la pasada *forward*
- El método para configurar el/los optimizadores *configure_optimizers*
- La definición de un *training_step*. Veamos su contenido en el código.
  https://github.com/diegoandradecanosa/Cesga2023Courses/blob/0f8e4b878f9e9b5865b573d4783e5300e539a9cb/pytorch_dist/lightning/000/sampleLI.py#L30-L37
- La definición de un *validation_step*. Veamos su contenido en el código.
  https://github.com/diegoandradecanosa/Cesga2023Courses/blob/0f8e4b878f9e9b5865b573d4783e5300e539a9cb/pytorch_dist/lightning/000/sampleLI.py#L39-L45

  Si proporcionamos una implementación para todos esos métodos, el uso de Lightning es realmente sencillo e involucra unas pocas líneas de código

  https://github.com/diegoandradecanosa/Cesga2023Courses/blob/0f8e4b878f9e9b5865b573d4783e5300e539a9cb/pytorch_dist/lightning/000/sampleLI.py#L48-L59

  Además, vemos que lightning es capaz de detectar y gestionar automáticamente el *world_size* y el *rank* proporcionados por SLURM.
  
