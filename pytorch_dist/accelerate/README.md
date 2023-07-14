# Ejemplo de accelarate

Este ejemplo ilustra el uso de **accelerate** en un entorno gobernado por SLURM.

El código proporcionado consta de:
- Dos scripts de lanzamiento, principal ([accelerate.sbatch](https://github.com/diegoandradecanosa/Cesga2023Courses/blob/main/pytorch_dist/accelerate/accelerate.sbatch))
y secundario ([accelerate.sh](https://github.com/diegoandradecanosa/Cesga2023Courses/blob/main/pytorch_dist/accelerate/accelerate.sh))
- Un script python con el código de ejemplo ([accelerate_sample.py](https://github.com/diegoandradecanosa/Cesga2023Courses/blob/main/pytorch_dist/accelerate/accelerate_sample.py))
- Un ejemplo de salida ([accel_dist_3431611.out](https://github.com/diegoandradecanosa/Cesga2023Courses/blob/main/pytorch_dist/accelerate/accel_dist_3431611.out))

El código se ejecutar del siguiente modo
```
sbatch accelerate.sbach
```
y la salida se genera en un fichero con el nombre *accel_dist_xxx.out* (ver ejemplo proporcionado).

Los scripts de lanzamiento son similares a los usados en otras tecnologías como Lightning o Fabric, aunque tienen algunas peculiaridades aplicables a accelerate.
La más obvia es que el lanzamiento del script de entrenamiento se realizar usando la herramienta **accelerate**.

https://github.com/diegoandradecanosa/Cesga2023Courses/blob/d259b8ec95168a97a7e15d86246d0126727e3fb0/pytorch_dist/accelerate/accelerate.sh#L23

El código propio de accelerate está contenido en la función *training_function* del script de entrenamiento. Al principio, se debe instanciar la clase
*Accelerator*

https://github.com/diegoandradecanosa/Cesga2023Courses/blob/d259b8ec95168a97a7e15d86246d0126727e3fb0/pytorch_dist/accelerate/accelerate_sample.py#L60

Luego, hay que preparar el *accelerator* con el modelo, el optimizador, los cargadores de los conjuntos de entrenamiento y evaluación y el planificador del *Learning Rate*.

https://github.com/diegoandradecanosa/Cesga2023Courses/blob/d259b8ec95168a97a7e15d86246d0126727e3fb0/pytorch_dist/accelerate/accelerate_sample.py#L149-L151

El bucle principal de entrenamiento contiene algunos elementos condicionados por el uso de *accelerate*

https://github.com/diegoandradecanosa/Cesga2023Courses/blob/d259b8ec95168a97a7e15d86246d0126727e3fb0/pytorch_dist/accelerate/accelerate_sample.py#L181-L203



