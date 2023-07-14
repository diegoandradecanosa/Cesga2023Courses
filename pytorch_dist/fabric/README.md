# Ejemplo de Lightning Fabric

Este ejemplo ilustra el uso básico de la librería **Lightning Fabric** en una plataforma ordenada con SLURM.

Los 2 scripts de lanzamiento, el principal ([submit.sbatch](https://github.com/diegoandradecanosa/Cesga2023Courses/blob/main/pytorch_dist/fabric/submit.sbatch)) 
y el secundario ([mnist.sh](https://github.com/diegoandradecanosa/Cesga2023Courses/blob/main/pytorch_dist/fabric/mnist.sh)), 
no difieren sustancialmente de los usados en los ejemplos de **Lightning**.

El lanzamiento del código se realiza con el comando.
```
sbatch submit.sbatch
```

La salida se obtiene en un fichero que sigue la nomenclatura habitual *slurm-xxx.out* y el fichero [slurm-3431901.out](https://github.com/diegoandradecanosa/Cesga2023Courses/blob/main/pytorch_dist/fabric/slurm-3431902.out) 
contiene un ejemplo de salida.

En el script de entrenamiento, el uso de Fabric implica en primer lugar la creación de un objeto *Fabric* indicando los recursos computacionales disponibles
y su lanzamiento (*launch*).

https://github.com/diegoandradecanosa/Cesga2023Courses/blob/aed1fe37fab274e01c0abcd8a26109d22c48970b/pytorch_dist/fabric/mnist.py#L114-L115

A continuación debemos hacer la configuración de fabric usando el modelo (Pytorch) y el optimizador que queremos asociar.

https://github.com/diegoandradecanosa/Cesga2023Courses/blob/aed1fe37fab274e01c0abcd8a26109d22c48970b/pytorch_dist/fabric/mnist.py#L141-L142

El bucle de entrenamiento principal también consta de algunos elementos puntuales que están condicionados por el uso de fabric, como que la pasada backward
se llama a través del objeto *fabric*.

https://github.com/diegoandradecanosa/Cesga2023Courses/blob/aed1fe37fab274e01c0abcd8a26109d22c48970b/pytorch_dist/fabric/mnist.py#L39-L54









