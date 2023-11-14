
# Actividad un nodo dos GPUs

Conéctate al FT3 

```bash
ssh tuusuario@ft3.cesga.es
```



Abre una sesión interactiva con 1 nodo y 2 GPUs A100

```bash
./interactive_1node_2gpus.sh 
```

Activar el entorno, entrar en el directorio del ejemplo y arrancar jupyter lab

```bash
source $STORE/mytf/bin/activate
cd $STORE/Cesga2023Courses/tf_dist/TF/003
jupyter lab --ip `hostname -i` --no-browser
```

Abrir sucesivamente los notebook simple2GPUsOrig, simple2GPUsSol y simple2GPUsSol2 y apreciar las diferencias entre ellos.

Las diferencia apreciables se deben a que:

- simple2GPUsOrig no está configurado para usar ninguna estrategia de entrenamiento distribuido.
- simple2GPUsSol está configurado para usar una estrategia espejo del tipo MirroredStrategy (no recomendada, incluso dentro de un único nodo)
- simple2GPUSol2 está configurado para usar una estrategia espejo del tipo MultiworkerMirroredStrategy (la recomendada para proporcionar paralelismo de datos)

También se pueden ejecutar las versiones .py de los mismos códigos usando ipython

- ipython simple2GPUsOrig.py 
- ipython simple2GPUsSol.py 
- ipython simple2GPUsSol2.py 


