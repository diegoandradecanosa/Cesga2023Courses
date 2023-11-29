import tensorflow as tf
import ray

print(f"Version de TensorFlow: {tf.__version__}")
print(f"Version de Ray: {ray.__version__}")

# Inicializa Ray y verifica​
ray.init(ignore_reinit_error=True)
print("Ray iniciado con éxito: ", ray.is_initialized())