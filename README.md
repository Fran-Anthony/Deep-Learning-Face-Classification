# FACE-OCLUSION_DEEP_LEARNING

## Trabajo de Fin de Unidad 02
**Estudio comparativo de algoritmos de detección de rostros basados en HOG y CNN con análisis de técnicas de oclusión**

### Integrantes
- CHOQUE QUISPE JADYRA CH'ASKA - 204795  
- HANCCO CHAMPI FRAN ANTHONY - 204797  
- JALLO PACCAYA YASUMY MARICELY - 204799  

### Descripción
Este proyecto fue desarrollado como parte de un curso universitario y consiste en un análisis comparativo de algoritmos de detección de rostros, evaluando cómo afectan las oclusiones (gafas, sombreros, desenfoque, etc.) en la precisión de los modelos. Se entrenaron modelos CNN (MobileNetV2 y ResNet50) usando el dataset CelebA para clasificar imágenes de rostros con diferentes tipos de oclusión.

El objetivo principal es:
- Comparar el rendimiento de modelos de detección de rostros entrenados con y sin datos de oclusión.
- Analizar métricas de precisión y pérdida.
- Preparar modelos para inferencia en dispositivos móviles mediante TensorFlow Lite.

---

### Dataset
- **Fuente:** [CelebA Dataset](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)  
- **Uso en el proyecto:** Descarga y carga mediante la biblioteca `kagglehub`.  
- **Contenido:** Imágenes de rostros de celebridades con atributos como Eyeglasses, Blurry, Wearing_Hat, etc.

```python
import kagglehub

# Descargar dataset
path = kagglehub.dataset_download("jessicali9530/celeba-dataset")
print("Path to dataset files:", path)
