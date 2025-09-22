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
- Analizar métricas de precisión y pérdida durante el entrenamiento y validación.  
- Preparar modelos para inferencia en dispositivos móviles mediante TensorFlow Lite.

---

### Dataset
- **Fuente:** [CelebA Dataset](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)  
- **Uso en el proyecto:** Descarga y carga mediante la biblioteca `kagglehub`.  
- **Contenido:** Imágenes de rostros de celebridades con atributos como `Eyeglasses`, `Blurry`, `Wearing_Hat`, `No_Occlusion`, entre otros.

```python
import kagglehub

# Descargar dataset
path = kagglehub.dataset_download("jessicali9530/celeba-dataset")
print("Path to dataset files:", path)
Procedimiento Experimental
Preprocesamiento de datos
Etiquetado de imágenes según el tipo de oclusión.

División en conjuntos de entrenamiento (80%) y validación (20%).

Modelos utilizados
MobileNetV2: Fine-tuning de las últimas capas para clasificación de oclusiones.

ResNet50: Evaluación comparativa para medir desempeño frente a MobileNetV2.

Generación de datos aumentados mediante ImageDataGenerator.

Entrenamiento y Evaluación
Monitoreo de curvas de pérdida y precisión.

Comparación entre modelos entrenados con y sin oclusiones.

Medición del tiempo promedio de inferencia por imagen.

Exportación del modelo
Conversión a TensorFlow Lite (.tflite) para uso en dispositivos móviles o embebidos.

Requisitos
Python 3.10 o superior

TensorFlow

Keras

OpenCV

Pandas

Numpy

Matplotlib

Seaborn

scikit-learn

Kagglehub

Uso
Clonar el repositorio:

bash
Copiar código
git clone <URL_DEL_REPOSITORIO>
Instalar dependencias:

bash
Copiar código
pip install -r requirements.txt
Descargar y preparar el dataset usando el snippet de Kagglehub (ver sección de Dataset).

Ejecutar los notebooks de entrenamiento y evaluación incluidos en el repositorio.

Resultados
Precisión final MobileNetV2 con oclusiones: ~94.10%

Precisión final MobileNetV2 sin oclusiones: 100%

Tiempo promedio de inferencia por imagen: 0.35 segundos
