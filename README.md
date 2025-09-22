# FACE-OCLUSION_DEEP_LEARNING

## Descripción
Proyecto realizado como trabajo de fin de la Unidad 02 en el curso de Ingeniería de Sistemas.  
Se trata de un **estudio comparativo de algoritmos de detección de rostros** basados en HOG y CNN, con análisis de técnicas de oclusión.  
El proyecto utiliza el dataset **CelebA** para entrenar modelos de detección de rostros y clasificación de oclusiones como gafas, sombrero, desenfoque y ausencia de oclusión.

## Integrantes
- CHOQUE QUISPE JADYRA CH'ASKA - 204795  
- HANCCO CHAMPI FRAN ANTHONY - 204797  
- JALLO PACCAYA YASUMY MARICELY - 204799  

## Contenido del proyecto
1. **Importación de datos:**  
   Descarga del dataset CelebA mediante la librería `kagglehub` y carga de imágenes y atributos.  

2. **Preprocesamiento:**  
   - Etiquetado de imágenes según tipo de oclusión (`Eyeglasses`, `Blurry`, `Hat`, `No Occlusion`).  
   - División en conjuntos de entrenamiento (80%) y validación (20%).  

3. **Modelos utilizados:**  
   - **MobileNetV2**: Para clasificación de rostros con oclusiones, con fine-tuning de las últimas 20 capas.  
   - Generación de datos aumentados con `ImageDataGenerator`.  
   - Entrenamiento y evaluación del modelo con y sin datos de oclusiones.  

4. **Evaluación del modelo:**  
   - Métricas de precisión y pérdida de entrenamiento y validación.  
   - Comparación del desempeño entre modelos entrenados con y sin oclusiones.  
   - Medición del tiempo promedio de detección.  

5. **Predicción de imágenes individuales:**  
   Función `predict_image_class(image_path)` para clasificar imágenes en las 4 categorías de oclusión.

6. **Exportación del modelo:**  
   Conversión del modelo entrenado a **TensorFlow Lite** (`.tflite`) para uso en dispositivos móviles o embebidos.

## Requisitos
- Python 3.10 o superior  
- TensorFlow  
- Keras  
- OpenCV  
- Pandas  
- Numpy  
- Matplotlib  
- Seaborn  
- scikit-learn  
- Kagglehub  


## Resultados
# Precisión final de MobileNetV2 con oclusiones: ~94.10%
# Precisión final de MobileNetV2 sin oclusiones: 100%
# Tiempo promedio de inferencia por imagen: 0.35 segundos
