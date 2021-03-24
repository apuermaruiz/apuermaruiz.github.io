---
title: Términos de Machine Learning en Español (Parte 1)
header:
  teaser: https://images.unsplash.com/1/work-station-straight-on-view.jpg?ixlib=rb-1.2.1&q=85&fm=jpg&crop=entropy&cs=srgb&w=6000
  image: https://images.unsplash.com/1/work-station-straight-on-view.jpg?ixlib=rb-1.2.1&q=85&fm=jpg&crop=entropy&cs=srgb&w=6000
  og_image: https://images.unsplash.com/1/work-station-straight-on-view.jpg?ixlib=rb-1.2.1&q=85&fm=jpg&crop=entropy&cs=srgb&w=6000
category: 
  - machine learning
tags: 
  - machine learning
  - aprendizaje supervisado
  - regresión
  - clasificación
  - regularización
  - traducción
toc: true
---

Durante mi aprendizaje de `Machine Learning` todo lo que he escuchado, visto o leido sobre el tema ha sido en *inglés* en su mayoría. Esto me hizo pensar que no sabía realmente cómo referirme a todo lo relacionado con este campo en mi lengua materna, el español.

Por ello, voy a redactar una serie de posts con pequeños *glosarios* de términos relacionados con el `Aprendizaje Automático` que he ido aprendiendo hasta ahora, explicando brevemente lo que significa cada uno.

## General

- *Machine Learning* - Aprendizaje Automático

    Campo de estudio que da al ordenador la habilidad de aprender sin necesidad de escribir código explícito para ello.

- *Supervised Learning* - Aprendizaje supervisado

    Subtipo dentro del Aprendizaje automático. Para realizar el aprendizaje o predicción, utiliza un conjunto de datos de entrenamiento compuestos por datos de entrada (X) y el resultado deseado (y). 

- *Unsupervised Learning* - Aprendizaje no supervisado

    Subtipo dentro del Aprendizaje automático. Para realizar el aprendizaje o predicción, utiliza un conjunto de datos de entrenamiento compuesto solo por datos de entrada (X) sin proporcionar el resultado deseado (y).

- *Reinforcement learning* - Aprendizaje por refuerzo o reforzado

    Subtipo dentro del Aprendizaje automático. Para realizar el aprendizaje o predicción, se somete al ordenador a un proceso continuo de acciones y recompensas, con el fin de que aprenda una habilidad.

- *Training dataset* - Conjunto de datos de entrenamiento

- *Validation dataset* - Conjunto de datos de validación

- *Test dataset* - Conjunto de datos de prueba o testeo

<br/>

## Aprendizaje Supervisado

### Regresión

- *Regression* - Regresión

    Modelo de algoritmo de Aprendizaje Automático que predice valores de salida continuos.  

- *Linear Regression* - Regresión linear

    Modelo de regressión cuya representación gráfica es una línea recta continua basada en los valores estimados en la predicción. 

- *Polynomial regression* - Regresión polinómica

    Modelo de regressión cuya representación gráfica es una línea continua formada por una función polinómica y basada en los valores estimados en la predicción. 

- *Cost Function (J(θ))* - Función de coste

    Función que determina la diferencia entre el valor estimado en la predicción y el valor real. Cuanto menor sea el valor de la función de coste, mejor será el algoritmo predictivo.

- *Gradient Descent* - Gradiente descendente

    Método para minimizar el valor de la función de coste. El gradiente descendente busca los parámetros de la funcion de coste que minimicen la diferencia entre el valor estimado y el valor real.

- *Batch Gradient Descent* - Gradiente descendente por lotes

    En cada paso del gradiente descendente se evaluan todos los datos de entrada

- *Mini-Batch Gradient Descent* - Gradiente descendente por mini-lotes

    En cada paso del gradiente descendente se evalua un subgrupo dentro de los datos de entrada, acumulandose a lo largo del proceso completo.

- *Stochastic Gradient Descent* - Gradiente descendente estocástico o incremental

    En cada paso del gradiente descendente se evalua un caso de los datos de entrada, acumulandose a lo largo del proceso completo.

- *Learning Rate (α)* - Grado o ratio de aprendizaje

    Determina la velocidad de descenso del gradiente buscando el mínimo. Si es muy pequeño, la velocidad de convergencia sera menor . Si es muy grande, descenderá más rapido con la posibilidad de no converger o divergir.

- *Feature Scaling* - Ajuste de características o datos de entrada

    En el caso de un modelo con multiples variables de entrada, consiste en ajustar todas los variables a una misma escala. Las escalas más habituales son [0, 1] o [-1, 1]

- *Mean Normalization* - Normalización promedio

    Normalización o ajuste en el que las variables de entrada tienen una media de 0, por ejemplo el intervalo [-0.5, 0.5]. 

    Se calcula con la siguiente formula: 
    
    $$x = \dfrac{(x-\mu)}{\sigma}$$
    
    siendo <var>\mu<var> el valor medio de la variable sin normalizar, y <var>\sigma</var> la desviación estándar (rango entre el valor mínimo y el máximo). 
    
    <br/>

### Clasificación

- *Classification* - Clasificación

    Modelo de algoritmo de Aprendizaje Automático que predice valores de salida discretos.  

- Binary Classification - Clasifiación binaria

    Modelo de Clasificación con dos posibles valores discretos de salida. Por ejemplo *y = 0 e y = 1*.

- *Logistic Regression* - Regresión Logística

    Modelo de Clasificación Binaria cuya representación gráfica es una línea recta continua que separa los dos posibles valores discretos de salida.

- *Decision boundary* - Límite de decisión

    Función que delimita la separación entre los dos posibles valores discretos de salida. Si es linear, correspondería a Regresión Logística. 

- *Multi-class Classification* - Clasificación multiclase o con multiples datos de salida.

    Modelo de Clasificación con tres o más valores discretos de salida. Por ejemplo *y = 1*, *y = 2* e *y = 3*.

- *One-vs-all* - Uno frente a todos

    Método que separa los modelos de clasificación multiclase en todos los modelos de clasificación binaria posibles. Por ejemplo en el caso de *y = 1*, *y = 2* e *y = 3* se crearían: 
    - *y = 1* frente a *y = 2*, *y = 3*
    - *y = 2* frente a *y = 1*, *y = 3*
    - *y = 3* frente a *y = 1*, *y = 2.*

<br/>

### Regularización

- *Regularization* - Regularización

    Técnica del Aprendizaje Automático que optimiza la obtención de la función de coste ideal para un modelo predictivo. Depende del parámetro de regularización $\lambda$. Cuanto mayor sea $\lambda$, mayor efecto tendra la regularización.

- *Overfitting* - Sobreajuste o ajuste por encima

    Efecto que se produce cuando se entrena el modelo de Aprendizaje Automático ajustandolo en exceso a los datos de entrada. Esto provoca que pierda la habilidad de generalizar y falle cuando se comprueba el modelo con datos de validación y de prueba.

- *Underfitting* - Infraajuste o ajuste por debajo

    Efecto que se produce cuando se entrena el modelo de Aprendizaje Automático ajustándolo pobremente a los datos de entrada. Esto provoca que pierda mucha exactitud en la predicción tanto en los datos de entrenamiento como en datos de validación y de prueba.

- *High Bias / Low Bias* - Sesgo alto o bajo

    El sesgo de un modelo predictivo es la diferencia entre la predicción esperada del modelo y la predicción real. 

    - Si hay sesgo alto, el modelo es más simple y rígido. No se ajusta correctamente a los valores de los datos de entrenamiento y se produce un *infraajuste*.
    - Si hay sesgo bajo, el modelo es más flexible y se ajusta mejor a los valores reales.
    
- *High variance / Low Variance* - Varianza alta o baja

    La varianza de un modelo predictivo es la sensibilidad en la predicción del modelo en función de pequeñas variaciones en los datos de entrada.

    - Si hay varianza alta, el modelo es demasiado complejo y flexible. Se ajusta en exceso a los valores de entrenamiento y se produce un *sobreajuste*.
    - Si hay varianza baja, el modelo no es demasiado complejo y se ajusta en términos más generales a los valores de los datos de entrenamiento
    

---

Gracias por leer hasta aquí y, ¡nos vemos en el siguiente post!
