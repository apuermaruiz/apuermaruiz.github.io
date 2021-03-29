---
title: Cómo hice en Python el curso de Machine Learning de Andrew Ng - Ejercicio 1
header:
  teaser: /images/header-ml-ex1.png
  image: /images/header-ml-ex1.png
  og_image: /images/header-ml-ex1.png
category: 
  - machine learning
tags: 
  - python
  - machine learning
  - linear regression
  - cost funcion
  - gradient descent
  - jupyter notebook
toc: true
---

Mi camino en el aprendizaje de Machine Learning comenzó con el curso de Andrew Ng de Coursera, tal y como comenté el post [*Como empecé a aprender Machine Learning*](https://apuerma.github.io/vivencia/como-aprendi-machine-learning/) del blog.

Este curso está dividido en bloques semanales que contienen videos explicativos y ejercicios evaluables escritos para realizarse en `Octave` o `Matlab`. 

Actualmente uno de los lenguajes más popular para desarrollar los modelos de Machine Learning es `Python`. Por lo tanto, me he propuesto volver a realizar estos ejercicios evaluables, pero en esta ocasión en `Python` y usando los ficheros `Jupyter Notebook` que facilitan el desarrollo de una memoria escrita añadiendo líneas de código y su resultado.

Puedes ver el resultado [aquí](https://github.com/apuerma/python-machine-learning) en mi repositorio de Github.

---

# Ejercicio 1 - Regresión Linear

El primer ejercicio del curso consiste en implementar modelos de regresión linear con una variable y con multiples variables, así como visualizarlos y como se comportan respecto a los datos de entrada.

## Regresión Linear con una variable

En este primer apartado del ejercicio, hay que implementar un modelo de regresión linear con una variable para predecir los beneficios de un comión de comidas de una franquicia de restaurantes. La cadena tiene varios camiones por varias ciudades y tiene datos de beneficios en función de las poblaciones. 

### Visualizar los datos

En primer lugar voy a visualizar los datos que tengo almacenados en el archivo que me proporcionan. La primera columna es la población de la ciudad y la segunda, el beneficio (si es positivo) o la pérdida (si es negativo).

|     | population | profit  |
|:----|:----------:|:-------:|
| 1   | 6.1101	   | 17.5920 |
| 2   | 5.5277     | 9.1302  |
| 3   | 8.5186     | 13.6620 |
| 4   | 7.0032     | 11.8540 |
| 5   | 5.8598     | 6.8233  |

![ml-ex1-image1]({{ site.url }}{{ site.baseurl }}/images/ml-ex1-image1.png){: .align-right}

Con las siguientes líneas de código, usando `matplotlib` y una gráfica de dispresión, podemos representar los datos proporcionados:

```python
data.plot(kind='scatter', x='population', y='profit', figsize=(10,6), color='red')

plt.title('Profits by Population')
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')

plt.show()
```


### Función de coste

El objetivo de la regresión linear es minimizar la función de coste J(𝜃), es decir, minimizar la diferencia entre los datos reales y la hipótesis, tal y como muestran las siguientes ecuaciones:

![ml-ex1-image2]({{ site.url }}{{ site.baseurl }}/images/ml-ex1-image2.png)

Para tener en cuenta el valor de 𝜃<sub>0</sub>, es necesario añadir a X una columna adicional de unos al principio. Por tanto las variables `X` e `y` quedarían de la siguiente manera:

```python
X = data['population'].to_numpy()
X = np.stack([np.ones(m), X], axis=1)
y = data['profit'].to_numpy().reshape((m, 1))
```

Para calcular la función de coste en `Python`, dados los parámetros de entrada `X`, `y` y `theta` la función `computeCost` devuelve `J` como el valor de la función de coste para esos valores:

```python
def computeCost(X, y, theta):
	# number of training examples  
	m = len(y)
	h = np.dot(X, theta)
	J = (1/(2*m)) * np.sum(np.square(h - y))

	return J
```

Para comprobar si está correcto, en el ejercicio original del curso prueban con dos valores de 𝜃 y los valores concuerdan.

| 𝜃        | coste calculado | coste esperado |
|:--------:|:---------------:|:--------------:|
| [0 ; 0]  | 32.0727	       | 32.07          |
| [-1 ; 2] | 54.2425         | 54.24          |


### Gradiente Descendente

La finalidad del gradiente descendente es ayudar a minimizar la función de coste ajustando los valores de 𝜃 mediante la repetición de actualizaciones de los valores de 𝜃, según la siguiente ecuación:

![ml-ex1-image3]({{ site.url }}{{ site.baseurl }}/images/ml-ex1-image3.png)

En cada iteración del gradiente descendente se actualizan simultaneamente todos los valores de 𝜃 y el valor de la función de coste en cada caso se va acercando al valor mínimo.  

Para implementar el proceso del gradiente descendente, dados los parámetros `X`, `y`, `theta`, `alpha` e `iterations` la función `gradientDescent` devuelve `theta`, con los últimos valores de 𝜃, y `J_history`, con el registro de todos los valores de la función de coste a lo largo del gradiente.

```python
def gradientDescent(X, y, theta, alpha, iterations):
	# number of training examples  
	m = len(y)                               
  # cost over the iterations
	J_history = np.zeros((iterations, 1))   
  # Make a copy of theta, which will be updated by gradient descent
  theta = theta.copy()

  for i in range(iterations):
    # Perform a single gradient step on the parameter vector theta. 
    h = np.dot(X, theta)
    t0 = theta[0] - ( alpha * (1/m) * np.sum(h - y) )
    t1 = theta[1] - ( alpha * (1/m) * np.sum(np.dot( X[:,1].T , (h - y) ) ) )
    theta[0] = t0
    theta[1] = t1
    # Save the cost J in every iteration    
    J_history[i,:] = computeCost(X, y, theta)

  return theta, J_history
```

Para comprobar si está correcto, en el ejercicio original inicializan los valores de 𝜃 a cero, α a 0.01 y el número de iteraciones a 1500. Después de realizar el gradiente descendente, el valor de la función de coste es **4.4834**, mucho menor que el valor anterior de **32.0727**. Asimismo, los valores de 𝜃 que se esperan y los obtenidos concuerdan.

| 𝜃 calculada | 𝜃 esperada     |
|:-----------:|:--------------:|
| -3.6303     | -3.6303	       | 
| 1.1664      | 1.1664         | 

### Visualizar el ajuste linear

![ml-ex1-image4]({{ site.url }}{{ site.baseurl }}/images/ml-ex1-image4.png){: .align-right}

Para visualizar el ajuste de modelo de regresión superponemos la función linear a los datos de entrada, y quedaría de la siguiente manera:

```python
# Plot the data in a scatter plot
data.plot(kind='scatter', x='population', y='profit', figsize=(10,6), color='red')
plt.title('Profits by Population')
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')

# Plot the linear fit over the scatter plot
h = np.dot(X, theta)
plt.plot(X[:,1], h, color='blue')

plt.show()
```

### Predecir nuevos valores

Una vez que tenemos el modelo de regresión linear podemos predecir valores de beneficios con nuevos datos de población. En el ejercicio original, nos piden calcular la predicción de beneficios en áreas de 35000 y 70000 personas. Con estos datos y los valores de 𝜃 obtenidos, podemos calcular las predicciones de la siguiente forma:

```python
# Predict values for population sizes of 35,000 and 70,000
predict1 = np.dot([1, 3.5] , theta) * 10000
predict2 = np.dot([1, 7] , theta) * 10000
```

| población  | predicción   |
|:----------:|:------------:|
| 35000      | 4519.7679    | 
| 70000      | 45342.4501   | 


### Visualizar la función de coste

Para entender mejor la función de coste, se puede representar en función de los valores de 𝜃, tanto en dos como en 3 dimensiones, utilizando las herramientas de `surface` y `contour` de `matplotlib`. Puedes ver cómo lo he implementado con más detalle [aquí](https://github.com/apuerma/python-machine-learning/blob/main/ML_Exercise_1_Linear_Regression.ipynb) en mi repositorio en Github.

![ml-ex1-image5]({{ site.url }}{{ site.baseurl }}/images/ml-ex1-image5.png){: .align-right}

---

Hasta aquí la primera parte del primer ejercicio del curso de Machine Learning de Andrew Ng. La primera vez que realicé este ejercicio aprendí muchísimo y me enfrenté a pequeñas dificultades, tanto en lo matemático como en la programación para `Octave`. En esta ocasión, si bien los conocimientos estaban mucho más claros, también me he enfrentado a pequeños obstáculos al adaptar el código a Python. No obstante, todos ellos me han servido para afianzar mis habilidades programando en `Python`.

Proximamente iré implementando en `Python` el resto de ejercicios del curso y escribiré nuevos post sobre ellos.

---

Gracias por leer hasta aquí y, ¡nos vemos en el siguiente post!
