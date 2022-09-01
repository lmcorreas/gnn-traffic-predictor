# gnn-traffic-predictor

# Predicción de tráfico mediante redes neuronales basadas en grafos 

Máster Universitario en Ingeniería y Ciencia de Datos, UNED.

Trabajo de fin de máster.

Septiembre de 2022.

**Autor: Luis Manuel Correas Ramos**

**Tutora: Rocío Muñoz Mansilla**

**Co-tutor: Agustín Carlos Caminero Herráez**

## Resumen

La predicción del tráfico ha tornado a ser una tarea crucial con multitud de intereses tanto en aspectos económicos como de salud, más aún con el rápido crecimiento en el nivel de circulación debido al incremento del tamaño de las grandes ciudades. En este trabajo se propone un sistema de predicción del nivel de tráfico mediante la propagación de información en una red de carreteras compleja, como lo es la ciudad de Madrid.

A través de una simple entrada que permita identificar un instante en el futuro y condiciones básicas de la naturaleza de ese día, el sistema ofrecerá una estimación del nivel de densidad circulatoria en cada uno de los arcos que conforman la red.

El sistema propuesto es sencillo y extensible, permitiendo añadir nuevas características tanto a la entrada como a la salida, simplemente incluyéndolas en la fase de procesado de la información y ajustando los parámetros que marcan la estructura de la red.

Constará de dos partes principales y diferenciadas por su cometido y estructura: la primera encargada de procesar los valores que identifican un punto en el tiempo y ofrecer una estimación del uso de vía para cada arco en el sistema; la segunda, en su lugar, llevará a cabo la propagación espacial del estado de los arcos usando una red neuronal aplicada a grafos, con el fin de estabilizar las estimaciones incluso en los arcos donde no existe un medidor físico de tráfico. De este modo, es posible aprender de las tendencias y relaciones de ocupación de vía entre nodos y arcos vecinos, condicionadas por las características físicas de cada carretera.

La fase de aprendizaje donde se hace uso de una red neuronal aplicada a grafos ha sido diseñada mediante el uso de un framework que puede adaptarse a varios backends de aprendizaje automático, otorgando una mayor versatilidad a la aplicación y la posibilidad de computación mediante unidades de procesamiento de gráficos o sistemas distribuidos. Este sistema permite el paso de mensajes entre nodos y arcos según su conectividad y, de este modo, lograr el aprendizaje basado en la cercanía entre elementos. Gracias a esto, tiene la habilidad de ofrecer estimaciones dadas en concordancia para el resto de puntos de la red, aunque no dispongan de un medidor y por lo tanto no haya un valor de referencia con el que entrenarlos.

El sistema ha sido capaz de demostrar una precisión razonablemente buena, con un error medio inferior a 10 puntos, aun contando con información muy básica como entrada al sistema y, por tanto, pocos datos con los que tomar decisiones. Además, en pruebas de validación de la propagación espacial, tras ser retiradas la mediciones del 10% de los arcos de la red, el incremento del error global es muy pequeño, subiendo hasta los 12-13 puntos de separación media para los puntos de medición inhibidos.

## Keywords

Predicción de tráfico, Madrid, redes neuronales, python, OpenStreetMap, Deep Graph Library, MXNet, Pytorch, grafo, GNN, PEMS-BAY.