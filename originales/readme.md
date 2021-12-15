## Ficheros origiales de Jorge
###EL MMODELO QUE FUNCIONA ESTÁ EN MODEL_ML100K_DOT.PY

##model_ml100k_dot_generos_estaticos.py
GEN en el artículo. Este prograna es el que hace el producto escalar con los géneros originales

###model_ml100k_MF.py
FM_oh en el artículo. Hace factorización de matrices con one-hot

###model_ml100k_MF2.py
FM_oh+g en el artículo. Hace factorización de matrices con one-hot pero añadiendo los géneros de las películas

###model_ml100k_dot.py
STE en el artículo. El nuestro

Lo que se pretende es mostrar que se puede hacer una recomendación que venga acompañada
de una explicación gráfica. En este caso se pretende utilizar gráficos radiales utilizando
los géneros. La idea es poder mostrar para cada usuario el gráfico radial de géneros que le
representa (a partir de las películas que ha puntuado) y cuando se le recomienda una película
se le podrá mostrar también el gráfico radial de géneros de esa película bajo el punto de vista
del usuario.

¿cómo se hace esto? Se aprende a partir de pares (usuario, película) representados por one-hot y
se pretende aprender de manera simultánea la valoración que da el usuario a la película y los géneros
de la película. Los géneros de entrada son un vector de ceros y unos y la predicción será un vector
de valores entre 0 y 1 que podemos considerar como la pertenencia a cada género de esa película
desde el punto de vista del usuario. Puede haber películas que tengan un 0 en acción y un 1 en aventuras
pero que bajo el punto de vista de usuario sean 0.3 y 0.9 respectivamente.
