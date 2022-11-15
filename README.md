# Hack For Good 2022 Granada
### _Identificación única para combatir la desnutrición_

---
### :pushpin: Introducción
Código implementado y documentos generados durante la participación del Hackatón [HackForGood 2022](https://hackforgood.net/) durante los días 21-22 de octubre de 2022, donde se obtuvo el 2º premio local en Granada.

Se planteó una solución al reto [Identificación única para combatir la desnutrición](https://hackforgood.net/2022/10/15/identificacion-unica-para-combatir-la-desnutricion/) en la que se plantea identificar a las personas que carezcan de un carnet o tarjeta de identidad por medio de diferentes partes del cuerpo, siendo las más relevantes el iris, las plantas de las manos y los pies. 

Con el objetivo de determinar la viabilidad de la propuesta se desarrolló un pequeño ejemplo para la identificación de irises, es decir, saber si un iris es de la misma persona o de una diferente. 

Se desarolló una red siamesa sobre el framework de [Keras](https://keras.io/), las imágenes de los iris utilizadas fueron extraídas de la base de datos pública [MMU2 Iris](https://www.kaggle.com/datasets/naureenmohammad/mmu-iris-dataset). Una vez entrenada y validada la red, la inferencia se realizó por medio de una interfaz web simple desarrollada con [Streamlit](https://docs.streamlit.io/).

La presentación de la propuesta puede observarse [aquí](https://github.com/RhinoBlindado/hackforgood-grx-2022/blob/main/docs/presentacion-identificacion-unica.pdf), y el vídeo [aquí](https://youtu.be/UQNsCV2DFLc).

### :busts_in_silhouette: Equipo
El equipo estuvo formado por [Alejandro Alonso](https://github.com/aalonso99), [Pilar Navarro](https://github.com/pilarnavarro) y [Valentino Lugli](https://github.com/RhinoBlindado).

### :gear: Ejecución
El código del entrenamiento se encuentra desarrollado en una libreta de Google Colab por lo que solo es necesario subir este fichero a Google Drive, es necesario subir también los datos que se encuentran en el fichero `.zip`. Una vez en la libreta, se debe de modificar la ruta de acceso a los datos.

Para la inferencia se debe de iniciar el servicio de Streamlit con el fichero `web_interface.py`.