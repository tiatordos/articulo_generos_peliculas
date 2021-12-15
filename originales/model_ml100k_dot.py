import pickle
import pandas as pd
from math import pi
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import tensorflow as tf


def save_to_pickle(name, item):
    filehandler = open(name+".pkl", "wb")
    pickle.dump(item, filehandler)
    filehandler.close()


def plot_metric(history, metric, valida):
    train_metrics = history.history[metric]
    # if valida:
    #    val_metrics = history.history['val_'+metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics)
    if valida:
        val_metrics = history.history['val_' + metric]
        plt.plot(epochs, val_metrics)
        plt.title('Training and validation ' + metric)
    else:
        plt.title('Training ' + metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    if valida:
        plt.legend(["train_"+metric, 'val_' + metric])
    plt.show()


# ME FALTA PASAR COMO PARÁMETRO NUM_GENRES
def crear_modelo(K, LR, num_users, num_pelis, dim_users, dim_pelis, peso_score):
    user_input = tf.keras.layers.Input(shape=(dim_users,), name="input_usr")  # Variable-length sequence of ints
    usuarios_emb = tf.keras.layers.Embedding(input_dim=num_users, output_dim=K, name="embedding_usr")(user_input)
    usuarios = tf.keras.layers.Flatten(name="flatten_usr")(usuarios_emb)  # para pasar de (?,1,K) a (?,K)
    usr_emb_genr = tf.keras.layers.Dense(num_genres, activation='sigmoid', name="usr_dot")(usuarios)  # (?, num_genres)
    #usr_emb_genr = tf.keras.layers.Dense(num_genres, activation='tanh', name="usr_dot")(usuarios)

    pelis_input = tf.keras.layers.Input(shape=(dim_pelis,), name="input_pelis")  # Variable-length sequence of ints
    pelis_emb = tf.keras.layers.Embedding(input_dim=num_pelis, output_dim=K, name="embedding_pelis")(pelis_input)
    pelis = tf.keras.layers.Flatten(name="flatten_pelis")(pelis_emb)  # para pasar de (?,1,K) a (?,K)
    genre_pred = tf.keras.layers.Dense(num_genres, activation='sigmoid', name="genres")(pelis)  # (?, num_genres)
    # genre_pred = tf.keras.layers.Dense(num_genres, activation='tanh', name="genres")(pelis)  # (?, num_genres)

    score_pred = tf.keras.layers.Dot(axes=1, name="scores")([usr_emb_genr, genre_pred])

    model = tf.keras.Model(
        inputs=[user_input, pelis_input],
        outputs=[score_pred, genre_pred],
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR),
        # optimizer=keras.optimizers.RMSprop(1e-3),
        loss={
            "scores": tf.keras.losses.MeanSquaredError(),
            # "scores": keras.losses.MeanAbsoluteError(),
            "genres": tf.keras.losses.BinaryCrossentropy(),  # (from_logits=True),
        },
        loss_weights=[peso_score, 1 - peso_score],
        metrics=['mean_absolute_error', tf.keras.metrics.BinaryAccuracy()],
    )

    model_usr = tf.keras.Model(
        inputs=[user_input],
        outputs=[usr_emb_genr],
    )

    '''model2.summary()
    output_array2 = model2.predict([valoraciones.user_id, valoraciones.item_id])
    print(output_array2.shape)'''

    return model, model_usr


def entrenar(VALIDATION, K, epochs, batch_size, learning_rate, peso_score, num_users, num_pelis, dim_input_users,
             dim_input_pelis, X_train, X_train_dev, X_test, y_train, y_train_dev, y_test):
    # SE CREA EL MODELO
    model, mod_usr = crear_modelo(K, learning_rate, num_users, num_pelis, dim_input_users, dim_input_pelis, peso_score)

    # ENTRENAMOS EL MODELO Y EVALUAMOS EN EL CONJUNTO DE TEST
    if VALIDATION:
        # DECLARAMOS UN OBJETO EARLY STOPPING PARA DECIDIR LA MEJOR EPOCH
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
        # early_stop = EarlyStopping(monitor='val_scores_loss', patience=20)

        checkpoint = tf.keras.callbacks.ModelCheckpoint('model/model_T_{epoch:06d}.h5', save_best_only=True)
        history = model.fit(x=[X_train.user_id, X_train.item_id], y=[y_train.rating, y_train[col_items]],
                            validation_data=([X_dev.user_id, X_dev.item_id], [y_dev.rating, y_dev[col_items]]),
                            epochs=epochs, batch_size=batch_size, callbacks=[early_stop, checkpoint])
        print('Mejor epoch:', early_stop.stopped_epoch)
        media_en_train = y_train['rating'].mean()
    else:
        if epochs == 0:
            epochs = 1000
        checkpoint = tf.keras.callbacks.ModelCheckpoint('model/model_T+D_{epoch:06d}.h5')
        history = model.fit(x=[X_train_dev.user_id, X_train_dev.item_id],
                            y=[y_train_dev.rating, y_train_dev[col_items]],
                            epochs=epochs, batch_size=batch_size, callbacks=[checkpoint])
        media_en_train = y_train_dev['rating'].mean()
        usr_emb_g = mod_usr.predict([X_test.user_id])

    losses = model.evaluate([X_test.user_id, X_test.item_id], [y_test.rating, y_test[col_items]])
    print('\t*-* Test                         loss: %.4f' % losses[0])
    print('\t*-* Test                   score_loss: %.4f' % losses[1])
    print('\t*-* Test                  gender_loss: %.4f' % losses[2])
    print('\t*-* Test             MAE sobre scores: %.4f' % losses[3])
    print('\t*-* Test binary accuracy sobre genres: %.4f' % losses[6])

    print("excel: %.4f\t%.4f\t%.4f\t%.4f\t%.4f" % (losses[0], losses[1], losses[2], losses[3], losses[6]))

    prediction = model.predict([X_test.user_id, X_test.item_id])
    print('Test MAE    modelo: %.4f' % (sum(abs(y_test['rating'] - prediction[0].flatten())) / y_test.shape[0]))
    print('Test MAE Sys-Media: %.4f' % (sum(abs(y_test['rating'] - media_en_train)) / y_test.shape[0]))

    # GRÁFICOS
    # plot_metric(history, 'loss', valida=VALIDATION)
    # plot_metric(history, 'scores_loss', valida=VALIDATION)
    plot_metric(history, 'genres_loss', valida=VALIDATION)
    plot_metric(history, 'scores_mean_absolute_error', valida=VALIDATION)
    # plot_metric(history, 'genres_binary_accuracy', valida=VALIDATION)

    if VALIDATION:
        return model, early_stop.stopped_epoch
    else:
        return model, prediction, usr_emb_g


def radar_chart(line_out, df_train_dev, items, names, unames, pnames, prnames):
    print("-------------------------------------------------------------------------------------------------")
    print("----- radar_chart: genera el radar_chart ")
    print("-------------------------------------------------------------------------------------------------")
    num_user = line_out.user_id  #.iloc[line]
    num_peli = line_out.item_id  #.iloc[line]
    titulo_peli = items[items.item_id == num_peli].title.values[0]

    # number of variable
    categories = names  # list(df)[2:]  # para quitar user_id y rating de las etiquetas
    N = len(categories)

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Initialise the spider plot
    plt.figure(figsize=(6, 7))
    ax = plt.subplot(111, polar=True)

    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # Draw one axe per variable
    plt.xticks(angles[:-1], categories)

    # ------- PART 2: Add plots
    # Plot each individual = each line of the data
    # I don't make a loop, because plotting more than 3 groups makes the chart unreadable

    # Usuario
    # values_user = line_out[unames].iloc[line].values.tolist()
    values_user = line_out[unames].values.tolist()
    values_user += values_user[:1]
    ax.plot(angles, values_user, linewidth=1, linestyle='solid', label="Usuario - predicho")
    ax.fill(angles, values_user, 'b', alpha=0.1)

    # Predicción géneros película
    values_peli = line_out[pnames].values.tolist()
    values_peli += values_peli[:1]
    ax.plot(angles, values_peli, linewidth=1, linestyle='solid', label="Predicho - "+titulo_peli)
    ax.fill(angles, values_peli, 'r', alpha=0.1)

    # géneros película verdaderos
    true_values_peli = line_out[prnames].values.tolist()
    true_values_peli += true_values_peli[:1]
    ax.plot(angles, true_values_peli, linewidth=1, linestyle='solid', label="Real - "+titulo_peli)
    ax.fill(angles, true_values_peli, 'g', alpha=0.1)

    # Draw ylabels
    ax.set_rlabel_position(0)
    v = max(max(values_user), max(values_peli)) / 4
    plt.yticks([v, v*2, v*3], [str(round(v, 4)), str(round(v*2, 4)), str(round(v*3, 4))], color="grey", size=7)
    plt.ylim(0, max(max(values_user), max(values_peli)))

    # Add legend
    plt.legend(loc='lower left', bbox_to_anchor=(-0.15, -0.23))

    nota_media_users = df_train_dev['rating'].groupby(df_train_dev['user_id']).mean().reset_index()
    nota_media_pelis = df_train_dev['rating'].groupby(df_train_dev['item_id']).mean().reset_index()
    nota_media_usr = nota_media_users[nota_media_users.user_id == num_user].rating.values[0]
    nota_media_peli = nota_media_pelis[nota_media_pelis.item_id == num_peli].rating.values[0]
    titulo = 'Usuario ' + str(int(num_user)) + ' (nota media: ' + str(np.round(nota_media_usr, 2)) + ')\n' + \
             titulo_peli + ' (nota media: ' + str(np.round(nota_media_peli, 2)) + ')\n' + \
             ' Nota predicha: ' + str(np.round(line_out.rating, 2))

    plt.title(titulo)
    # Show the graph
    plt.show()


def topNrankings(N, line_out, names, unames, pnames, prnames, verbose=False):
    # line_out = df_out.iloc[7]
    usuario = line_out[unames].copy()
    usuario.rename(dict(zip(unames, names)), axis='columns', inplace=True)
    peli = line_out[pnames].copy()
    peli.rename(dict(zip(pnames, names)), axis='columns', inplace=True)
    peli_real = line_out[prnames].copy()
    peli_real.rename(dict(zip(prnames, names)), axis='columns', inplace=True)
    top5user = usuario.sort_values(ascending=False)[0:N]
    top5peli = peli.sort_values(ascending=False)[0:N]
    top5real = peli_real.sort_values(ascending=False)[0:N]
    inters_genres = top5user.index & top5peli.index
    if verbose:
        print("-------------------------------------------------------------------------------------------------")
        print("----- topNrankings: muestra los TopN y calcula la intersección ")
        print("-------------------------------------------------------------------------------------------------")
        print("Top", N, "de géneros del usuario según el modelo:")
        print(top5user)
        print("Top", N, "de géneros de la película según el modelo:")
        print(top5peli)
        print("Top", N, "de géneros de la película según MovieLens:")
        print(top5real)
        print("Intersección usuario-película en top", N, "según el modelo:")
        print(inters_genres)
        print("Número de elementos en la intersección:", len(inters_genres))
        print("Puntuación predicha (real) para la película: %.2f (%.0f)" % (line_out.rating, line_out.rat_real))
    return len(inters_genres)


def genres_bar_chart(genre_names, gen_reales, gen_predichos):
    x = np.arange(len(genre_names))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, gen_reales, width, label='Géneros reales')
    rects2 = ax.bar(x + width / 2, gen_predichos, width, label='Géneros predichos')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    # ax.set_ylabel('media')
    ax.set_title('Media de géneros de las películas')
    ax.set_xticks(x)
    ax.set_xticklabels(genre_names, rotation='vertical')
    ax.legend()

    #ax.bar_label(rects1, padding=3)
    #ax.bar_label(rects2, padding=3)

    fig.tight_layout()
    plt.show()


def compara_generos(df_out, pnames, prnames):
    dfo = df_out[pnames + prnames].groupby(df_out['item_id']).mean().reset_index()
    # genres_bar_chart(names, dfo[prnames].sum(), dfo[pnames].sum())
    genres_bar_chart(names, dfo[prnames].mean(), dfo[pnames].mean())
    print('Correlación: ', np.corrcoef(dfo[prnames].mean(), dfo[pnames].mean())[0,1])
    dfo[dfo[pnames] < 0.5] = 0
    dfo[dfo[pnames] >= 0.5] = 1
    genres_por_peli_pred = np.sum(dfo[pnames], axis=1)
    genres_por_peli_real = np.sum(dfo[prnames], axis=1)
    print("-------------------------------------------------------------------------------------------------")
    print("----- compara_generos: genera el barchart y muestra media de géneros por película")
    print("-------------------------------------------------------------------------------------------------")
    print("Media de géneros por película en test (real):", genres_por_peli_real.mean())
    print("Media de géneros por película en test (pred):", genres_por_peli_pred.mean())
    dfo['dif_genres'] = genres_por_peli_pred - genres_por_peli_real
    dfo.to_csv('dif_genres.csv', sep=';', decimal=',')


def explica(linea, df_out, df_train_dev, items, names, unames, pnames, prnames):
    print("-------------------------------------------------------------------------------------------------")
    print("----- explica: muestra qué géneros influyen más en la valoración")
    print("-------------------------------------------------------------------------------------------------")
    importancia_genres = np.multiply(df_out[unames], df_out[pnames])
    print("Valor predicho: %.2f" % (importancia_genres.iloc[linea].sum()))
    genres_ordenados = importancia_genres.iloc[linea].sort_values(ascending=False)
    print("Géneros que más aportan:")
    print(genres_ordenados[0:5])
    suma = genres_ordenados[0:5].sum()
    print("Aportan el %.2f%% de la nota predicha (%.2f de %.2f)" % (suma*100 / importancia_genres.iloc[linea].sum(),
                                                                    suma, importancia_genres.iloc[linea].sum()))
    # aportación de los géneros con predicción >=05
    mask = df_out.iloc[linea][pnames] >= 0.5
    gen_ge_05 = importancia_genres.iloc[linea, mask.values.tolist()]
    print("Aportación de los géneros >= 0.5:")
    print(gen_ge_05.sort_values(ascending=False))
    suma = gen_ge_05.sum()
    print("Aportan el %.2f%% de la nota predicha (%.2f de %.2f)" % (suma * 100 / importancia_genres.iloc[linea].sum(),
                                                                    suma, importancia_genres.iloc[linea].sum()))


'''def radar_chart_user_peli(df, df_pred, line, items):
    num_user = df.user_id[line]
    pred = df_pred[df_pred.user_id == num_user].sort_values('rating', ascending=False)

    titulo_peli = items[items.item_id == pred.iloc[0].item_id].title.values[0]
    # print(titulo_peli)

    # number of variable
    categories = list(df)[2:]  # para quitar user_id y rating de las etiquetas
    N = len(categories)

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Initialise the spider plot
    plt.figure(figsize=(6, 7))
    ax = plt.subplot(111, polar=True)
    # ax.set_title('unio\ndos\ntres')

    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # Draw one axe per variable
    plt.xticks(angles[:-1], categories)

    # ------- PART 2: Add plots
    # Plot each individual = each line of the data
    # I don't make a loop, because plotting more than 3 groups makes the chart unreadable

    # Usuario
    # values_user = df.iloc[line].drop(['user_id', 'item_id', 'rating']).values.flatten().tolist()
    values_user = df.iloc[line].drop(['user_id', 'rating']).values.flatten().tolist()
    values_user += values_user[:1]
    ax.plot(angles, values_user, linewidth=1, linestyle='solid', label="Usuario - Media")
    ax.fill(angles, values_user, 'b', alpha=0.1)

    # Predicción géneros película
    values_peli = pred.iloc[0].drop(['user_id', 'item_id', 'rating']).values.flatten().tolist()
    values_peli += values_peli[:1]
    ax.plot(angles, values_peli, linewidth=1, linestyle='solid', label="Usuarios - Predicho")
    # ax.plot(angles, values_peli, linewidth=1, linestyle='solid', label="predicho")
    ax.fill(angles, values_peli, 'r', alpha=0.1)

    # géneros película verdaderos
    true_values_peli = items.loc[items.item_id == pred.iloc[0].item_id, items.columns[2:]].values.flatten().tolist()
    true_values_peli += true_values_peli[:1]
    # titulo_peli = items[items.item_id == pred.iloc[0].item_id].title.values[0]
    ax.plot(angles, true_values_peli, linewidth=1, linestyle='solid', label="real - "+titulo_peli)
    ax.fill(angles, true_values_peli, 'g', alpha=0.1)

    # Draw ylabels
    ax.set_rlabel_position(0)
    v = max(max(values_user), max(values_peli)) / 4
    plt.yticks([v, v*2, v*3], [str(round(v, 4)), str(round(v*2, 4)), str(round(v*3, 4))], color="grey", size=7)
    # print([v, v*2, v*3])
    # print([str(v), str(v*2), str(v*3)])
    plt.ylim(0, max(max(values_user), max(values_peli)))

    # Add legend
    # plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.legend(loc='lower left', bbox_to_anchor=(-0.15, -0.23))

#    titulo = 'user ' + str(df.user_id[line]) + ' (nota media: ' + str(np.round(df.rating[line], 2)) + ') Película ' + \
#             str(int(pred.iloc[0].item_id)) + ' (nota predicha: ' + str(np.round(pred.iloc[0].rating, 2)) + ')'
    titulo = 'user ' + str(df.user_id[line]) + ' (nota media: ' + str(np.round(df.rating[line], 2)) + ')\n' + \
             titulo_peli + ' (nota predicha: ' + str(np.round(pred.iloc[0].rating, 2)) + ')'

    plt.title(titulo)
    # Show the graph
    plt.show()'''


# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------------------------------------------------

PATH = './cjtos/ml-100k/'

# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
# VALORACIONES
# ---------------------------------------------------------------------------------------------------------------------
valoraciones = pd.read_csv(PATH+'u.data', sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
print(valoraciones)
# quito el timestamp porque no me interesa
valoraciones.drop(['timestamp'], axis='columns', inplace=True)
# de momento dejo el rating como un problema de regresión
print(valoraciones)

num_ejem = valoraciones.shape[0]
num_users = valoraciones.user_id.max()+1  # 943, del 1 al 943. El +1 es porque empieza en 0
num_pelis = valoraciones.item_id.max()+1  # 1682, del 1 al 1682

# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
# ITEMS (PELÍCULAS)
# ---------------------------------------------------------------------------------------------------------------------
items = pd.read_csv(PATH+'u.item', sep='|', names=['item_id', 'title', 'RD', 'VRD', 'IMDB', 'unknown', 'Action',
                                                   'Adventure', 'Animation', 'Children', 'Comedy', 'Crime',
                                                   'Documentary', 'Drama', 'Fantasy', 'FilmNoir', 'Horror', 'Musical',
                                                   'Mystery', 'Romance', 'SciFi', 'Thriller', 'War', 'Western'])
print(items)
# de las películas me quedo sólo con los géneros de las mismas. El resto de campos los elimino
# dejo el item_id para comprobaciones
# titles = items.loc[:, ['item_id', 'title']]
items.drop(['RD', 'VRD', 'IMDB', 'unknown'], axis='columns', inplace=True)
print(items)

num_genres = items.shape[1]-2  # el -2 es para no contar el item_id ni el título. Son 18 géneros


# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
# PREPARANDO EL CONJUNTO DE DATOS
# ---------------------------------------------------------------------------------------------------------------------
cjto = valoraciones

# añado las columnas para el item ceros
col_items = items.columns.drop(['item_id', 'title'])
# col_items = col_items.drop(['item_id', 'title'])
cjto[col_items] = np.zeros((cjto.shape[0], col_items.shape[0]), dtype='int32')

# relleno los items con sus valores
for i in range(items.shape[0]):
    num_item = items.iloc[i].item_id
    if i % 20 == 0:
        print("item:", num_item)
    cjto.loc[cjto.item_id == num_item, col_items] = items.iloc[i, 2:].to_list()  # se salta item_id y title

# se almacenas en un fichero los ejemplos
# cjto.to_pickle('SPARSES/ML100K_CASOS_USO.pkl')

# X / y
X = cjto[['user_id', 'item_id']].copy()
y = cjto[col_items.insert(0, 'rating')].copy()

TEST = 0.10
DEV = 0.10

# TRAIN/DEV/TEST SPLIT (sin estratificar para prevenir errores)
X_train_dev, X_test, y_train_dev, y_test = train_test_split(X, y, stratify=None, test_size=TEST, random_state=2032)
X_train, X_dev, y_train, y_dev = train_test_split(X_train_dev, y_train_dev, stratify=None,
                                                  test_size=DEV, random_state=2032)

print("% test =", TEST, "\n% dev =", DEV)
print("X_train:", X_train.shape)
print("X_dev  :", X_dev.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_dev  :", y_dev.shape)
print("y_test :", y_test.shape)


# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
# ENTRENAMIENTO
# ---------------------------------------------------------------------------------------------------------------------
K = 32
learning_rate = 1e-3
dim_input_users = 1  # el entero a partir del cual se calcula el embedding
dim_input_pelis = 1  # el entero a partir del cual se calcula el embedding
tam_batch = 512
max_epochs = 1000
peso_score = 0.5  # valor entre 0 y 1.0

# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
# TRAIN -> DEV Y TEST
# ---------------------------------------------------------------------------------------------------------------------
VALIDATION = True
model, n_epochs = entrenar(VALIDATION, K, max_epochs, tam_batch, learning_rate, peso_score, num_users, num_pelis,
                           dim_input_users, dim_input_pelis, X_train, X_train_dev, X_test, y_train, y_train_dev, y_test)

# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
# TRAIN+DEV -> TEST
# ---------------------------------------------------------------------------------------------------------------------
VALIDATION = False
model, prediction, usr_emb_genres = entrenar(VALIDATION, K, n_epochs, tam_batch, learning_rate, peso_score, num_users,
                                        num_pelis, dim_input_users, dim_input_pelis,
                                        X_train, X_train_dev, X_test, y_train, y_train_dev, y_test)

save_to_pickle('ml100k_prediction', prediction)

# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------

# se crea un df con los datos de TRAIN+DEV [idusr, iditem, rating, genres]
df_train_dev = X_train_dev.copy()
df_train_dev[y_test.columns] = y_train_dev

# unos que hay en la predicción y en el test
print('UNOS en la predicción:', sum(sum(prediction[1] > 0.5)))
print('UNOS en el test:      ', sum(sum(y_test.values[:, 1:] > 0.5)))

# se crea un df con los datos de TEST [idusr, iditem, rating, genres]
# df_test = X_test.copy()
# df_test[y_test.columns] = y_test

# se crea un df con los datos PREDICHOS [idusr, iditem, rating, genres]
# df_pred = X_test.copy()
# df_pred['rating'] = prediction[0]
# df_pred[y_test.columns[1:]] = prediction[1]

# se crea un dataframe con toda la salida
df_out = X_test.copy()
df_out['rating'] = prediction[0]
df_out['rat_real'] = y_test['rating']
unames = ['u_' + g for g in y_test.columns[1:]]
df_out[unames] = usr_emb_genres  # esto es el mapeado de usuarios en el espacio de géneros
pnames = ['p_' + g for g in y_test.columns[1:]]
df_out[pnames] = prediction[1]  # esto es el mapeado de películas en el espacio de géneros
prnames = ['pr_' + g for g in y_test.columns[1:]]  # pr = películas real
df_out[prnames] = y_test[y_test.columns.drop('rating')]  # esto son los géneros de las películas según MovieLens

# lista con los nombre de los géneros
names = y_test.columns[1:]

# esto es para calcular la intersección de géneros en el top 5 y 10
# esto tarda un poco, así que lo comento y **sólo hay que descomentarlo para los RESULTADOS FINALES**
#df_out['intersecTop5'] = df_out.apply(lambda row: topNrankings(5, row, names, unames, pnames, prnames), axis=1)
#df_out['intersecTop10'] = df_out.apply(lambda row: topNrankings(10, row, names, unames, pnames, prnames), axis=1)
#df_out.to_csv('salida.csv', sep=';', decimal=',')


# hace el barchart
compara_generos(df_out, pnames, prnames)

EXPLICA = True
if EXPLICA:
    # Este código es para ver la recomendación que se hace a un usuario sobre las películas que ha valorado en el test
    # Se cogen las películas que ha valorado y se ordenan por el valor predicho por el modelo
    # Después se analiza y explica la recomendación de la película con mayor valoración (línea 0)
    num_usuario = 542  # 712 puede ser un buen ejemplo, 4 starwars, 8 pulpfiction, 9 elpacienteingles

    print("  ---***--- PREDICCIÓN PARA EL USUARIO %d ---***---" %(num_usuario))
    out_by_user = df_out[df_out.user_id == num_usuario].sort_values(by='rating', ascending=False)
    linea = 0  # en la línea 0 de out_by_user está la más valorada y por tanto la recomendada
    # radar_chart(df_out.iloc[linea], df_train_dev, items, names, unames, pnames, prnames)
    radar_chart(out_by_user.iloc[linea], df_train_dev, items, names, unames, pnames, prnames)
    topNrankings(5, out_by_user.iloc[linea], names, unames, pnames, prnames, verbose=True)
    explica(linea, out_by_user, df_train_dev, items, names, unames, pnames, prnames)

# esta es la media de géneros en las películas evaluadas por los usuarios
# NO tiene en cuenta las valoraciones
# gustos_usuarios_train = df_train_dev[y_test.columns].groupby(df_train_dev['user_id']).mean().reset_index()


# la predicción que se hace del género de las películas puede ser ligeramente diferente
# de los géneros reales. Eso haría que la proyección de los usuarios en el espacio de los
# géneros sirva para hacer predicciones mejores
# mostar: la proyección del usuario y la película en el espacio de géneros. Ahí se verá que
# géneros son importantes para el usuario a la hora de puntuar la película. Además se puede
# comparar el vector de géneros verdadero de la película con el predicho.
# Hay que ver cómo se hace para que el modelo mapee los usuarios en el espacio de los géneros
# para poder mostrarlos en la explicación

print('hiperparámetros: K=%d, batch_size=%d, learning_rate=%f, peso_score=%.2f' % (K, tam_batch, learning_rate,
                                                                                   peso_score))
print("FIN")
