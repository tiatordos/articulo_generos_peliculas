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

    pelis_input = tf.keras.layers.Input(shape=(dim_pelis,), name="input_pelis")  # Variable-length sequence of ints
    pelis_emb = tf.keras.layers.Embedding(input_dim=num_pelis, output_dim=K, name="embedding_pelis")(pelis_input)
    pelis = tf.keras.layers.Flatten(name="flatten_pelis")(pelis_emb)  # para pasar de (?,1,K) a (?,K)

    capa_concat = tf.keras.layers.concatenate([usuarios, pelis])  # (?, k*2)

    score_pred = tf.keras.layers.Dense(1, name="scores")(capa_concat)  # (?, 1)
    # score_hidden = Dense(16, activation='relu', name="hidden_scores")(capa_concat)  # (?, 10)
    # score_pred = Dense(1, name="scores")(score_hidden)  # (?, 1)

    genre_pred = tf.keras.layers.Dense(num_genres, activation='sigmoid', name="genres")(capa_concat)  # (?, num_genres)

    model = tf.keras.Model(
        inputs=[user_input, pelis_input],
        outputs=[score_pred, genre_pred],
    )

    '''model2 = keras.Model(
        inputs=[user_input, pelis_input],
        # inputs=[user_input],
        # outputs=[capa_concat],
        outputs=[usuarios, pelis],
        # outputs=[usuarios]
    )
    model2.summary()
    output_array2 = model2.predict([valoraciones.user_id, valoraciones.item_id])
    print(output_array2.shape)'''

    # keras.utils.plot_model(model, "multi_input_and_output_model.png", show_shapes=True)

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

    return model


def entrenar(VALIDATION, K, epochs, batch_size, learning_rate, peso_score, num_users, num_pelis, dim_input_users,
             dim_input_pelis, X_train, X_train_dev, X_test, y_train, y_train_dev, y_test):
    # SE CREA EL MODELO
    model = crear_modelo(K, learning_rate, num_users, num_pelis, dim_input_users, dim_input_pelis, peso_score)

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
        checkpoint = tf.keras.callbacks.ModelCheckpoint('model/model_T+D_{epoch:06d}.h5')
        history = model.fit(x=[X_train_dev.user_id, X_train_dev.item_id],
                            y=[y_train_dev.rating, y_train_dev[col_items]],
                            epochs=epochs, batch_size=batch_size, callbacks=[checkpoint])
        media_en_train = y_train_dev['rating'].mean()

    losses = model.evaluate([X_test.user_id, X_test.item_id], [y_test.rating, y_test[col_items]])
    print('\t*-* Test                         loss: %.4f' % losses[0])
    print('\t*-* Test                   score_loss: %.4f' % losses[1])
    print('\t*-* Test                  gender_loss: %.4f' % losses[2])
    print('\t*-* Test             MAE sobre scores: %.4f' % losses[3])
    print('\t*-* Test binary accuracy sobre genres: %.4f' % losses[6])

    '''if VALIDATION:
        print('Mejor epoch:', early_stop.stopped_epoch)
        media_en_train = y_train['rating'].mean()
    else:
        media_en_train = y_train_dev['rating'].mean()'''

    prediction = model.predict([X_test.user_id, X_test.item_id])
    print('Test MAE    modelo: %.4f' % (sum(abs(y_test['rating'] - prediction[0].flatten())) / y_test.shape[0]))
    print('Test MAE Sys-Media: %.4f' % (sum(abs(y_test['rating'] - media_en_train)) / y_test.shape[0]))

    # GRÁFICOS
    plot_metric(history, 'loss', valida=VALIDATION)
    plot_metric(history, 'scores_loss', valida=VALIDATION)
    plot_metric(history, 'genres_loss', valida=VALIDATION)
    plot_metric(history, 'scores_mean_absolute_error', valida=VALIDATION)
    plot_metric(history, 'genres_binary_accuracy', valida=VALIDATION)

    if VALIDATION:
        return model, early_stop.stopped_epoch
    else:
        return model, prediction


def radar_chart_user_peli(df, df_pred, line, items):
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
    ax.plot(angles, values_user, linewidth=1, linestyle='solid', label="Usuario")
    ax.fill(angles, values_user, 'b', alpha=0.1)

    # Predicción géneros película
    values_peli = pred.iloc[0].drop(['user_id', 'item_id', 'rating']).values.flatten().tolist()
    values_peli += values_peli[:1]
    ax.plot(angles, values_peli, linewidth=1, linestyle='solid', label="predicho - "+titulo_peli)
    # ax.plot(angles, values_peli, linewidth=1, linestyle='solid', label="predicho")
    ax.fill(angles, values_peli, 'r', alpha=0.1)

    # géneros película verdaderos
    '''true_values_peli = items.loc[items.item_id == pred.iloc[0].item_id, items.columns[2:]].values.flatten().tolist()
    true_values_peli += true_values_peli[:1]
    # titulo_peli = items[items.item_id == pred.iloc[0].item_id].title.values[0]
    ax.plot(angles, true_values_peli, linewidth=1, linestyle='solid', label="real - "+titulo_peli)
    ax.fill(angles, true_values_peli, 'g', alpha=0.1)'''

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
    plt.show()


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
learning_rate = 1e-2
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
model, prediction = entrenar(VALIDATION, K, n_epochs, tam_batch, learning_rate, peso_score, num_users, num_pelis,
                             dim_input_users, dim_input_pelis,
                             X_train, X_train_dev, X_test, y_train, y_train_dev, y_test)

save_to_pickle('ml100k_prediction', prediction)

# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------

# se crea un df con los datos de TRAIN+DEV [idusr, iditem, rating, genres]
df_train_dev = X_train_dev.copy()
df_train_dev[y_test.columns] = y_train_dev

# se crea un df con los datos de TEST [idusr, iditem, rating, genres]
df_test = X_test.copy()
df_test[y_test.columns] = y_test

# se crea un df con los datos PREDICHOS [idusr, iditem, rating, genres]
df_pred = X_test.copy()
df_pred['rating'] = prediction[0]
df_pred[y_test.columns[1:]] = prediction[1]

gustos_usuarios = df_train_dev[y_test.columns].groupby(df_train_dev['user_id']).mean().reset_index()

radar_chart_user_peli(gustos_usuarios, df_pred, 3, items)

# unos que hay en la predicción y en el test
print('UNOS en la predicción:', sum(sum(prediction[1] > 0.5)))
print('UNOS en el test:      ', sum(sum(y_test.values[:, 1:] > 0.5)))

print('hiperparámetros: K=%d, batch_size=%d, learning_rate=%f, peso_score=%.2f' % (K, tam_batch, learning_rate,
                                                                                   peso_score))

print("FIN")
