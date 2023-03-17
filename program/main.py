import tensorflow as tf
import numpy as np

model=tf.keras.models.load_model("model/model.h5")
temperature=input("temperature=")
precipitation=input("precipitation(estimated)=")

pred = np.argmax(model.predict([[float(temperature),float(precipitation)]]))

# 0: it's hot you don't need to wear anything but tshirt
# 1: wear something light
# 2: wear raincoat
# 3: it seems cold, wear coat
# 4: it's freezing cold, bundle up

dic={"0":"çok sıcak üstüne bir şey almana gerek yok", "1":"üstüne ince bir şey al",
    "2":"üstüne yağmurluk al", "3":"hava soğuk görünüyor üstüne mont al", "4":"hava buz gibi sıkı sıkı giyin"}
print(dic[f"{pred}".format(pred=pred)])