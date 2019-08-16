import numpy as np
import pickle as pkl
from tensorflow import keras
import socket
import json


model = keras.models.load_model('/home/app/function/model/model.model')
f_ex = pkl.load(open('/home/app/function/model/model.f_ex'))


def predict(data):
	transform_data = f_ex.transform(data)
	return {"median_price" : float(np.exp(model.predict(transform_data))[0][0])}

server = socket.socket()
port = 12345
server.bind(('', port))
server.listen(5)
while True:
   client, addr = server.accept()
   client.send(json.dumps(predict(json.loads(client.recv(1024)))))
   client.close()