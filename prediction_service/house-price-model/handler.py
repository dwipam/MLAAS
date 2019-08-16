import socket

client = socket.socket()
port = 12345
client.connect(('', port))

def handle(req):
    """handle a request to the function
    Args:
        req (str):
        request body
    """
    client.send(req)
    return client.recv(1024)