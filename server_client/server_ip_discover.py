import socket
import json
import os
import xmltodict

# Source for getting my IP: https://stackoverflow.com/a/1267524/3158519
# (Unrolled the single liner)

def ip_responder():

    PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    for ip in socket.gethostbyname_ex(socket.gethostname())[2]:
        if ip.startswith('127.'):
            for s in [socket.socket(socket.AF_INET, socket.SOCK_DGRAM)]:
                s.connect(('8.8.8.8', 53))
                my_ip = s.getsockname()[0]
                s.close()
                break
        else:
            my_ip = ip


    with open(os.path.join(PARENT_DIR, 'parameters.xml')) as p:
        config = xmltodict.parse(p.read())
    port_ip_disc = int(config['params']['ports']['server_port_ip_disc'])

    server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # UDP
    server.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    server.bind(("0.0.0.0", port_ip_disc))

    data = {'return_port': port_ip_disc, 'ip_addr': my_ip}
    data_return = bytes(json.dumps(data).encode('utf-8'))

    while True:
        # Sit and wait forever for a message. 
        data, addr = server.recvfrom(1024)
        # Break down the JSON of the message to get port and IP. 
        data = json.loads(data)
        return_port = int(data['return_port'])
        return_ip = data['ip_addr']
        print("received message: %s"%data)
        # Respond with our own information to the
        # IP and port that was sent to us.
        server.sendto(data_return, (return_ip, return_port))

if __name__ == '__main__':
    ip_responder()