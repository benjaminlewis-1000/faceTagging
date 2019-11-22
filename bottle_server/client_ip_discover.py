import socket
import json
import xmltodict
import os


# Source for getting my IP: https://stackoverflow.com/a/1267524/3158519
# (Unrolled the single liner)

def find_external_server():

    for ip in socket.gethostbyname_ex(socket.gethostname())[2]:
        if ip.startswith('127.'):
            for s in [socket.socket(socket.AF_INET, socket.SOCK_DGRAM)]:
                s.connect(('8.8.8.8', 53))
                my_ip = s.getsockname()[0]
                s.close()
                break
        else:
            my_ip = ip

    PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    client.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    IP = socket.gethostbyname_ex(socket.gethostname())[2]
    # Set a timeout so the socket does not block
    # indefinitely when trying to receive data.
    client.settimeout(0.2)
    client.bind(("0.0.0.0", 44444)) # This is my port
    message = bytes(my_ip.encode('utf-8'))
    data = {'return_port': 44444, 'ip_addr': my_ip}
    data_return = bytes(json.dumps(data).encode('utf-8'))

    with open(os.path.join(PARENT_DIR, 'parameters.xml')) as p:
        config = xmltodict.parse(p.read())
    port_ip_disc = int(config['params']['server']['port_ip_disc'])

    try:
        # Broadcast a message. 127.0.0.1 should be '<broadcast>'.
        client.sendto(data_return, ('127.0.0.1', port_ip_disc))
        # print("message sent!")
        data, addr = client.recvfrom(1024)
        data = json.loads(data)
        return_port = int(data['return_port'])
        return_ip = data['ip_addr']
        # print(data)
        return return_ip
    except socket.timeout:
        return None

if __name__ == "__main__":
    print(find_external_server())