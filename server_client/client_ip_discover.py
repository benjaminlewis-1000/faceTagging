import socket
import json
import xmltodict
import os
import re
from time import sleep 
import ipaddress

# Source for getting my IP: https://stackoverflow.com/a/1267524/3158519s
# (Unrolled the single liner)

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class server_finder():
    def __init__(self, client_port=None, logger=None):

        # self.server_ip = None
        self.logger = logger
        with open(os.path.join(PARENT_DIR, 'parameters.xml')) as p:
            config = xmltodict.parse(p.read())

        self.port_ip_disc = int(config['params']['ports']['server_port_ip_disc'])
        if client_port is None:
            if 'CLIENT_FACE_PORT' in os.environ.keys():
                self.client_port = int(os.environ['CLIENT_FACE_PORT'])
            else:
                self.client_port = int(config['params']['ports']['client_return_port'])
        else:
            self.client_port = client_port
        print(self.client_port)

        self.get_my_ip()

        if self.logger is not None:
            self.logger.debug(f"My ip is {self.my_ip}")
        else:
            print(f"printing: My ip is {self.my_ip}")
        self.my_subnet = re.match('(\d+\.\d+\.\d+\.)\d+', self.my_ip).group(1)

        self.find_external_server()

    def get_my_ip(self):
        if 'IN_DOCKER' in os.environ.keys() and os.environ['IN_DOCKER']:
            self.my_ip = os.environ['DOCKER_HOST_IP']
        else:
            ip_cmd = os.popen("ip route | grep default | awk '{print $9}'")
            self.my_ip = ip_cmd.read().strip()
            if not re.match('\d+\.\d+\.\d+\.\d+', self.my_ip):
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("8.8.8.8", 80))
                self.my_ip = s.getsockname()[0]
                s.close()
                            

    def find_external_server(self):

        # print("Finding external GPU server...")

        client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        client.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        # IP = socket.gethostbyname_ex(socket.gethostname())[2]
        # Set a timeout so the socket does not block
        # indefinitely when trying to receive data.

        client.bind(("0.0.0.0", self.client_port)) # This is my port
        message = bytes(self.my_ip.encode('utf-8'))
        data = {'return_port': self.client_port, 'ip_addr': self.my_ip}
        data_return = bytes(json.dumps(data).encode('utf-8'))

        return_found = False

        def find_ip(timeout, sleeptime):
            # Get subnet of ip

            msg = f"Trying timeout value {timeout} with delay {sleeptime}."
            if self.logger is not None:
                self.logger.debug(msg)
            else:
                print(msg)
                
            client.settimeout(timeout)
            # client.setblocking(0)
            # sleep(1)
            # my_subnet = ipaddress.ip_network(my_ip + '/255.255.255.0', strict)
            for i in range(255):
                ip_test = f'{self.my_subnet}{i+1}'
                # print(ip_test) 
                try:
                    # Broadcast a message. 127.0.0.1 should be '<broadcast>'.
                    tmp = client.sendto(data_return, (ip_test, self.port_ip_disc))
                    # print(tmp)
                    data, addr = client.recvfrom(1024)
                    data = json.loads(data)
                    return_port = int(data['return_port'])
                    return_ip = data['ip_addr']
                    return_found = True
                    print(data)
                    return return_ip, return_port, return_found
                except socket.timeout:
                    pass

                # Needs a small time to... do something, not sure. I imagine
                # that it's flushing the queue from non-received data or something.
                if i % 25 == 0 and i > 0:
                    sleep(1000 * timeout)
            return None, None, None
        # return_ip, return_port, return_found = find_ip(0.0001, 0.1)
        # Try with a slower timeout
        for delay in [0.0001, 0.001, 0.005]:
            if not return_found:
                # print(delay)
                return_ip, return_port, return_found = find_ip(delay, delay * 500)
                # print(return_ip, return_found)
        if return_found:
            self.server_ip = return_ip
        else:
            self.server_ip = None

    def check_ip(self):

        if self.server_ip is None:
            return False

        client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)

        client.bind(("0.0.0.0", self.client_port)) # This is my port
        message = bytes(self.my_ip.encode('utf-8'))
        data = {'return_port': self.client_port, 'ip_addr': self.my_ip}
        data_return = bytes(json.dumps(data).encode('utf-8'))

        return_found = False

        try:
            client.settimeout(1)
            tmp = client.sendto(data_return, (self.server_ip, self.port_ip_disc))
            data, addr = client.recvfrom(1024)
            data = json.loads(data)
            return_port = int(data['return_port'])
            return_ip = data['ip_addr']
            return_found = True
        except socket.timeout:
            return_found = False

        return return_found

if __name__ == "__main__":
    s = server_finder()
    print("Found: ", s.server_ip)
    print("Still there: ", s.check_ip())
    print("Still there: ", s.check_ip())
    print("Still there: ", s.check_ip())
    print("Still there: ", s.check_ip())
