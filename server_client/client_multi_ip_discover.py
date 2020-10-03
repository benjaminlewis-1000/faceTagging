import socket
import json
import xmltodict
import os
import re
from time import sleep 
import ipaddress
import random

# Source for getting my IP: https://stackoverflow.com/a/1267524/3158519
# (Unrolled the single liner)

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class server_finder():
    def __init__(self, logger=None):

        # self.server_ip = None
        # print("Init of server finder")
        self.logger = logger
        with open(os.path.join(PARENT_DIR, 'parameters.xml')) as p:
            config = xmltodict.parse(p.read())

        self.port_ip_disc = int(config['params']['ports']['server_port_ip_disc'])
        # print(f"Port is {self.port_ip_disc}")
        if 'CLIENT_FACE_PORT' in os.environ.keys():
            self.client_port = int(os.environ['CLIENT_FACE_PORT'])
        else:
            self.client_port = int(config['params']['ports']['client_return_port'])
        # print(self.client_port)

        self.get_my_ip()
        # print(f"My ip is {self.my_ip}")

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
                
            # print(msg)

            client.settimeout(timeout)
            # client.setblocking(0)
            # sleep(1)
            # my_subnet = ipaddress.ip_network(my_ip + '/255.255.255.0', strict)
            valid_ips = []
            for i in range(255):
                ip_test = f'{self.my_subnet}{i+1}'
                try:
                    # Broadcast a message. 127.0.0.1 should be '<broadcast>'.
                    tmp = client.sendto(data_return, (ip_test, self.port_ip_disc))
                    # print(tmp)
                    data, addr = client.recvfrom(1024)
                    data = json.loads(data)
                    return_port = int(data['return_port'])
                    return_ip = data['ip_addr']
                    return_found = True
                    # print(data, return_ip, return_port)
                    # return return_ip, return_port, return_found
                    valid_ips.append((return_ip, return_port))
                except socket.timeout:
                    pass


                # Needs a small time to... do something, not sure. I imagine
                # that it's flushing the queue from non-received data or something.
                if i % 25 == 0 and i > 0:
                    sleep(1000 * timeout)
            return valid_ips
            return None, None, None
        # return_ip, return_port, return_found = find_ip(0.0001, 0.1)
        # Try with a slower timeout
        ret_data = []
        for delay in [0.0001, 0.001]:
            if len(ret_data) == 0:
                # print(delay)
                # return_ip, return_port, return_found = 
                ret_data = find_ip(delay, delay * 500)
                # print(return_ip, return_found)
        if len(ret_data):
            ret_data = list(set(ret_data))
            self.server_ips = [x[0] for x in ret_data]
        else:
            self.server_ips = None


    def check_ip(self, ip_address=None):

        if isinstance(ip_address, int):
            ip_address = self.server_ips[ip_address]

        if ip_address is not None:
            assert ip_address in self.server_ips
            ips_check = [ip_address]
        else:
            ips_check = self.server_ips

        # print(ips_check)

        if self.server_ips is None:
            return False

        client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)

        client.bind(("0.0.0.0", self.client_port)) # This is my port

        message = bytes(self.my_ip.encode('utf-8'))
        data = {'return_port': self.client_port, 'ip_addr': self.my_ip}
        data_return = bytes(json.dumps(data).encode('utf-8'))

        return_found = []

        for s in ips_check:
            try:
                client.settimeout(1)
                tmp = client.sendto(data_return, (s, self.port_ip_disc))
                data, addr = client.recvfrom(1024)
                data = json.loads(data)
                return_port = int(data['return_port'])
                return_ip = data['ip_addr']
                return_found.append(True)
            except socket.timeout:
                return_found.append(False)

        return return_found

    def __len__(self):
        return len(self.server_ips)

if __name__ == "__main__":
    s = server_finder()
    print("Found: ", s.server_ips)
    print("Still there: ", s.check_ip('192.168.1.16'))
#    print("Still there: ", s.check_ip(0))
#    print("Still there: ", s.check_ip(1))
#    print("Still there: ", s.check_ip())
#    print("Still there: ", s.check_ip())
