import socket
import json
import xmltodict
import os
import re
from time import sleep 
import ipaddress

# Source for getting my IP: https://stackoverflow.com/a/1267524/3158519
# (Unrolled the single liner)

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class server_finder():
    def __init__(self, logger=None):

        # self.server_ip = None
        self.logger = logger
        with open(os.path.join(PARENT_DIR, 'parameters.xml')) as p:
            config = xmltodict.parse(p.read())

        self.port_ip_disc = int(config['params']['ports']['server_port_ip_disc'])
        self.client_port = int(config['params']['ports']['client_return_port'])

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
            valid_ips = []
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
                print(ret_data)
                # print(return_ip, return_found)
        if len(ret_data):
            ret_data = list(set(ret_data))
            print(ret_data)
            self.server_ips = [x[0] for x in ret_data]
        else:
            self.server_ips = None

    def check_ip(self, index=None):

        if index is not None:
            assert index in list(range(len(self.server_ips)))
            ips_check = [self.server_ips[index]]
        else:
            ips_check = self.server_ips

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

if __name__ == "__main__":
    s = server_finder()
    print("Found: ", s.server_ips)
    print("Still there: ", s.check_ip(0))
    print("Still there: ", s.check_ip(1))
    print("Still there: ", s.check_ip())
    print("Still there: ", s.check_ip())
