import socket
from threading import Thread
from urllib.parse import urlparse, parse_qs
import json
import pickle
import requests
from detector.sql_detector import detect_sql_injection
from detector.sql_detector import BertLSTMClassifier

class ProxyServer:
    def __init__(self, host, port, target_host, target_port, report_host, report_port):
        self.host = host
        self.port = port
        self.target_host = target_host
        self.target_port = target_port
        self.report_host = report_host
        self.report_port = report_port

        # Load the model
        model_path = "../pickles/sqli_detector.pkl"
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

    def handle_client(self, client_socket):
        request = client_socket.recv(4096)

        # Extract the query part of the request
        parsed_url = urlparse(request.decode().split('\r\n')[0].split()[1])
        query_params = parse_qs(parsed_url.query)

        # Prepare report data
        client_ip, client_port = client_socket.getpeername()
        request_path = parsed_url.path

        # Initialize status as accepted
        status = 'accepted'
        payload = ""  # Initialize payload

        # Check for SQL injection
        for values in query_params.values():
            for value in values:

                if value:
                    res = detect_sql_injection(value, self.model)
                    payload = value  # Update payload with the detected value
                    if res == 1:
                        print(f"Rejected the request with SQL Injection: {value}")
                        status = 'rejected'  # Update status if SQL injection detected

                        client_socket.sendall(b"HTTP/1.1 403 Forbidden\r\n\r\nRejected due to SQL Injection")
                        break
                    else:
                        print(f"SQL Injection Detected for {value}: {res}")

        # Prepare report data
        report_data = {
            'client_ip': client_ip,
            'client_port': client_port,
            'destination_port': self.target_port,
            'request_path': request_path,
            'status': status , # Set status based on whether SQL injection is detected or not
            'payload': payload  # Include payload in report data
        }

        # Send report to Flask server
        try:
            requests.post(f'http://{self.report_host}:{self.report_port}/report', json=report_data)
        except Exception as e:
            print(f"Error sending report to Flask server: {e}")

        # Forward the modified request to the target server
        modified_request = request.replace(b'altoro.testfire.net/search.jsp?query=' + value.encode(),
                                           self.target_host.encode())
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            server_socket.connect((self.target_host, self.target_port))
            server_socket.sendall(modified_request)
            response = server_socket.recv(4096)
            client_socket.sendall(response)

    def start(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as proxy_socket:
            proxy_socket.bind((self.host, self.port))
            proxy_socket.listen(5)
            print(f"Proxy server listening on {self.host}:{self.port}...")

            while True:
                client_socket, addr = proxy_socket.accept()
                print(f"Accepted connection from {addr[0]}:{addr[1]}")
                client_handler = Thread(target=self.handle_client, args=(client_socket,))
                client_handler.start()

if __name__ == "__main__":
    proxy = ProxyServer('127.0.0.1', 8090, 'demo.testfire.net', 80, '127.0.0.1', 5000)
    proxy.start()
