"""Launch the inference server."""

import os
import sys

from sglang.srt.server import launch_server
from sglang.srt.server_args import prepare_server_args
from sglang.srt.utils import kill_process_tree

import nacos
import configparser
from urllib.parse import urlparse

if __name__ == "__main__":
    server_args = prepare_server_args(sys.argv[1:])
    def register_service(client,service_name,service_ip,service_port,cluster_name,health_check_interval,weight,http_proxy,domain,protocol,direct_domain):
        try:
            # 初始化 metadata
            metadata = {
                "model": server_args.served_model_name,
                "port": server_args.port
            }
            
            # 如果 http_proxy 为 True，添加额外的 metadata 键值对
            if http_proxy:
                metadata["http_proxy"] = True
                if direct_domain:
                    metadata["domain"] = f"{protocol}://{service_ip}:{service_port}"
                else:
                    metadata["domain"] = f"{domain}/port/{service_port}"
            else:
                metadata["http_proxy"] = False
                metadata["domain"] = f"{protocol}://{service_ip}:{service_port}"
            response = client.add_naming_instance(
                service_name,
                service_ip,
                service_port,
                cluster_name,
                weight,
                metadata,
                enable=True,
                healthy=True,
                ephemeral=True,
                heartbeat_interval=health_check_interval
            )
            return response
        except Exception as e:
            print(f"Error registering service to Nacos: {e}")
            return True
    try:
        # 创建配置解析器
        config = configparser.ConfigParser()
        # 读取配置文件
        if not config.read('config.ini'):
            raise RuntimeError("配置文件不存在")
        # Nacos server and other configurations
        NACOS_SERVER = config['nacos']['nacos_server']
        NAMESPACE = config['nacos']['namespace']
        CLUSTER_NAME = config['nacos']['cluster_name']
        client = nacos.NacosClient(NACOS_SERVER, namespace=NAMESPACE, username=config['nacos']['username'], password=config['nacos']['password'])
        SERVICE_NAME = config['nacos']['service_name']
        HEALTH_CHECK_INTERVAL = int(config['nacos']['health_check_interval'])
        if config.has_option('nacos', 'weight'):
            WEIGHT = int(config.get('nacos', 'weight'))
        else:
            WEIGHT = 1
        HTTP_PROXY = config.getboolean('server', 'http_proxy')
        DOMAIN = config['server']['domain']
        PROTOCOL = config['server']['protocol']
        DIRECT_DOMAIN = config['server']['direct_domain']
        # Parse AutoDLServiceURL
        autodl_url = os.environ.get('AutoDLServiceURL')
        AutoDLContainerUUID = os.environ.get('AutoDLContainerUUID')
        if not autodl_url:
            raise RuntimeError("Error: AutoDLServiceURL environment variable is not set.")

        parsed_url = urlparse(autodl_url)
        SERVICE_IP = parsed_url.hostname
        SERVICE_PORT = parsed_url.port
        if not SERVICE_IP or not SERVICE_PORT:
            raise RuntimeError("Error: Invalid AutoDLServiceURL format.")

        print(f"Service will be registered with IP: {SERVICE_IP} and Port: {SERVICE_PORT}")
        if not register_service(client,SERVICE_NAME,SERVICE_IP,SERVICE_PORT,CLUSTER_NAME,HEALTH_CHECK_INTERVAL,WEIGHT,HTTP_PROXY,DOMAIN,PROTOCOL,DIRECT_DOMAIN):
            raise RuntimeError("Service is healthy but failed to register.")
        
        launch_server(server_args)
    finally:
        try:
            client.remove_naming_instance(SERVICE_NAME, SERVICE_IP, SERVICE_PORT, CLUSTER_NAME)
        finally:
            pass 
        kill_process_tree(os.getpid(), include_parent=False)