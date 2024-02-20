from pymodbus.client import ModbusTcpClient

scada_port = 502
scada_ip = "137.110.108.161"
scada_client = ModbusTcpClient(scada_ip, scada_port)
scada_client.connect()
scada_client.write_register(address=0, value=5)
scada_client.close()
