import paho.mqtt.client as paho
import json


broker="localhost"
port=1883

class plot_juggler():
    def __init__(self,
                 client_name: str = "my_client",
                 topic_name: str = "/my/topic") -> None:

        self.client_name = client_name
        self.topic_name = topic_name

        self.client = paho.Client(self.client_name)                           #create client object
        # self.client = paho.Client(paho.CallbackAPIVersion.VERSION1, self.client_name)       #create client object for version paho.mttq >= 2.0.0
        self.client.on_publish = self.on_publish                          #assign function to callback
        self.client.connect(broker,port)  


    def on_publish(self, client,userdata,result):             #create function for callback
        # print("data published \n")
        pass

    def publish(self, msg):
        MQTT_MSG=json.dumps(msg)
        result = self.client.publish(self.topic_name,MQTT_MSG)


