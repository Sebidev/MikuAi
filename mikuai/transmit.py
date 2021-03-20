"""
 MikuAi - Is a toolkit for creating Conversational AI applications.
 Copyright by Sebastian Bürger [sebidev]
"""

import time
import serial
import paho.mqtt.client as mqtt

def mqtt_publish(server, mqtt_port, topic, msg):
    def on_connect(client, userdata, flags, rc):
        print("Connected with result code " + str(rc))

    client = mqtt.Client()
    client.on_connect = on_connect
    client.connect(server, mqtt_port, 60)
    client.loop_start()
    client.publish(topic, msg)

def mqtt_subscribe(server, mqtt_port, topic):
    def on_connect(client, userdata, flags, rc):
        print("mikuai.tranmit -> Connected with result code " + str(rc))

    client.subscribe(topic)

    def on_message(client, userdata, msg):
        print("mikuai.tranmit -> " + msg.topic + " " + str(msg.payload))
        message_puffer = msg.topic + " " + str(msg.payload)

    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(server, mqtt_port, 60)
    client.loop_forever()

def serial_read(port, baud):
    ser = serial.Serial(port, baud)
    serial_puffer = ser.readline()
    ser.close()

def serial_write(port, baud, msg):
    ser = serial.Serial(port, baud)
    ser.write(msg)
    ser.close()
