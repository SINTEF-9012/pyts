# Import KafkaProducer from Kafka library
from kafka import KafkaProducer
import json

# Define server with port
bootstrap_servers = ['localhost:9092']

# Define topic name where the message will publish
topicName = 'bhp_input'

# Initialize producer variable
producer = KafkaProducer(bootstrap_servers = bootstrap_servers)

# Opening JSON file
f = open('data/broken_input.json')
  
# returns JSON object as a dictionary
msg = json.load(f)
print("Sample message: ", msg)

# Publish text in defined topic
producer.send(topicName, json.dumps(msg).encode('utf-8'))
producer.flush()
# producer.close()

# Print message
print("Message Sent")