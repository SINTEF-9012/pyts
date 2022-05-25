import sys
import json
import argparse
from infer_fas import *
from kafka import KafkaConsumer
from kafka import KafkaProducer

# Instantiate the parser
parser = argparse.ArgumentParser(description='Kafka consumer for BHP')

# Required argument
parser.add_argument('host', help='A required argument for Kafka broker host and port (e.g. localhost:9092)')

# Required argument
parser.add_argument('input_topic', help='A required argument for Kafka input topic')

# Required argument
parser.add_argument('output_topic', help='A required argument for Kafka output topic')

args = parser.parse_args()
print("Argument values: ", args.host, args.input_topic, args.output_topic)
kafka_servers = [args.host]
input_topic = args.input_topic
output_topic = args.output_topic

# Load name of input columns
input_columns = pd.read_csv(
        "data/input_columns.csv", index_col=0, header=None
    ).index.tolist()

producer = KafkaProducer(bootstrap_servers=kafka_servers)

# To consume latest messages and auto-commit offsets
consumer = KafkaConsumer(input_topic, #'bhp_input',
                         group_id='bhp_group',
                         auto_offset_reset='earliest',
                         bootstrap_servers=kafka_servers)
for message in consumer:
    # message value and key are raw bytes -- decode if necessary!
    # e.g., for unicode: `message.value.decode('utf-8')`
    print(json.loads(message.value))
    #print ("%s:%d:%d: key=%s value=%s" % (message.topic, message.partition,
    #                                      message.offset, message.key,
    #                                      message.value))
    output_json = preprocess_and_infer(
        message.value, "model/input_scaler.z", "model/model.h5", input_columns
    )
    print("Inference results: ", output_json)
    
    # Publish result in defined topic
    producer.send(output_topic, output_json.encode('utf-8'))
    producer.flush()
    
    # Print message
    print("Result sent back.")

# Close the connection and terminate the script
producer.close()
sys.exit()