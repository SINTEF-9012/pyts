import sys
import json
import argparse
from kafka import KafkaConsumer

# Instantiate the parser
parser = argparse.ArgumentParser(description='Kafka consumer for BHP')

# Required argument
parser.add_argument('host', help='A required argument for Kafka broker host and port (e.g. localhost:9092)')

# Required argument
parser.add_argument('topic', help='A required argument for Kafka topic')

args = parser.parse_args()
print("Argument values: ", args.host, args.topic)

# To consume latest messages and auto-commit offsets
consumer = KafkaConsumer(args.topic, #'bhp_input',
                         group_id='bhp_group',
                         auto_offset_reset='earliest',
                         bootstrap_servers=[args.host])
for message in consumer:
    # message value and key are raw bytes -- decode if necessary!
    # e.g., for unicode: `message.value.decode('utf-8')`
    print(json.loads(message.value))
    #print ("%s:%d:%d: key=%s value=%s" % (message.topic, message.partition,
    #                                      message.offset, message.key,
    #                                      message.value))


# Terminate the script
sys.exit()