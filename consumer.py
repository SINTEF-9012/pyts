import time
from kafka import KafkaConsumer

consumer = KafkaConsumer(
     bootstrap_servers=['localhost:9092'],
     auto_offset_reset='earliest',
     group_id='my-consumer-1',
)
consumer.subscribe(['first_kafka_topic'])

while True:
    try: 
        message = consumer.poll(10.0)

        if not message:
            print("Consumer: no message")
            time.sleep(10) # Sleep for 2 minutes

        if message.error():
            print(f"Consumer error: {message.error()}")
            continue

        print("Received message")
        # TODO insert the ML magic here...
    except:
        # Handle any exception here
        ...
    finally:
        consumer.close()
        print("Goodbye")
        time.sleep(10) # Sleep for 10 sec
