import json
import os
import time
import requests
from websocket import create_connection

#pairs = requests.get('https://api.cryptowat.ch/pairs).json()
cwd = os.path.dirname(os.path.abspath(__file__))
with open(cwd+'/pairs.json','r') as pairs_file:
    pairs = json.load(pairs_file)

my_pairs = ['BTCEUR', 'BTCGBP', 'BTCPLN', 'BTCUSD', 'BCHEUR', 'ETHEUR', 'LTCEUR', 'XLMEUR', 'XRPEUR']
my_pair_ids = []
for pair in pairs:
    if pair['symbol'].upper() in my_pairs:
        my_pair_ids.append(pair['id'])

mykey = '3961ZQ10CGAF0ZGSIJV7'
client_conn = create_connection('wss://stream.cryptowat.ch/connect?apikey={}'.format(mykey))
sub_msg = {
    'subscribe': {
        'subscriptions': []
    }
}

# https://docs.cryptowat.ch/websocket-api/data-subscriptions/order-books
for pid in my_pair_ids:
    resource = 'instruments:{}:book:deltas'.format(pid)
    subscription = {
        'streamSubscription': {
          'resource': resource
        }
    }
    sub_msg['subscribe']['subscriptions'].append(subscription)

client_conn.send(json.dumps(sub_msg))

#start_time = time.time()
#duration = 30
while True: #time.time() < start_time + duration:
    res = client_conn.recv()
    message = json.loads(res)
    if 'hb' not in message: # ignore heartbeats
        print('received {}'.format(message))

# These are reached when using a fixed duration
ws.close()
print('done')
