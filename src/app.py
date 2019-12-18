import dynet_config
dynet_config.set_gpu()

import dynet as dy
import parse
import vocabulary
import util
import latent
from flask import Flask, request, Response
import json

app = Flask(__name__, static_url_path='')


label_vocab = vocabulary.Vocabulary()

label_list = util.load_label_list('../data/labels.txt')

model = dy.ParameterCollection()
[parser] = dy.load('../models/chartdyRBTC-model_addr_dytree_giga_0.4_200_1_chartdyRBTC_dytree_1_houseno_0_0_dev=0.90', model)

for item in label_list:
    label_vocab.index((item, ))
label_vocab.index((parse.EMPTY,))
for item in label_list:
    label_vocab.index((item + "'",))


label_vocab.freeze()
latent_tree = latent.latent_tree_builder(label_vocab, 'city')


@app.route("/api")
def get():
    addr = request.args.get('text', '')
    sentence = [('XX', ch) for ch in addr]
    result = parser.parse(sentence)[0].convert().to_chunks()
    response = [[k[0], ''.join(k[3])] for k in result]
    response = [f'{k[0]}: {k[1]}' for k in response]
    text = '\n'.join(response)
    return Response(json.dumps({'status': "ok", 'message': text, 'request': addr},
                               ensure_ascii=False), mimetype="application/json")


@app.route('/')
def index():
    return app.send_static_file('index.html')


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=25678, debug=True)