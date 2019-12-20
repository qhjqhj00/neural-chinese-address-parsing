from lensnlp.models import SequenceTagger
from flask import Flask, request, Response
import json
from lensnlp.utils.data_preprocess import cn_prepare


app = Flask(__name__, static_url_path='')


tagger = SequenceTagger.load('address.pt')


@app.route("/api")
def get():
    addr = request.args.get('text', '')
    sentences = cn_prepare(addr)
    tagger.predict(sentences)

    result = []
    for s in sentences:
        neural_result = s.to_dict(tag_type='ner')
        result.append(neural_result)

    return Response(json.dumps({'status': "ok", 'message': str(result), 'request': addr},
                               ensure_ascii=False), mimetype="application/json")


@app.route('/')
def index():
    return app.send_static_file('add.html')


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=25679, debug=True)