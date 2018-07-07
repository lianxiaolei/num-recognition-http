# coding: utf-8

import sys
reload(sys)
sys.setdefaultencoding('utf8')
sys.path.append('../')
import numpy as np
import cv2
from core.cnn import CNN
from server.server_tools import *
from gevent.pywsgi import WSGIServer
from multiprocessing import cpu_count, Process
from tornado.wsgi import WSGIContainer
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
import time
import base64
from flask import Flask, jsonify, request
from flask_cors import CORS
import traceback

cnn = CNN()
cnn.load_session('../model/Test_CNN_Model.ckpt')
print 'load model done'

app = Flask(__name__)

CORS(app, resource=r'/*')
@app.route('/ai/cv/numreco',methods=['POST'])
def num_reco():
    """
    :return:
    """
    result = []
    data = []
    json_obj = request.json

    try:
        for i in range(0, 10):
            imgbs64 = json_obj['%s' % i]
            if not imgbs64.strip():
                return jsonify({"code": "401", "message": "%s位置数字缺失" % (i + 1)})

            imgstr = base64.b64decode(json_obj['%s' % i])
            img_array = np.fromstring(imgstr, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
            img = preprocessing(img)
            data.append(img)

        result = cnn.predict_new(np.array(data))

    except Exception as e:
        print('repr(e):\t', repr(e))
        print('e.message:\t', e.message)
        print('traceback.format_exc():\n%s' % traceback.format_exc())
        return jsonify({"code": "402", "message": e.message})

    return jsonify({"code": "200", "message": "成功", "data": result.tolist()})


def server_forever():
    server.start_accepting()
    server._stop_event.wait()


if __name__ == '__main__':
    # server = WSGIServer(('0.0.0.0', 2222), app)
    # server.start()
    # for i in range(cpu_count()):
    #     p = Process(target=server_forever)
    #     p.start()

    http_server = HTTPServer(WSGIContainer(app))
    http_server.listen(2222)
    IOLoop.instance().start()

    # app.run(host='0.0.0', port=2222, debug=True, threaded=False)
