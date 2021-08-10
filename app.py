from flask import Flask, render_template, request, send_file
from flask_pymongo import PyMongo
import json
import sg_core_api as sgapi
import os
import pathlib
import numpy as np
from bson.json_util import dumps
from bson.objectid import ObjectId
from datetime import datetime
from scipy.interpolate import CubicSpline

app = Flask(__name__)

gesture_generator = sgapi.get_gesture_generator()
root_path = pathlib.Path(__file__).parent

app.config["MONGO_URI"] = "mongodb://localhost"  # setup your own db to enable motion library and rule functions
mongo = PyMongo(app)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/motion', methods=['GET', 'POST'])
def motion_library():
    if request.method == 'POST':
        json = request.get_json()
        json["motion"] = sgapi.convert_pose_coordinate_for_ui(np.array(json["motion"])).tolist()
        result = {}
        try:
            mongo.db.motion.insert_one(json)
            result['msg'] = "success"
        except Exception as e:
            result['msg'] = "fail"
        return result
    elif request.method == 'GET':
        cursor = mongo.db.motion.find().sort("name", 1)
        motions = sgapi.convert_pose_coordinate_for_ui_for_motion_library(list(cursor))
        return dumps(motions)
    else:
        assert False


@app.route('/api/delete_motion/<id>', methods=['GET'])
def delete_motion_library(id):
    result = mongo.db.motion.delete_one({'_id': ObjectId(id)})
    msg = {}
    if result.deleted_count > 0:
        msg['msg'] = "success"
    else:
        msg['msg'] = "fail"
    return msg


@app.route('/api/rule', methods=['GET', 'POST'])
def rule():
    if request.method == 'POST':
        json = request.get_json()
        result = {}
        try:
            json['motion'] = ObjectId(json['motion'])
            mongo.db.rule.insert_one(json)
            result['msg'] = "success"
        except Exception as e:
            print(json)
            print(e)
            result['msg'] = "fail"
        return result
    elif request.method == 'GET':
        pipeline = [{'$lookup':
                         {'from': 'motion',
                          'localField': 'motion',
                          'foreignField': '_id',
                          'as': 'motion_info'}},
                    ]

        cursor = mongo.db.rule.aggregate(pipeline)
        rules = sgapi.convert_pose_coordinate_for_ui_for_rule_library(cursor)
        rules = dumps(rules)
        return rules
    else:
        assert False


@app.route('/api/delete_rule/<id>', methods=['GET'])
def delete_rule(id):
    result = mongo.db.rule.delete_one({'_id': ObjectId(id)})
    msg = {}
    if result.deleted_count > 0:
        msg['msg'] = "success"
    else:
        msg['msg'] = "fail"
    return msg


@app.route('/api/input', methods=['POST'])
def input_text_post():
    content = request.get_json()
    input_text = content.get('text-input')
    if input_text is None or len(input_text) == 0:
        return {'msg': 'empty'}

    print('--------------------------------------------')
    print('request time:', datetime.now())
    print('request IP:', request.remote_addr)
    print(input_text)

    kp_constraints = content.get('keypoint-constraints')
    if kp_constraints:
        pose_constraints_input = np.array(kp_constraints)
        pose_constraints = sgapi.convert_pose_coordinate_for_model(np.copy(pose_constraints_input))
    else:
        pose_constraints = None
        pose_constraints_input = None

    style_constraints = content.get('style-constraints')
    if style_constraints:
        style_constraints = np.array(style_constraints)
    else:
        style_constraints = None

    result = {}
    result['msg'] = "success"
    result['input-pose-constraints'] = pose_constraints_input.tolist() if pose_constraints_input is not None else None
    result['input-style-constraints'] = style_constraints.tolist() if style_constraints is not None else None
    result['input-voice'] = content.get('voice')
    result['is-manual-scenario'] = content.get('is-manual-scenario')

    if content.get('is-manual-scenario'):
        # interpolate key poses
        n_frames = pose_constraints_input.shape[0]
        n_joints = int((pose_constraints_input.shape[1] - 1) / 3)
        key_idxs = [i for i, e in enumerate(pose_constraints_input) if e[-1] == 1]

        if len(key_idxs) >= 2:
            out_gesture = np.zeros((n_frames, n_joints * 3))
            xs = np.arange(0, n_frames, 1)

            for i in range(n_joints):
                pts = pose_constraints_input[key_idxs, i * 3:(i + 1) * 3]
                cs = CubicSpline(key_idxs, pts, bc_type='clamped')
                out_gesture[:, i * 3:(i + 1) * 3] = cs(xs)

            result['output-data'] = out_gesture.tolist()
            result['audio-filename'] = os.path.split(result['input-voice'])[
                1]  # WARNING: assumed manual mode uses external audio file
        else:
            result['msg'] = "fail"
    else:
        # run gesture generation model
        output = gesture_generator.generate(input_text, pose_constraints=pose_constraints,
                                            style_values=style_constraints, voice=content.get('voice'))

        if output is None:
            # something wrong
            result['msg'] = "fail"
        else:
            gesture, audio, tts_filename, words_with_timestamps = output
            gesture = sgapi.convert_pose_coordinate_for_ui(gesture)

            result['audio-filename'] = os.path.split(tts_filename)[1]  # filename without path
            result['words-with-timestamps'] = words_with_timestamps
            result['output-data'] = gesture.tolist()

    return result


@app.route('/media/<path:filename>/<path:new_filename>')
def download_audio_file(filename, new_filename):
    return send_file(os.path.join('./cached_wav', filename), as_attachment=True, attachment_filename=new_filename,
                     cache_timeout=0)


@app.route('/mesh/<path:filename>')
def download_mesh_file(filename):
    mesh_path = root_path.joinpath("static", "mesh", filename)
    return send_file(str(mesh_path), as_attachment=True, cache_timeout=0)


@app.route('/upload_audio', methods=['POST'])
def upload():
    upload_dir = './cached_wav'
    file_names = []

    for key in request.files:
        file = request.files[key]
        _, ext = os.path.splitext(file.filename)
        print('uploaded: ', file.filename)
        try:
            upload_path = os.path.join(upload_dir, "uploaded_audio" + ext)
            file.save(upload_path)
            file_names.append(upload_path)
        except:
            print('save fail: ' + os.path.join(upload_dir, file.filename))

    return json.dumps({'filename': [f for f in file_names]})


if __name__ == '__main__':
    app.run()
