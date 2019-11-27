from os import path

import fraud_detection

base_path = path.dirname(path.dirname(fraud_detection.__file__))
workspace_path = path.join(base_path, 'workspace')
data_path = path.join(workspace_path, 'data')
models_path = path.join(workspace_path, 'models')


