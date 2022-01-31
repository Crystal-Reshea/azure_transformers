from azureml.core import Workspace
ws = Workspace.from_config()
datastore = ws.get_default_datastore()
datastore.upload(src_dir='/Users/reshea/Work/nlp-poc/train-v2.0.json',
                 target_path='datasets/squad_train',
                 overwrite=True)
datastore.upload(src_dir='/Users/reshea/Work/nlp-poc/dev-v2.0.json',
                 target_path='datasets/squad_val',
                 overwrite=True)