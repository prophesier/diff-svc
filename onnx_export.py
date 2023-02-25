import utils
from infer_tools.infer_tool import SvcOnnx



if __name__ == "__main__":
    project_name = "Yua"
    model_path = f'./checkpoints/{project_name}/model.ckpt'
    config_path = f'./checkpoints/{project_name}/config.yaml'
    hubert_gpu = False

    model = SvcOnnx(project_name, config_path, hubert_gpu, model_path)
    model = model.model
    model.cpu()
    model.fs2.cpu()

    model.OnnxExport(project_name)

