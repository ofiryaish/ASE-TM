import copy
import os 
import sys

SCRIPT_FOLDER = os.path.dirname(os.path.realpath(__file__))
PARENT_FOLDER = f"{SCRIPT_FOLDER}/.."
sys.path.append(PARENT_FOLDER)


sys.path.append('/a/home/cc/students/cs/mishaly1/repos/data/code/speechbrain')
sys.path.append('/a/home/cc/students/cs/mishaly1/repos/data/code/speechbrain/_separation/compare/sha_rnn')
sys.path.append('/a/home/cc/students/cs/mishaly1/repos/data/code/speechbrain/_separation/ours')


from tools.models import Model, ModelNoD
from tools.helpers import get_device, load_model, RMSNormalizer, get_hparams
from data_utils.data_tools import sef

from ours.model import OurModel, OurModelAdd

from compare.FxLMS import predict_signal
from compare.ARN.model_c import SHARNN
from compare.ARN.utils import predict
from compare.DeepANC.model import get_model


device = get_device()

def get_ours(name):
    yaml_dir = "/a/home/cc/students/cs/mishaly1/repos/data/code/speechbrain/_separation/ours/hparams"
    
    if name in ["mamaba_S_simv3_random_hpf", "oas00S_mamaba_S_simv3_random_hpf"]:
        yaml_path = f"{yaml_dir}/dpmamba_S.yaml"
        hparams = get_hparams(yaml_path)
        our_model = Model(hparams).to(device)
    elif name in ["mamaba_M_simv3_random_hpf", "oasL0S_mamaba_M_simv3_random_hpf"]:
        yaml_file = f"{yaml_dir}/dpmamba_M.yaml"
        hparams = get_hparams(yaml_file)
        our_model = Model(hparams).to(device)
    elif name in ["05l_oas6_3bands_2S1M_simv3_inf","oas6_3bands_2S1M_simv3_inf","oas0_3bands_2S1M_simv3_inf","2bands_1S1M_simv3", "3bands_2S1M_simv3", "4bands_3S1M_simv3", "3bands_2S1M_simv3_inf", "oas0_3bands_2S1M_simv3_wsj"]:
        s_path = f"{yaml_dir}/dpmamba_S.yaml"
        s_hparams = get_hparams(s_path)
        s_model = ModelNoD(s_hparams).to(device)

        m_path = f"{yaml_dir}/dpmamba_M.yaml"
        m_params = get_hparams(m_path)
        m_model = ModelNoD(m_params).to(device)

        ours_path = f"{yaml_dir}/exp/{name}.yaml"
        hparams = get_hparams(ours_path)
        encoder_mamaba = [copy.deepcopy(s_model) for _ in range(len(hparams.cutoffs) - 1)] + [m_model]
        our_model = OurModel(hparams, encoder_mamaba=encoder_mamaba).to(device)
    elif name in ["pesq_3bands_2S1M_simv3"]:
        ours_path = f"{yaml_dir}/exp/{name}.yaml"
        hparams = get_hparams(ours_path)
        our_model = OurModel(hparams).to(device)

    our_model = load_model(our_model,name, strict=False)[0].to(device)
    our_model.eval()
    return our_model


def get_predict_callbacks(names, simulator, gama="inf",mu=None):
    cbs = []
    ours_dict = {}
    for method_name, model_name in names:
        if method_name == "ARN":
            model_name = model_name if model_name is not None else "arn_nonorm_simv3_hpf_random_v2"
            print("Loading model",model_name)

            arn_model = SHARNN(256, 512, 4*512, 4, 0.05)
            arn_model = load_model(model=arn_model,model_name=model_name)[0].to(device)
            arn_model.eval()
            cb = lambda x: sef(predict(x, arn_model),gama)
        elif method_name == "FxLMS":
            cb = lambda x: predict_signal(x, mu,simulator, gama)
        elif method_name == "THF-FxLMS":
            cb = lambda x: predict_signal(x, mu,simulator, gama,thf=True)
        elif "Ours" in method_name:
            name = model_name if model_name is not None else "oas6_3bands_2S1M_simv3_inf"
            print("Loading model",model_name)
            ours_dict[name] = get_ours(name)
            cb = lambda x, name_=name: sef(ours_dict[name_](x),gama)

        elif method_name == "SEPFORMER":
            yaml_path = "/a/home/cc/students/cs/mishaly1/repos/data/code/speechbrain/_separation/sepformer/hparams/yaml_flex/sepformer_50net.yaml"
            hparams = get_hparams(yaml_path)

            sep_model = Model(hparams).to(device)
            print("Loading model",model_name)

            sep_model = load_model(sep_model,model_name)[0].to(device)
            sep_model.eval()
            cb = lambda x: sef(sep_model(x),gama)
        elif method_name == "DeepANC":
            model_name = model_name if model_name is not None else "random-sef|0-delay|20k_3s|rirgen|ep-40|0.0005|bch-32|dec-0.7-10|_best"

            print("Loading model",model_name)
            deep_anc_model, _, _, _ = get_model(model_name)
            deep_anc_model.net.eval()
            cb = lambda x: deep_anc_model.predict(x, gama=gama, denorm=True)[1]
            
        cbs.append(cb)

    return cbs
