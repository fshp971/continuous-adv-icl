import torch, transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig
from typing import Union, Tuple


def _parse_peft_module_name(module_name: str, peft_name: str) -> str:
    name_new = None
    spt = module_name.rsplit(".", 2)

    if len(spt) == 3 and spt[1] == peft_name:
        name_new = f"{spt[0]}.{spt[2]}"

    elif len(spt) >= 2 and spt[-1] == peft_name:
        if len(spt) == 2:
            name_new = spt[0]
        else:
            name_new = f"{spt[0]}.{spt[1]}"

    return name_new


def named_peft_parameters(model, peft_name: str = None):
    if peft_name is None:
        return
    for name, pp in model.named_parameters():
        if _parse_peft_module_name(name, peft_name) is not None:
            yield name, pp


def peft_parameters(model, peft_name: str):
    for _, pp in named_peft_parameters(model, peft_name):
        yield pp


def peft_state_dict(model, peft_name: str):
    state_dict = dict()
    for name, pp in named_peft_parameters(model, peft_name):
        state_dict[name] = pp
    return state_dict


def load_peft_state_dict(model, peft_name: str, state_dict: dict):
    for name, pp in model.named_parameters():
        spt = name.rsplit(".", 2)
        name_new = _parse_peft_module_name(name, peft_name)
        if name_new is not None:
            pp_new = state_dict.get(name, None)
            if pp_new is None:
                pp_new = state_dict.get(name_new, None)
            if pp_new is not None:
                pp.data.copy_(pp_new.data)


class PEFTWrapper:
    def __init__(self, model, peft_name: str = None):
        self.model = model
        self.peft_name = peft_name

    @property
    def active_adapter(self):
        return self.model.active_adapter

    @property
    def vocab_size(self):
        return self.model.vocab_size

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    def save_pretrained(self, *args, **kwargs):
        self.model.save_pretrained(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def to(self, *args, **kwargs):
        self.model.to(*args, **kwargs)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def named_parameters(self):
        return named_peft_parameters(self.model, self.peft_name)

    def parameters(self):
        return peft_parameters(self.model, self.peft_name)

    def state_dict(self) -> dict:
        return peft_state_dict(self.model, self.peft_name)

    def load_state_dict(self, state_dict):
        load_peft_state_dict(self.model, self.peft_name, state_dict)


def get_model(
        model_id: str,
        device: Union[str, torch.device] = None,
        lora_cfg: dict = None
    ) -> Tuple[Union[PEFTWrapper, transformers.PreTrainedModel], transformers.PreTrainedTokenizerBase]:

    if device is None:
        device = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map=device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # from peft import PeftModel
    # if model_id == "ContinuousAT/Llama-2-7B-CAT":
    #     base_model_id = "meta-llama/Llama-2-7b-chat-hf"
    #     base_model = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype=torch.bfloat16, device_map=device)
    #     tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    #     model = PeftModel.from_pretrained(base_model, model_id)
    # else:
    #     model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map=device)
    #     tokenizer = AutoTokenizer.from_pretrained(model_id)

    if "vicuna" in model_id:
        tokenizer.chat_template = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = system_message + '\\nUSER: ' + message['content'] %}{% else %}{% set content = 'USER: ' + message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ bos_token + ' ' + content.strip() + '\n' }}{% elif message['role'] == 'assistant' %}{{ 'ASSISTANT: '  + content.strip() + ' ' + eos_token + '\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'ASSISTANT: ' }}{% endif %}"

    if lora_cfg is not None:
        adapter_name = lora_cfg.pop("adapter_name", "default")
        lora_config = LoraConfig(**lora_cfg)
        model = get_peft_model(model, lora_config, adapter_name=adapter_name)
        model = PEFTWrapper(model, adapter_name)

    return model, tokenizer
