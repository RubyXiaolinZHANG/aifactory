import os
import torch
import numpy as np


class ModuleCollector:
    collection = None
    module_id = 0
    save_numpy = True
    _device = None
    _save_dir = None

    def __init__(self, save_numpy=False, device=None, detach=False, save_dir=None):
        self.collection = None
        self.save_numpy = save_numpy
        self._device = device
        self._save_dir = save_dir
        self._detach = detach

    @property
    def detach(self):
        return self._detach

    @property
    def device(self):
        return self._device

    def __call__(self, module, module_in, module_out):
        if self.collection is None:
            self.collection = {}
        if self.save_numpy:
            if isinstance(module_out, torch.Tensor):
                module_out =  module_out.cpu().detach().numpy()
            elif isinstance(module_out, (list, tuple)):
                module_out = [x.cpu().detach().numpy() for x in module_out]
            else:
                raise ValueError("module outputs should be tensor, list or tuple of tensor")
            self.collection[self.module_id] = {"id": self.module_id,
                                               "module_name": module.name if hasattr(module, "name") else "",
                                               "module_class_name": module.__class__.__name__,
                                               "module_in": [data.cpu().detach().numpy() for data in module_in if data is not None],
                                               "module_out": module_out}
        else:
            self._device = module_in[0].device if self._device is None else self._device
            if self._detach:
                module_in = module_in.detach()
                module_out = module_out.detach()
            if isinstance(module_out, torch.Tensor):
                module_out =  module_out.to(self._device)
            elif isinstance(module_out, (list, tuple)):
                module_out = [x.to(self._device) for x in module_out]
            else:
                raise ValueError("module outputs should be tensor, list or tuple of tensor")
            self.collection[self.module_id] = {"id": self.module_id,
                                               "module_name": module.name if hasattr(module, "name") else "",
                                               "module_class_name": module.__class__.__name__,
                                               "module": module,
                                               "module_in": [data.to(self._device) for data in module_in if data is not None],
                                               "module_out": module_out}
        if self._save_dir is not None:
            save_name = os.path.join(self._save_dir, "{}~{}~{}~output".format(self.module_id,
                                                                              module.name,
                                                                              module.__class__.__name__))
            os.makedirs(os.path.dirname(save_name), exist_ok=True)
            np.save(save_name, module_out.cpu().detach().numpy())

        self.module_id += 1

    def set_save_dir(self, save_dir):
        self._save_dir = save_dir

    def clear(self):
        self.module_id = 0
        self.collection = None

    def name_indexes(self):
        named_collection = None
        for key, val in self.collection.items():
            assert hasattr(val["module"], "name")
            if named_collection is None:
                named_collection = {}
            named_collection[val["module"].name] = val
        return named_collection


class ModuleCollector4Test(ModuleCollector):

    def __init__(self, save_numpy=True, device=None, save_dir=None):
        super().__init__(save_numpy, device, save_dir)

    def __call__(self, module, module_in, module_out):
        this_id = self.module_id
        if True:
            src_dir = r"F:\_result\sd\gen_images\quant_kv8\result\direct_conv\quant_ops_output_mixdq"
            key = "{}~QuantLayer~input.npy".format(module.name)
            is_loaded = False
            for file in os.listdir(src_dir):
                if key in file:
                    src_file = os.path.join(src_dir, file)
                    data = np.load(src_file)
                    data = torch.from_numpy(data).to(module_in[0].device)
                    data_list = [data]
                    for i in range(1,len(module_in)):
                        data_list.append(module_in[i])
                    module_in = tuple(data_list)
                    is_loaded = True
                    break
            if not is_loaded:
                print("{} is not loaded!".format(module.name))
        super().__call__(module, module_in, module_out)
        test_result = self.call_op_test(module, module_in, module_out)
        if test_result is None:
            pass
        else:
            self.collection[this_id]["test"] = test_result
            if self._save_dir is not None:
                for name, val in test_result.items():
                    save_name = os.path.join(self._save_dir, "{}~{}~{}~{}".format(this_id,
                                                                                  module.name,
                                                                                  module.__class__.__name__,
                                                                                  name))
                    os.makedirs(os.path.dirname(save_name), exist_ok=True)
                    np.save(save_name, val.cpu().detach().numpy())

    @staticmethod
    def call_op_test(module, module_in, module_out):
        if hasattr(module, "op_test"):
            return module.op_test(module_in, module_out)
        else:
            return None


def add_forward_hooks(modules, forward_hook, handles=None):
    if isinstance(modules, dict):
        for name, module in modules.items():
            add_forward_hooks(module, forward_hook, handles)
    elif isinstance(modules, (list, tuple)):
        for module in modules:
            assert isinstance(module, torch.nn.Module)
            add_forward_hooks(module, forward_hook, handles)
    elif isinstance(modules, torch.nn.Module):
        handle = modules.register_forward_hook(forward_hook)
        if isinstance(handles, list):
            handles.append(handle)
    return handles


def add_backward_hooks(modules, backward_hook, handles=None):
    if isinstance(modules, dict):
        for name, module in modules.items():
            add_backward_hooks(module, backward_hook, handles)
    elif isinstance(modules, (list, tuple)):
        for module in modules:
            assert isinstance(module, torch.nn.Module)
            add_backward_hooks(module, backward_hook, handles)
    elif isinstance(modules, torch.nn.Module):
        handle = modules.register_full_backward_hook(backward_hook)
        if isinstance(handles, list):
            handles.append(handle)
    return handles