from .hooks import add_backward_hooks, add_forward_hooks, ModuleCollector

class ModuleMonitor:

    def __init__(self, target_modules):

        self._forward_hooks = ModuleCollector()
        add_forward_hooks(target_modules, self._forward_hooks)
        self._backward_hooks = ModuleCollector()
        add_backward_hooks(target_modules, self._backward_hooks)


