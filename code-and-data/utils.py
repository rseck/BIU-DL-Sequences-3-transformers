def get_module_device(module):
    return next(module.parameters()).device
