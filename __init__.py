# Info displayed in the add-on overview.
bl_info = {
    'name': "JAX Cloth",
    'description': "",
    'author': "Victor-Louis De Gusseme",
    'version': (0, 1),
    'blender': (2, 80, 0),
    'location': "View3D > Tool Shelf",
    'wiki_url': 'https://github.com/Victorlouisdg/jax_cloth_blender',
    'support': "COMMUNITY",
    'category': "Import"
}

# Importing in this init file is a bit weird.
# if "bpy" in locals():
#     print("Force reloading the plugin.")
#     import importlib

#     importlib.reload(node_utils)
#     importlib.reload(hdri)

# else:
#     from . import node_utils, \
#         hdri

import bpy


# -------------------------------------------------------------------
#   Register & Unregister
# -------------------------------------------------------------------
def make_annotations(cls):
    """Converts class fields to annotations if running with Blender 2.8"""
    bl_props = {k: v for k, v in cls.__dict__.items() if isinstance(v, tuple)}
    if bl_props:
        if '__annotations__' not in cls.__dict__:
            setattr(cls, '__annotations__', {})
        annotations = cls.__dict__['__annotations__']
        for k, v in bl_props.items():
            annotations[k] = v
            delattr(cls, k)
    return cls

# All classes to register.
classes = ()

# Register all classes + the collection property for storing lightfields
def register():
    # Classes
    for cls in classes:
        make_annotations(cls)
        bpy.utils.register_class(cls)

# Unregister all classes
# This is done in reverse to 'pop the register stack'.
def unregister():
    # Unregister classes
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)