custom_imports = dict(
    imports=['cpu_augmentor', 'tensor_augmentor', 'bbox'],
    allow_failed_imports=False
)
transforms = dict(type='SPNAugmentation', n=2, p=0.8)
model = dict(type='CombinedAugmentation')
transforms2 = dict(type='SetFullImageBBox')