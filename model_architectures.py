from torchvision import models

# supported_models = ["vgg11", "vgg13", "vgg16", "vgg19", \
#                     "vgg11_bn", "vgg13_bn", "vgg16_bn", "vgg19_bn"]

# available_models = {
#     "vgg11": models.vgg11(pretrained=True), 
#     "vgg13": models.vgg13(pretrained=True), 
#     "vgg16": models.vgg16(pretrained=True), 
#     "vgg19": models.vgg19(pretrained=True), 
#     "vgg11_bn": models.vgg11_bn(pretrained=True), 
#     "vgg13_bn": models.vgg13_bn(pretrained=True), 
#     "vgg16_bn": models.vgg16_bn(pretrained=True), 
#     "vgg19_bn": models.vgg19_bn(pretrained=True)
# }

supported_models = ["vgg16_bn", "vgg19_bn"]

available_models = {
    "vgg16_bn": models.vgg16_bn(pretrained=True), 
    "vgg19_bn": models.vgg19_bn(pretrained=True)
}

model_input_sizes = {
    "vgg16_bn": 25088,
    "vgg19_bn": 25088
}

def get_pretrained_model(model_architecture, freeze=False):
    model = available_models.get(model_architecture, None)
    if model and freeze:
        for param in model.parameters():
            param.requires_grad = False
    return model

def get_model_input_size(model_architecture):
    return model_input_sizes.get(model_architecture, None)