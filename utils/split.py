import torch 

def split_support_query(image, way, shot, query, episodes=1):
    image_shape = image.shape[1:]
    image = image.view(episodes, way, shot + query, *image_shape)
    x_shot, x_query = image.split([shot, query], dim=2)
    x_shot = x_shot.contiguous()
    x_query = x_query.contiguous().view(episodes, way * query, *image_shape)
    return x_shot, x_query 

def make_label(way, shot, episodes=1):
    label = torch.arange(way).unsqueeze(1).expand(way, shot).reshape(-1)
    label = label.repeat(episodes)
    return label 
