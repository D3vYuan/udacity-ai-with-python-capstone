import os
from PIL import Image
from torchvision import transforms

class ImageFile:
    def __init__(self, path):
        self.path = path

    def generate_resize_width_height(self, image_width, image_height, new_shortest_size):
        is_width_shorter = image_width < image_height
        if is_width_shorter:
            resize_width = new_shortest_size 
            resize_height = round(image_height / ( image_width / new_shortest_size ))
        else:
            resize_width = round(image_width / ( image_height / new_shortest_size ))
            resize_height = new_shortest_size
        return resize_width, resize_height

    def generate_centre_crop_dimension(self, image_width, image_height, crop_width, crop_height):
        left = (image_width - crop_width)/2
        top = (image_height - crop_height)/2
        right = (image_width + crop_width)/2
        bottom = (image_height + crop_height)/2
        
        return left, top, right, bottom

    def convert_image_to_tensor(self, shortest_size = 256, cropped_width = 224, cropped_height = 224):
        ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an tensor array
        '''
        if not self.path or not os.path.exists(self.path):
            print(f"No image loaded as [{self.path}] does not exists")
            return None
        
        try:
            with Image.open(self.path) as original_img:
                # resize the images where the shortest side is 256 pixels, keeping the aspect ratio
                print(f"{self.path} original size - {original_img.width} x {original_img.height}")
                print(original_img.width, original_img.height, shortest_size)
                resize_width, resize_height = self.generate_resize_width_height(original_img.width, original_img.height, shortest_size)
                resize_img = original_img.resize((resize_width, resize_height))
                print(f"{self.path} re-size - {resize_img.width} x {resize_img.height}")
                
                # crop out the center 224x224 portion of the image
                left, top, right, bottom = self.generate_centre_crop_dimension(resize_img.width, resize_img.height, cropped_width, cropped_height)
                crop_img = resize_img.crop((left, top, right, bottom))
                print(f"{self.path} crop - {crop_img.width} x {crop_img.height}")
                
                # Color channels of images are typically encoded as integers 0-255, but the model expected floats 0-1
        #         np_img = np.array(img)

                # For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations `[0.229, 0.224, 0.225]`. 
                # You'll want to subtract the means from each color channel, then divide by the standard deviation. 
                pil_to_tensor_transforms = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],
                                                                    [0.229, 0.224, 0.225])])
                
                # PyTorch expects the color channel to be the first dimension but it's the third dimension in the PIL image and Numpy array. 
                # You can reorder dimensions using [`ndarray.transpose`](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.transpose.html). 
                # The color channel needs to be first and retain the order of the other two dimensions.
                return pil_to_tensor_transforms(crop_img)
        except Exception as e:
            print(f"{self.path} not converted to tensor due to {e}")
        