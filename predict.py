# Image Classifier Script
# Predict.py
# Inputs: 'path_to_image', 'path_to_saved_model' 
# Options: 
# --top_k, Top K most likely classes
# --category_names, Path to a JSON file mapping labels to flower names

# Call: python predict.py /path/to/image saved_model --category_names map.json

import argparse
# import functions.py
from functions import predict

parser = argparse.ArgumentParser(description='Given an flower image with shape (224,224,3), program predicts top K probabilities of flower class')
parser.add_argument('img_path', help='Filepath to Image')
parser.add_argument('model_path', help='Filepath to Model .H5')
parser.add_argument('-k','--top_k', type=int, help='Top k probabilities to return along with the prediction')
parser.add_argument('-c','--category_names', help='Filepath to class names JSON file')

args = parser.parse_args()

if __name__=='__main__':
    print('Arg1: ', args.img_path)
    print('Arg2: ', args.model_path)
    print('Arg3: ',args.top_k)
    print('Arg4: ',args.category_names)
    prob, classes = predict(args.img_path, args.model_path, args.top_k, args.category_names)
    print('Probabilities: ',prob)
    print('Classes: ', classes)