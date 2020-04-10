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
import numpy as np

parser = argparse.ArgumentParser(description='Given an flower image with shape (224,224,3), program predicts top K probabilities of flower class')
parser.add_argument('img_path', help='Filepath to Image')
parser.add_argument('model_path', help='Filepath to Model .H5')
parser.add_argument('-k','--top_k', type=int, help='Top k probabilities to return along with the prediction')
parser.add_argument('-c','--category_names', help='Filepath to class names JSON file')

args = parser.parse_args()

if __name__=='__main__':
    
    if args.top_k == None:
        top_k = 1
    else:
        top_k = args.top_k
    print(args.top_k)
    
    catg = True
    if args.category_names == None:
        catg = False
       
    prob, n_class = predict(args.img_path, args.model_path, top_k, args.category_names)
    
    
    print('\n\n')
    print('{:30}'.format('---------------------------------------------------'))
    print('{:30}'.format('Class Name') ,'{:30}'.format('Probability'))
    print('{:30}'.format('---------------------------------------------------'))
   
    for i in range(top_k):
        if catg == False:
            print('{:30}'.format(str(n_class[i])), '{:6f}'.format(prob[i]))
        else:
            print('{:30}'.format(n_class[i]), '{:6f}'.format(prob[i]))
   
    max_index = np.argmax(prob, axis=0)    
    print('\n\n* The image "', args.img_path, '" is probable to be a class: {:20}'.format(n_class[max_index]), ', with likehood of {:6f}'.format(prob[max_index]))