from abc import ABC,abstractmethod
import os
import cv2
from PIL import Image
import concurrent.futures
from torchvision.models import resnet50, ResNet50_Weights
import torch
import torch.nn as nn
from torchvision.transforms import v2
from enum import Enum
import argparse
import torch.nn.functional as F
import math
import shutil
from collections import defaultdict
import json

#######################################################################################################################

# SOURCES:

    # https://docs.python.org/3/library/abc.html
    # https://docs.python.org/3/library/enum.html
    # https://www.geeksforgeeks.org/python/python-pillow-using-image-module/
    # https://www.geeksforgeeks.org/python/defaultdict-in-python/

    # https://www.geeksforgeeks.org/python/command-line-option-and-argument-parsing-using-argparse-in-python/

    # https://docs.python.org/3/tutorial/errors.html

    # https://docs.pytorch.org/vision/main/models.html
    # https://discuss.pytorch.org/t/feature-extraction-in-torchvision-models-vit-b-16/148029
    # https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.v2.ToTensor.html
    # https://medium.com/%40abolarinwaotun/unsqueeze-method-in-pytorch-a-useful-tool-for-deep-learning-aaa4e5e588f9
    # https://medium.com/@sofiasondh/why-use-typing-override-decorator-in-python-4362ffd253

    # https://www.geeksforgeeks.org/python/python-os-makedirs-method/

    # https://www.geeksforgeeks.org/python/python-os-listdir-method/

    # https://www.geeksforgeeks.org/python/python-os-path-basename-method/

    # https://www.geeksforgeeks.org/python/python-shutil-copy-method/

    # https://docs.pytorch.org/docs/stable/generated/torch.stack.html
    # https://docs.pytorch.org/docs/stable/generated/torch.squeeze.html
    # https://docs.pytorch.org/docs/stable/generated/torch.unsqueeze.html

    # https://www.geeksforgeeks.org/python/python-process-images-of-a-video-using-opencv/
    # https://www.geeksforgeeks.org/python/convert-a-numpy-array-to-an-image/
    # https://www.geeksforgeeks.org/python/convert-bgr-and-rgb-with-python-opencv/
    # https://medium.com/@michael71314/python-lesson-47-image-rotation-ai-pt-13-e345fcab4


    # https://docs.python.org/3/library/concurrent.futures.html
    # https://www.geeksforgeeks.org/python/python-shutil-copy-method/

    # https://www.geeksforgeeks.org/python/reading-and-writing-json-to-a-file-in-python/


#######################################################################################################################

# INPUT

class Regime(Enum):

    """enum class for input arg strings -> framed video or find augmentation parameter"""

    FRAME = "Frame",
    SPLIT = "Split",
    BALANCE = "Balance",
    MAPING  = "Maping"

# user can choose get best augmentation parameter and save that model or framed given videos
parser = argparse.ArgumentParser()
parser.add_argument("--regime", help="Choose regime (Frame or Augmentation_parameter):",default = Regime.MAPING)


#######################################################################################################################

# EXCEPTIONS

class Errors(Exception,ABC):
    def __init__(self,what_error,part_message,*args):
        super().__init__(*args)

        self.message = f"{what_error} {part_message}"

class ExtraVideo(Errors):
    def __init__(self, video_name, PART_MESSAGE="video is extra and program has no testing data"):

        super().__init__(video_name,PART_MESSAGE)

class PlayVideoError(Errors):
    def __init__(self,video_name,PART_MESSAGE = "Video can not be played"):
        super().__init__(video_name,PART_MESSAGE)

class InputError(Errors):
    def __init__(self,argument,PART_MESSAGE = "argument is not correct"):

        super().__init__(argument,PART_MESSAGE)


#######################################################################################################################

# EMBEDINGS

class ImageEmbedder:

    """model for getting embedidngs"""

    DEVICE = "cuda"
    
    def __init__(self):
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.embedding_extractor = nn.Sequential(*list(model.children())[:-1]).to(self.DEVICE)
        self.transform = v2.Compose([v2.ToImage(),v2.ToDtype(torch.float32, scale=True)])
    
    # getting image embedding from image model
    def get_embedding(self,data):
        embedding = None
        transform_img = self.transform(data).unsqueeze(0).to(self.DEVICE)

        with torch.no_grad():
            embedding = self.embedding_extractor(transform_img).squeeze(0)
        
        return embedding

#######################################################################################################################

# FOLDERS

class Folder(ABC):

    """abstract class for folders"""

    def __init__(self,path):
        super().__init__()

        self.path = path

        os.makedirs(path,exist_ok=True)

# abstract method for nested folders. This class can represent folders with video frames
class NestedFolder(Folder,ABC):

    IMAGE_FORMAT = ".png"
    
    def __init__(self,path):
        super().__init__(path)


#######################################################################################################################

# PROCESSING NOT VIDEO FRAMES

class NotVideoFrames(Folder):

    """In this folder are folders include not video frames"""

    def __init__(self,NAME = "not_video_frames"):
        super().__init__(NAME)
    
    # all img paths in class, count of validating numbers in class and class name
    def get_info_for_splitting(self,name):

        for class_name in os.listdir(name):
            imgs_paths = []
            class_path = os.path.join(name,class_name)

            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path,img_name)
                imgs_paths.append(img_path)
            
            count_of_validate_images = math.ceil(len(imgs_paths) / 3)
        
            yield imgs_paths,count_of_validate_images,class_name
    
    # main method for spliting not video data into validate images and testing images
    def split(self):

        for imgs_paths,count_of_validate_images,class_name in self.get_info_for_splitting(self.path):
            validate_frames = ValidateFrames(class_name)
            test_frames = TestFrames(class_name)

            for i,img_path in enumerate(imgs_paths):

                if i < count_of_validate_images:
                    validate_frames.add(class_name,img_path)

                else:
                    test_frames.add(class_name,img_path)

class PartNotVideoFrames(Folder,ABC):

    """abstract class for validating and test class"""

    def __init__(self,folder_name,class_name):
        path = os.path.join(folder_name,class_name)
        super().__init__(path)
    
    # add img to goal folder validating/testing
    def add(self,class_name,img_path):
        class_path = os.path.join(self.path,class_name)
        os.makedirs(class_path,exist_ok=True)

        img_name = os.path.basename(img_path)
        img_destination = os.path.join(class_path,img_name)
        
        shutil.copy(img_path,img_destination)

class ValidateFrames(PartNotVideoFrames):

    """creating validate frame folder"""

    NAME = "validate_frames"

    def __init__(self,class_name):
        super().__init__(self.NAME,class_name)

class TestFrames(PartNotVideoFrames):

    """creating testing frame folder"""

    NAME = "test_frames"

    def __init__(self,class_name):
        super().__init__(self.NAME,class_name)


#######################################################################################################################

# BALANCE DATASET

class BalanceClassFrames(NestedFolder):
    def __init__(self, path):
        super().__init__(path)

    # add img to balance folder
    def add(self,img_path,img_name,copy_number):
        img_new_name = img_name + f"_{copy_number}{self.IMAGE_FORMAT}"
        img_destination_path = os.path.join(self.path,img_new_name)
        shutil.copy(img_path,img_destination_path)

class BalanceTrainingFrames(Folder):
    NAME = "balance_training_frames"

    def __init__(self):
        super().__init__(self.NAME)
    
    # for each class method will return count of imgs
    def get_count_img_classes(self):

        counts_imgs_classes = defaultdict(int)

        for class_name in os.listdir(TrainFrames.NAME):
            class_path = os.path.join(TrainFrames.NAME,class_name)
            count_imgs_class = 0

            for _ in os.listdir(class_path):
                count_imgs_class += 1
            
            
            counts_imgs_classes[class_name] = count_imgs_class
        
        return counts_imgs_classes

    # returns maximum counts of imgs in class of all classes
    def get_max_imgs_in_class(self,counts_imgs_classes):

        max_imgs_class = float("-inf")

        for name,count_imgs_classes in counts_imgs_classes.items():

            if count_imgs_classes > max_imgs_class :
                max_imgs_class = count_imgs_classes
                print(name)
        
        return max_imgs_class
    
    # duplicate images for balancing dataset
    def duplicate(self,copies_per_class):

        for class_name in os.listdir(TrainFrames.NAME):

            class_path = os.path.join(TrainFrames.NAME,class_name)
            balance_class_path = os.path.join(self.path,class_name)
            balance_class_folder = BalanceClassFrames(balance_class_path)

            copies_class = copies_per_class[class_name]

            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path,img_name)

                for i in range(copies_class):
                    balance_class_folder.add(img_path,img_name,i)

    # main method for balancing dataset               
    def balance(self):

        copies_per_class = defaultdict(int)
        counts_imgs_classes = self.get_count_img_classes()
        max_count_imgs_class = self.get_max_imgs_in_class(counts_imgs_classes)

        for class_name,counts in counts_imgs_classes.items():
            copies_class = math.ceil(max_count_imgs_class / counts)
            copies_per_class[class_name] = copies_class
        
        self.duplicate(copies_per_class)

#######################################################################################################################

# CREATE DATASET FROM VIDEO

# class which representing folder with folders
class ClassFrames(NestedFolder):

    def __init__(self,path):
        super().__init__(path)

        self.count_imgs = 0
        self.embeddings = []
    
    # save and framing given video
    def save_video_frames(self,video_path,imageModel):
        cap = cv2.VideoCapture(video_path)

        while (cap.isOpened()):
            ret, frame = cap.read()

            if ret:
                frame_rotated = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                image_rgb = cv2.cvtColor(frame_rotated, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(image_rgb)
                self.add(img,imageModel)
            
            else:
                break
        
        cap.release()
    
    # save img to device on disk
    def save_img_to_folder(self,img:Image,img_embedding):
        img_path_no_extension = os.path.join(self.path,str(self.count_imgs))
        img_path = f"{img_path_no_extension}{self.IMAGE_FORMAT}"
        self.count_imgs += 1

        img.save(img_path)

        self.embeddings.append(img_embedding)

    # checking if we can save img, if yes then program will save it 
    def add(self, img: Image,imageModel,THRESHOLD_SIMILARITY=1):
        img_embedding = imageModel.get_embedding(img)

        if self.count_imgs == 0:
            self.save_img_to_folder(img, img_embedding)

            return 

        accepted_embeddings = torch.stack(self.embeddings).to(ImageEmbedder.DEVICE)

        similarities = F.cosine_similarity(img_embedding.unsqueze(0),accepted_embeddings, dim=1)

        best_similarity = torch.max(similarities).item()

        if best_similarity < THRESHOLD_SIMILARITY:
            self.save_img_to_folder(img, img_embedding)

class TrainFrames(Folder):

    """Parent folder of video frames. All nested folders holded video frames"""

    NAME = "balance_training_frames"

    def __init__(self):
        super().__init__(self.NAME)
        
        self.nested_folders: dict[ClassFrames] = {}
        not_video_frames = NotVideoFrames()

        for test_class_name in os.listdir(not_video_frames.path):
            classFrames_path = os.path.join(self.NAME,test_class_name)
            classFrames = ClassFrames(classFrames_path)
            self.nested_folders[test_class_name] = classFrames
    
    # check if video is correct and if yes, than video is framed 
    def process_video(self,video_name,imageModel,folder_video_name,SEPERATOR = "."):
        classFrames : ClassFrames = None
        class_name = ""
        exception = None
        error_message = ""

        try:
            class_name = video_name.split(SEPERATOR)[0]
            classFrames = self.nested_folders[class_name]

        except Exception:
            exception = ExtraVideo(video_name)
            error_message = str(exception)
            print(f"{error_message}")

            return 

        try:
            video_path = os.path.join(folder_video_name,video_name)
            classFrames.save_video_frames(video_path,imageModel)

        except Exception:
            exception = PlayVideoError(video_name)
            error_message = str(exception)
            print(f"{error_message}")
    
    # for processing all videos faster program uses multi threading 
    def process_all_videos(self,imageModel,FOLDER_VIDEO_NAME = "videos",COUNT_THREADS = 10):
            with concurrent.futures.ThreadPoolExecutor(max_workers=COUNT_THREADS) as executor:

                future_frames = {executor.submit(self.process_video,video_name,imageModel,FOLDER_VIDEO_NAME): 
                                video_name for video_name in os.listdir(FOLDER_VIDEO_NAME)}

                for future in concurrent.futures.as_completed(future_frames):
                    future.result()


#######################################################################################################################

# CREATE MAPING


class Maping:
    MAPING_NAME = "maping.json"

    """create maping between model labels and true class_names"""

    @classmethod
    def create_class_maping(clf):
        label_class_name_maping = {}

        for i,class_name in enumerate(os.listdir(TrainFrames.NAME)):
            label_class_name_maping[i] = class_name
        
        with open(clf.MAPING_NAME, "w", encoding="utf-8") as maping_json:
            json.dump(label_class_name_maping, maping_json)
        

#######################################################################################################################

# INPUT PROCESSING

class InputProcessor(ABC):
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def process(self,input_argument):
        pass 

class RegimeProcessor(InputProcessor):
    
    """Here we process input regime"""

    # We will get frames from video
    @classmethod
    def __frame_videos(clf):
        trainFrames = TrainFrames()
        imageModel = ImageEmbedder()
        trainFrames.process_all_videos(imageModel)

    # Main method for processing input
    @classmethod
    def process(clf,input_argument,WRONG_PARAMETER_TEXT ="Your input parameter was wrong!"):

        match input_argument:

            case Regime.FRAME:
                clf.__frame_videos()

            case Regime.SPLIT:
                not_video_frames = NotVideoFrames()
                not_video_frames.split()

            case Regime.BALANCE:
                balance_dataset = BalanceTrainingFrames()
                balance_dataset.balance()

            case Regime.MAPING:
                Maping.create_class_maping()

            case _:
                raise ValueError(WRONG_PARAMETER_TEXT)

#######################################################################################################################

if __name__ == "__main__":
    args = parser.parse_args()
    RegimeProcessor.process(args.regime)