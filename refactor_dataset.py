from pathlib import Path
from shutil import copyfile
from bisect import bisect_right
import json

class DatasetRefactorer():

    def __init__(self, dataset_srcs : list[Path], dataset_dest : Path, labels : Path):
        self.dataset_srcs = dataset_srcs
        self.dataset_dest = dataset_dest
        self.label_path = labels
        self.label_ranges = []
                
        #initialize range of labels for each src directory
        for data_path in self.dataset_srcs:
            low, high = self.get_label_range(data_path)
            self.label_ranges.append((low, high, data_path))
       
        # initialize lookup list for video directories
        self.start_list = [int(i[0]) for i in self.label_ranges]

        self.label_data = None


    def refactor_dataset(self):
        label_path = self.label_path 
        label_data = []

        #load in label data
        with open(label_path, 'r') as file:
            label_data = json.load(file)
        
        for label in label_data:

            #get the video directory using video_path
            video_path = Path(label['video_path'])
            video_name = video_path.stem
            video_location = self.get_video_directory(int(video_name)) / f'{video_name}.mp4'
            
            #get the video category_label
            video_category = label['label']

            #get video split
            video_split = label['split']

            # compute dest using root/split/category_label/video_path
            dest_path = self.dataset_dest / str(video_split) / str(video_category).replace(' ', '_') / f'{video_name}.mp4'
            if not dest_path.parent.exists():
                dest_path.parent.mkdir(parents=True, exist_ok=True)
            if not dest_path.exists():
                copyfile(video_location, dest_path)
                print(f'Wrote file {video_location} to {dest_path}')   

    def get_label_range(self, data_path : Path):
        #returns inclusive range of video names
        low = None
        high = None
        for video_path in data_path.glob('*.mp4'):
            vid_num = int(video_path.stem)
            if(low is None):
                low = vid_num
            else:
                low = min(vid_num, low)
            
            if(high is None):
                high = vid_num
            else:
                high = max(vid_num, high)
        return (low, high)
    
    def get_video_directory(self, video_num : int):
        index = bisect_right(self.start_list, video_num) - 1
        return self.label_ranges[index][2]

def main():

    srcs = []
    srcs.append(Path('./dataset/QEVD-FIT-300k-Part-1'))
    srcs.append(Path('./dataset/QEVD-FIT-300k-Part-2'))
    srcs.append(Path('./dataset/QEVD-FIT-300k-Part-3'))
    srcs.append(Path('./dataset/QEVD-FIT-300k-Part-4'))
    dest = Path('./full_dataset')

    refactor = DatasetRefactorer(srcs, dest, Path('fine_grained_labels_release.json') )
    refactor.refactor_dataset()

if __name__ == '__main__':
    main()


