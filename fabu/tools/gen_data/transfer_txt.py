import os,sys
import glob

CLASSES12 = ('__background__', # always index 0
          'car', 'van', 'bus', 'truck', 'person', 'person-sitting', 'bicycle', 'motor', 'open-tricycle', 'close-tricycle', 'forklift')   # 12
CLASSES9 = ('__background__','car', 'bus', 'truck', 'person', 'bicycle', 'motor', 'tricycle', 'block')
CLASSES7 = ('__background__','car', 'bus', 'truck', 'person', 'bicycle_motor', 'tricycle')
CLASSES = CLASSES9

cls9_map = {
    'car': 'car',
    'van': 'car',
    'bus': 'bus',
    'truck': 'truck',
    'forklift': 'truck',
    'person': 'person',
    'person-sitting': 'person',
    'bicycle': 'bicycle',
    'motor': 'bicycle',
    'parked-bicycle': 'bicycle',
    'parked-motor': 'bicycle',
    'open-tricycle': 'tricycle',
    'close-tricycle': 'tricycle',
    'water-block': 'block',
    'cone-block': 'block',
    'crash-block': 'block',
    'triangle-block': 'block',
    'small-block': 'block',
    'warning-block': 'block',
    'large-block': 'block',
    'other-block': 'block',
}

list_path = ""
annotations_path = "/wangwenxiao/adas_dataset/dataset_list/adas_train.txt"
out_path = "/wangwenxiao/adas_dataset/adas_annotations/"

def transfer(data_root):
    ''' transfer the annotation format
    from: [xmin ymin xmax ymax occlusion_level class_string]
    to: [class_id xmin ymin xmax ymax]
    '''
    global out_path
    global cls9_map
    dataset = ''
    out_root = out_path

    # get txt list
    annot_dir = os.path.join(data_root, dataset, 'Annotations')
    splits = ['train', 'val']
    for split in splits:
        listfile = os.path.join(annotations_path)
        print(listfile)
        with open(listfile, 'r') as f:
            lines = f.read().splitlines()

        for line in lines:
            print(annot_dir)
            print(line + '.txt')
            from_txt = os.path.join(annot_dir, line + '.txt')
            to_txt = os.path.join(out_root, line + '.txt')
            if os.path.exists(to_txt):
                continue

            if not os.path.exists(os.path.dirname(to_txt)):
                os.makedirs(os.path.dirname(to_txt))
            with open(from_txt, 'r') as f_from, open(to_txt, 'w') as f_to:
                for l in f_from.readlines():
                    label = l.split()

                    # skip heavy occlusion instance
                    if int(label[4]) == 2:
                        continue

                    try:
                        cls = cls9_map[label[5]]
                        if cls == 'block' or cls == 'tricycle':
                            continue

                        w = float(label[2]) - float(label[0])
                        h = float(label[3]) - float(label[1])

                        if (label == 'car' and (w < 45 or h < 45)) or \
                           (w < 24 or h < 24):
                           continue

                        f_to.write('{} {}\n'.format(CLASSES.index(cls), ' '.join(label[:4])))
                    except:
                        print('{} not in class map'.format(label[5]))
                        pass

        with open(os.path.join(out_root, dataset, split+'.txt'), 'w') as f:
            for l in lines:
                f.write('{}.jpg {}.txt\n'.format(os.path.join(data_root, 'JPGImages', l), os.path.join(out_root, l)))
                # f.write('{}.jpg {}.txt\n'.format(os.path.join(dataset, 'JPGImages', l), os.path.join(out_root, 'annotations', l)))
        if 'val' in split or 'test' in split:
            with open(os.path.join(out_root, dataset, split+'_name_size.txt'), 'w') as f:
                for l in lines:
                    f.write('{} {} {}\n'.format(os.path.basename(l), 400, 640))
        print(split + ' done!')


if __name__ == '__main__':
    #data_root = '/home/ningqingqun/codes/caffe_ssd/data/zy_data'
    #data_root = '/home/ningqingqun/codes/caffe_ssd/data'
    # data_root = '/home/yaowanchao/code/mobilenet_ssd/data'
    data_root = "/private/ningqingqun/vision_detector_data/vision_data/"
    transfer(data_root)
