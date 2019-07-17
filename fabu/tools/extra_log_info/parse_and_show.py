#coding:utf-8
#!/usr/bin/env python
import inspect
import random
import matplotlib.cm as cmx
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.legend as lgd
import matplotlib.markers as mks
import os 
import numpy as np 
import sys
import argparse

def get_log_parsing_script():
    dirname = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parser_py=dirname + '/parse_log.py'
    print("the parse_log script is : {}".format(parser_py))
    return parser_py

def parse_log(log_file):
	'''
	inout: train.log
	output: train.log.test
			train.log.test
	'''
	output_file_path=os.path.dirname(log_file)
	os.system('%s %s %s %s' % ("python",get_log_parsing_script(), log_file, output_file_path))

def get_log_file_suffix():
    return '.log'

def get_chart_type_description_separator():
    return '  vs. '

def is_x_axis_field(field):
    x_axis_fields = ['Iters', 'Seconds']
    return field in x_axis_fields

def create_field_index():
    train_key = 'Train'
    test_key = 'Test'
    field_index = {train_key:{'Iters':0, 'Seconds':1, train_key + ' learning rate':2,
                              train_key + ' loss':3},
                   test_key:{'Iters':0, 'Seconds':1, test_key + ' learning rate':2,
                             test_key + ' acc1':3, test_key + ' acc5':4, test_key + ' loss':5}}
    fields = set()
    for data_file_type in field_index.keys():
        fields = fields.union(set(field_index[data_file_type].keys()))
    fields = list(fields)
    fields.sort()
    return field_index, fields

def get_supported_chart_types():
    field_index, fields = create_field_index()
    num_fields = len(fields)
    supported_chart_types = []
    for i in xrange(num_fields):
        if not is_x_axis_field(fields[i]):
            for j in xrange(num_fields):
                if i != j and is_x_axis_field(fields[j]):
                    supported_chart_types.append('%s%s%s' % (
                        fields[i], get_chart_type_description_separator(),
                        fields[j]))
    return supported_chart_types

def get_chart_type_description(chart_type):
    supported_chart_types = get_supported_chart_types()
    chart_type_description = supported_chart_types[chart_type]
    return chart_type_description

def get_data_file_type(chart_type):
    description = get_chart_type_description(chart_type)
    data_file_type = description.split()[0]
    return data_file_type

def get_data_file(chart_type, path_to_log):
    data_file_name=os.path.basename(path_to_log) + '.' + get_data_file_type(chart_type).lower()
    data_file_name_dir=os.path.join(os.path.dirname(path_to_log),data_file_name)
    return data_file_name_dir

def get_field_descriptions(chart_type):
    description = get_chart_type_description(chart_type).split(
        get_chart_type_description_separator())
    y_axis_field = description[0]
    x_axis_field = description[1]
    return x_axis_field, y_axis_field    

def get_field_indecies(chart_type,x_axis_field, y_axis_field):    
    data_file_type = get_data_file_type(chart_type)
    fields = create_field_index()[0][data_file_type]
    return fields[x_axis_field], fields[y_axis_field]

def load_data(data_file, field_idx0, field_idx1):
    data = [[], []]
    with open(data_file, 'r') as f:
        for line in f:
            line = line.strip()
            if "NumIters" in line:
            	continue
            else:
                fields = line.split(",")
                data[0].append(float(fields[field_idx0].strip()))
                data[1].append(float(fields[field_idx1].strip()))
    return data

def random_marker():
    markers = mks.MarkerStyle.markers
    num = len(markers.values())
    idx = random.randint(0, num - 1)
    return markers.values()[idx]

def get_data_label(path_to_log):
    label = path_to_log[path_to_log.rfind('/')+1 : path_to_log.rfind(
        get_log_file_suffix())]
    return label

def get_legend_loc(chart_type):
    x_axis, y_axis = get_field_descriptions(chart_type)
    loc = 'lower right'
    if y_axis.find('accuracy') != -1:
        pass
    if y_axis.find('loss') != -1 or y_axis.find('learning rate') != -1:
        loc = 'upper right'
    return loc

def plot_chart(chart_type, path_to_png, path_to_log):
    #for path_to_log in path_to_log_list:
        # os.system('%s %s' % (get_log_parsing_script(), path_to_log))
    # get log_dir.test/train
    data_file = get_data_file(chart_type, path_to_log)
    # get  iter/seconds  loss/train 
    x_axis_field, y_axis_field = get_field_descriptions(chart_type)
    # get the num of  x_axis_field  and the num of y 
    x, y = get_field_indecies(chart_type,x_axis_field, y_axis_field)
    # get the x data and y data from log_dir.test or log_dir.train
    data = load_data(data_file, x, y)
    ## TODO: more systematic color cycle for lines
    # plot picture
    color = [random.random(), random.random(), random.random()]
    label = get_data_label(path_to_log)
    linewidth = 0.75
    ## If there too many datapoints, do not use marker.
##        use_marker = False
    use_marker = True
    if not use_marker:
        plt.plot(data[0], data[1], label = label, color = color,
                 linewidth = linewidth)
    else:
        ok = False
        ## Some markers throw ValueError: Unrecognized marker style
        while not ok:
            try:
                marker = random_marker()
                plt.plot(data[0], data[1], label = label, color = color,
                         marker = marker, linewidth = linewidth)
                ok = True
            except:
                pass
    legend_loc = get_legend_loc(chart_type)
    plt.legend(loc = legend_loc, ncol = 1) # ajust ncol to fit the space
    plt.title(get_chart_type_description(chart_type))
    plt.xlabel(x_axis_field)
    plt.ylabel(y_axis_field)  
    plt.savefig(path_to_png)     
    plt.show()


# def plot_all(train_log_name,test_log_name):







def print_help():
    print"""This script mainly serves as the basis of your customizations.
Customization is a must.
You can copy, paste, edit them in whatever way you want.
Be warned that the fields in the training log may change in the future.
You had better check the data files and change the mapping from field name to
 field index in create_field_index before designing your own plots.
Usage:
    ./plot_log.sh chart_type[0-%s] /where/to/save.png /path/to/first.log ...
Notes:
    1. Supporting multiple logs.
    2. Log file name must end with the lower-cased "%s".
Supported chart types:""" % (len(get_supported_chart_types()) - 1,
                             get_log_file_suffix())
    supported_chart_types = get_supported_chart_types()
    num = len(supported_chart_types)
    for i in xrange(num):
        print '    %d: %s' % (i, supported_chart_types[i])
    exit

def is_valid_chart_type(chart_type):
    return chart_type >= 0 and chart_type < len(get_supported_chart_types())
 

def main():
    description = ('Parse a Caffe training log into two CSV files '
                   'And create the acc and loss pic for training & test process')
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--logfile',
                        help='Path to log file')
    args = parser.parse_args()
    log_file=args.logfile
    # 首先解析logfile得到.train and .test 解析文件
    parse_log(log_file)
    # 解析得到的.train and .test 的解析文件生成需要的图片
    train_log_name=log_file+".train"
    test_log_name=log_file+".test"
    assert os.path.basename(train_log_name) in os.listdir(os.path.dirname(log_file)),"the train log file is not exists"
    assert os.path.basename(test_log_name) in os.listdir(os.path.dirname(log_file)),"the test log file is not exists"
    # 调用parser_training_log.py里面的思路进行文件解析生成图片
    print_help()
    print("-------------Now we create all the pic we need-------------")
    supported_chart_types = get_supported_chart_types()
    num = len(supported_chart_types)
    # create all the picture
    for i in range(num):
    	description_type= supported_chart_types[i]
    	print('    %d: %s' % (i, description_type))
    	save_pic_name=description_type.replace(" ","_")+".png"
    	print("sace picture name is :{} ".format(save_pic_name))
    	save_pic_dir=os.path.join(os.path.dirname(log_file),save_pic_name)
        plot_chart(i, save_pic_dir, log_file)

if __name__=="__main__":
	main()

