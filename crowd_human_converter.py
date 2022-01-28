import json
import os
from PIL import Image

# load ogdt data
def load_file(fpath):  #fpath is the path of the file, to convert string to list
    assert os.path.exists(fpath)  #assert() raise-if-not
    with open(fpath,'r') as fid:
        lines = fid.readlines()
    records = [json.loads(line.strip('\n')) for line in lines] #str to list
    return records

def crowdhuman2coco(odgt_path,json_path,image_root,prefix):#input: ogdt_path, output: json_path
    records = load_file(odgt_path)
    #preprocess
    json_dict = {"images":[], "annotations": [], "categories": []} # define a dictionary with coco data annotation format
    START_B_BOX_ID = 1 # set the starting id of box
    image_id = 1
    bbox_id = START_B_BOX_ID
    image = {} # dictionay to record image
    annotation = {} # dictionary to record annotation
    categories = {}  # dictionary to record categories
    record_list = len(records)  
    print(record_list)
    
    # process line by line
    for i in range(record_list):
        file_name = records[i]['ID']+'.jpg'  
        #print(file_name)
        im = Image.open(image_root+file_name)
        image = {'file_name': prefix+file_name, 'height': im.size[1], 'width': im.size[0],'id':image_id} #im.size[0]，im.size[1] are width and height
        json_dict['images'].append(image) # convdrt one line to jason

        gt_box = records[i]['gtboxes']  
        gt_box_len = len(gt_box) 
        for j in range(gt_box_len):
            category = gt_box[j]['tag']
            
            #if category not in categories:  
            #    new_id = len(categories) + 1 
            #    categories[category] = new_id
            #category_id = categories[category]
              
            if category == "person":
                category_id = 1
                #fbox = gt_box[j]['fbox']  
                fbox = gt_box[j]['hbox']  # modified by crpan: area is calculated as hbox width * height
                if fbox[0] < 0 or fbox[1] < 0 or fbox[0]+fbox[2] > im.size[0] or fbox[1]+fbox[3] > im.size[1]:
                    continue
                else:
                    ignore = 0 
                    if "ignore" in gt_box[j]['head_attr']:
                        ignore = gt_box[j]['head_attr']['ignore']
                    if "ignore" in gt_box[j]['extra']:
                        ignore = gt_box[j]['extra']['ignore']
            
                    annotation = {'area': fbox[2]*fbox[3], 'iscrowd': ignore, 'image_id':
                                  image_id, 'bbox':gt_box[j]['hbox'], 'category_id': category_id,'id': bbox_id,'ignore': ignore,'segmentation': []}  
                    json_dict['annotations'].append(annotation)
                    bbox_id += 1
        categories["head"] = 1 # Let originally "person" category renamed as "head"
        image_id += 1 
    for cate, cid in categories.items():
        cat = {'supercategory': 'none', 'id': cid, 'name': cate}
        json_dict['categories'].append(cat)
    json_fp = open(json_path, 'w')
    json_str = json.dumps(json_dict,indent=4) 
    json_fp.write(json_str)
    json_fp.close()

crowdhuman2coco("./annotation_train.odgt", 
                "./train.json",
                "./Images/train/",
                "CrowdHuman/Images/train/")
