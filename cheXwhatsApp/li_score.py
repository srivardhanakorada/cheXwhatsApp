import pandas as pd
import shutil
import os
import warnings
warnings.filterwarnings('ignore')

def recreate_folder(folder_path):
    # Check if the folder exists
    if os.path.exists(folder_path):
        # Remove the folder and all its contents
        shutil.rmtree(folder_path)
        # print(f"Folder '{folder_path}' and all its contents have been removed.")
    # else:
    #     print(f"Folder '{folder_path}' does not exist, so it will be created.")
    
    # Create the folder again
    os.makedirs(folder_path)


def _calculate_single_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'y1', 'x2', 'y2'}
        Keys: {xMinCorr, yMaxCorr, xMaxCorr, yMinCorr}
        The (x1, y1) position is at the bottom left corner,
        the (x2, y2) position is at the top right corner
    bb2 : dict
        Keys: {'x1', 'y1', 'x2', 'y2'}
        The (x1, y1) position is at the bottom left corner,
        the (x2, y2) position is at the top right corner

    Returns
    -------
    float
        in [0, 1]
    """
    # print("BB1",bb1)
    # print(bb1['x1'])
    # print(bb1['x2'])
    # print(bb1['y1'])
    # print(bb1['y2'])
    # print(bb1, bb2)
    if(bb2['x1']==0 and bb2['x2']==0 and bb2['y1']==0 and bb2['y2']==0):
        return 0.0
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = abs((x_right - x_left) * (y_bottom - y_top))

    # compute the area of both AABBs
    bb1_area = abs((bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1']))
    bb2_area = abs((bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1']))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def get_iou_for_bbox(df_original, filename, df_gt):
    # print(df_original.head())
    df_original['iou_score'] = 0
    # print(df_original.head())
    
    # df_gt = pd.read_csv(gt_file)
    
    for index, row in df_original.iterrows():
        # print(row['Name'], row['label'])
        df_gt_sub = df_gt[(df_gt['Name']==row['Name']) & (df_gt['class']==row['label'])]
        # print(df_gt_sub.shape)
        if df_gt_sub.shape[0] > 0:
            max_iou_score = 0
            for index2, row2 in df_gt_sub.iterrows():
                bb1 = {'x1':row['x_min'], 'y1':row['y_min'], 'x2':row['x_max'], 'y2':row['y_max']}
                bb2 = {'x1':row2['x_min'], 'y1':row2['y_min'], 'x2':row2['x_max'], 'y2':row2['y_max']}
                iou_score = _calculate_single_iou(bb1, bb2)
                if iou_score > max_iou_score:
                    max_iou_score = iou_score
                    # print(row['Name'], max_iou_score)
            df_original.loc[index,'iou_score'] = max_iou_score
    df_original.to_csv(filename, index=False)            

def group_by_max(df, group_col):
    """Groups a DataFrame and merges other columns using max for each group.   
    Args:       df (pd.DataFrame): The DataFrame to group.       
    group_col (str): The column name to group by.   
    Returns:       pd.DataFrame: A new DataFrame with grouped and merged data.   """
    # Group by the specified column  
    grouped_df = df.groupby(group_col)   
    # Apply max aggregation to all columns except the grouping column  
    aggregated_df = grouped_df.agg(max) 
    return aggregated_df


def create_labeled_list(class_list):
    labeled_list = []
    for item in class_list:
        # Use f-string for concise string formatting with the label
        labeled_list.append(f"label_{int(item)}")
    return labeled_list


def create_thresholdwise_labels(df, filename, thresholds, class_list):
    # print(df.shape)
#     df = df[df['score'] > 0.4]
    # print(df.shape)
    
    for t in thresholds:
        df_new = pd.DataFrame(columns=['Name']+class_list)
    

        for index, row in df.iterrows():
            row_new = {}
            row_new['Name'] = row['Name']
            for label in class_list:
                row_new[label] = 0
            
            # print(row['Name'], row['iou_score'])
            # print(row['Name'], row['iou_score'])
            if float(row['iou_score'])>=t:
                
                row_new['label_'+str(int(row['label']))] = 1
            df_new = df_new._append(row_new, ignore_index=True)
        # print(df_new.head())   
        grouped_by_max_df = group_by_max(df_new.copy(), 'Name') 
        # print(grouped_by_max_df.head())   
        grouped_by_max_df.to_csv(filename + "_threshold" +str(t)+".csv")


def get_lis_score(dir_original, dir_whatsapp, thresholds, class_list):
    ans = dict()
    for t in thresholds:
        df_orig = pd.read_csv(os.path.join(dir_original, "hr_threshold" + str(t)+".csv"))
#         print(df_orig.head())
        df_wa = pd.read_csv(os.path.join(dir_whatsapp, "lr_threshold" + str(t)+".csv"))
#         print(df_wa.head())
        total = 0
        instabile = 0
        for index, row in df_orig.iterrows():
            df_wa_sub = df_wa[df_wa['Name']==row['Name']]
            # print(df_wa_sub)
            for index, row_wa in df_wa_sub.iterrows():
                total += 1
                for label in class_list:
                    if row[label] != row_wa[label]:
#                         print("Instable")
                        instabile += 1
                        break
        ans[t] = instabile/ total
        # print("Threshold : {} , instability = {}".format(t, instabile/ total))                
    return ans            
    

def li_score(df_hr, df_lr, df_gt_hr, df_gt_lr, iou_thresholds = [0.5]):
    current_dir = os.getcwd()
    temp_path = os.path.join(current_dir,'li_score_temp')
    recreate_folder(temp_path)
    get_iou_for_bbox(df_hr, os.path.join(temp_path,'iou_score_hr.csv'), df_gt_hr)
    get_iou_for_bbox(df_lr, os.path.join(temp_path,'iou_score_lr.csv'), df_gt_lr)
    class_list = list(set(df_gt_hr['class'].to_list()))
    class_list = create_labeled_list(class_list)
    recreate_folder(os.path.join(temp_path,'hr'))
    recreate_folder(os.path.join(temp_path,'lr'))
    df_iou_hr = pd.read_csv(os.path.join(temp_path,'iou_score_hr.csv'))
    create_thresholdwise_labels(df_iou_hr, os.path.join(os.path.join(temp_path,'hr'), 'hr'), thresholds=iou_thresholds, class_list=class_list)
    df_iou_lr = pd.read_csv(os.path.join(temp_path,'iou_score_lr.csv'))
    create_thresholdwise_labels(df_iou_lr, os.path.join(os.path.join(temp_path,'lr'), 'lr'), thresholds=iou_thresholds, class_list=class_list)
    ans = get_lis_score(os.path.join(temp_path,'hr'), os.path.join(temp_path,'lr'), iou_thresholds, class_list)
    return ans
