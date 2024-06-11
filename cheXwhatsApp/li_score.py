import pandas as pd
import warnings

warnings.filterwarnings('ignore')

def test():
    print("Sahil")
    return

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

def li_score(df_pred, df_pred_wp, df_gt, iou_thresholds = [0.5]):
    
    df_iou_org = pd.DataFrame(columns=iou_thresholds)
    df_iou_org.columns = iou_thresholds
    
    for i in (range(len(df_gt))):
        name1 = df_gt['file_name'][i].split('_')[0]
        for j in range(len(df_pred)):
            name2 = df_pred['file_name'][j].split('_')[0]
            if name1==name2:
                # print(name1)
                bb1 = {'x1':df_gt.iloc[i]['x_min'], 'y1':df_gt.iloc[i]['y_min'], 'x2':df_gt.iloc[i]['x_max'],'y2': df_gt.iloc[i]['y_max']}
                bb2 = {'x1':df_pred.iloc[j]['x_min'], 'y1':df_pred.iloc[j]['y_min'], 'x2':df_pred.iloc[j]['x_max'],'y2': df_pred.iloc[j]['y_max']}
                
                if df_gt['class'][i] == df_pred['class'][j]:
                    score = _calculate_single_iou(bb1,bb2)

                else:
                    score = 0
                cnt_list = []
                
                for iou_threshold in iou_thresholds:
                    cnt = 0
                    if score>=iou_threshold:
                        cnt=1
                    cnt_list.append(cnt)
                # print(cnt_list)
                df_iou_org.loc[df_gt['file_name'][i]] = cnt_list
        #     break
        # break



    df_iou_wp = pd.DataFrame(columns=iou_thresholds)
    df_iou_wp.columns = iou_thresholds
    
    for i in (range(len(df_gt))):
        name1 = df_gt['file_name'][i].split('_')[0]
        for j in range(len(df_pred_wp)):
            name2 = df_pred_wp['file_name'][j].split('_')[0]
            if name1==name2:
                # print(name1)
                bb1 = {'x1':df_gt.iloc[i]['x_min'], 'y1':df_gt.iloc[i]['y_min'], 'x2':df_gt.iloc[i]['x_max'],'y2': df_gt.iloc[i]['y_max']}
                bb2 = {'x1':df_pred_wp.iloc[j]['x_min'], 'y1':df_pred_wp.iloc[j]['y_min'], 'x2':df_pred_wp.iloc[j]['x_max'],'y2': df_pred_wp.iloc[j]['y_max']}

                if df_gt['class'][i] == df_pred_wp['class'][j]:
                    score = _calculate_single_iou(bb1,bb2)
                    
                else:
                    score = 0

                cnt_list = []
                # cnt = 0
                for iou_threshold in iou_thresholds:
                    cnt = 0
                    if score>=iou_threshold:
                        cnt=1
                    cnt_list.append(cnt)
                # print(cnt_list)
                df_iou_wp.loc[df_gt['file_name'][i]] = cnt_list
        #     break
        # break
        
    df_iou_org = df_iou_org.reset_index()
    df_iou_wp = df_iou_wp.reset_index()
    
    
    df_pi = pd.DataFrame(columns=iou_thresholds)
    
    for i in (range(len(df_iou_org))):
        row_org = df_iou_org.iloc[[i]]
        row_wp = df_iou_wp.iloc[[i]]
        # print(row_org["Name"])
        # print(row_wp["Name"])
        cols = iou_thresholds
        ans = []
        for c in cols:
            # if(int(row_org[c]))
            if (int(row_org[c])>0 and int(row_wp[c])==0) or (int(row_org[c])==0 and int(row_wp[c])>0):
                ans.append(1)
            else:
                ans.append(0)
        
        df_pi.loc[df_iou_org.iloc[[i]]['index'].values [0]] = ans
        
    df_pi = df_pi.reset_index()
    
    unique_names = []
    for i in (range(len(df_pi))):
        if df_pi.iloc[i]['index'].split('_')[0] not in unique_names:
            unique_names.append(df_pi.iloc[i]['index'].split('_')[0])
    
    
    df_li = pd.DataFrame(columns=df_pi.columns)
    
    df_li = df_li.set_index('index')
    
    for i in (range(len(df_pi))):
        values = list(df_pi.iloc[i][iou_thresholds])
        if df_pi.iloc[i]['index'].split('_')[0] in df_li.index:
            # print(df2.loc[df.iloc[i]['index'].split('_')[0]] + values)
            df_li.loc[df_pi.iloc[i]['index'].split('_')[0]]  = df_li.loc[df_pi.iloc[i]['index'].split('_')[0]]  + values
        else:
            df_li.loc[df_pi.iloc[i]['index'].split('_')[0]]  = values
        
    for name in unique_names:
        org_values = list(df_li.loc[name])
        for i in range(len(org_values)):
            org_values[i] = int(org_values[i] > 0)
        df_li.loc[name] = org_values
        
    return df_li.mean().to_dict()