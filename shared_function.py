import shapely
from shapely.geometry import Polygon, MultiPoint
import numpy as np
from mean_average_precision import MetricBuilder


def polygon_iou(list1, list2):
    """
    Intersection over union between two shapely polygons.
    """
    polygon_points1 = np.array(list1).reshape(4, 2)
    poly1 = Polygon(polygon_points1).convex_hull
    polygon_points2 = np.array(list2).reshape(4, 2)
    poly2 = Polygon(polygon_points2).convex_hull
    union_poly = np.concatenate((polygon_points1, polygon_points2))
    if not poly1.intersects(poly2): # this test is fast and can accelerate calculation
        iou = 0
    else:
        try:
            inter_area = poly1.intersection(poly2).area
            #union_area = poly1.area + poly2.area - inter_area
            union_area = MultiPoint(union_poly).convex_hull.area
            if union_area == 0:
                return 0
            iou = float(inter_area) / union_area
        except shapely.geos.TopologicalError:
            print('shapely.geos.TopologicalError occured, iou set to 0')
            iou = 0
    return iou


def change_four_corners_to_bbox(four_corners_list):
    x_top_left, y_top_left, x_top_right, y_top_right,\
        x_bottom_right, y_bottom_right, x_bottom_left, y_bottom_left = four_corners_list

    xmin = min(x_top_left, x_bottom_left)
    ymin = min(y_top_left, y_top_right)
    xmax = max(x_bottom_right, x_top_right)
    ymax = max(y_bottom_right, y_bottom_left)

    return [xmin, ymin, xmax, ymax]


def change_bbox_to_four_corners(bbox_list):
    xmin, ymin, xmax, ymax = bbox_list

    x_top_left = xmin
    y_top_left = ymin
    x_top_right = xmax
    y_top_right = ymin
    x_bottom_right = xmax
    y_bottom_right = ymax
    x_bottom_left = xmin
    y_bottom_left = ymax

    return [x_top_left, y_top_left, x_top_right, y_top_right,
                x_bottom_right, y_bottom_right, x_bottom_left, y_bottom_left]


def calculate_AP(ovthresh, recs, detpath, classname, imagenames, cal_type='bbox'):
    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()
    if any(lines) == 1:
        # exchange dets as key-value dic
        # 1. pred bbox with gt bbox; 2. bbox from pred four corners with gt bbox
        preds_dic = {}
        splitlines = [x.strip().split(' ') for x in lines]
        for x in splitlines:
            bbox_used = None
            if cal_type == 'bbox':
                bbox_used = x[2:6]
            elif cal_type == 'bbox_from_fc':
                bbox_used = change_four_corners_to_bbox(x[6:14])
            else:
                assert False, "wrong AP calculation type"
            if x[0] in preds_dic.keys():
                preds_dic[x[0]] = np.vstack((preds_dic[x[0]], np.append(np.array([float(z) for z in bbox_used]), [int(0), float(x[1])])))
            else:
                preds_dic[x[0]] = np.append(np.array([float(z) for z in bbox_used]), [int(0), float(x[1])])
                preds_dic[x[0]] = np.expand_dims(preds_dic[x[0]], axis=0)

        # create metric
        metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=True, num_classes=1)
        # extract gt objects for this class
        for imagename in imagenames:
            R = [obj for obj in recs[imagename] if obj['name'] == classname]
            GT_bbox = np.array([x['bbox'] for x in R])
            GT_append = np.zeros((GT_bbox.shape[0], 3), dtype=np.int32)
            GTs = np.hstack((GT_bbox, GT_append))
            if imagename not in preds_dic.keys():
                preds = np.array([[]])
            else:
                preds = preds_dic[imagename]
            metric_fn.add(preds, GTs)

        # metrics = metric_fn.value(iou_thresholds=ovthresh)
        # rec = metrics[ovthresh][0]['recall']
        # prec = metrics[ovthresh][0]['precision']
        # ap = metrics[ovthresh][0]['ap']
        metrics = metric_fn.value(iou_thresholds=np.arange(0.5, 1.0, 0.05), recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')
        map = metrics["mAP"]

    return metrics, map


def calculate_F1_score(ovthresh, recs, detpath, classname, cal_type='fc'):
    class_recs = {}
    npos = 0
    for imagename in recs.keys():
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        four_corners = np.array([x['four_corners'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool_)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'four_corners': four_corners,
                                'difficult': difficult,
                                'det': det}

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()
    if any(lines) == 1:
        splitlines = [x.strip().split(' ') for x in lines]
        image_ids = [x[0] for x in splitlines]
        confidence = np.array([float(x[1]) for x in splitlines])
        conf_idx = confidence > 0.5
        image_ids = np.array(image_ids)[conf_idx]
        image_ids = image_ids.tolist()
        confidence = confidence[conf_idx]
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines])
        BB = BB[conf_idx, :]

        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d in range(nd):
            R = class_recs[image_ids[d]]
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R['four_corners'].astype(float)
            if cal_type == 'fc':
                bb = bb[4:]
            elif cal_type == 'fc_from_bbox':
                bb = np.array(change_bbox_to_four_corners(bb[:4]))
            else:
                assert False, "wrong f1 calculation type"
            if BBGT.size > 0:
                # compute overlaps
                overlaps = np.zeros(BBGT.shape[0])
                for idx in range(BBGT.shape[0]):
                    overlap = polygon_iou(bb.tolist(), BBGT[idx, :].tolist())
                    overlaps[idx] = overlap
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not R['difficult'][jmax]:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.
            else:
                fp[d] = 1.

        tp_num = np.sum(tp)
        fp_num = np.sum(fp)
        rec = tp_num / float(npos)
        prec = tp_num / float(tp_num + fp_num)
        F1 = 2.*prec*rec/(prec+rec)

    return rec, prec, F1
