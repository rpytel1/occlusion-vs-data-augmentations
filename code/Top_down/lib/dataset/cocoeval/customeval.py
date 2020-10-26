from pycocotools.cocoeval import COCOeval

from lib.utils.cutout_util import get_cords_with_width, is_within_box
import copy
import numpy as np

from lib.utils.keypoints_constants import part_mapping, keypoint_names


class CustomEval(COCOeval):

    def __init__(self, cocoGt=None, cocoDt=None, iouType='segm'):
        super().__init__(cocoGt, cocoDt, iouType)
        self.params.occlusion = ""
        self.params.keypoint = ""
        self.params.print = True

    def _prepare(self):
        super()._prepare()
        self._gts = copy.deepcopy(self._gts)
        self._dts = copy.deepcopy(self._dts)

        for key in self._gts.keys():
            image_ann_gts = self._gts[key]
            image_ann_dts = self._dts[key]
            ids_to_remove = self.calc_ids_to_remove(image_ann_gts)
            self._gts[key] = self.remove_anns(ids_to_remove, image_ann_gts)
            self._dts[key] = self.remove_anns(ids_to_remove, image_ann_dts)

    def calc_ids_to_remove(self, gts):
        ids_list = []
        for ann in gts:
            new_ids = self.calc_new_ids_to_remove(ann['keypoints'])
            ids_list = ids_list + new_ids
        return list(set(ids_list))

    def calc_new_ids_to_remove(self, keypoints):
        indexes = []
        if self.params.occlusion in part_mapping["coco"].keys():
            indexes = self.occluded_keypoints(keypoints)
        if self.params.keypoint in keypoint_names["coco"]:
            keypoint = keypoint_names["coco"].index(self.params.keypoint)
            targetted = list(range(17))
            targetted.remove(keypoint)
            indexes.extend(targetted)
            indexes = list(set(indexes))
        return indexes

    def occluded_keypoints(self, keypoints):
        ids_to_remove = []

        keypoint_dict = self.extract_keypoint_list(keypoints)
        cords_with_width = get_cords_with_width(keypoint_dict, self.params.occlusion, dataset="coco")
        if cords_with_width:
            center_pos, widths = cords_with_width
            for id in range(len(keypoints) // 3):
                pos = (keypoints[3 * id], keypoints[3 * id + 1])
                if is_within_box(pos, center_pos, widths):
                    ids_to_remove.append(id)
        return ids_to_remove

    def remove_anns(self, ids_to_remove, image_ann):
        new_img_ann = []
        for ann in image_ann:
            keypoints = ann['keypoints']
            for e in ids_to_remove:
                keypoints[3 * e] = 0
                keypoints[3 * e + 1] = 0
                keypoints[3 * e + 2] = 0
            ann['keypoints'] = keypoints
            new_img_ann.append(ann)
        return new_img_ann

    def extract_keypoint_list(self, keypoints):
        new_elem = {}
        for i in range(len(keypoints) // 3):
            x, y = keypoints[3 * i], keypoints[3 * i + 1]
            if x != 0 and y != 0:
                new_elem[keypoint_names["coco"][i]] = (x, y)
        return new_elem

    def summarize(self):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        '''

        def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=100):
            p = self.params
            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap == 1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, :, aind, mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, aind, mind]
            if len(s[s > -1]) == 0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s > -1])
            if self.params.print:
                print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
            return mean_s

        def _summarizeKps():
            stats = np.zeros((10,))
            stats[0] = _summarize(1, maxDets=20)
            stats[1] = _summarize(1, maxDets=20, iouThr=.5)
            stats[2] = _summarize(1, maxDets=20, iouThr=.75)
            stats[3] = _summarize(1, maxDets=20, areaRng='medium')
            stats[4] = _summarize(1, maxDets=20, areaRng='large')
            stats[5] = _summarize(0, maxDets=20)
            stats[6] = _summarize(0, maxDets=20, iouThr=.5)
            stats[7] = _summarize(0, maxDets=20, iouThr=.75)
            stats[8] = _summarize(0, maxDets=20, areaRng='medium')
            stats[9] = _summarize(0, maxDets=20, areaRng='large')
            return stats

        if not self.eval:
            raise Exception('Please run accumulate() first')
        self.stats = _summarizeKps()
