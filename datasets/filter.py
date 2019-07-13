

    def filter_ids(self):
        # filter bad annotations
        ids = []
        for ann_id in self.ids:
            ann = self.coco.anns[ann_id]
            if not has_valid_annotation([ann]):
                continue
            if ann["area"] < self.area_threshold:
                continue
            if ann["score"] < self.score_threshold:
                continue
            ids.append(ann_id)
        self.ids = ids

        # correct image path
        for img_id in self.coco.imgs:
            img = self.coco.imgs[img_id]
            if "ade_challenge/images/" in img["file_name"]:
                img["file_name"] = img["file_name"].replace("ade_challenge/images/", "")

    def filter_category(self, cat_name):
        ids = []
        cat_name_to_cat_id = {
            cat["name"]: cat["id"] for cat in self.coco.dataset["categories"]
        }
        cat_id = cat_name_to_cat_id[cat_name]
        for ann_id in self.ids:
            ann = self.coco.anns[ann_id]
            if ann["category_id"] == cat_id:
                ids.append(ann_id)
        self.ids = ids
        

    def prepare_input_with_mask(self, img, ann):
        image_path = os.path.join(self.root, img["file_name"])
        image = Image.open(image_path).convert('RGB')
        mask = mask_utils.decode(ann["segmentation"])  # [h, w, n]
        bbox = ann["bbox"]

        image = np.array(image)
        mask = mask * 255
        image = crop_bbox(image, bbox, margin=0.2)
        mask = crop_bbox(mask, bbox, margin=0.2)

        if self.transform:
            image = Image.fromarray(image)
            mask = Image.fromarray(mask)
            # Hack to ensure transform is the same
            seed = random.randint(0,2**32)
            random.seed(seed)
            image = self.transform(image)
            random.seed(seed)
            mask = self.transform(mask)
            image = torch.cat([image, mask])
        return image