# Deform rate detect
The project is divided into two parts:
## 1. label_statistics
Get a specific labeled json file and make statistic about the details of the deform rate.

Run label_statistics/file_process.ipynb.

## 2. auto_scoring
Get deform rate scores for a group of input images in a given folder.

For methods that have not yet been added, a default_prob is temporarily given as 0.5.
```
python auto_scoring.py \
--image_dir_file 'test_images/' \
--score_save_dir 'deform_rate_score/'
```
### 2.1 Result presentation
| image_id | image | result |
|---------|---------|---------|
| 9855   | ![9855.jpg](auto_scoring/test_images/9855.jpg)   | [9855.json](auto_scoring/deform_rate_score/9855.json)   |
| 9856   | ![9856.jpg](auto_scoring/test_images/9856.jpg)   | [9856.json](auto_scoring/deform_rate_score/9856.json)   |
| 9857   | ![9857.jpg](auto_scoring/test_images/9857.jpg)   | [9857.json](auto_scoring/deform_rate_score/9857.json)   |
