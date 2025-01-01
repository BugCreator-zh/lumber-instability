import time

import numpy as np
from PIL import Image
import os
from yolo import YOLO

file_path = r'C:\JLUstudy\code\python\project\data\515_dataset\total_TRA'
save_path = r'C:\JLUstudy\code\python\project\data\515_dataset\cropped_TRA'
yolo = YOLO()

if not os.path.exists(save_path):
    os.mkdir(save_path)

for i in os.listdir(file_path):
    img = file_path+'\\' + i
    image = Image.open(img)
    result = yolo.detect_image(image, crop = True, count=True,sample_id = i[:5])
    result.save(save_path+'\\'+i)
    
    
    
    
    
    
"""
SAG：
11
23
52
53
78
108
113
132
139
177
194
195
201
230
243
245
254
265
297
322
333
336
339
371
372
379
380
383
391
396
401
410
442
451
454
461
472
487
497
501
510
523
533
542
550
557
560
"""