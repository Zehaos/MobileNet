import numpy as np

image_set_dir = './KITTI/ImageSets'
trainval_file = image_set_dir+'/trainval.txt'
train_file = image_set_dir+'/train.txt'
val_file = image_set_dir+'/val.txt'

idx = []
with open(trainval_file) as f:
  for line in f:
    idx.append(line.strip())
f.close()

idx = np.random.permutation(idx)

train_idx = sorted(idx[:len(idx)/2])
val_idx = sorted(idx[len(idx)/2:])

with open(train_file, 'w') as f:
  for i in train_idx:
    f.write('{}\n'.format(i))
f.close()

with open(val_file, 'w') as f:
  for i in val_idx:
    f.write('{}\n'.format(i))
f.close()

print 'Trainining set is saved to ' + train_file
print 'Validation set is saved to ' + val_file