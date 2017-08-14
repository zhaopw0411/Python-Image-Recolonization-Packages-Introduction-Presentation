from PIL import Image
import glob
import numpy as np
from pandas import Series, DataFrame
import pandas as pd
from sklearn.cluster import KMeans
from shutil import copyfile
from sklearn import mixture


class PixelCounter(object):
  ''' loop through each pixel and average rgb '''
  def __init__(self, imageName):
      self.pic = Image.open(imageName)
      # load image data
      self.imgData = self.pic.load()
  def averagePixels(self):
      r, g, b = 0, 0, 0
      count = 0
      for x in xrange(self.pic.size[0]):
          for y in xrange(self.pic.size[1]):
              tempr,tempg,tempb = self.imgData[x,y]
              r += tempr
              g += tempg
              b += tempb
              count += 1
      # calculate averages
      return [r/count,g/count,b/count]
      # return (r/count), (g/count), (b/count), count

  
image_list = []
image_name= []
for filename in glob.glob('/Users/zeyuanli/Desktop/6310/trends market place/Pictures/2lake/*.jpeg'): #assuming gif
    a=PixelCounter(filename)
    image_list.append(a.averagePixels())
    image_name.append(filename)

a=np.array(image_list)

list1 = [
         ('R', a[:,0]),
         ('G', a[:,1]),
         ('B', a[:,2]),
         ]
         
df = pd.DataFrame.from_items(list1)
df['filename']=image_name
         
f1=df['R']
f2=df['G']
f3=df['B']

X=np.matrix(zip(f1,f2,f3))   
gmix = mixture.GMM(n_components=4, covariance_type='full')
gmix.fit(X)
print gmix.means_

df['Lables']=gmix.predict(X)

for i in df['filename'].index.values:

    copyfile(df['filename'][i],'/Users/zeyuanli/Desktop/6310/trends market place/Pictures/'+str(df['Lables'][i])+'/'+df['filename'][i].split('/')[-1])