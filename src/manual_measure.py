from PIL import Image
from pylab import *

# im = array(Image.open('.\\images\\calibrated\\-962.png'))
# im = array(Image.open('.\\images\\-962.png'))
im = array(Image.open('./src/images/-962.png'))
imshow(im)
print('Please click 2 points')
imshow(im)
x = ginput(2)
print( 'You clicked:', x)
plt.plot(x)
show()

