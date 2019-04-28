from models import models

# utility params
fig_mode = None
embed_plot_epoch=10

# model params
use_gpu = True
dataset_mean = (0.5, 0.5, 0.5)
#dataset_mean = (104.0,117.0,123.0)
dataset_std = (0.5, 0.5, 0.5)

batch_size = 64
epochs = 10
gamma = 10
theta = 1

# path params
data_root = '/home/yhl/dataset'

mnist_path = data_root + '/MNIST'
mnistm_path = data_root + '/MNIST_M'
mnistm_5_path = data_root + '/MNIST_M_5'
svhn_path = data_root + '/SVHN'
syndig_path = data_root + '/SynthDigits'
amazon_path = data_root + "/Office/amazon"
dslr_path = data_root + "/Office/dslr"
webcam_path = data_root + "/Office/webcam"
art_path = data_root + "/Office_Home/Art"
clipart_path = data_root + "/Office_Home/Clipart"
product_path = data_root + "/Office_Home/Product"
realworld_path = data_root + "/Office_Home/Real_World"


save_dir = './experiment'


# specific dataset params
extractor_dict = {'MNIST_MNIST_M': models.Extractor(),
                  'MNIST_MNIST_M_5': models.Extractor(),
                  'SVHN_MNIST': models.SVHN_Extractor(),
                  'SynDig_SVHN': models.SVHN_Extractor(),
                  'ResNet50':models.Res50_Extractor()}

class_dict = {'MNIST_MNIST_M': models.Class_classifier(),
              'MNIST_MNIST_M_5': models.Class_classifier(),
              'SVHN_MNIST': models.SVHN_Class_classifier(),
              'SynDig_SVHN': models.SVHN_Class_classifier(),
              'ResNet50': models.Res50_Class_classifier(class_num=65)}

domain_dict = {'MNIST_MNIST_M': models.Small_Domain_classifier(),
                'MNIST_MNIST_M_5': models.Domain_classifier(),
               'SVHN_MNIST': models.SVHN_Domain_classifier(),
               'SynDig_SVHN': models.SVHN_Domain_classifier(),
               'Stand': models.Domain_classifier()}
