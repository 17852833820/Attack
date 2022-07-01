from codes.train_offline_conv_black import DNN_offine_conv_black
from codes.train_offline_conv_white import DNN_offine_conv_white
from codes.train_offline_fcnn_black import DNN_offine_fcnn_black
from codes.train_offline_fcnn_white import DNN_offine_fcnn_white
class Location():
    def __init__(self,mode,DNN):
        self.mode=mode
        self.DNN=DNN
    def run(self):
        if self.mode=="white" and self.DNN=='CNN':
            location = DNN_offine_conv_white()
        elif self.mode=="white" and self.DNN=='FCNN':
            location = DNN_offine_fcnn_white()
        elif self.mode == "black" and self.DNN == 'CNN':
            location = DNN_offine_conv_black()
        elif self.mode=="black" and self.DNN=='FCNN':
            location = DNN_offine_fcnn_black()
        location.run()

if __name__ == '__main__':
    mode='white'
    DNN='CNN'
    location=Location(mode,DNN)
    location.run()