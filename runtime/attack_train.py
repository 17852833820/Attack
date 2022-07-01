from codes.attack_offline_conv_white_new import T_offine_conv_white
from codes.attack_offline_fcnn_white_new import T_offine_fcnn_white
from codes.ut_attack_offline_fcnn_white_new import DNN_offine_fcnn_black
from codes.train_offline_fcnn_white import DNN_offine_fcnn_white
class Location():
    def __init__(self,mode,DNN,target):
        self.mode=mode
        self.DNN=DNN
        self.target=target
    def run(self):
        if self.mode=="white" and self.DNN=='CNN':
            if self.target:
                attacker = DNN_offine_conv_white()
            else:
                attacker = DNN_offine_conv_white()
        elif self.mode=="white" and self.DNN=='FCNN':
            attacker = DNN_offine_fcnn_white()
        elif self.mode == "black" and self.DNN == 'CNN':
            attacker = DNN_offine_conv_black()
        elif self.mode=="black" and self.DNN=='FCNN':
            attacker = DNN_offine_fcnn_black()
        attacker.run()

if __name__ == '__main__':
    mode='white'
    DNN='CNN'
    target=True
    location=Location(mode,DNN,target)
    location.run()