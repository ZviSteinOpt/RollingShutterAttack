import matplotlib.pyplot as plt
import torch
from PIL import Image
import scipy.io
import numpy as np
from torchvision import transforms
from torchvision import models
from torch.utils.data import DataLoader

# Setup device as 'mps' if mps backend is built, else use 'cpu'
if torch.backends.mps.is_built():
    device = 'mps'
elif torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


class JamGenerator:
    # Initialize constants
    N_r = 720  # Total number of rows
    h_N_r = 240  # Number of hidden rows
    t_read = 32 * 10 ** -6  # Readout time in seconds
    R_n = 37  # Total number of repeats

    def __init__(self, image_path, pulse_path, pn=4):
        # Initializing parameters and preprocessing pipeline for images
        self.image_path = image_path
        self.laser_pulse_path = pulse_path
        self.P_n = pn  # Number of pulses
        self.loss_f = torch.nn.CrossEntropyLoss()
        self.t_exp = self.R_n * self.t_read  # Total exposure time in seconds
        self.T_rows = int(self.h_N_r + self.N_r + (self.t_exp / self.t_read))
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.local_dip_con_val = 4
        self.real_exp = False
    def preprocess_im(self):
        # Preprocess input image
        im = Image.open(self.image_path)
        self.tran = transforms.Compose([transforms.ToTensor()])
        self.Img = self.tran(im)

    def get_laser_pulse(self):
        # Load and preprocess laser pulse from a .mat file
        mat = scipy.io.loadmat(self.laser_pulse_path)
        laser_puls = list(mat.items())[3][1]
        p_laser_puls = Image.fromarray(laser_puls.astype('uint8'), 'RGB')
        self.laser_puls = self.tran(p_laser_puls) * torch.tensor([1, 0.5, 1]).view(3, 1, 1)

    def set_DNN(self):
        # Set up pre-trained ResNet50 and evaluate it on the preprocessed image
        self.net = models.resnet50(pretrained=True).to(device)
        r_img = torch.unsqueeze(self.preprocess(self.Img), 0).to(device)
        self.net.eval()
        self.score_r = torch.nn.functional.softmax(self.net(r_img), dim=1)[0]

    def shutter_con(self, red_laser_eff):
        # Compute shutter control signal
        l_t = torch.zeros(int(self.h_N_r + self.N_r) - 1, self.P_n)
        l_tc = torch.zeros(int(self.h_N_r + self.N_r), self.P_n)
        l = torch.zeros(int(self.h_N_r + self.N_r))
        conv_size = int(self.t_exp / self.t_read)
        shutt = torch.nn.Conv1d(1, 1, conv_size)
        shutt.weight.data = torch.full([1, 1, conv_size], 1 / conv_size, dtype=torch.float)
        shutt.bias.data = torch.zeros(1, dtype=torch.float)

        # Calculate l values for each pulse and accumulate the result in l
        for i in np.arange(0, self.P_n):
            l_t[:, i] = torch.arange(1, int(self.h_N_r + self.N_r))
            l_t[:, i] = self.stretch((1 / (l_t[:, i] - red_laser_eff[i]) ** 4))
            l_tc[:, i] = self.stretch(
                shutt(torch.cat((l_t[:, i], torch.zeros(conv_size))).view(1, 1, conv_size + len(l_t[:, i]))).view(
                    int(self.h_N_r + self.N_r)))
            l += l_tc[:, i]

        # Convolve the l signal with a Gaussian kernel
        sig = 3
        h = torch.exp((-torch.arange(-18, 18) ** 2) / (2 * sig ** 2)) / (np.sqrt(2 * torch.pi) * sig)
        conv = torch.nn.Conv1d(1, 1, 36)
        conv.weight.data = h.view(1, 1, 36)

        return torch.tanh(100 * l[130:850])

    def grad_method(self, red_laser, itr, max_score=0):
        # Optimization method to find the optimal red laser signal
        red_l_max = red_laser
        realx_ind = 0
        idx = 0
        s_loss = []
        for i in range(itr):
            red_laser.requires_grad = True
            optimizer = torch.optim.Adam([red_laser], lr=0.05)
            score = self.score_cal(red_laser)
            inv_score = 1 / score  # Inverse of score to calculate maximum loss
            inv_score.backward()
            with torch.no_grad():
                optimizer.step()
                optimizer.zero_grad()
                s_loss.append(score)
                if i>self.local_dip_con_val and realx_ind>self.local_dip_con_val:
                    local_dip_con = torch.sum(torch.sign(torch.diff(torch.tensor(s_loss[-self.local_dip_con_val:]))))
                    local_dip_con = local_dip_con<2
                else:
                    local_dip_con = False
                    realx_ind +=1
                if s_loss[i] > max_score:
                    print(score)
                    max_score = s_loss[i]
                    red_l_max = red_laser
                    idx +=1
                    if idx>4:
                        self.score_cal(red_laser,plt_o=True)

                if local_dip_con:
                    red_laser = torch.round(torch.rand(self.P_n) * self.T_rows)
                    red_laser = torch.atanh(2 * (red_laser / self.T_rows) - 1)
                    realx_ind = 0
            red_laser.requires_grad = True

        self.score_cal(red_laser, plt_o=True)
        return red_l_max, max_score

    def score_cal(self, red_laser,plt_o=False):
        # Calculate score based on given red_laser signal
        red_laser_eff = int(self.h_N_r + self.N_r) * (torch.tanh(red_laser) + 1) / 2

        n_r = np.arange(0, self.h_N_r + self.N_r, 72)
        p_image = []
        images_p = []
        for phase in n_r:
            l = self.shutter_con((red_laser_eff + phase) % int(self.h_N_r + self.N_r))
            lp = self.laser_puls * l.view(self.N_r, 1)
            if plt_o:
                images_p.append(self.Img + lp)
            p_image.append(torch.unsqueeze(self.preprocess((self.Img + lp)), 0))

        data = DataLoader(p_image, batch_size=len(p_image), shuffle=False)
        labels = torch.ones(len(n_r)).to(device) * torch.argmax(self.score_r).view(-1)
        for ll, batch in enumerate(data):
            batch = (batch[:, 0, ...]).to(device)
            self.net.eval()
            pred = self.net(batch)
            loss = self.loss_f(pred, labels.long())

        if plt_o:
            with open(r'imagenet_classes.txt') as f:
                classes = [line.strip() for line in f.readlines()]
            classes = np.array(classes)

            predNp = np.argmax(pred.cpu().numpy(), axis=1)
            indx = np.argmax(predNp-torch.argmax(self.score_r).cpu().numpy())
            im_show = np.transpose(images_p[indx], (1, 2, 0))
            plt.subplot(1, 2, 1)  # 1 row, 2 columns, index 1
            plt.imshow(im_show)
            plt.axis('off')  # Turn off axis numbers and ticks
            plt.title('Class name = ' + classes[predNp[indx]])

            plt.subplot(1, 2, 2)  # 1 row, 2 columns, index 1
            plt.imshow(np.transpose(self.Img, (1, 2, 0)))
            plt.axis('off')  # Turn off axis numbers and ticks

            plt.title('Class name = ' + classes[torch.argmax(self.score_r).cpu().numpy()])
            plt.show(block=False)  # Non-blocking show
            input("Press Enter to continue to the next graph...")  # Wait for user input
            plt.close()  # Close the current plot
            return
        return loss

    def Pulse2Ard(self, pulses):
        # Convert pulses to Arduino compatible format
        f = np.zeros(int(self.h_N_r + self.N_r + (self.t_exp / self.t_read)))
        f[pulses.astype(int)] = 1
        g = np.arange(1, self.h_N_r + self.N_r + 1)
        g = g - 1
        g = g * self.t_read
        gg = g + self.t_read + self.t_exp
        l = np.zeros((self.h_N_r + self.N_r, 1))
        for t in range(int(self.h_N_r + self.N_r + (self.t_exp / self.t_read))-1):
            on = (gg > (t * self.t_read)) * (g < (t * self.t_read))* f[t]
            l += on[:,np.newaxis]

        # swithing units to mic sec
        t_read = 32 # musec
        ff = np.copy(f)
        for i in range(len(f)):
            if (f[i]==1):
                ff[i:(i+17)] = 1

        f = ff
        df = np.diff(f)

        count = 1
        stateV = []
        milV = []
        micV = []
        stateV.append(f[1])

        for i in range(len(f) - 1):
            if (df[i] == 0):
                count += 1

            if (df[i] == -1):
                stateV.append(0)
                if self.pulse_width:
                    milV.append(0)
                    micV.append(self.pulse_width)

            if (df[i] == 1):
                count += 1
                stateV.append(1)
                milV.append(np.floor((count * t_read-self.pulse_width) / 1000))
                micV.append(count * t_read - 1000 * milV[-1]-self.pulse_width)
                count = 1


        milV.append(np.floor((count*t_read)/1000))
        micV.append(count*32-1000*milV[-1])

        micV[-1] = micV[-1]+33128 - sum(1000*milV+micV)-10*len(stateV)

        return stateV, milV, micV

    def stretch(self,l):

            l = .5 * (torch.tanh(l) + 1)

            m = 1 / (max(l) - min(l))
            n = -m * min(l)

            if min(l) == max(l):
                m = 1 / max(l)
                n = 0

            if min(l) == max(l):
                m = 0
                n = 0

            l = m * l + n

            return l

    def print_ard_code(self,stateV, milV, micV):
        j = 0
        for i in range(len(stateV)):
            if stateV[i] and not self.pulse_width:
                s = 'digitalWrite(pin,'+ str(stateV[i]) + ');'
                print(s)
            else:
                s = 'digitalWrite(pin,'+ str(stateV[i]) + ');'
                print(s)
                mic = 'delayMicroseconds(' + str(micV[j]) + ');'
                print(mic)
                mil = 'delay(' + str(milV[j])  + ');'
                print(mil)
                j+=1

    def run(self):

        # Init
        self.preprocess_im()
        self.get_laser_pulse()
        self.set_DNN()

        # Random start
        red_gen = torch.round( torch.rand(self.P_n)* self.T_rows )
        red_gen = torch.atanh(2 * (red_gen / self.T_rows) - 1)

        # Optimization
        pulses_grad,score = self.grad_method(red_gen,itr=300)
        pulses = torch.round((torch.tanh(pulses_grad) + 1) * self.T_rows / 2).detach().numpy()

        # Generate Arduino pulses intervals
        if self.real_exp:
            stateV, milV, micV = self.Pulse2Ard(pulses)
            self.print_ard_code(stateV, milV, micV)
            print(torch.round((torch.tanh(pulses_grad) + 1) * self.T_rows / 2).detach().numpy())


if __name__ == "__main__":
    image_path = r'im.tif'
    pulse_path = r'laser_pulse.mat'
    new_teq = JamGenerator(image_path,pulse_path)
    new_teq.run()







