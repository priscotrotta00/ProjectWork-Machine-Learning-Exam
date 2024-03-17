import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchvision import transforms
import albumentations
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

class MobileNet_Binary(nn.Module):
  """
  Classe che implementa una rete neurale binaria basata su MobileNet.

  """

  def __init__(self):
    super().__init__()
    self.__net = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
    
    self.__net.classifier[-1] = nn.Linear(1280, 1280)
    self.__net.classifier.append(nn.Hardswish())
    self.__net.classifier.append(nn.Dropout(p=0.2, inplace=True))
    self.__net.classifier.append(nn.Linear(1280, 1))

  def train(self, mode=True):
    super().train(mode)
    self.__net.train(mode)

  def eval(self):
    super().eval()
    self.__net.eval()

  def forward(self, x):
    if x.dim() == 5:
      batch_size = x.size(0)
      num_frames_per_batch = x.size(1)
      x = x.reshape(batch_size * x.size(1), x.size(2), x.size(3), x.size(4))
    else:
      batch_size = 0
      num_frames_per_batch = x.size(0)

    results = self.__net(x)
    results = self.__sigmoid(results)
    if batch_size > 0:
      return results.reshape(batch_size, num_frames_per_batch)
    else:
      return results.reshape(num_frames_per_batch)

class CNNLSTM(nn.Module):
    def __init__(self, weights_path):
        super(CNNLSTM, self).__init__()
        self.cnn = self.__importMobileNetV3(weights_path)

        self.num_features = self.cnn.classifier[0].in_features
        self.lstm_hidden_size = 100
        self.lstm_num_layers = 2

        self.cnn.classifier = nn.Identity()

        self.lstm = nn.LSTM(input_size=self.num_features, hidden_size=100, num_layers=2, bidirectional=True, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(self.lstm_num_layers * self.lstm_hidden_size, 1))

    def __importMobileNetV3(self, weights_path):
      mnb = MobileNet_Binary()

      if not torch.cuda.is_available():
        mnb.load_state_dict(torch.load(weights_path, map_location='cpu'))
      else:
        mnb.load_state_dict(torch.load(weights_path))


      for param in mnb.parameters():
        param.requires_grad = False

      cnn = next(mnb.children())

      return cnn

    def train(self, mode=True):
      super().train(mode)
      self.cnn.train(mode)
      self.lstm.train(mode)
      self.fc.train(mode)

    def eval(self):
      self.train(False)

    def forward(self, x_3d):
        out = None
        hidden = None

        if (x_3d.dim() == 5):
          batch_size, frames, c, h, w = x_3d.size()
          x_3d = x_3d.view(batch_size * frames, c, h, w)

          with torch.no_grad():
            cnn_out = self.cnn(x_3d)
            cnn_out = cnn_out.view(batch_size, frames, self.num_features)
          lstm_out, hidden = self.lstm(cnn_out)
          fc_out = self.fc(lstm_out[:,-1])

          return F.sigmoid(fc_out).squeeze(-1)
        else:
          with torch.no_grad():
            cnn_out = self.cnn(x_3d)
          lstm_out, hidden = self.lstm(cnn_out)
          fc_out = self.fc(lstm_out[-1])

          return F.sigmoid(fc_out).squeeze(-1)

def create_CNN_LSTM():
  return CNNLSTM('./MobileNetV3_best_model.pth')

def transformation():
  return albumentations.Compose([
      albumentations.Resize(height=232, width=232, always_apply=True),
      albumentations.CenterCrop(224,224,always_apply=True),
      albumentations.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225],
                                max_pixel_value=255.,
                                always_apply=True),
  ])

class ImglistOrdictToTensor(torch.nn.Module):
    """
    Converts a list or a dict of numpy images to a torch.FloatTensor
    of shape (NUM_IMAGES x CHANNELS x HEIGHT x WIDTH).
    Can be used as first transform for ``VideoFrameDataset``.
    """
    @staticmethod
    def forward(img_list_or_dict):
        """
        Converts each numpy image in a list or a dict to
        a torch Tensor and stacks them into a single tensor.

        Args:
            img_list_or_dict: list or dict of numpy images.
        Returns:
            tensor of size ``NUM_IMAGES x CHANNELS x HEIGHT x WIDTH``
        """
        if isinstance(img_list_or_dict, list):
            return torch.stack([transforms.functional.to_tensor(img)
                                for img in img_list_or_dict])
        else:
            return torch.stack([transforms.functional.to_tensor(img_list_or_dict[k])
                                for k in img_list_or_dict.keys()])

if __name__ == '__main__':
  print(create_CNN_LSTM())