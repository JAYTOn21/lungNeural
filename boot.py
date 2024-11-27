import torch
from PIL import Image
from matplotlib import pyplot as plt
from tensorboard import summary

from createNtrain import ResidualNetwork, ResidualBlock, bacha_size
from torchvision import transforms
from torchvision.transforms import functional as F


def predict(path):
    model = ResidualNetwork(
        learning_rate=0.01,
        batch_size=bacha_size,
        block=ResidualBlock,
        layers=[3, 5, 21, 5]
    )

    test_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img = Image.open(path).convert('RGB')
    device = torch.device("cpu")
    img = test_transforms(img).unsqueeze(0).to(device)

    model.load_state_dict(torch.load("myModel.pt", weights_only=True, map_location=device))
    model.to(device)
    model.eval()
    prediction = model(img)
    ills = ["Bacterial Pneumonia", "Normal", "Tuberculosis"]
    # print(prediction)
    # for i in range(3):
    #     print(ills[i], ": ", prediction.detach().cpu().numpy()[0][i])
    needIMG = F.to_pil_image(img[0])
    return needIMG, prediction.detach().cpu().numpy()[0]


# def show_images(img):
#     plt.imshow(transforms.functional.to_pil_image(img))
#     plt.show()

# show_images(img[0])