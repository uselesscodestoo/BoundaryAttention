from module.Boundary import BoundaryAttentionModule
import torch
import cv2
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='model.pth')
    parser.add_argument('--image', type=str, default='test.jpg')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--noise', type=float, default=0.2, help='noise level of image, [0, 1]')
    parser.add_argument('--resize', type=float, default=0.2, help='resize image, [0, 1]')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    device = torch.device(args.device)
    model = BoundaryAttentionModule(3, 64, 8, device=device)
    model.load_state_dict(torch.load(args.model, weights_only=True))
    model.eval()

    image = cv2.imread(args.image)
    H, W, _ = image.shape
    image = cv2.resize(image, (int(W*args.resize), int(H*args.resize)))
    canny = cv2.Canny(image, 100, 200)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.0
    with torch.no_grad():
        image = torch.from_numpy(image).float().unsqueeze(0).to(device)
        noise_image = image + torch.normal(0, args.noise, size=image.shape, device=device)
        noise_image = torch.clamp(noise_image, 0, 1)
        pred_x, pred_d = model(noise_image)

    pred_x = pred_x.cpu().squeeze(0).numpy()
    b = 1 / (1 + (pred_d / 0.3)**2)
    b = b.cpu().squeeze(0).numpy()
    pred_x = (pred_x * 255).astype(np.uint8)
    pred_image = cv2.cvtColor(pred_x, cv2.COLOR_RGB2BGR)
    cv2.imshow('pred_x', pred_x)
    cv2.imshow('canny', canny)
    cv2.imshow('b', b)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

