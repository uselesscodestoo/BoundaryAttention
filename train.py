from module.Boundary import BoundaryAttentionModule
from info import count_parameters
import torch
from dataset import load_dataset
from tqdm import tqdm
import os
import argparse

MIN_SCALE = 0.2
OFFSET_SCALE = 0.6

def rand_scale():
    return MIN_SCALE + OFFSET_SCALE * torch.rand(1).item()

def parse_args():
    parser = argparse.ArgumentParser()
    # module
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    # optimizer
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    # dataset
    parser.add_argument('--min_scale', type=float, default=0.1, help='min noise scale of image')
    parser.add_argument('--max_scale', type=float, default=0.8, help='max noise scale of image')
    # other
    parser.add_argument('--dataset_dir', type=str, default='./data/21pix')
    parser.add_argument('--output_dir', type=str, default='./run/train')
    parser.add_argument('--resume_model', type=str, default='')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval', type=bool, default=True)
    args = parser.parse_args()

    global MIN_SCALE, OFFSET_SCALE
    MIN_SCALE = args.min_scale
    OFFSET_SCALE = args.max_scale - args.min_scale
    return args

def ensure_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

if __name__ == "__main__":
    args = parse_args()

    torch.random.manual_seed(args.seed)
    device = torch.device(args.device)
    model = BoundaryAttentionModule(3, 64, 8, device=device)
    if os.path.exists(args.resume_model):
        if args.resume_model.endswith('.pth'):
            model.load_state_dict(torch.load(args.resume_model, weights_only=True))
        elif args.resume_model.endswith('.pt'):
            model = torch.load(args.resume_model)
        else:
            print("resume_model must be .pth or .pt")
            exit(1)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    epoch = args.epochs
    dataset, dataloader = load_dataset(args.dataset_dir, batch_size=args.batch_size)
    ensure_dir(args.output_dir)

    for i in range(1,epoch+1):
        print(f"epoch: {i} start")
        inner = tqdm(dataloader)
        model.train()
        for images, labels in inner:
            images = images.permute(0, 2, 3, 1).to(device)
            noised_images = images + torch.normal(0, rand_scale(), size=images.shape, device=device)
            noised_images = torch.clamp(noised_images, 0, 1)
            labels = labels.to(device)
            optimizer.zero_grad()
            pred_x, pred_d = model(noised_images)
            loss = model.loss(images, labels, pred_x, pred_d)
            loss.backward()
            optimizer.step()
            inner.set_description(f"loss:{loss.item():8.5f}")
        torch.save(model.state_dict(), os.path.join(args.output_dir, f'model_{i}.pth'))
        if args.eval:
            model.eval()
            with torch.no_grad():
                l = 0
                loss = 0
                for images, labels in tqdm(dataloader):
                    if l % 10 != 0:
                        l += 1
                        continue
                    images = images.permute(0, 2, 3, 1).to(device)
                    noised_images = images + torch.normal(0, rand_scale(), size=images.shape, device=device)
                    noised_images = torch.clamp(noised_images, 0, 1)
                    labels = labels.to(device)
                    pred_x, pred_d = model(noised_images)
                    loss += model.loss(images, labels, pred_x, pred_d).item()
                    l += 1
                loss /= (l + 9) // 10
                print(f"eval loss: {loss:.5f}")
                os.rename(os.path.join(args.output_dir, f'model_{i}.pth'), os.path.join(args.output_dir, f'model_{i}_{loss:.5f}.pth'))


    torch.save(model.state_dict(), os.path.join(args.output_dir, 'model_final.pth'))