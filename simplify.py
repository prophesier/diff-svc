from argparse import ArgumentParser

import torch


def simplify_pth(pth_name, project_name):
    model_path = f'./checkpoints/{project_name}'
    checkpoint_dict = torch.load(f'{model_path}/{pth_name}')
    torch.save({'epoch': checkpoint_dict['epoch'],
                'state_dict': checkpoint_dict['state_dict'],
                'global_step': None,
                'checkpoint_callback_best': None,
                'optimizer_states': None,
                'lr_schedulers': None
                }, f'./clean_{pth_name}')


def main():
    parser = ArgumentParser()
    parser.add_argument('--proj', type=str)
    parser.add_argument('--steps', type=str)
    args = parser.parse_args()
    model_name = f"model_ckpt_steps_{args.steps}.ckpt"
    simplify_pth(model_name, args.proj)


if __name__ == '__main__':
    main()
