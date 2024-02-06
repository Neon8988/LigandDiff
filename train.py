import argparse
import os
import pwd
import sys
import wandb
import yaml

from datetime import datetime
from pytorch_lightning import Trainer, callbacks, loggers

from src.const import NUMBER_OF_ATOM_TYPES
from src.lightning import DDPM
from src.utils import disable_rdkit_logging, Logger


def find_last_checkpoint(checkpoints_dir):
    epoch2fname = [
        (int(fname.split('=')[1].split('.')[0]), fname)
        for fname in os.listdir(checkpoints_dir)
        if fname.endswith('.ckpt')
    ]
    latest_fname = max(epoch2fname, key=lambda t: t[0])[1]
    return os.path.join(checkpoints_dir, latest_fname)


def main(args):
    start_time = datetime.now().strftime('date%d-%m_time%H-%M-%S.%f')
    run_name = f'{os.path.splitext(os.path.basename(args.config))[0]}_{pwd.getpwuid(os.getuid())[0]}_{args.exp_name}_bs{args.batch_size}_{start_time}'
    experiment = run_name if args.resume is None else args.resume
    checkpoints_dir = os.path.join(args.checkpoints, experiment)
    os.makedirs(os.path.join(args.logs, "general_logs", experiment),exist_ok=True)
    sys.stdout = Logger(logpath=os.path.join(args.logs, "general_logs", experiment, f'log.log'), syspart=sys.stdout)
    sys.stderr = Logger(logpath=os.path.join(args.logs, "general_logs", experiment, f'log.log'), syspart=sys.stderr)

    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(args.logs, exist_ok=True)

    samples_dir = os.path.join(args.logs, 'sample_chain', experiment)
    torch_device = 'cuda:0' if args.device == 'gpu' else 'cpu'


    wandb_logger = loggers.WandbLogger(
        save_dir=args.logs,
        project='LigandDiff',
        name=experiment,
        id=experiment,
        resume='must' if args.resume is not None else 'allow',
        entity=args.wandb_entity,
    )

    in_node_nf = NUMBER_OF_ATOM_TYPES
    ligand_group_node_nf = 6

    ddpm = DDPM(
        data_path=args.data,
        train_data=args.train_data,
        val_data=args.val_data,
        in_node_nf=in_node_nf,
        n_dims=3,
        ligand_group_node_nf=ligand_group_node_nf,
        hidden_nf=args.hidden_nf,
        attention=args.attention,
        n_layers=args.n_layers,
        normalization_factor=args.normalization_factor,
        normalize_factors=args.normalize_factors,

        drop_rate=args.drop_rate,

        activation=args.activation,
        tanh=args.tanh,
        norm_constant=args.norm_constant,
        inv_sublayers=args.inv_sublayers,
        sin_embedding=args.sin_embedding,
        aggregation_method=args.aggregation_method,
        normalization=args.normalization,

        diffusion_steps=args.diffusion_steps,
        diffusion_noise_schedule=args.diffusion_noise_schedule,
        diffusion_noise_precision=args.diffusion_noise_precision,
        diffusion_loss_type=args.diffusion_loss_type,
       
        lr=args.lr,
        batch_size=args.batch_size,
        torch_device=torch_device,
        model=args.model,
        test_epochs=args.test_epochs,
        center_of_mass=args.center_of_mass,
        clip_grad=args.clip_grad,
        samples_dir=samples_dir)
    
    checkpoint_callback = callbacks.ModelCheckpoint(
        dirpath=checkpoints_dir,
        filename=experiment + '_{epoch:02d}',
        monitor='loss/val',
        save_top_k=1,
    )
    trainer = Trainer(
        max_epochs=args.n_epochs,
        logger=wandb_logger,
        callbacks=checkpoint_callback,
        accelerator=args.device,
        devices=1,
        num_sanity_val_steps=0,
        enable_progress_bar=False,
    )

    if args.resume is None:
        last_checkpoint = None
    else:
        last_checkpoint = find_last_checkpoint(checkpoints_dir)
        print(f'Training will be resumed from the latest checkpoint {last_checkpoint}')

    wandb_logger.watch(ddpm, log='gradients', log_freq=1,log_graph=True)
    trainer.fit(model=ddpm, ckpt_path=last_checkpoint)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='LigandDiff')
    p.add_argument('--config', type=argparse.FileType(mode='r'), default='config.yml')
    p.add_argument('--exp_name', type=str, default='YourName')
    p.add_argument('--checkpoints', action='store', type=str, default='checkpoints')
    p.add_argument('--logs', action='store', type=str, default='logs')
    p.add_argument('--n_epochs', type=int, default=200)
    p.add_argument('--resume', type=str, default=None, help='')
    p.add_argument('--wandb_entity', type=str, default='geometric', help='Entity (project) name')
    
    ## DDPM args <--
    p.add_argument('--data', action='store', type=str,  default="datasets")
    p.add_argument('--train_data', action='store', type=str, default='train_onehot')
    p.add_argument('--val_data', action='store', type=str,  default='val_onehot')
    p.add_argument('--hidden_nf', type=int, default=128,  help='number of layers')
    p.add_argument('--attention', type=eval, default=True, help='use attention in the EGNN')
    p.add_argument('--n_layers', type=int, default=6,   help='number of layers')
    p.add_argument('--normalization_factor', type=float, default=1,help="Normalize the sum aggregation")
    p.add_argument('--normalize_factors', type=eval, default=[1, 4, 1], help='normalize factors for [x, categorical, integer]')
    ##gvp_dynamics
    p.add_argument('--drop_rate', type=float, default=0.0, help='Dropout rate')
    ##egnn_dynamics
    p.add_argument('--activation', type=str, default='silu', help='silu')
    p.add_argument('--tanh', type=eval, default=True, help='use tanh in the coord_mlp')
    p.add_argument('--norm_constant', type=float, default=1,help='diff/(|diff| + norm_constant)')
    p.add_argument('--inv_sublayers', type=int, default=1, help='number of layers')
    p.add_argument('--sin_embedding', type=eval, default=False, help='whether using or not the sin embedding')
    p.add_argument('--aggregation_method', type=str, default='sum',help='"sum" or "mean"')
    p.add_argument('--normalization', type=str, default='batch_norm', help='batch_norm')

    p.add_argument('--diffusion_steps', type=int, default=500)
    p.add_argument('--diffusion_noise_schedule', type=str, default='polynomial_2', help='learned, cosine')
    p.add_argument('--diffusion_noise_precision', type=float, default=1e-5, )
    p.add_argument('--diffusion_loss_type', type=str, default='l2', help='vlb, l2')
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--device', action='store', type=str, default='gpu')
    p.add_argument('--model', type=str, default='gvp_dynamics',help='egnn_dynamics |gvp_dynamics')
    p.add_argument('--test_epochs', type=int, default=1)
    p.add_argument('--center_of_mass', type=str, default='context', help='Where to center the data: context | coord_site')
    p.add_argument('--clip_grad', type=eval, default=True,help='True | False')
    
    disable_rdkit_logging()

    args = p.parse_args()
    if args.config:
        config_dict = yaml.load(args.config, Loader=yaml.FullLoader)
        arg_dict = args.__dict__
        for key, value in config_dict.items():
            if isinstance(value, list) and key != 'normalize_factors':
                for v in value:
                    arg_dict[key].append(v)
            else:
                arg_dict[key] = value
        args.config = args.config.name
    else:
        config_dict = {}
    main(args=args)
