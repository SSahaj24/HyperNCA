import argparse
import torch
import time
import numpy as np
import cma
import pickle
import gym
import psutil 
import pathlib
import yaml
import datetime
import gym
import wandb
import os

from fitness_functions import fitnessRL
from policies import MLPn
from NCA_3D import CellCAModel3D
from utils_and_wrappers import generate_seeds3D, plots_rewads_save, policy_layers_parameters, dimensions_env

from cma.optimization_tools import EvalParallel2

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.set_default_dtype(torch.float64)

def load_config_from_saved_model(model_id):
    """
    Load configuration from a saved model using its ID.
    
    Args:
        model_id (str): The ID of the saved model
        
    Returns:
        dict: The configuration dictionary
    """
    config_path = os.path.join("saved_models", model_id, "nca.config")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    with open(config_path, "rb") as f:
        config = pickle.load(f)
    
    return config

def x0_sampling(dist, nb_params):
    if dist == 'U[0,1]':
        return np.random.rand(nb_params)
    elif dist == 'U[-1,1]':
        return 2*np.random.rand(nb_params)-1
    elif dist == 'N[0,1]':
        return np.random.randn(nb_params)
    else:
        raise ValueError('Distribution not available')

def train(args):
    
    # Initialize wandb
    if args['use_wandb']:
        wandb.init(
            project="nca-rl-optimization",
            config={
                "generations": args['generations'],
                "popsize": args['popsize'],
                "NCA_steps": args['NCA_steps'],
                "NCA_dimension": args['NCA_dimension'],
                "policy_layers": args['policy_layers'],
                "living_threshold": args['living_threshold'],
                "NCA_channels": args['NCA_channels'],
                "update_net_channel_dims": args['update_net_channel_dims'],
                "size_substrate": args['size_substrate'],
                "environment": args['environment'],
                "seed_type": args['seed_type'],
                "sigma_init": args['sigma_init'],
                "plastic": args['plastic'],
                "normalize": args['normalise'],
                "NCA_bias": args['NCA_bias'],
                "neighborhood": args['neighborhood'],
                "cell_noise": args['cell_noise'],
                "nca_dropout_rate": args['nca_dropout_rate'],
                "network_dropout_rate": args['network_dropout_rate'],
            }
        )
    
    if args['save_model']: # prevent printing during experiments
        print('\n')
        for key, value in args.items():
            if key != 'seeds':
                print(key,':',value)
        print('\n')
    
    environments = args['environment']
    seeds_type = ['ones', '-ones'] if args['seed_type'] == '-1+1' else len(environments)*[args['seed_type']]
    seeds = [] 
    policies_sizes = []
    for i, environment in enumerate(environments):
        
        # Check if selected env is pixel or state-vector and its dimensions
        input_dim, action_dim, pixel_env = dimensions_env(environment)
        
        if (args['NCA_dimension'] == 3 and args['size_substrate'] < max(input_dim,action_dim)) or (args['NCA_dimension'] == 3 and args['size_substrate'] == 0) :
            args['size_substrate'] = max(input_dim,action_dim)
            print(f'A bigger substrate is needed for the 3D NCA for the chosen environemnts, increasing substrate size to {max(input_dim,action_dim)}')
        
        # Initialise policy network just so we know how many parameters it has
        if args['NCA_dimension'] == 2:
            raise NotImplementedError
        elif args['NCA_dimension'] == 3:
            p = MLPn(input_space=input_dim, action_space=action_dim, hidden_dim=args['size_substrate'], bias=False, layers=args['policy_layers']) 
        
        if seeds_type[i] == 'activations':
            env = gym.make(environment)
            observation = env.reset() 
        else:
            observation = None
            
        # Count the number of parameters of the policy (!= than the seed for NCA 3D)
        if args['NCA_dimension'] == 3:
            policies_sizes.append(input_dim*args['size_substrate'] + args['size_substrate']**2 + args['size_substrate']*action_dim)
        
        if not args['random_seed']:
            if args['NCA_dimension'] == 2:
                raise NotImplementedError
            elif args['NCA_dimension'] == 3:
                seed = generate_seeds3D(policy_layers_parameters(p), seeds_type[i], args['NCA_channels'], observation, environment)

        else:
            seed = None
            
        if 'single' in args['seed_type'] or 'one' in  args['seed_type']:
            bits_per_seed = 1
        elif 'two' in args['seed_type']:
            bits_per_seed = 2
        elif 'three' in args['seed_type']:
            bits_per_seed = 3
        elif 'four' in args['seed_type']:
            bits_per_seed = 4
        elif 'eight' in args['seed_type']:
            bits_per_seed = 8  
        elif 'activation' in args['seed_type']:
            bits_per_seed = 0
        elif 'random' in args['seed_type']:
            bits_per_seed = policies_sizes
        
        seeds.append(seed)
        
        seeds_flatten = []
        if args['co_evolve_seed']:
            for seed in seeds:
                if args['NCA_dimension'] == 2:
                    raise NotImplementedError
                elif args['NCA_dimension'] == 3:
                    seed_flatten = seed.flatten().detach().numpy()
                seeds_flatten.append(seed_flatten)
        
    # NCA config
    nca_config = {
        "cell_noise": args['cell_noise'],     # Adjust noise level
        "nca_dropout_rate": args['nca_dropout_rate'], # Adjust dropout probability
        "network_dropout_rate": args['network_dropout_rate'], # Adjust network dropout probability
        "environment" : args['environment'],
        "popsize" : args['popsize'],
        "generations" : args['generations'],
        "NCA_channels" : args['NCA_channels'],
        "living_channel_dim" : 1 if args['NCA_channels'] > 1 else 0, # channel that encodes aliveness, aka alpha channel
        "NCA_steps" : args['NCA_steps'],    
        "update_net_channel_dims" : args['update_net_channel_dims'],    
        "batch_size" : 1,                   
        "alpha_living_threshold": args['living_threshold'] if args['living_threshold'] != 0 else -np.inf, # alive: percentage thereshold of abs mean value
        "seed_type" : seeds_type,
        "random_seed" : args['random_seed'],
        "seeds" : seeds,
        "plastic" : args['plastic'],
        "normalise" : args['normalise'],
        "replace" : args['replace'],
        "NCA_dimension" : args['NCA_dimension'],
        "reading_channel" : args['reading_channel'],
        "size_substrate" : args['size_substrate'],
        "penalty_off_topology" : None,
        "debugging" : 0,
        "RANDOM_SEED" : np.random.randint(10**5),
        "random_seed_env" :args['random_seed_env'],
        "co_evolve_seed" : args['co_evolve_seed'],
        "NCA_MLP" : False,
        "NCA_bias" : args['NCA_bias'],
        "neighborhood" : args['neighborhood'],
        "seeds_size" : [seeds_flatten[i].shape[0] if args['co_evolve_seed'] else None for i in range(len(seeds_flatten))],
        "nb_envs" : len(environments),
        "policy_layers" : args['policy_layers'],
        "layerwise_dropout_rate" : args['layerwise_dropout_rate']
    }
    
    if nca_config['NCA_dimension'] == 2:
        raise NotImplementedError
    elif nca_config['NCA_dimension'] == 3:
        if nca_config['NCA_MLP']:
            raise NotImplementedError
        else:
            ca = CellCAModel3D(nca_config)    
    
    # Parameters dimensions
    nca_nb_weights = torch.nn.utils.parameters_to_vector(ca.parameters()).shape[0]
    seed_size = nca_config['NCA_channels']*nca_config['policy_layers']*nca_config['size_substrate']*nca_config['size_substrate']
    policy_nb_functional_params = torch.nn.utils.parameters_to_vector(p.parameters()).shape[0]
    
    flatten_seed_size =  np.sum(nca_config['seeds_size']) if args['co_evolve_seed'] else 0

    # CMAEvolutionStrategy(x0, sigma0, options)
    x0 = x0_sampling(args['x0_dist'], nca_nb_weights + flatten_seed_size)
    es = cma.CMAEvolutionStrategy(x0, args['sigma_init'], {'verb_disp': args['print_every'],
                                                            'popsize' : args['popsize'], 
                                                            'maxiter' : args['generations'], 
                                                            }) 


    
    print('\n.......................................................')
    print('\nInitilisating CMA-ES for Hypernet NCA for ' + str(args['environment']) + ' with', nca_nb_weights, 'trainable parameters controlling a policy(/ies)', str(p)[:8], 'with', policy_nb_functional_params, 'effective weights with a', nca_config['NCA_dimension'] ,'dimensional seed of size', seed_size, 'sampled from', args['seed_type'],'\n')
    if args['plastic']: print('Plastic Policy network') 
    
    print('\n ♪┏(°.°)┛┗(°.°)┓ Starting Evolution ┗(°.°)┛┏(°.°)┓ ♪ \n')
    tic = time.time()
    
    args_fit = (nca_config,False, False, False, False, True)
    solution_best_reward = np.inf
    solution_mean_reward = np.inf
    rewards_mean = []
    rewards_best = []
    gen = 0
        
    num_cores = psutil.cpu_count(logical = False) if args['threads'] == -1 else args['threads']    
    print(f'\nUsing {num_cores} cores\n')
    
    # Optimisation loop
    fitness = fitnessRL
    with EvalParallel2(fitness, number_of_processes=num_cores ) as eval_all:
        while not es.stop() or gen < args['generations']:
            
            try:
                # Generate candidate solutions
                X = es.ask()                          
                
                # Evaluate in parallel
                fitvals = eval_all(X, args=args_fit)
                
                # Inform CMA optimizer of fitness results
                es.tell(X, fitvals)                    
                
                if gen%args['print_every'] == 0:
                    es.disp()
                
                solution_current_best = es.best.f
                rewards_best.append(solution_current_best)
                if solution_current_best <= solution_best_reward:
                    solution_best_reward = solution_current_best
                    solution_best = es.best.x
                
                solution_current_mean_eval = es.fit.fit.mean() 
                rewards_mean.append(solution_current_mean_eval)
                if solution_current_mean_eval <= solution_mean_reward:
                    solution_mean_reward = solution_current_mean_eval
                    solution_mean = es.mean

                # Log to wandb
                if args['use_wandb'] and gen % args['wandb_log_interval'] == 0:
                    wandb.log({
                        "generation": gen,
                        "best_reward": -solution_best_reward,
                        "mean_reward": -solution_mean_reward,
                        "current_best_reward": -solution_current_best,
                        "current_mean_reward": -solution_current_mean_eval
                    })

                gen += 1

            except KeyboardInterrupt: # Only works with python mp
                print('\n'+20*'*')
                print(f'\nCaught Ctrl+C!\nStopping evolution\n')
                print(20*'*'+'\n')
                break
            
            # Early stopping of evolution
            environment = args['environment'][0] if isinstance(args['environment'], list) else args['environment']
            if ( ('Lander' in environment ) and (gen == 1000 and solution_current_best > 0) ) or ( ('Ant' in environment ) and (gen == 1000 and solution_current_best > -500) ):
                print(f'\nReward too low {solution_current_best} at generation {gen}.\n\nUnpromising run! Stopping evolution.\n')
                print(20*'*'+'\n')
                break
            
                
                
    rewards_mean = np.array(rewards_mean)
    rewards_best = np.array(rewards_best)
    # solution = es.result # unsed since ask&tell
                
    toc = time.time()
    nca_config['training time'] = str(int(toc-tic)) + ' seconds'
    nca_config['seed_size'] = seed_size
    nca_config['policies_sizes'] = policies_sizes
    nca_config['bits_per_seed'] = bits_per_seed
    print('\nEvolution took: ', int(toc-tic), ' seconds\n')
    print(f'Best single reward found was {-solution_best_reward}\n')
    print(f'Best mean reward found was {-solution_mean_reward}\n')

    # Save path
    id_ = str(int(time.time()))
    plastic = '_PLASTIC' if args['plastic'] else ''
    
    # Save solutions
    if args['save_model']:
        path = 'saved_models' + '/' + id_ 
        
        pathlib.Path(path).mkdir(parents=True, exist_ok=False)

        # Save best and best mean solution
        np.save(path + '/' + str(args['environment']) + plastic + "_pop_" + str(args['popsize']) + "__rew_" + str(int(-solution_best_reward)) + '_bestsolution', solution_best)
        np.save(path + '/' + str(args['environment']) + plastic + "_pop_" + str(args['popsize']) + '_meansolution', solution_mean)
        # Save config file
        pickle.dump( nca_config, open( "saved_models" + "/"+ id_ + '/nca.config', "wb" ) )
        # Save rewards
        np.save(path + '/' + 'rewards_mean', rewards_mean)
        np.save(path + '/' + 'rewards_best', rewards_best)
        
        # Save reward plot
        plots_rewads_save(id_, rewards_mean, rewards_best)
            
        with open(path + '/nca.config', "wb" ) as outfile:
            pickle.dump( nca_config, outfile )    
        with open(path + '/nca_config.yml', 'w') as outfile: # For quickly see the parameters
            del nca_config['seeds']
            nca_config['date'] = datetime.date.today()
            nca_config['id'] = id_
            yaml.dump(nca_config, outfile, default_flow_style=False)
            
    if args['save_model'] and args['generations'] > 10:
        es.result_pretty()
        print(f'\nid_: {id_}\n')
        for key, value in nca_config.items():
            if key != 'seeds':
                print(key,':',value)
    
    # Log final results to wandb
    if args['use_wandb']:
        wandb.log({
            "final_best_reward": -solution_best_reward,
            "final_mean_reward": -solution_mean_reward,
            "training_time_seconds": int(toc-tic)
        })
        wandb.finish()
    
    return rewards_best, rewards_mean

def wandb_sweep(model_id="1645447353", trial_count=20):
    """
    Run a sweep with wandb to optimize hyperparameters based on a saved model config.
    
    Args:
        model_id (str): The ID of the saved model to use as base config
        trial_count (int): Number of trials to run in the sweep
    """
    # Load configuration from saved model
    saved_config = load_config_from_saved_model(model_id)
    
    # Set up sweep configuration focusing on cell_noise and nca_dropout_rate
    sweep_config = {
        'method': 'bayes',  # Bayesian optimization
        'metric': {
            'name': 'final_best_reward',
            'goal': 'maximize'
        },
        'parameters': {
            'cell_noise': {
                'min': 0.0,
                'max': 0.4
            },
            'nca_dropout_rate': {
                'min': 0.0,
                'max': 0.4
            },
            'network_dropout_rate': {
                'min': 0.0,
                'max': 0.4
            },
            'policy_layers': {
                'values': [i for i in range(1,11)] + [j for j in range(12,30,3)] 
        },
            'NCA_channels':{
                'values': [i for i in range(1,10)]
        }
    }
    }
    
    sweep_id = wandb.sweep(sweep_config, project="NCA_channel_sweep")
    
    def sweep_train():
        wandb.init()
        
        # Get hyperparameters from wandb
        wandb_config = wandb.config
        
        # Create args dictionary with values from saved config
        args = {
            'environment': saved_config.get('environment', ['LunarLander-v2']),
            'generations': 10000,
            'popsize': saved_config.get('popsize', 64),
            'print_every': 10,
            'x0_dist': 'U[-1,1]',
            'sigma_init': saved_config.get('sigma_init', 0.1),
            'threads': 16,
            'seed_type': saved_config.get('seed_type', ['randomU2'])[0],
            'NCA_steps': saved_config.get('NCA_steps', 20),
            'NCA_dimension': saved_config.get('NCA_dimension', 3),
            'size_substrate': saved_config.get('size_substrate', 0),
            #'NCA_channels': saved_config.get('NCA_channels', 2),
            'reading_channel': saved_config.get('reading_channel', 0),
            'update_net_channel_dims': saved_config.get('update_net_channel_dims', 4),
            'living_threshold': saved_config.get('alpha_living_threshold', 0),
            'policy_layers': saved_config.get('policy_layers', 4),
            'NCA_bias': saved_config.get('NCA_bias', False),
            'neighborhood': saved_config.get('neighborhood', 'Moore'),
            'save_model': True,
            'random_seed': saved_config.get('random_seed', False),
            'random_seed_env': saved_config.get('random_seed_env', True),
            'normalise': saved_config.get('normalise', True),
            'replace': saved_config.get('replace', False),
            'co_evolve_seed': saved_config.get('co_evolve_seed', False),
            'plastic': saved_config.get('plastic', False),
            'use_wandb': True,
            'wandb_log_interval': 10,
            
            # Use the swept hyperparameters
            # 'policy_layers': wandb_config.policy_layers,
            'cell_noise': saved_config.get('cell_noise', 0), #wandb_.cell_noise,
            'nca_dropout_rate': saved_config.get('nca_dropout_rate', 0), #nca_dropout_rate
            'network_dropout_rate': saved_config.get('network_dropout_rate',0),
            'NCA_channels' : wandb_config.NCA_channels,
        
        }
        
        train(args)
    
    # Run the sweep
    wandb.agent(sweep_id, sweep_train, count=trial_count)  # Run the specified number of trials

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--environment', type=str, default=['LunarLander-v2'], nargs='+', metavar='', help='Environment: any state-vector OpenAI Gym or pyBullet environment may be used')
    # parser.add_argument('--environment', type=str, default=['AntBulletEnv-v0'] , nargs='+', metavar='', help='Environments: any OpenAI Gym or pyBullet environment may be used')
    parser.add_argument('--generations', type=int, default=10000, metavar='', help='Number of generations that the ES will run.')
    parser.add_argument('--popsize', type=int,  default=64, metavar='', help='Population size.')
    parser.add_argument('--print_every', type=int, default=10, metavar='', help='Print every N steps.') 
    parser.add_argument('--x0_dist', type=str, default='U[-1,1]', metavar='', help='Distribution used to sample intial value for CMA-ES') 
    parser.add_argument('--sigma_init', type=float,  default=0.1 , metavar='', help='Initial sigma: modulates the amount of noise used to populate each new generation. Choose carefully: NCA very senstive to weights > 1')
    parser.add_argument('--threads', type=int, metavar='', default=-1, help='Number of threads used to run evolution in parallel: -1 uses all physical cores available, 0 uses a single core and avoids the multiprocessing library.')
    parser.add_argument('--seed_type', type=str,  default='randomU2', metavar='', help='Seed type: activations, zeros, ones, -ones, -1+1, single_seed, two_seeds, three_seeds, four_seeds, randomU [0,1], randomU2 [-1,1], randomU3 [-0.5,0.5], randomU4 [0,0.1], randomU5 [-0.1,0.1], randomN') 
    # parser.add_argument('--seed_type', type=str,  default='single_seed', metavar='', help='Seed type: activations, zeros, ones, -ones, -1+1, single_seed, two_seeds, three_seeds, four_seeds, randomU [0,1], randomU2 [-1,1], randomU3 [-0.5,0.5], randomU4 [0,0.1], randomU5 [-0.1,0.1], randomN') 
    parser.add_argument('--NCA_steps', type=int,  default=20, metavar='', help='NCA steps')
    parser.add_argument('--NCA_dimension', type=int,  default=3, metavar='', help='NCA dimension: 3 uses a single 3D seed and 3DConvs')
    parser.add_argument('--size_substrate', type=int,  default=0, metavar='', help='Size of hidden layers (2D), Size of every fc layer (3D). For 3D: if 0, it takes the smallest size needed.')
    parser.add_argument('--NCA_channels', type=int,  default=2, metavar='', help='NCA channels')
    parser.add_argument('--reading_channel', type=int,  default=0, metavar='', help='Seed channel from which the pattern will be taken to become the NN weigths.')
    parser.add_argument('--update_net_channel_dims', type=int,  default=4, metavar='', help='Number of channels produced by the convolution in the update network (ie. the CA rule). It modulates the number of parameter of the neural update rule.')
    parser.add_argument('--living_threshold', type=float,  default=0, metavar='', help='Cells below living_threshold*(average value of living cells) will be set to zero. If living_threshold=0, all cells are alive.')
    parser.add_argument('--policy_layers', type=int,  default=4, metavar='', help='Number of layers of the policy.')
    parser.add_argument('--NCA_bias', type=bool,  default=False, metavar='', help='Whether the NCA has bias')
    parser.add_argument('--neighborhood', type=str,  default='Moore', metavar='', help='Neighborhood definition: Whether to use "Von Neumann" neighborhood (4+1) or "Moore" (8+1), both with Manhattan and Chebyshev distance respectively of 1.')
    parser.add_argument('--save_model', default=True, action=argparse.BooleanOptionalAction, help='If called, it will not save the resulting model')
    parser.add_argument('--random_seed', default=False, action=argparse.BooleanOptionalAction, help='If true and seed is type random, the NCA uses a random seed at each episode')
    parser.add_argument('--random_seed_env', default=True, action=argparse.BooleanOptionalAction, help='If true is uses a random seed to run the gym environments at each episode')
    parser.add_argument('--normalise', default=True, action=argparse.BooleanOptionalAction, help='Normalise NCA output')
    parser.add_argument('--replace', default=False, action=argparse.BooleanOptionalAction, help='If true, NCA values are replaced by new values, not added')
    parser.add_argument('--co_evolve_seed', default=False, action=argparse.BooleanOptionalAction, help='If true, it co-evolve the initial seed')
    parser.add_argument('--plastic', default=False, action=argparse.BooleanOptionalAction, help='Makes the policy netowork plastic')
    
    # Add wandb-related arguments
    parser.add_argument('--use_wandb', default=False, action=argparse.BooleanOptionalAction, help='Use Weights & Biases for experiment tracking')
    parser.add_argument('--wandb_log_interval', type=int, default=10, metavar='', help='Log to wandb every N generations')
    parser.add_argument('--run_sweep', default=False, action=argparse.BooleanOptionalAction, help='Run a wandb sweep for hyperparameter optimization')
    parser.add_argument('--model_id', type=str, default="1645447353", metavar='', help='ID of the saved model to use as base configuration for sweep')
    parser.add_argument('--trial_count', type=int, default=20, metavar='', help='Number of trials to run in the sweep')

    parser.add_argument('--cell_noise', type=float, default=0.00, metavar='', help='Gaussian noise standard deviation applied during NCA updates')
    parser.add_argument('--nca_dropout_rate', type=float, default=0.00, metavar='', help='Cell dropout probability during NCA updates')
    parser.add_argument('--network_dropout_rate', type=float, default=0.00, metavar='', help='Cell dropout probability during NCA updates')
    parser.add_argument('--layerwise_dropout_rate', type=float, default=0.0, metavar='', help='Fraction of weights to dropout in each policy layer')

    args = parser.parse_args()
    
    if args.run_sweep:
        wandb_sweep(args.model_id, args.trial_count)
    else:
        train(vars(args))
