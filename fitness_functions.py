import gym
from gym import wrappers as w
from gym.spaces import Discrete, Box
import pybullet_envs
import numpy as np
import torch
import torch.nn as nn
import pickle
import os
import time
import multiprocessing as mp
from policies import MLPn
from utils_and_wrappers import FireEpisodicLifeEnv, ScaledFloatFrame
from utils_and_wrappers import generate_seeds3D, policy_layers_parameters, dimensions_env
from NCA_3D import CellCAModel3D

gym.logger.set_level(40)
torch.set_default_dtype(torch.float64)

def fitnessRL(evolved_parameters, nca_config, archive, lock, render=False, debugging=False, 
             visualise_weights=False, visualise_network=False, training=True):
    """Returns the NEGATIVE episodic fitness of agents using LSH-accelerated novelty calculation"""
    
    # Local LSH parameters
    L = 5                # Number of hash functions
    WIDTH = 1.0          # Bucket width
    LSH_SEED = 42        # Fixed seed for reproducibility

    try:
        novelty_alpha = nca_config['novelty_alpha']
        novelty_k = nca_config['novelty_k']
    except KeyError:
        novelty_alpha = 0
        novelty_k = 0

    with torch.no_grad():
        cum_reward = 0
        patterns = []
        seed_offset = 0
        
        for i, environment in enumerate(nca_config['environment']):
            # Environment setup
            try:
                env = gym.make(environment, verbose=0)
            except Exception as e:
                env = gym.make(environment)
                
            if not nca_config['random_seed_env']:
                env.seed(nca_config['RANDOM_SEED'])
            
            if environment[-12:-6] == 'Bullet' and render:
                env.render()
            
            mujoco_env = False
            if 'AntBullet' in environment:
                mujoco_env = True

            # Environment wrappers
            try:
                if 'FIRE' in env.unwrapped.get_action_meanings():
                    env = FireEpisodicLifeEnv(env)
            except AttributeError:
                pass

            # Get environment dimensions
            input_dim, action_dim, pixel_env = dimensions_env(environment)
            if pixel_env:
                env = w.ResizeObservation(env, 32)
                env = ScaledFloatFrame(env)

            # Policy network setup
            if nca_config['NCA_dimension'] == 3:
                p = MLPn(nca_config['size_substrate'], action_dim, 
                        nca_config['size_substrate'], bias=False, 
                        layers=nca_config['policy_layers'])
                for param in p.parameters():
                    param.requires_grad = False
            else:
                raise NotImplementedError("Only 3D NCA supported")

            # Initialize NCA
            ca = CellCAModel3D(nca_config)
            nca_nb_weights = torch.nn.utils.parameters_to_vector(ca.parameters()).shape[0]
            if render:
                nca_nb_weights = torch.nn.utils.parameters_to_vector(ca.parameters()).shape[0]
                seed_size = nca_config['NCA_channels']*nca_config['policy_layers']*nca_config['size_substrate']*nca_config['size_substrate']
                policy_nb_functional_params = torch.nn.utils.parameters_to_vector(p.parameters()).shape[0]
                print(f'Policy has: {policy_nb_functional_params}')
                print('\n.......................................................')
                print('\n' + str(nca_config['environment']) + ' with', nca_nb_weights, 'trainable parameters controlling a policy', str(p)[:3], 'with', policy_nb_functional_params, 'effective weights with a seed size of', seed_size, ' and seed type', nca_config['seed_type'], '\n')
                if nca_config['plastic']: print('Plastic Policy network') 
                print('.......................................................\n')
            
            nn.utils.vector_to_parameters(torch.tensor(evolved_parameters[:nca_nb_weights], 
                                      dtype=torch.float64), ca.parameters())

            # Seed handling
            observation = env.reset().astype(np.float64)
            if pixel_env:
                observation = observation.flatten()

            if nca_config['random_seed']:
                seed = generate_seeds3D(policy_layers_parameters(p), 
                                      nca_config['seed_type'][i],
                                      nca_config['NCA_channels'],
                                      observation, environment)
            elif nca_config['co_evolve_seed']:
                sp = nca_config['seeds_shapes'][i]
                evolved_seed = torch.tensor(evolved_parameters[nca_nb_weights + seed_offset:
                                              nca_nb_weights + seed_offset + nca_config['seeds_size'][i]])
                seed = torch.reshape(evolved_seed, sp[0])
                seed_offset += nca_config['seeds_size'][i]
            else:
                seed = nca_config['seeds'][i]

            # Generate policy weights
            new_pattern, _ = ca.forward(seed, steps=nca_config['NCA_steps'],
                                      reading_channel=nca_config['reading_channel'],
                                      policy_layers=nca_config['policy_layers'],
                                      run_pca=False, visualise_weights=visualise_weights,
                                      visualise_network=visualise_network,
                                      inOutdim=[input_dim, action_dim], training=training)
            
            generated_policy_weights = new_pattern.detach()[0]
            flattened_pattern = generated_policy_weights.flatten().cpu().numpy()
            patterns.append(flattened_pattern)

            # Load weights into policy network
            reading_channel = nca_config['reading_channel']
            for layer_idx in range(nca_config['policy_layers']):
                layer_weights = generated_policy_weights[reading_channel][layer_idx]
                if layer_idx == nca_config['policy_layers'] - 1:
                    layer_weights = layer_weights[:action_dim, :]
                if training and nca_config.get('torch_dropout_rate', 0) > 0:
                    dropout_mask = (torch.rand_like(layer_weights) > 
                                  nca_config['torch_dropout_rate'])
                    scale = 1.0 / (1.0 - nca_config['torch_dropout_rate'])
                    layer_weights = layer_weights * dropout_mask.to(layer_weights.device) * scale
                p.out[2*layer_idx].weight = nn.Parameter(layer_weights, requires_grad=False)

            # Environment interaction loop
            if 'AntBullet' in environment:
                action = np.zeros(8)
                for _ in range(40):
                    __ = env.step(action)  # Burn-in phase
            
            obs = observation
            episode_reward = 0
            neg_count = 0
            for t in range(1000):
                obs_tensor = torch.tensor(obs)
                action = p(obs_tensor).detach().numpy()
                
                if 'Bullet' in environment or isinstance(env.action_space, Box):
                    action = np.tanh(action)
                elif isinstance(env.action_space, Discrete):
                    action = np.argmax(action)
                
                obs, reward, done, _ = env.step(action)
                episode_reward += reward
                
                if pixel_env:
                    obs = obs.flatten()
                obs = obs.astype(np.float64)
                
                # Early stopping conditions
                if t > 100:
                    threshold = 1.01 if mujoco_env else 0.01
                    neg_count = neg_count + 1 if reward < threshold else 0
                    if done or (neg_count > 50 if mujoco_env else neg_count > 30):
                        break
            
            env.close()
            cum_reward += episode_reward

    # LSH-based novelty calculation
    pattern_buckets = []
    if len(patterns) > 0:
        # Validate pattern dimensions
        pattern_dim = len(patterns[0])
        if any(len(p) != pattern_dim for p in patterns):
            raise ValueError("Inconsistent pattern dimensions for LSH")
        
        # Generate LSH hash functions
        np.random.seed(LSH_SEED)
        hash_fns = [(np.random.randn(pattern_dim), 
                   np.random.uniform(0, WIDTH)) for _ in range(L)]
        
        # Compute bucket signatures
        pattern_buckets = []
        for p in patterns:
            signature = []
            for a, b in hash_fns:
                hash_val = np.floor((np.dot(p, a) + b) / WIDTH)
                signature.append(str(int(hash_val)))
            pattern_buckets.append(signature)

    novelty_score = 0.0
    with lock:
        current_archive = list(archive)
        
        if current_archive and patterns:
            total_novelty = 0.0
            valid_patterns = 0
            
            for pattern, buckets in zip(patterns, pattern_buckets):
                # Find candidates through bucket matching
                candidates = []
                for archived_p, archived_buckets in current_archive:
                    if any(b in archived_buckets for b in buckets):
                        candidates.append(archived_p)
                
                # Calculate distances to candidates
                distances = [np.linalg.norm(pattern - cand) for cand in candidates]
                distances.sort()
                k_actual = min(novelty_k, len(distances))
                
                if k_actual > 0:
                    avg_distance = sum(distances[:k_actual]) / k_actual
                    total_novelty += avg_distance
                    valid_patterns += 1
            
            if valid_patterns > 0:
                novelty_score = total_novelty / valid_patterns

        # Add new patterns to archive
        for p, b in zip(patterns, pattern_buckets):
            archive.append((p.copy(), b))

    adjusted_fitness = -cum_reward - novelty_alpha * novelty_score
    return (adjusted_fitness, novelty_score)

def evaluate(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--id', type=str, default='1645447353', metavar='', help='Run id')   # lander 5 layers
    # parser.add_argument('--id', type=str, default='1646940683', metavar='', help='Run id')   # lander single seed
    # parser.add_argument('--id', type=str, default='1645360631', metavar='', help='Run id')   # ant 3 layers
    # parser.add_argument('--id', type=str, default='1645605120', metavar='', help='Run id')   # ant 30 layers deep one
    # parser.add_argument('--id', type=str, default='1647084085', metavar='', help='Run id')   # ant single seed
    
    parser.add_argument('--render', type=bool, default=1)
    parser.add_argument('--visualise_weigths', type=bool, default=0) 
    parser.add_argument('--visualise_network', type=bool, default=0)
    parser.add_argument('--mean_solution', type=bool, default=1,  help='Whether to use the best population mean, else it will use best individual solution')
    parser.add_argument('--evaluation_runs', type=int, default=1,  help='Number of runs to evaluate model')


    args = parser.parse_args()
    
    if args.visualise_weigths and args.visualise_network:
        raise ValueError('Can not visualise both weight matrix and network at the same time')
    
    # Load NCA config
    nca_config = pickle.load( open( 'saved_models/' + args.id + '/nca.config', "rb" ) )
    print(nca_config['plastic'])
    
    for key, value in nca_config.items():
        if key != 'seeds':
            print(key,':',value)
    
    # Load evolved NCA weightsl
    for root, dirs, files in os.walk("saved_models/" + args.id):
        for file in files:
            if args.mean_solution and file and 'meansolution' in file:
                evolved_parameters = np.load('saved_models/' + args.id + '/' + file) 
                print(f"\nUsing best MEAN solution") 
            elif not args.mean_solution and file and 'bestsolution' in file:
                evolved_parameters = np.load('saved_models/' + args.id + '/' + file) 
                print(f"\nUsing BEST individual solution") 

    evals = []
    runs = args.evaluation_runs
    for _ in range(runs):
        evals.append(-1*fitnessRL(evolved_parameters=evolved_parameters, nca_config=nca_config, archive=[], lock=mp.Manager().Lock(),render=args.render, visualise_weights=args.visualise_weigths, visualise_network=args.visualise_network, training=False))
    evals = np.array(evals)
    print(f'mean reward {np.mean(evals)}. Var: {np.std(evals)}. Shape {evals.shape}')

if __name__ == '__main__':
    import argparse
    import sys
    evaluate(sys.argv)