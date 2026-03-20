from enad_ga import run_enad_binary_stack_all, run_enad_binary_stack_transfer
import numpy as np
import argparse
import sys

class Solution:
    def __init__(self, binary_list):
        self.binary_list = binary_list
        self.auroc = 0
        self.aupr = 0
        self.unified_metric = 0
        self.f1_score = 0

def evaluate_fitness(population, config):
    for solution in population:
        try:
            if config['mode'] == 'single':
                auroc, aupr, f1_score = run_enad_binary_stack_all(
                    ds_name=config['dataset'],
                    net_type=config['net_type'],
                    adv_type=config['adv_type'],
                    binary_list=solution.binary_list,
                    outf=config['outf'], 
                    eval_on_test=config['eval_on_test'])
            else:  # transfer mode
                auroc, aupr, f1_score = run_enad_binary_stack_transfer(
                    ds_name=config['dataset'], 
                    net_type=config['net_type'], 
                    adv_type=config['adv_type'], 
                    adv_transfer_type=config['transfer_type'], 
                    binary_list=solution.binary_list, 
                    outf=config['outf'], 
                    eval_on_test=config['eval_on_test'])
            
            solution.auroc = auroc
            solution.aupr = aupr
            solution.f1_score = f1_score
            
            # Calculate unified metric based on chosen strategy
            if config['fitness_function'] == 'auroc_aupr':
                solution.unified_metric = auroc * aupr
            elif config['fitness_function'] == 'weighted_sum':
                solution.unified_metric = config['auroc_weight'] * auroc + config['aupr_weight'] * aupr + config['f1_weight'] * f1_score
            elif config['fitness_function'] == 'auroc_only':
                solution.unified_metric = auroc
            elif config['fitness_function'] == 'aupr_only':
                solution.unified_metric = aupr
            else:  # f1_only
                solution.unified_metric = f1_score
        
        except Exception as e:
            print(f"Error evaluating solution {solution.binary_list}: {e}")
            solution.auroc = 0
            solution.aupr = 0
            solution.f1_score = 0
            solution.unified_metric = 0

def uniform_crossover(parent1, parent2):
    offspring1 = []
    offspring2 = []
    for i in range(len(parent1.binary_list)):
        if np.random.random() < 0.5:
            offspring1.append(parent1.binary_list[i])
            offspring2.append(parent2.binary_list[i])
        else:
            offspring1.append(parent2.binary_list[i])
            offspring2.append(parent1.binary_list[i])
            
    if sum(offspring1) == 0:
        idx = np.random.randint(0, 8)
        offspring1[idx] = 1
    if sum(offspring2) == 0:
        idx = np.random.randint(0, 8)
        offspring2[idx] = 1
        
    return Solution(offspring1), Solution(offspring2)

def mutation(solution, mutation_rate=0.4):
    mutated = solution.binary_list.copy()
    for i in range(len(mutated)):
        if np.random.random() < mutation_rate:
            mutated[i] = 1 - mutated[i]

    if sum(mutated) == 0:
        idx = np.random.randint(0, 8)
        mutated[idx] = 1
    return Solution(mutated)
    
def generate_offspring(parents, mutation_rate):
    offsprings = []
    for parent in parents:
        offspring1, offspring2 = uniform_crossover(parent[0], parent[1])
        offspring1 = mutation(offspring1, mutation_rate)
        offspring2 = mutation(offspring2, mutation_rate)
        offsprings.extend([offspring1, offspring2])
    return offsprings

def dominates(ind1, ind2):
    return ind1.unified_metric > ind2.unified_metric
    
def tournament(population, tournament_size):
    participants = np.random.choice(population, size=(tournament_size,), replace=False)
    best = participants[0]
    for participant in participants[1:]:
        if (dominates(participant, best)):
            best=participant
    return best

def tournament_selection(population, tournament_size):
    parents = []
    while len(parents) < len(population)//2:
        parent1 = tournament(population, tournament_size)
        parent2 = tournament(population, tournament_size)

        while parent1 is parent2:
            parent2 = tournament(population, tournament_size)
            
        parents.append([parent1, parent2])
    return parents

def initialize_population(population_size, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    population = []
    for _ in range(population_size):
        binary_list = np.random.randint(0, 2, 8).tolist()
        while sum(binary_list) == 0:
            binary_list = np.random.randint(0, 2, 8).tolist()
        population.append(Solution(binary_list))
    return population

def genetic_algorithm(config):
    population_size = config['population_size']
    tournament_size = config['tournament_size']
    num_generations = config['num_generations']
    mutation_rate = config['mutation_rate']
    verbose = config['verbose']

    population = initialize_population(population_size, config.get('seed'))
    evaluate_fitness(population, config)

    if verbose:
        print("=" * 80)
        print("🧬 GENETIC ALGORITHM STARTED")
        print("=" * 80)
        print(f"Dataset: {config['dataset']}")
        print(f"Network: {config['net_type']}")
        print(f"Mode: {config['mode']}")
        if config['mode'] == 'single':
            print(f"Attack Type: {config['adv_type']}")
        else:
            print(f"Source Attack: {config['adv_type']}")
            print(f"Target Attack: {config['transfer_type']}")
        print(f"Population Size: {population_size}")
        print(f"Tournament Size: {tournament_size}")
        print(f"Generations: {num_generations}")
        print(f"Mutation Rate: {mutation_rate}")
        print(f"Fitness Function: {config['fitness_function']}")
        print("=" * 80)

    # Track best solutions across generations
    best_solutions_history = []

    for generation in range(num_generations):
        if verbose:
            print(f"\n🔄 GENERATION {generation + 1}/{num_generations}")
            print("-" * 50)
        
        parents = tournament_selection(population, tournament_size)
        offsprings = generate_offspring(parents, mutation_rate)

        evaluate_fitness(offsprings, config)

        combined_population = population + offsprings
        combined_population.sort(key=lambda x: x.unified_metric, reverse=True)

        # Truncated selection
        population = combined_population[:population_size]

        best_solution = population[0]
        best_solutions_history.append(best_solution)
        
        if verbose:
            print(f"🏆 Best Solution: {best_solution.binary_list}")
            print(f"📊 Performance Metrics:")
            print(f"   • AUROC:    {best_solution.auroc:.4f}")
            print(f"   • AUPR:     {best_solution.aupr:.4f}")
            print(f"   • F1 Score: {best_solution.f1_score:.4f}")
            print(f"   • Unified:  {best_solution.unified_metric:.4f}")
            
            # Show top 3 solutions for this generation
            print(f"\n📈 Top 3 Solutions:")
            for i, sol in enumerate(population[:3]):
                print(f"   {i+1}. {sol.binary_list} → Unified: {sol.unified_metric:.4f}")
        else:
            # Compact output for non-verbose mode
            feature_str = ''.join(map(str, best_solution.binary_list))
            print(f"Gen{generation+1:02d},{feature_str},{best_solution.auroc:.4f},{best_solution.aupr:.4f},{best_solution.f1_score:.4f},{best_solution.unified_metric:.4f}")
    
    if verbose:
        print("\n" + "=" * 80)
        print("🎉 GENETIC ALGORITHM COMPLETED")
        print("=" * 80)
        
        best_solution = population[0]
        print(f"🏅 FINAL BEST SOLUTION")
        print("-" * 40)
        print(f"Binary Vector:     {best_solution.binary_list}")
        print(f"AUROC:            {best_solution.auroc:.4f}")
        print(f"AUPR:             {best_solution.aupr:.4f}")
        print(f"F1 Score:         {best_solution.f1_score:.4f}")
        print(f"Unified Metric:   {best_solution.unified_metric:.4f}")
        print("=" * 80)
    
    return population[0], best_solutions_history

def main():
    parser = argparse.ArgumentParser(
        description='Genetic Algorithm for ENAD Feature Selection',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="""
Examples:
  # Single attack optimization
  python GA.py --dataset cifar10 --adv-type DeepFool --generations 10 --population 16
  
  # Transfer learning optimization
  python GA.py --dataset svhn --mode transfer --adv-type FGSM --transfer-type BIM --generations 5
  
  # Custom fitness function
  python GA.py --dataset cifar10 --adv-type CWL2 --fitness weighted_sum --auroc-weight 0.6 --aupr-weight 0.4
  
  # Save results to CSV
  python GA.py --dataset cifar10 --adv-type DeepFool --generations 20 > ga_results.csv
        """)
    
    # Dataset and model configuration
    parser.add_argument('--dataset', '--ds', type=str, 
                       choices=['cifar10', 'svhn'], 
                       required=True,
                       help='Dataset to use')
    
    parser.add_argument('--net-type', '--net', type=str, 
                       default='resnet',
                       help='Network architecture type')
    
    # Attack configuration
    parser.add_argument('--mode', type=str,
                       choices=['single', 'transfer'],
                       default='single',
                       help='Optimization mode: single attack or transfer learning')
    
    parser.add_argument('--adv-type', '--adv', type=str,
                       choices=['DeepFool', 'FGSM', 'BIM', 'CWL2'],
                       required=True,
                       help='Adversarial attack type (source for transfer mode)')
    
    parser.add_argument('--transfer-type', '--transfer', type=str,
                       choices=['DeepFool', 'FGSM', 'BIM', 'CWL2'],
                       help='Target attack type for transfer mode (required if mode=transfer)')
    
    # Genetic Algorithm parameters
    parser.add_argument('--population-size', '--population', type=int, 
                       default=8,
                       help='Population size')
    
    parser.add_argument('--num-generations', '--generations', type=int, 
                       default=5,
                       help='Number of generations')
    
    parser.add_argument('--tournament-size', '--tournament', type=int, 
                       default=2,
                       help='Tournament size for selection')
    
    parser.add_argument('--mutation-rate', '--mutation', type=float, 
                       default=0.4,
                       help='Mutation rate (0.0 to 1.0)')
    
    # Fitness function configuration
    parser.add_argument('--fitness-function', '--fitness', type=str,
                       choices=['auroc_aupr', 'weighted_sum', 'auroc_only', 'aupr_only', 'f1_only'],
                       default='auroc_aupr',
                       help='Fitness function: auroc_aupr (product), weighted_sum, or single metric')
    
    parser.add_argument('--auroc-weight', type=float, default=0.5,
                       help='Weight for AUROC in weighted_sum fitness (only used with --fitness weighted_sum)')
    
    parser.add_argument('--aupr-weight', type=float, default=0.5,
                       help='Weight for AUPR in weighted_sum fitness (only used with --fitness weighted_sum)')
    
    parser.add_argument('--f1-weight', type=float, default=0.0,
                       help='Weight for F1 score in weighted_sum fitness (only used with --fitness weighted_sum)')
    
    # Evaluation configuration
    parser.add_argument('--eval-test', action='store_true',
                       help='Evaluate on test set (default: evaluate on GA validation set)')
    
    # Output configuration
    parser.add_argument('--outf', type=str, 
                       default='/kaggle/working',
                       help='Output folder')
    
    parser.add_argument('--seed', type=int,
                       help='Random seed for reproducibility')
    
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Quiet mode: minimal output suitable for CSV format')
    
    # Multiple runs
    parser.add_argument('--runs', type=int, default=1,
                       help='Number of independent runs')
    
    args = parser.parse_args()
    
    # Validation
    if args.mode == 'transfer' and args.transfer_type is None:
        parser.error("--transfer-type is required when --mode is 'transfer'")
    
    if args.mode == 'transfer' and args.adv_type == args.transfer_type:
        parser.error("Source and target attack types should be different for transfer learning")
    
    if args.fitness_function == 'weighted_sum':
        total_weight = args.auroc_weight + args.aupr_weight + args.f1_weight
        if abs(total_weight - 1.0) > 1e-6:
            parser.error(f"Weights must sum to 1.0, got {total_weight}")
    
    # Configuration dictionary
    config = {
        'dataset': args.dataset,
        'net_type': args.net_type,
        'mode': args.mode,
        'adv_type': args.adv_type,
        'transfer_type': args.transfer_type,
        'population_size': args.population_size,
        'num_generations': args.num_generations,
        'tournament_size': args.tournament_size,
        'mutation_rate': args.mutation_rate,
        'fitness_function': args.fitness_function,
        'auroc_weight': args.auroc_weight,
        'aupr_weight': args.aupr_weight,
        'f1_weight': args.f1_weight,
        'eval_on_test': args.eval_test,
        'outf': args.outf,
        'seed': args.seed,
        'verbose': not args.quiet
    }
    
    # Print CSV header for quiet mode
    if args.quiet and args.runs == 1:
        print("Generation,Features,AUROC,AUPR,F1,Unified")
    elif args.quiet and args.runs > 1:
        print("Run,Generation,Features,AUROC,AUPR,F1,Unified")
    
    all_results = []
    
    # Run the genetic algorithm
    for run in range(args.runs):
        if args.runs > 1 and config['verbose']:
            print(f"\n{'='*20} RUN {run+1}/{args.runs} {'='*20}")
        
        # Update seed for multiple runs
        if args.seed is not None:
            config['seed'] = args.seed + run
        
        best_solution, history = genetic_algorithm(config)
        all_results.append((best_solution, history))
        
        # Output results for multiple runs in quiet mode
        if args.quiet and args.runs > 1:
            for gen, sol in enumerate(history):
                feature_str = ''.join(map(str, sol.binary_list))
                print(f"{run+1},{gen+1:02d},{feature_str},{sol.auroc:.4f},{sol.aupr:.4f},{sol.f1_score:.4f},{sol.unified_metric:.4f}")
    
    # Summary for multiple runs
    if args.runs > 1 and config['verbose']:
        print(f"\n{'='*20} SUMMARY OF {args.runs} RUNS {'='*20}")
        unified_scores = [result[0].unified_metric for result, _ in all_results]
        print(f"Best Unified Score: {max(unified_scores):.4f}")
        print(f"Mean Unified Score: {np.mean(unified_scores):.4f}")
        print(f"Std Unified Score:  {np.std(unified_scores):.4f}")
        
        # Best overall solution
        best_idx = np.argmax(unified_scores)
        best_overall = all_results[best_idx][0]
        print(f"\nBest Overall Solution (Run {best_idx+1}):")
        print(f"Features: {best_overall.binary_list}")
        print(f"AUROC: {best_overall.auroc:.4f}")
        print(f"AUPR: {best_overall.aupr:.4f}")
        print(f"F1: {best_overall.f1_score:.4f}")
        print(f"Unified: {best_overall.unified_metric:.4f}")

if __name__ == '__main__':
    main()