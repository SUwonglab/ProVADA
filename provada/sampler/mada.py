import os
import numpy as np
import pandas as pd
from collections import Counter
from tqdm.auto import tqdm
from provada.fitness_score import eval_fitness_scores
from provada.utils import (
    predict_location_from_seqs,
    get_ESM_perplexity_one_pass_multiseqs,
    fill_mpnn,
    generate_masked_seq_arr,
    append_csv_line,
    arr_to_aa,
    get_mismatch_fraction_multiseqs
)
from provada.base_sequence_info import BaseSequenceInfo
from provada.sampler import SamplerParams, SamplerResult, TopProteinTracker



# -------------------------------
# Elite selection + resample
# -------------------------------
def resample_population_stochastic(seqs, scores, top_k_frac, temp, prev_temp) -> tuple:
    """
    Two-stage resampling:
      1) Use annealed weights to pick K elites (with replacement).
      2) Uniformly replicate those K to rebuild N sequences.

    Args:
        seqs (np.ndarray): shape (N, L)
        scores (np.ndarray): shape (N,)
        top_k_frac (float): fraction K/N
        temp (float): current temperature T
        prev_temp (float): previous temperature T_prev

    Returns:
        elite_indices (np.ndarray): the K indices picked as elites
        counts (dict): mapping elite_index -> how many copies in new population
    """
    N = len(seqs)
    
    scores = np.asarray(scores, dtype=float)
    # Annealed log‐weights
    if prev_temp is np.inf:
        logw = scores * (1.0/temp)
    else:
        logw = scores * (1.0/temp - 1.0/prev_temp)
    logw -= logw.max()           # stabilize
    w = np.exp(logw)
    w /= w.sum()

    # Number of elites to sample
    K = max(1, int(N * top_k_frac))

    # Sample K elites stochastically by weight
    elite_idx = np.random.choice(N, size=K, replace=True, p=w)

    # Rebuild N by uniform resample from those K
    elite_idx = np.random.choice(elite_idx, size=N, replace=True)

    # Count how many times each elite was used
    counts = dict(Counter(elite_idx))

    elite_scores = scores[elite_idx]
    print("Elite indices:", elite_idx)

    return elite_idx, elite_scores, counts



def resample_population_greedy(seqs, scores, top_k_frac, temp, prev_temp) -> tuple:
    """
    Two‐stage resampling:
      1) Use annealed weights to pick K elites (with replacement).
      2) Uniformly replicate those K to rebuild N sequences.

    Args:
        seqs (np.ndarray): shape (N, L)
        scores (np.ndarray): shape (N,)
        top_k_frac (float): fraction K/N
        temp (float): current temperature T
        prev_temp (float): previous temperature T_prev

    Returns:
        elite_indices (np.ndarray): the K indices picked as elites
        counts (dict): mapping elite_index -> how many copies in new population
    """
    scores = np.asarray(scores, dtype=float)
    N = len(seqs)

    # Number of elites
    K = max(1, int(np.ceil(N * top_k_frac)))

    # Deterministically pick top K by raw score
    topk_idx = np.argsort(scores)[-K:]  # indices of the K best

    # Compute annealed log-weights for those elites
    if np.isinf(prev_temp):
        logw = scores[topk_idx] * (1.0 / temp)
    else:
        logw = scores[topk_idx] * (1.0 / temp - 1.0 / prev_temp)
    logw -= logw.max()
    w = np.exp(logw)
    w /= w.sum()

    # Resample N times from topk_idx according to w
    resampled_idx = np.random.choice(topk_idx, size=N, replace=True, p=w)
    resampled_scores = scores[resampled_idx]

    # Count copies per elite
    counts = dict(Counter(resampled_idx))

    return resampled_idx, resampled_scores, counts




# -------------------------------
# Single SIS iteration
# -------------------------------
def sis_iteration(seqs, 
                  scores,
                  top_k_frac = 0.1, 
                  temperature = None,
                  prev_temperature = None,
                  greedy = False):
    """
    Perform a single iteration of the SIS/CEM algorithm.
    Args:
        seqs (List[np.ndarray]): List of sequences to evaluate.
        fitness_fn (callable): Fitness function to evaluate sequences.
        params_dict (dict): Parameters for the fitness function.
        top_k_frac (float): Fraction of top sequences to select.
        temperature (float): Temperature for sampling.
    Returns:
        List[np.ndarray]: List of new sequences.
        np.ndarray: Array of scores for the new sequences.
    """
    if not isinstance(scores, np.ndarray):
        scores = np.asarray(scores)
    
    if greedy:
        # Use greedy resampling
        pop_idx, elite_scores, counts = resample_population_greedy(seqs, 
                                                                   scores, 
                                                                   top_k_frac, 
                                                                   temperature, 
                                                                   prev_temperature)
    else:
        # Use stochastic resampling
        pop_idx, elite_scores, counts = resample_population_stochastic(seqs, 
                                                                       scores, 
                                                                       top_k_frac, 
                                                                       temperature, 
                                                                       prev_temperature)
    new_pop = [seqs[pop_idx[i]] for i in range(len(pop_idx))]
    return new_pop, elite_scores, counts, pop_idx


def rejection_sampler(seqs, scores, top_k_frac=0.05):
    """
    Rejection sampling for generating sequences.
    
    Args:
        seqs (List[np.ndarray]): List of sequences to evaluate.
        fitness_fn (callable): Function to evaluate fitness of a sequence.
        params_dict (dict): Dictionary of parameters for the fitness function.
        top_k_frac (float): Percentage of top sequences to select.
    """
    # Sort sequences by their scores
    top_k = max(1, int(len(seqs) * top_k_frac))
    
    # Select top-k sequences
    sorted_indices = np.argsort(scores)[::-1][:top_k]
    
    # Resample sequences based on their scores
    retained_seqs = [seqs[i] for i in sorted_indices]
    retained_scores = [scores[i] for i in sorted_indices]

    return retained_seqs, retained_scores


def mh_step(proposal_seqs: np.ndarray, 
            orig_seq: np.ndarray, 
            proposal_scores: np.ndarray, 
            orig_score: float,
            temperature: float,
            sample_attributes: list=None,
            orig_attributes: float=None,
            verbose: bool=False):
    """
    Perform a Metropolis-Hastings step for each proposal sequence.
    Args:
        proposal_seqs (np.ndarray): Array of proposed sequences.
        orig_seq (np.ndarray): Original sequence.
        proposal_scores (np.ndarray): Scores for the proposed sequences.
        orig_score (float): Score for the original sequence.
        temperature (float): Temperature for the MH step.
        sample_attributes (list): Attributes for the proposed sequences. Current
                                  implementation: global score.
        orig_attributes (float): Attributes for the original sequence.
        verbose (bool): Whether to print debug information.
    """

    mh_proposals = []
    mh_proposal_scores = []
    if (sample_attributes is not None) and (orig_attributes is not None):
        mh_sample_attributes = []
        keep_attributes = True
    else:
        keep_attributes = False

    for i in range(len(proposal_seqs)):
        # run MH step for acceptance
        new_seq = proposal_seqs[i]
        new_score = proposal_scores[i]

        if verbose:
            print("new score: %.3f, orig score: %.3f" % (new_score, orig_score))
        
        mh_ratio = np.exp((new_score - orig_score) / temperature)
        
        paccept = mh_ratio
        # accept or reject
        if verbose:
            print(paccept)

        if np.random.rand() < paccept:
            # accept the new sequence
            mh_proposals.append(new_seq)
            mh_proposal_scores.append(float(new_score))
            if keep_attributes:
                mh_sample_attributes.append(sample_attributes[i])
        else:
            # reject the new sequence
            mh_proposals.append(orig_seq)
            mh_proposal_scores.append(float(orig_score))
            if keep_attributes:
                mh_sample_attributes.append(orig_attributes)

    if keep_attributes:
        return mh_proposals, mh_proposal_scores, mh_sample_attributes
    return mh_proposals, mh_proposal_scores


def make_mismatch_schedule(T, 
                           lambda_init, 
                           lambda_max, 
                           alpha=2.0):
    """
    Build a length-T schedule for the mismatch penalty.
    The schedule is a power law from lambda_init to lambda_max.
    Args:
        T (int): Length of the schedule.
        lambda_init (float): Initial mismatch penalty.
        lambda_max (float): Maximum mismatch penalty.
        alpha (float): Power law exponent.

    Returns:
        function: A function that takes an integer t (0 <= t < T) and returns
                  the mismatch penalty for that step.
    """
    t = np.arange(T, dtype=float)       
    frac = t / (T - 1)                   
    return lambda_init + (lambda_max - lambda_init) * (frac ** alpha)


def make_mask_schedule(T, init_mask_frac, min_mask_frac, alpha=2.0):
    """
    Returns a length-T array mask_frac[t] = init_mask_frac 
    annealed down to min_mask_frac via a power law with exponent alpha.

    Args:
        T (int): Length of the schedule.
        init_mask_frac (float): Initial masking fraction.
        min_mask_frac (float): Minimum masking fraction.
        alpha (float): Power law exponent.
    Returns:
        np.ndarray: An array of length T with the masking fractions.
    """
    # Fractional progress from 0 to 1
    t = np.arange(T, dtype=float)
    s = t / (T - 1)
    return init_mask_frac - (init_mask_frac - min_mask_frac) * (s ** alpha)


def MADA(
    sequence: str,
    num_seqs: int,
    num_iter: int,
    sampler_params: SamplerParams,
    base_protein_info: BaseSequenceInfo,
    verbose: bool = True,
    save_sample_traj: bool = True,
    trajectory_path: str = './',
    trajectory_file: str = None,
) -> SamplerResult:
    """
    Perform population-based sampling with iterative mutation, evaluation, and selection.
    """

    assert num_iter >= 2, "Number of iterations must >= 2"
    seq_len = len(sequence)
    init_mask_frac = sampler_params.init_mask_frac
    min_mask_frac = sampler_params.min_mask_frac
    
    top_k_frac = sampler_params.top_k_frac
    alpha = sampler_params.alpha

    T_schedule = (
        sampler_params.T_schedule
        if hasattr(sampler_params, 'T_schedule') and sampler_params.T_schedule is not None
        else np.geomspace(2.0, 0.1, num=num_iter-1, endpoint=True)
    )
    T_schedule = np.concatenate(([np.inf], T_schedule))  

    
    # Anneal the mismatch penalty
    if sampler_params.anneal_mismatch_penalty:
        mismatch_penalty_schedule = (
            sampler_params.mismatch_penalty_schedule
            if hasattr(sampler_params, 'mismatch_penalty_schedule') and sampler_params.mismatch_penalty_schedule is not None
            else make_mismatch_schedule(num_iter, 
                                        lambda_init=sampler_params.mismatch_penalty,
                                        lambda_max=sampler_params.max_mismatch_penalty,
                                        alpha=sampler_params.alpha)
        )
    else:
        mismatch_penalty_schedule = np.ones(num_iter) * sampler_params.mismatch_penalty

    sampler_params.mismatch_penalty_schedule = mismatch_penalty_schedule
    print(f"Mismatch penalty schedule: {mismatch_penalty_schedule}")
    print(f"Temperature schedule: {T_schedule}")

    masking_frac_schedule = make_mask_schedule(num_iter, 
                                               init_mask_frac, 
                                               min_mask_frac, 
                                               alpha=alpha)
    print(f"Masking fraction schedule: {masking_frac_schedule}")
    
    
    # Initialize the population with copies of the initial sequence
    population = [np.asarray(arr_to_aa(sequence)) for _ in range(num_seqs)]
    tracker = TopProteinTracker(max_size=num_seqs)
    traj_rows = []

    if save_sample_traj:
        if trajectory_file is None:
            trajectory_file = "population_trajectory.csv"
        trajectory_path = os.path.join(trajectory_path, trajectory_file)
        os.makedirs(os.path.dirname(trajectory_path), exist_ok=True)

        empty_row = {
                        "chain": "NA",
                        "iteration": "NA",
                        "temperature": "NA",
                        "percent_mutated": "NA",
                        f"prob_{base_protein_info.target_label}": "NA",
                        "pppl": "NA",
                        "MPNN_score": "NA",
                        "fitness": "NA",
                        "sequence": "NA",
                    }

        append_csv_line(
            path = trajectory_path,
            line = empty_row
            )

    for t, temperature in enumerate(tqdm(T_schedule, desc="Population iterations")):
        mismatch_penalty = mismatch_penalty_schedule[t]
        mask_frac = masking_frac_schedule[t]
        proposals = []
        proposal_scores = []
        hard_fixed_positions = base_protein_info.hard_fixed_positions

         # mask & fill each chain
        if t == 0:
            # For the first iteration, we use the initial sequence
            masked_seq = generate_masked_seq_arr(
                num_fixed_positions = max(1, (1 - mask_frac) * (seq_len - len(hard_fixed_positions))),
                input_seq = sequence,
                hard_fixed_positions = hard_fixed_positions
            )

            # Start from the initial sequence
            # Call MPNN once to fill the initial sequence x with population size
            mpnn_output = fill_mpnn(
                masked_seq,
                pdb_path=base_protein_info.input_pdb,
                protein_chain=base_protein_info.protein_chain,
                num_seqs_gen=num_seqs,
                mpnn_sample_temp=sampler_params.mpnn_sample_temp,
            )

            filled_arrs = mpnn_output['filled_seqs']

            mpnn_scores = mpnn_output['new_global_scores']

            fitness_scores = eval_fitness_scores(filled_arrs, base_protein_info, mpnn_scores, mismatch_penalty)
            proposals = filled_arrs
            proposal_scores = fitness_scores
            proposal_mpnn_scores = mpnn_scores

            probs = predict_location_from_seqs(
                seqs=proposals,
                target_label=base_protein_info.target_label,
                clf_name=base_protein_info.clf_name,
                classifier=base_protein_info.classifier,
                ESM_model=base_protein_info.ESM_model,
                tokenizer=base_protein_info.tokenizer,
                device=base_protein_info.device,
            )

            pppls = get_ESM_perplexity_one_pass_multiseqs(
                seqs=proposals,
                model=base_protein_info.ESM_model,
                tokenizer=base_protein_info.tokenizer,
                device=base_protein_info.device,
            )

            frac_mismatches = get_mismatch_fraction_multiseqs(seqs = proposals, 
                                                              ref_seq= sequence)
        
            # logging
            for i in range(len(proposals)):
                row = {
                    "chain": i,
                    "iteration": t,
                    "temperature": np.inf,
                    "percent_mutated": frac_mismatches[i],
                    f"prob_{base_protein_info.target_label}": float(probs[i]),
                    "pppl": pppls[i],
                    "MPNN_score": proposal_mpnn_scores[i],
                    "fitness": float(proposal_scores[i]),
                    "sequence": arr_to_aa(proposals[i]),
                }
                traj_rows.append(row)
                tracker.add(row)

                if save_sample_traj:
                    append_csv_line(
                        path=trajectory_path,
                        line=row
                    )


        if t > 0:
            for idx, idx_num_seq in counts.items():
                # idx : index of the chain to fill
                # idx_num_seq : number of population to fill
                cur_seq = population[int(idx)]

                # Generate a masked sequence for the current chain
                masked_seq = generate_masked_seq_arr(num_fixed_positions = max(1, (1 - mask_frac) * (seq_len - len(hard_fixed_positions))),
                                                     input_seq = arr_to_aa(cur_seq),
                                                     hard_fixed_positions = hard_fixed_positions)  
                
                # Call MPNN once to fill the initial sequence x with population size
                mpnn_output = fill_mpnn(
                    masked_seq,
                    pdb_path=base_protein_info.input_pdb,
                    protein_chain=base_protein_info.protein_chain,
                    num_seqs_gen=idx_num_seq,
                    mpnn_sample_temp=sampler_params.mpnn_sample_temp,
                )

                filled_arrs = mpnn_output['filled_seqs']

                mpnn_scores = mpnn_output['new_global_scores']

                fitness_scores = eval_fitness_scores(filled_arrs, base_protein_info, mpnn_scores, mismatch_penalty)

                cur_sample_atributes = mpnn_scores
                orig_sample_attributes = mpnn_output['old_global_score']

                cur_proposals, cur_proposal_scores, cur_mpnn_scores = mh_step(
                    proposal_seqs = filled_arrs, 
                    orig_seq = cur_seq, 
                    proposal_scores = fitness_scores, 
                    orig_score = scores[int(idx)],
                    sample_attributes = cur_sample_atributes,
                    orig_attributes = orig_sample_attributes,
                    temperature = temperature,
                    verbose = verbose
                )

                probs = predict_location_from_seqs(
                    seqs=cur_proposals,
                    target_label=base_protein_info.target_label,
                    clf_name=base_protein_info.clf_name,
                    classifier=base_protein_info.classifier,
                    ESM_model=base_protein_info.ESM_model,
                    tokenizer=base_protein_info.tokenizer,
                    device=base_protein_info.device,
                )

                pppls = get_ESM_perplexity_one_pass_multiseqs(
                    seqs=cur_proposals,
                    model=base_protein_info.ESM_model,
                    tokenizer=base_protein_info.tokenizer,
                    device=base_protein_info.device,
                )

                frac_mismatches = get_mismatch_fraction_multiseqs(seqs = cur_proposals, 
                                                                  ref_seq= sequence)
            
                # logging
                for i in range(len(cur_proposals)):
                    row = {
                        "chain": idx,
                        "iteration": t,
                        "temperature": temperature,
                        "percent_mutated": frac_mismatches[i],
                        f"prob_{base_protein_info.target_label}": float(probs[i]),
                        "pppl": pppls[i],
                        "MPNN_score": cur_mpnn_scores[i],
                        "fitness": float(cur_proposal_scores[i]),
                        "sequence": arr_to_aa(cur_proposals[i]),
                    }

                    traj_rows.append(row)
                    tracker.add(row)

                    if save_sample_traj:
                        append_csv_line(
                            path=trajectory_path,
                            line=row
                        )

                proposals.extend(cur_proposals)
                proposal_scores.extend(cur_proposal_scores)
                
        # Update 
        population = proposals
        scores = proposal_scores

        # 4) resample via SIS
        if t < num_iter - 1:
            prev_temperature = T_schedule[t] 
            temperature = T_schedule[t+1]

            population, scores, counts, idx = sis_iteration(
                population, 
                scores,
                top_k_frac=top_k_frac,
                temperature=temperature,
                prev_temperature = prev_temperature,
                greedy=sampler_params.greedy
            )
        else:
            # last step: rejection sampling
            # population, scores = rejection_sampler(
            #     population, 
            #     scores,
            #     top_k_frac=1.0
            # )
            pass
        
        print("Iteration %d, best score: %.3f" % (t, max(scores)))
        best_prob = max(tracker.get_all_scores(f"prob_{base_protein_info.target_label}"))
        print("Iteration %d, best probability: %.3f" % (t, best_prob))
        print("[Fraction of mismatches] Iteration %d, avg fraction of mismatches: %.3f" % (t, np.mean(frac_mismatches)))

    # Finalize results, outputting the top k sequences and scores
    final_sequences, final_fitness, final_mpnn_scores, final_cls_probs, final_perps = [], [], [], [], []

    final_entries = tracker.get_top(n = max(1, int(num_seqs * top_k_frac)),
                                    sort_key="fitness")
    for entry in final_entries:
        final_sequences.append(entry['sequence'])
        final_fitness.append(entry['fitness'])
        final_mpnn_scores.append(entry['MPNN_score'])
        final_cls_probs.append(entry[f"prob_{base_protein_info.target_label}"])
        final_perps.append(entry['pppl'])

    traj_df = pd.DataFrame(traj_rows)

    return SamplerResult(
            sampler_params=sampler_params,
            masked_sequences = None,
            filled_sequences=final_sequences,
            mpnn_scores=final_mpnn_scores,
            fitness_scores=final_fitness,
            classifier_probs=final_cls_probs,
            perplexities=final_perps,
            trajectory=traj_df,
            top_tracker=tracker
        )