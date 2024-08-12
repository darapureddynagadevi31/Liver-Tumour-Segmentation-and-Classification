import time
import numpy as np


#  improved Lemurs Optimizer: A New Metaheuristic Algorithm --- updates startsfrom Line no -> 37
def Proposed(swarm, fobj, lb, ub, Max_iter):
    [PopSize, dim] = swarm.shape[0], swarm.shape[1]
    jumping_rate_min = 0.1
    jumping_rate_max = 0.5
    runs = 2
    ObjVal = np.zeros((1, PopSize))
    BestResults = np.zeros((runs, 1))

    conv = np.zeros(Max_iter)
    cg_curve = np.zeros(Max_iter)
    Fitness = np.zeros(dim)
    for i in range(PopSize):
        Fitness[i] = fobj(swarm[i, :])
        # Fitness = calculateFitness(ObjVal)
        # ===================== loop ===================================
    ct = time.time()
    itr = 0
    while itr < Max_iter:

        jumping_rate = jumping_rate_max - itr * ((jumping_rate_max - jumping_rate_min) / Max_iter)
        # swarm looking to go away from killer "free risk started too high "
        sorted_objctive, sorted_indexes = np.sort(Fitness), np.argsort(Fitness)
        for i in range(PopSize):
            current_solution = np.where(sorted_indexes == i)
            near_solution_postion = current_solution - 1
            if near_solution_postion == 0:
                near_solution_postion = 1
            near_solution = sorted_indexes[near_solution_postion]
            cost, best_solution_Index = np.amin(ObjVal)
            NewSol = swarm[i, :]
            for j in range(dim):
                r = min(Fitness) / np.sqrt(pow(min(Fitness),2) + pow(max(Fitness),2))
                if (r < jumping_rate):
                    NewSol[j] = swarm(i, j) + np.abs(swarm(i, j) - swarm(near_solution, j)) * (
                            np.random.rand() - 0.5) * 2
                    # manipulate range between lb and ub
                    if lb.shape[2 - 1] != 1:
                        NewSol[j] = np.amin(np.amax(NewSol[j], lb[j]), ub[j])
                    else:
                        NewSol[j] = np.amin(np.amax(NewSol[j], lb), ub)
                else:
                    # for long jumbing will take from best solution
                    NewSol[j] = swarm[i, j] + np.abs(swarm[i, j] - swarm[best_solution_Index, j]) * (
                            np.random.random() - 0.5) * 2
                    # manipulate range between lb and ub
                    if lb.shape[2 - 1] != 1:
                        NewSol[j] = np.amin(np.amax(NewSol[j], lb[j]), ub[j])
                    else:
                        NewSol[j] = np.amin(np.amax(NewSol[j], lb), ub)
            # evaluate new solution
            FitnessSol = fobj(NewSol)
            # Update the curent solution  & Age of the current solution
            if (Fitness[i] > FitnessSol):
                swarm[i, :] = NewSol
                Fitness[i] = FitnessSol

        conv[:, itr] = np.amin(Fitness)
        BestResults[itr] = np.amin(Fitness)
        # best_val = np.min(NewSol)
        cg_curve[itr, :] = conv
        itr = itr + 1
    best_val = np.min(NewSol)
    ct = ct - time.time()

    return best_val, cg_curve, BestResults, ct
