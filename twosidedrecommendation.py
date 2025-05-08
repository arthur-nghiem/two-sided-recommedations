import numpy as np
from sklearn.neighbors import NearestNeighbors
import math
import random

# defining global variables
# number of times to run the simulation
N = 100
# performance metric (Survival time, Engagment sum, Turnover rate, Final engagement)
metric = "Engagement sum"
# algorithm to use (user centric, local clustering, creator ranking, filtered greedy)
method = "filtered greedy"
# starting number of users (default 200)
U_init_list = [*range(100, 301, 10)]
# starting number of creators (default 20)
C_init_list = [*range(15, 26, 1)]
# dimensionality constant (default 5)
D = 5
# user capacity (default 6)
K = 6
# number of repetitions in the simulation (default 20)
T = 20
# user's threshold for quality of recommendations (default 1.5)
x_user = 1.5
# steepness of user's quality vs retention curve (default 6.5)
k_user = 6.5
# creator's threshold for quantity of engagement (default 60)
x_creator = 60
# steepness of the creator's engagement vs retention curve (default 0.2)
k_creator = 0.2
# presence of discontinuous conditions
discontinuous = False
# detail setting for print statements
detailed = False

# model retention probabilities as logistic functions
def get_user_retention(quality, discontinuous):
    if discontinuous == False:
        return 1 / (1 + np.exp(-k_user * (quality - x_user)))
    if discontinuous == True:
        n = quality.shape
        threshold = np.full((n), x_user)
        judgements = np.greater_equal(quality, threshold)
        return judgements.astype(int)


def get_creator_retention(quantity, discontinuous):
    if discontinuous == False:
        return 1 / (1 + np.exp(-k_creator * (quantity - x_creator)))
    if discontinuous == True:
        n = quantity.shape
        threshold = np.full((n), x_user)
        judgements = np.greater_equal(quantity, threshold)
        return judgements.astype(int)

for U_init in U_init_list:
    for C_init in C_init_list:
        performance_log = np.zeros(N)
        for n in range(N):
            # instantiate users and creators
            U = U_init
            C = C_init
            user_types = np.random.normal(0, 1, size=(U, D))
            user_norms = np.linalg.norm(user_types, axis=1)
            user_types = user_types / user_norms[:,None]
            creator_types = np.random.normal(0, 1, size=(C, D))
            creator_norms = np.linalg.norm(creator_types, axis=1)
            creator_types = creator_types / creator_norms[:,None]
            engagement_log = np.zeros(T)
            if metric == "Turnover rate":
                replacement_log = np.zeros(T)

            for t in range(T):
                # break loop if platform is no longer viable
                if C < K:
                    break
                if U == 0:
                    break

                # local clustering algorithm
                if method == "local clustering":
                    nbrs = NearestNeighbors(n_neighbors=K, algorithm='ball_tree').fit(creator_types)
                    distances, indices = nbrs.kneighbors(user_types)
                    d = 2 * np.sin(np.arccos(x_user/D)/2)
                    user_order = np.arange(U)
                    np.random.shuffle(user_order)
                    R = np.full((U, K), -1)
                    for i in user_order:
                        if R[i, 0] == -1:
                            user_neighbors = []
                            creator_neighbors = []
                            for u in range(U):
                                distance = np.linalg.norm(user_types[u, :] - user_types[i, :])
                                if distance < d:
                                    user_neighbors.append(u)
                            for c in range(C):
                                distance = np.linalg.norm(creator_types[c, :] - user_types[i, :])
                                if distance < d:
                                    user_neighbors.append(c)
                            if len(user_neighbors) >= x_creator and len(creator_neighbors) >= K:
                                random.shuffle(creator_neighbors)
                                c_prime = creator_neighbors[:K]
                                for u in user_neighbors:
                                    for k in range(K):
                                        R[u][k] = c_prime[k]
                    for u in range(U):
                        if R[u][0] == -1:
                            R[u, :] = indices[u, :]
                    indices = R

                # creator ranking with potential audience size 1 algorithm
                elif method == "creator ranking":
                    d = 2 * np.sin(np.arccos(x_user/D)/2)
                    user_order = np.arange(U)
                    np.random.shuffle(user_order)
                    creator_list = list(range(0, C))
                    R = np.full((U, K), -1)
                    while True:
                        compatibility = np.zeros((C, U))
                        for c in range(C):
                            for u in range(U):
                                if np.linalg.norm(creator_types[c, :] - user_types[u, :]) < d and R[u, K-1] == -1:
                                    compatibility[c, u] = 1
                        audience_sizes = np.sum(compatibility, axis = 1)
                        if max(audience_sizes) < x_creator:
                            break
                        try:
                            j = min(i for i in audience_sizes[creator_list] if i >= x_creator)
                        except:
                            break
                        creator_smallest = -1
                        for i in creator_list[::-1]:
                            if audience_sizes[i] == j:
                                creator_smallest = i
                        potential_audience = np.nonzero(compatibility[creator_smallest, :] == 1)[0]
                        for u in potential_audience:
                            recommendations = R[u, :]
                            if recommendations[K-1] == -1:
                                slot = np.nonzero(recommendations == -1)[0][0]
                                R[u, slot] = creator_smallest
                        creator_list.remove(creator_smallest)
                        for c in range(C):
                            if audience_sizes[c] < x_creator and c in creator_list:
                                creator_list.remove(c)
                        if len(creator_list) == 0:
                            break
                    nbrs = NearestNeighbors(n_neighbors=K, algorithm='ball_tree').fit(creator_types)
                    distances, indices = nbrs.kneighbors(user_types)
                    for u in range(U):
                        greedy_solution = indices[u, :].tolist()
                        if R[u, K-1] == -1:
                            slot = np.nonzero(R[u, :] == -1)[0][0]
                            recommendations = R[u, :].tolist()
                            for i in recommendations:
                                if i in greedy_solution:
                                    greedy_solution.remove(i)
                            while slot < K:
                                R[u, slot] = greedy_solution[0]
                                greedy_solution.pop(0)
                                slot += 1
                    indices = R

                elif method == "filtered greedy":
                    nbrs = NearestNeighbors(n_neighbors=K, algorithm='ball_tree').fit(creator_types)
                    distances, indices = nbrs.kneighbors(user_types)
                    creator_engagement = np.zeros(C)
                    for c in range(C):
                        creator_engagement[c] = np.count_nonzero(indices == c)
                    creator_retention = get_creator_retention(creator_engagement, discontinuous)
                    filter = np.nonzero(creator_retention > 0.1)[0]
                    filtered_creators = creator_types[filter,:]
                    if filter.size >= K:
                        nbrs = NearestNeighbors(n_neighbors=K, algorithm='ball_tree').fit(filtered_creators)
                        distances, indices = nbrs.kneighbors(user_types)
                        for u in range(U):
                            for k in range(K):
                                indices[u][k] = filter[indices[u][k]]

                # greedy (user-centric) algorithm to generate recommendations
                else:
                    nbrs = NearestNeighbors(n_neighbors=K, algorithm='ball_tree').fit(creator_types)
                    distances, indices = nbrs.kneighbors(user_types)

                # compute creator engagement and retention probabilities
                creator_engagement = np.zeros(C)
                for c in range(C):
                    creator_engagement[c] = np.count_nonzero(indices == c)
                creator_retention = get_creator_retention(creator_engagement, discontinuous)

                # compute user engagement and retention probabilities
                user_engagement = np.zeros(U)
                for u in range(U):
                    for k in range(K):
                        user_engagement[u] += user_types[u, :] @ creator_types[indices[u, k], :].T
                user_retention = get_user_retention(user_engagement, discontinuous)

                # log user engagement at this time
                engagement_log[t] = np.sum(user_retention)

                # determine which agents stay on the platform
                user_status = np.repeat(True, U)
                user_random = np.random.uniform(size = U)
                for u in range(U):
                    if user_random[u] > user_retention[u]:
                        user_status[u] = False
                creator_status = np.repeat(True, C)
                creator_random = np.random.uniform(size = C)
                for c in range(C):
                    if creator_random[c] > creator_retention[c]:
                        creator_status[c] = False

                # update creators and users
                if metric != "Turnover rate":
                    user_rows = np.argwhere(user_status == True)
                    user_rows = user_rows.flatten()
                    user_types = user_types[user_rows, :]
                    creator_rows = np.argwhere(creator_status == True)
                    creator_rows = creator_rows.flatten()
                    creator_types = creator_types[creator_rows, :]
                    U, temp = user_types.shape
                    C, temp = creator_types.shape
                else:
                    replacement_log[t] = (C_init - np.argwhere(creator_status == True).size) / C_init

                # reintroduce creators and users with random types
                # if metric == "Turnover rate":
                #     U_add = U_init - U
                #     user_types_add = np.random.normal(0, 1, size=(U_add, D))
                #     user_types = np.vstack((user_types, user_types_add))
                #     U = U_init
                #     C_add = C_init - C
                #     creator_types_add = np.random.normal(0, 1, size=(C_add, D))
                #     creator_types = np.vstack((creator_types, creator_types_add))
                #     C = C_init
                #     replacement_log[t] = C_add / C_init

            # add to the performance log
            # survival time
            if metric == "Survival time":
                performance_log[n] = np.count_nonzero(engagement_log)
            # sum of engagement
            elif metric == "Engagement sum":
                performance_log[n] = np.sum(engagement_log)
            elif metric == "Turnover rate":
                performance_log[n] = np.average(replacement_log)
            elif metric == "Final engagement":
                performance_log[n] = engagement_log[-1]
            else:
                performance_log[n] = -1

            # print statement per simulation
            # indicate every x completions
            if detailed == True:
                x = N / 10
                if (n+1) % x == 0:
                    print("Simulation", n+1, "complete.")
            # include detail
            # if metric != "Turnover rate":
            #     print("Engagement log:")
            #     print(engagement_log)
            # else:
            #     print("Replacement log:")
            #     print(replacement_log)
            # print("Performance metric:", performance_log[n])
            # print("")

        # final report
        # result of eaach simulation
        # print("Performance log:")
        # print(performance_log)
        # print("")
        # overall result
        if detailed == True:
            print("")
            print(metric, "over", n+1, "simulations using", method, "algorithm:", np.average(performance_log))
            print("")
            print("Initial number of users:", U_init)
            print("Initial number of creators:", C_init)
            print("Dimensionality constant:", D)
            print("User capacity:", K)
            print("Repetitions per simulation:", T)
            print("User's threshold for quality of recommendations:", x_user)
            if discontinuous == False:
                print("Steepness of user's quality vs retention curve:", k_user)
            print("Creator's threshold for quantity of engagement:", x_creator)
            if discontinuous == False:
                print("Steepness of creator's engagement vs retention curve:", k_creator)
            print("Discontinuous case:", discontinuous)
        else:
            print(U_init, C_init, np.average(performance_log))
