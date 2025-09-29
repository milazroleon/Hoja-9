
import numpy as np
from policy import Policy
from lake_mdp import LakeMDP, ABSORB
from plot_utils import plot_policy
import matplotlib.pyplot as plt


class TabularPolicy(Policy):
    def __init__(self, mdp, rng, table=None):
        """
        A tabular deterministic policy represented as a dictionary mapping states to actions.

        Parameters
        ----------
        mdp : MDP
            The Markov Decision Process instance.
        rng : np.random.Generator
            Random number generator used for initialization.
        table : dict, optional
            A mapping from state to action to initialize the policy. If None, random actions are chosen.
        """

        super().__init__(mdp, rng)
        all_states = [
            ((i, j), cell)
            for i, row in enumerate(mdp.grid)
            for j, cell in enumerate(row)
        ]
        all_states.append((ABSORB, ABSORB))

        self.table = dict(table) if table is not None else {}
        for s in all_states:
            if s not in self.table:
                self.table[s] = rng.choice(list(mdp.actions(s)))

    def _decision(self, s):
        """
        Returns the action taken by the policy in state s.

        Parameters
        ----------
        s : State
            The current state.

        Returns
        -------
        Action
            The action chosen in state s.
        """

        return self.table[s]


def q_from_v(mdp, v, gamma):
    """
    Compute the Q-values from state-value estimates.

    Parameters
    ----------
    mdp : MDP
        The MDP for which Q-values are computed.
    v : dict
        State-value function mapping states to values.
    gamma : float
        Discount factor.

    Returns
    -------
    dict
        A nested dictionary q[s][a] representing the Q-values.
    """
    q={}

    for s in v.keys():
        q[s]={}
        for a in mdp.actions(s):
            q_sa=0
            for (s2,p) in mdp.transition(s,a):
                r=mdp.reward(s2)
                q_sa+=p*(r+gamma*v[s2])
            q[s][a]=q_sa
    return q


def make_greedy_policy(mdp, v, gamma):
    """
    Generate a greedy policy based on state-value estimates.

    Parameters
    ----------
    mdp : MDP
        The MDP on which the policy is based.
    v : dict
        State-value function mapping states to values.
    gamma : float
        Discount factor.

    Returns
    -------
    dict
        A mapping from states to greedy actions.
    """
    q=q_from_v(mdp,v,gamma)
    policy={}
    for s in v.keys():
        best_a=None
        best_q=-float('inf')
        for a in mdp.actions(s):
            if q[s][a]>best_q:
                best_q=q[s][a]
                best_a=a
        policy[s]=best_a
    return policy


def policy_mismatch(q_true, q_est):
    """
    Identify states where greedy actions under two Q-functions differ.

    Parameters
    ----------
    q_true : dict
        The true Q-value function.
    q_est : dict
        The estimated Q-value function.

    Returns
    -------
    list
        States where the greedy actions differ.
    """
    mismatched_states = []

    for s in q_true.keys():
        best_a_true = None
        best_q_true = -float('inf')
        for a in q_true[s].keys():
            if q_true[s][a] > best_q_true:
                best_q_true = q_true[s][a]
                best_a_true = a

        best_a_est = None
        best_q_est = -float('inf')
        for a in q_est[s].keys():
            if q_est[s][a] > best_q_est:
                best_q_est = q_est[s][a]
                best_a_est = a

        if best_a_true != best_a_est:
            mismatched_states.append(s)

    return mismatched_states


class ValueIteration:
    def __init__(self, gamma=0.99, epsilon=1e-3):
        """
        Initialize the Value Iteration algorithm.

        Parameters
        ----------
        gamma : float, optional
            Discount factor.
        epsilon : float, optional
            Convergence threshold (scaled by (1-gamma)/gamma).
        """
        self.gamma = gamma
        self.epsilon = epsilon

    def run(self, mdp):
        """
        Perform value iteration on the given MDP.

        Parameters
        ----------
        mdp : MDP
            The Markov Decision Process.

        Returns
        -------
        TabularPolicy
            The optimal policy derived from the converged value function.
        """
        v = {}
        all_states = [
            ((i, j), cell)
            for i, row in enumerate(mdp.grid)
            for j, cell in enumerate(row)
        ]
        all_states.append((ABSORB, ABSORB))
        for s in all_states:
            v[s] = 0.0

        while True:
            delta = 0.0
            v_new = v.copy()

            for s in all_states:
                if mdp.is_terminal(s):
                    continue

                max_q = -float('inf')
                for a in mdp.actions(s):
                    q_sa = 0.0
                    for (s2, p) in mdp.transition(s, a):
                        r = mdp.reward(s2)
                        q_sa += p * (r + self.gamma * v[s2])
                    if q_sa > max_q:
                        max_q = q_sa

                v_new[s] = max_q
                delta = max(delta, abs(v_new[s] - v[s]))

            v = v_new

            if delta < self.epsilon * (1 - self.gamma) / self.gamma:
                break

        policy_evaluation = {}
        rng=np.random.default_rng(0)

        for s in all_states:
            if mdp.is_terminal(s):
                continue

            best_a = None
            best_q = -float('inf')
            for a in mdp.actions(s):
                q_sa = 0.0
                for (s2, p) in mdp.transition(s, a):
                    r = mdp.reward(s2)
                    q_sa += p * (r + self.gamma * v[s2])
                if q_sa > best_q:
                    best_q = q_sa
                    best_a = a

            policy_evaluation[s] = best_a
        return TabularPolicy(mdp, rng, table=policy_evaluation)


class PolicyEvaluationFactory:
    def __init__(
        self,
        mdp,
        gamma,
        policy,
        async_mode=False,
        subset=None,
        initial_values=None,
    ):
        """
        Factory for performing policy evaluation using either synchronous or asynchronous updates.

        Parameters
        ----------
        mdp : MDP
            The MDP being evaluated.
        gamma : float
            Discount factor.
        policy : Policy
            The policy to evaluate.
        async_mode : bool, optional
            If True, use asynchronous updates. Otherwise, use synchronous updates.
        subset : list, optional
            Subset of states to update (used in async mode).
        initial_values : dict, optional
            Initial value estimates.
        """
        self.mdp = mdp
        self.gamma = gamma
        self.policy = policy
        self.v = {}
        self.async_mode = async_mode
        self.subset = subset

        all_states = [
            ((i, j), cell)
            for i, row in enumerate(mdp.grid)
            for j, cell in enumerate(row)
        ]
        all_states.append((ABSORB, ABSORB))
        
        if initial_values is None:
            self.v = {s: 0.0 for s in all_states}
        else:
            self.v = dict(initial_values)

    def synchronous_update(self):
        """
        Perform a synchronous update of the value function.

        Updates all states simultaneously using the Bellman expectation equation.
        """
        new_v = self.v.copy()
        for s in self.v.keys():
            if self.mdp.is_terminal(s):
                continue
            a = self.policy(s)
            val = 0.0
            for s2, p in self.mdp.transition(s, a):
                r = self.mdp.reward(s2)
                val += p * (r + self.gamma * self.v[s2])
            new_v[s] = val
        self.v = new_v


    def asynchronous_update(self):
        """
        Perform an asynchronous update of the value function.

        Only updates the specified subset of states using the Bellman expectation equation.
        """
        states_to_update = self.subset if self.subset is not None else self.v.keys()
        for s in states_to_update:
            if self.mdp.is_terminal(s):
                continue
            a = self.policy(s)
            val = 0.0
            for s2, p in self.mdp.transition(s, a):
                r = self.mdp.reward(s2)
                val += p * (r + self.gamma * self.v[s2])
            self.v[s] = val

    def step(self):
        """
        Perform one step of policy evaluation.

        Uses synchronous or asynchronous update depending on initialization.
        """
        if self.async_mode:
            self.asynchronous_update()
        else:
            self.synchronous_update()


class GeneralPolicyIteration:
    def __init__(
        self, mdp, gamma=0.99, steps_per_eval=5, async_mode=False, subset=None
    ):
        """
        Implements General Policy Iteration (GPI) combining policy evaluation and improvement.

        Parameters
        ----------
        mdp : MDP
            The Markov Decision Process.
        gamma : float, optional
            Discount factor.
        steps_per_eval : int, optional
            Number of evaluation steps per improvement iteration.
        async_mode : bool, optional
            Whether to use asynchronous evaluation.
        subset : list, optional
            Subset of states to evaluate in async mode.
        """
        self.mdp = mdp
        self.gamma = gamma
        self.steps_per_eval = steps_per_eval
        self.async_mode = async_mode
        self.subset = subset

    def run(self, init_policy=None):
        """
        Run the GPI algorithm starting from an initial policy.

        Returns
        -------
        TabularPolicy
            The final improved policy after convergence.
        """
        rng = np.random.default_rng(0)
        policy = init_policy if init_policy is not None else TabularPolicy(self.mdp, rng)

        while True:

            factory = PolicyEvaluationFactory(
                self.mdp,
                self.gamma,
                policy,
                async_mode=self.async_mode,
                subset=self.subset,
            )
            for _ in range(self.steps_per_eval):
                factory.step()

            new_policy = make_greedy_policy(self.mdp, factory.v, self.gamma)

            if new_policy.table == policy.table:
                break
            policy = new_policy

        return policy


# Example usage and testing
if __name__ == "__main__":
    # Create Lake MDP
    grid = [
        ["S", "F", "F", "F"],
        ["F", "H", "F", "H"],
        ["F", "F", "F", "H"],
        ["H", "F", "F", "G"],
    ]
    mdp = LakeMDP(grid)
    # General Policy Iteration
    gpi = GeneralPolicyIteration(mdp, gamma=0.9, steps_per_eval=10, async_mode=False)
    gpi_policy = gpi.run()
    plot_policy(gpi_policy)
    # Value Iteration
    vi = ValueIteration(gamma=0.9)
    vi_policy = vi.run(mdp)
    plot_policy(vi_policy)
    # Show plots
    plt.show()
