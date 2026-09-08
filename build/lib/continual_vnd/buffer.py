import torch
import numpy as np

class TrajectoryBuffer:
    def __init__(self, state_dim, action_dim, capacity=10000, context_len=10):
        self.capacity = capacity
        self.context_len = context_len
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)

        self.ptr = 0
        self.size = 0

    def add(self, state, action, next_state, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device="cpu"):
        """
        Samples transitions with their history context.
        Returns:
            context_states: (B, C, state_dim)
            context_actions: (B, C, action_dim)
            states: (B, state_dim)
            actions: (B, action_dim)
            next_states: (B, state_dim)
        """
        valid_indices = []
        attempts = 0
        max_attempts = batch_size * 10

        while len(valid_indices) < batch_size and attempts < max_attempts:
            attempts += 1
            idx = np.random.randint(0, self.size)

            # We want to form a context from idx - context_len to idx - 1.
            # However, if it crosses the buffer pointer or 0 boundary, it's invalid unless we carefully piece it together.
            # Also, we pad with zeros if the start of the episode is within the context window.
            # To simplify, we just track backwards.

            # Check if this index was overwritten recently
            # If ptr is within the window we are trying to read, it's invalid (wrap-around collision)
            # A safer way to check wrap-around:
            # The indices we need are: (idx - context_len) % capacity to (idx - 1) % capacity
            # If self.ptr is in this range, we reject.

            invalid = False
            for step in range(1, self.context_len + 1):
                check_idx = (idx - step) % self.capacity
                if check_idx == self.ptr and self.size < self.capacity:
                    # We haven't filled this far
                    invalid = True
                    break
                if check_idx == self.ptr:
                    # Overwritten
                    invalid = True
                    break

            if invalid:
                continue

            valid_indices.append(idx)

        # If we couldn't find enough after max_attempts, we fallback to just whatever valid we found or duplicate.
        # This prevents infinite loops.
        if len(valid_indices) == 0:
            # Very rare, just return something safe, e.g. the very last added element with padding
            valid_indices = [(self.ptr - 1) % self.size] * batch_size
        else:
            # Pad to batch_size if needed
            while len(valid_indices) < batch_size:
                valid_indices.append(valid_indices[0])

        valid_indices = np.array(valid_indices)

        context_states = np.zeros((batch_size, self.context_len, self.state_dim), dtype=np.float32)
        context_actions = np.zeros((batch_size, self.context_len, self.action_dim), dtype=np.float32)

        for i, idx in enumerate(valid_indices):
            # Extract history with padding for episode starts
            hist_states = []
            hist_actions = []

            curr_idx = idx
            # We go backwards
            for step in range(1, self.context_len + 1):
                prev_idx = (curr_idx - 1) % self.capacity

                # If the previous state was a terminal state (done), then the current step is the start of an episode.
                # So we pad the rest of the history with zeros.
                if self.dones[prev_idx]:
                    break

                hist_states.append(self.states[prev_idx])
                hist_actions.append(self.actions[prev_idx])
                curr_idx = prev_idx

            # Reverse to be chronological
            hist_states = hist_states[::-1]
            hist_actions = hist_actions[::-1]

            # Pad
            while len(hist_states) < self.context_len:
                hist_states.insert(0, np.zeros(self.state_dim, dtype=np.float32))
                hist_actions.insert(0, np.zeros(self.action_dim, dtype=np.float32))

            context_states[i] = np.array(hist_states)
            context_actions[i] = np.array(hist_actions)

        states = self.states[valid_indices]
        actions = self.actions[valid_indices]
        next_states = self.next_states[valid_indices]

        return (
            torch.FloatTensor(context_states).to(device),
            torch.FloatTensor(context_actions).to(device),
            torch.FloatTensor(states).to(device),
            torch.FloatTensor(actions).to(device),
            torch.FloatTensor(next_states).to(device)
        )

    def sample_initial_states(self, batch_size, device="cpu"):
        """
        Samples random states to start rollouts.
        """
        indices = np.random.randint(0, self.size, size=batch_size)
        return torch.FloatTensor(self.states[indices]).to(device)
