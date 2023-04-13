import torch 
import numpy as np 

class FewShotSampler():
    def __init__(self, label, n_batch, n_way, n_support, n_query, n_episodes):
        """
        Args:
        dataset: Dataset object
        n_batch: number of batches each time 
        n_support: number of support examples per class
        n_query: number of query examples per class
        n_way: number of classes in a few-shot classification task
        n_episodes: number of episode each batch
        """
        self.label = label 
        self.n_batch = n_batch  
        self.n_support = n_support
        self.n_query = n_query
        self.n_way = n_way
        self.n_episodes = n_episodes

        label = np.array(label)
        self.class_index = []
        # record index of different classes 
        for c in range(max(label) + 1):
            self.class_index.append(np.argwhere(label == c).reshape(-1))
        
    def __len__(self):
        return self.n_batch

    def __iter__(self):
        """
        get way * shot * episode images everytime 
        """
        # batch 的存在意义？
        for i_batch in range(self.n_batch):
            batch = []
            for i_ep in range(self.n_episodes):
                episode = []
                # random choose n_classes
                classes = np.random.choice(len(self.class_index), self.n_way, replace=False)
                for c in classes:
                    # random choose labels from the chosed classes 
                    l = np.random.choice(self.class_index[c], self.n_support + self.n_query, replace=False)
                    episode.append(torch.from_numpy(l))
                # list stack to tensor e.g. a list have three 1*2 tensor -> a 3*2 tensor
                episode = torch.stack(episode)
                batch.append(episode)
            batch = torch.stack(batch)
            yield batch.view(-1)