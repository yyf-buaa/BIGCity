class RoundRobinIterator:
    def __init__(self, dataloaders):
        self.dataloaders = list(dataloaders.items())
        self.iterators = {name: iter(dl) for name, dl in dataloaders.items()}
        self.remaining = set(dataloaders.keys())
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if len(self.remaining) == 0:
            raise StopIteration
        
        dataset_name, _ = self.dataloaders[0]
        
        try:
            batch = next(self.iterators[dataset_name])
        except StopIteration:
            self.remaining.remove(dataset_name)
            self.dataloaders.pop(0)
            return self.__next__()

        self.dataloaders.append(self.dataloaders.pop(0))

        return dataset_name, batch
    
    def __len__(self):
        return sum(len(dl) for dl in self.iterators.values())