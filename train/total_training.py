from phase1 import train_phase_1
from phase2 import train_phase_2
from utils import get_full_dataset
# This file will train the three phases of the model


dataset = get_full_dataset()
print(dataset.shape)
# Phase 1
print("Trainign Phase 1")
train_phase_1(dataset)

# Phase 2
print("Training Phase 2 - A")
train_phase_2("A", dataset)

print("Training Phase 2 - B")
train_phase_2("B", dataset)

print("Training Phase 2 - C")
train_phase_2("C", dataset)

