#No AI was used to generate this code, authored by Hadil Ghazal 6/9/26

"""
Majority Class Naive Baseline for comparing to my other models
keeping this very simple and minimum and goal is for other models to
do better than this one. using ~200 downdog photos and ~200 plank images.
the majority class expected to be plank
"""

# Dataset class#  counts sourced during dataset inspection
DOWNDOG_COUNT = 196
PLANK_COUNT = 197

#for capturing the majority class
if PLANK_COUNT > DOWNDOG_COUNT:
    majority_class = "Plank"
    majority_count = PLANK_COUNT
else:
    majority_class = "Downdog"
    majority_count = DOWNDOG_COUNT

total_images = DOWNDOG_COUNT + PLANK_COUNT


accuracy = majority_count / total_images # expected accuracy 

print("Majority Class Naive Baseline overview:")
print(f"Majority Class: {majority_class}")
print(f"Total Images: {total_images}")
print(f"Expected Accuracy: {accuracy:.4f}")
