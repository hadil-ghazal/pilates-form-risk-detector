#No AI was used to generate this code, authored by Hadil Ghazal 6/9/26



"""
Comparing  Model Results

Conducting a comparison of the three modeling
approaches used in the project (naive baseline vs random forest classical ML vs MobileNetV2 Deep Learning

"""

# -------------------------
# Model Results
# -------------------------

naive_accuracy = 0.5013

rf_accuracy = 0.8608
rf_precision = 0.8668
rf_recall = 0.8608
rf_f1 = 0.8603

dl_accuracy = 0.9241
dl_precision = 0.9253
dl_recall = 0.9241
dl_f1 = 0.9241



print("\nMODEL COMPARISON RESULTS:") # want to see everythign so printing 
print("-" * 50)

print(
    f"Naive Baseline Accuracy: {naive_accuracy:.4f}"
)

print(
    f"Random Forest Accuracy: {rf_accuracy:.4f}"
)
print(
    f"Random Forest Precision: {rf_precision:.4f}"
)
print(
    f"Random Forest Recall: {rf_recall:.4f}"
)
print(
    f"Random Forest F1: {rf_f1:.4f}"
)

print(
    f"MobileNetV2 Accuracy: {dl_accuracy:.4f}"
)
print(
    f"MobileNetV2 Precision: {dl_precision:.4f}"
)
print(
    f"MobileNetV2 Recall: {dl_recall:.4f}"
)
print(
    f"MobileNetV2 F1: {dl_f1:.4f}"
)


improvement_over_naive = (
    dl_accuracy - naive_accuracy
)

improvement_over_rf = (
    dl_accuracy - rf_accuracy
)

print(
    f"Deep Learning Improvement Over Naive: {improvement_over_naive:.4f}"
)

print(
    f"Deep Learning Improvement Over Random Forest: {improvement_over_rf:.4f}"
)

print()
#print("best model is: MobileNetV2 deep Learning")
## V2 decided to not hard code best model, better ML practice to have it flexible in case the best mdoel changes

results = {
    "Naive Baseline": naive_accuracy,
    "Random Forest": rf_accuracy,
    "MobileNetV2 Deep Learning": dl_accuracy
}

best_model = max(
    results,
    key=results.get
)

print(f"Best Model: {best_model}")