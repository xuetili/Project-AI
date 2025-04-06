import matplotlib.pyplot as plt

# Distribution of classes in the data set
def get_distribution_barplot(data, set):
    class_labels = data.class_indices
    labels = data.labels
    class_counts = {label: sum(labels == i) for i, label in enumerate(class_labels)}
    plt.bar(class_counts.keys(), class_counts.values())
    plt.xlabel('Class')
    plt.ylabel('Number of samples')
    plt.title('Class Distribution in {set} Set'.format(set = set))
    plt.show()

# plot 5 images with label
def plotImages(images_arr, labels):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for i in range(len(axes)):
        axes[i].imshow(images_arr[i])
        axes[i].set_title("COVID" if labels[i] == 1 else "Normal")
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()
