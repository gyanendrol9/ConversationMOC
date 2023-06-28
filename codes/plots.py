import matplotlib.pyplot as plt
plt.style.use("ggplot")
import numpy as np
import pandas as pd
from result_analysis import * 

def plot_author_stat(auth_idx,post_count,filepath):
    # Plotting the bar chart
    plt.bar(auth_idx,post_count)

    # Adding labels and title
    plt.xlabel('Users', fontsize=10)
    plt.ylabel('# of posts', fontsize=10)

    plt.xticks(rotation=90, ha='right', fontsize=5)
    plt.yticks(fontsize=5)  # Set font size of y-axis labels to 10

    # Save the histogram
    plt.savefig(filepath)

    # Display the plot
    return plt

def plot_gen(auth_idx,post_count,x_labels,y_labels,save_fig=False,axis_fontsize = 5, label_fontsize = 10, rotate = 'right', degree = 90, filepath='temp.pdf'):
    # Plotting the bar chart
    plt.bar(auth_idx,post_count)

    # Adding labels and title
    plt.xlabel(x_labels, fontsize=label_fontsize)
    plt.ylabel(y_labels, fontsize=label_fontsize)

    plt.xticks(rotation=degree, ha=rotate, fontsize=axis_fontsize)
    plt.yticks(fontsize=axis_fontsize)  # Set font size of y-axis labels to 10

    # Save the histogram
    if save_fig:
        plt.savefig(filepath)

    # Display the plot
    return plt

def plot_graph(G,iteration=20):
    pos = nx.spring_layout(G, k=0.3*1/np.sqrt(len(G.nodes())), iterations=iteration)
    plt.figure(3, figsize=(30, 30))
    nx.draw(G, pos=pos)
    # nx.draw_networkx_labels(G, pos=pos)
    plt.show()


def plot_subreddit_stat(auth_idx,post_count,filepath,y_labels):
    # Plotting the bar chart
    plt.bar(auth_idx,post_count)

    # Adding labels and title
    plt.xlabel('Subreddit')
    plt.ylabel(y_labels)

    plt.xticks(rotation=45, ha='right', fontsize=5)
    plt.yticks(fontsize=5)  # Set font size of y-axis labels to 10

    # Save the histogram
    plt.savefig(filepath)

    # Display the plot
    return plt

# Plot the training loss/accuracy
def plot_training_model_singletask(history, filepath='temp.pdf', savefig=False):
    fig, ax = plt.subplots()
    num_epochs = len(history.history['loss'])
    epochs = np.arange(1, num_epochs + 1)
    
    ax.plot(epochs, history.history['loss'], label="Training Loss")
    ax.plot(epochs, history.history['accuracy'], label="Training Accuracy")

    # plt.plot(history.history[plot_value.lower()])
    ax.set_title("Training Loss and Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss/Accuracy")
    ax.legend()

    if not savefig:
        plt.show()
    else:
        plt.savefig(filepath)

def plot_training_model_multitask(history, filepath='temp.pdf', savefig=False):
    fig, ax = plt.subplots()
    num_epochs = len(history.history['dense_1_loss'])
    epochs = np.arange(1, num_epochs + 1)
    
    ax.plot(epochs, history.history['dense_1_loss'], label="Training Loss Task 1")
    ax.plot(epochs, history.history['dense_2_loss'], label="Training Loss Task 2")
    ax.plot(epochs, history.history['dense_1_accuracy'], label="Training Accuracy Task 1")
    ax.plot(epochs, history.history['dense_2_accuracy'], label="Training Accuracy Task 2")

    # plt.plot(history.history[plot_value.lower()])
    ax.set_title("Training Loss and Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss/Accuracy")
    ax.legend()
    if not savefig:
        plt.show()
    else:
        plt.savefig(filepath)

def boxplot_folds(data,labels, score_key_label):
    fig, ax = plt.subplots()

    # Create the box plot
    bp = ax.boxplot(data)

    # Set the axis labels and title
    ax.set_xticklabels(labels)
    ax.set_ylabel('Score')
    ax.set_title(f'10-Fold Cross-Validation Results ({score_key_label})')

    # Display the plot
    return plt

def boxplot_folds_all(results,savepath='temp.pdf', savefig=False):
    score_keys = results.keys()
    columns = ['P', 'R', 'F']
    labels = ['Accuracy','Precision', 'Recall', 'F-score']

    tot_plots = len(results)-1

    # Create the figure and define the size
    fig, axs = plt.subplots(tot_plots, 1, figsize=(6, 18))
    # gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[2, 1])

    all_ax = {}
    i = 1
    for score_key in results:
        if score_key != 'accuracy':
            df = pd.DataFrame(results[score_key], columns=columns)
            data = [results['accuracy'], df['P'], df['R'], df['F']]

            all_ax[i] = fig.add_subplot(tot_plots, 1, i)
            all_ax[i].boxplot(data)
            # Set the axis labels and title
            all_ax[i].set_xticklabels(labels)
            all_ax[i].set_ylabel('Score')
            all_ax[i].set_title(f'{score_key}')

            i+=1

    plt.tight_layout()  # Adjust spacing between subplots
    if savefig:
        plt.savefig(savepath)
    else:
        plt.show()  # Display the combined plot
    


def boxplot_folds_all_task2(results2,savepath='temp.pdf', savefig=False):
    score_keys = results2.keys()
    columns = ['P', 'R', 'F']
    labels = ['Accuracy','Precision', 'Recall', 'F-score']

    tot_plots = len(results2)-1
    grids = int(tot_plots/5)+1
    # Create the figure and define the size
    fig, axs = plt.subplots(5,grids, figsize=(18, 18))
    # gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[2, 1])

    i = 0
    count= 0
    for score_key in results2:
        if score_key != 'accuracy':
            df = pd.DataFrame(results2[score_key], columns=columns)
            data = [results2['accuracy'], df['P'], df['R'], df['F']]

            if i>4:
                i = 0
                count+=1

            print(i,count)
    #         axs[i,j] = fig.add_subplot(tot_plots, j+1, i)
            axs[i,count].boxplot(data)
            # Set the axis labels and title
            axs[i,count].set_xticklabels(labels)
            axs[i,count].set_ylabel('Score')
            axs[i,count].set_title(f'{score_key}')

            i+=1

#     plt.tight_layout()  # Adjust spacing between subplots
    plt.subplots_adjust(hspace=0.5)  # Adjust vertical spacing between subplots
    if savefig:
        plt.savefig(savepath)
    else:
        plt.show()  # Display the combined plot


def boxplot_folds_models(model_pds, xticks, classes, metric='F-score', savepath='temp.pdf', savefig=False):
    tot_plots = len(classes)
    # Create the figure and define the size
    fig, axs = plt.subplots(tot_plots, 1, figsize=(6, 18))

    i = 0
    for class_to_find in classes:
        model_summary_df = get_class_specific_folds_all_models_metric(class_to_find, metric, model_pds)

        data = []
        for model in model_summary_df:    
            data.append(list(model_summary_df[model][metric]))

        axs[i].boxplot(data)
        axs[i].set_xlabel('Models')
        axs[i].set_ylabel(metric)
        axs[i].set_title(f'{class_to_find}')
        axs[i].set_xticklabels(xticks)
        axs[i].set_ylim(0, 1)  # Set the desired range of values
        i+=1

    plt.tight_layout()  # Adjust spacing between subplots
    if savefig:
        plt.savefig(savepath)
    else:
        plt.show()  # Display the combined plot