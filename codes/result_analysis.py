import pandas as pd
import statistics

import matplotlib.pyplot as plt
plt.style.use("ggplot")
import gc

def get_combined_results(results_models):
    columns = ['P', 'R', 'F']
    cResults = []
    for model in results_models:
        result1, result2 = results_models[model]
        for score_key in result1:
            if score_key != 'accuracy':
                df = pd.DataFrame(result1[score_key], columns=columns)
            res =  [model, score_key, max(result1['accuracy']), max(df['P']), max(df['R']), max(df['F']), statistics.mean(result1['accuracy']), statistics.mean(df['P']), statistics.mean(df['R']), statistics.mean(df['F']), statistics.stdev(result1['accuracy']), statistics.stdev(df['P']), statistics.stdev(df['R']), statistics.stdev(df['F'])]
            cResults.append(res)    

    labels = ['Model', 'Class', 'Mx-Accuracy', 'Mx-Precision',  'Mx-Recall',  'Mx-F-score', 'Mn-Accuracy', 'Mn-Precision',  'Mn-Recall',  'Mn-F-score', 'Std-Accuracy', 'Std-Precision',  'Std-Recall',  'Std-F-score']
    df = pd.DataFrame(cResults, columns=labels)

    result_df = pd.DataFrame()
    gdf = df.groupby('Class')
    for name, group in gdf:
        result_df = pd.concat([result_df, group])
    return result_df


def summarize(value_to_find, plot_col, result_df, order, metric, savefig=False, savepath='Temp.pdf'):
    summary_df = result_df[result_df['Class'] == value_to_find]

    # Create a Categorical data type with the desired order
    cat_dtype = pd.CategoricalDtype(categories=order, ordered=True)

    # Apply the Categorical data type to the 'Name' column
    summary_df['Model'] = summary_df['Model'].astype(cat_dtype)

    sorted_df = summary_df.sort_values('Model')
    sorted_df.index = range(1, len(sorted_df) + 1)

    # Plot a bar chart for the 'Age' column
    sorted_df[plot_col].plot.bar()

    # Set the x-axis label
    plt.xlabel('Models')

    # Set the y-axis label
    plt.ylabel(f'{metric}')

    # Set the title of the plot
    plt.title(f'{value_to_find} ({plot_col})')
    
    plt.xticks(range(1, len(order) + 1))
    if savefig:
        plt.savefig(f'{savepath}-{value_to_find}.pdf')
    else:
        # Display the plot
        plt.show()


def get_df(result1, score_key, columns, model):
    df = pd.DataFrame(result1[score_key], columns=columns)
    tdf = pd.concat([pd.Series([model]*len(df)), pd.Series([score_key]*len(df)), pd.Series(result1['accuracy']), pd.Series(df['P']), pd.Series(df['R']), pd.Series(df['F'])], axis=1, ignore_index=True)
    return tdf
                
def get_all_results(results_models):
    columns = ['P', 'R', 'F']
    labels = ['Model', 'Class', 'Accuracy', 'Precision',  'Recall',  'F-score']
    dc = 0
    for model in results_models:
        results1,results2 = results_models[model]
        for score_key in results1:
            if score_key != 'accuracy':
                tdf = get_df(results1, score_key, columns, model)
                tdf.columns = labels
                
                if dc == 0:
                    cdf = tdf
                    dc+=1
                else:
                    cdf =  pd.concat([cdf,tdf],axis=0, ignore_index=True)
                    dc+=1
                del tdf
        del results1, results2
        gc.collect()
                
    return cdf

def get_class_specific_folds_all_models_metric(class_to_find, metric, model_pds):
    model_summary = {}
    for model in model_pds:
        filtered_df = model_pds[model]
        model_summary[model] = filtered_df[filtered_df['Class'] == class_to_find]
    return model_summary


def get_class_specific_folds_specific_network_models_metric(model_pds, models, model_compare_keys, network):
    model_summary = {}
    for numnet in models[network]:
        if numnet not in model_summary:
            model_summary[numnet] = []
        for model in models[network][numnet]:
            filtered_df = model_pds[model]
            model_summary[numnet].append(filtered_df)

    model_summary_df = {}
    model_summary_df[0] = model_pds[model_compare_keys[0]]
    model_summary_df[1] = model_pds[model_compare_keys[1]]
    for numnet in model_summary:
        model_summary_df[numnet+1] = pd.concat(model_summary[numnet], axis=0, ignore_index=True) 
    model_summary_df[5] = model_pds[model_compare_keys[2]]
    
    return model_summary_df


def get_coverage_combined_results(results_models, idx2moc):
    columns = ['Model', 'Class', 'Coverage P (max)','Coverage R (max)', 'Coverage P (mean)','Coverage R (mean)', 'Coverage P (Stdev)','Coverage R (Stdev)']
    cResults = []
    for model in results_models:
        folds = results_models[model]
        res_label = ['0-CP', '0-CR','1-CP', '1-CR','2-CP', '2-CR']
        res = []
        for fold in folds:
            res.append([folds[fold][0][0],folds[fold][0][1],folds[fold][1][0],folds[fold][1][1],folds[fold][2][0],folds[fold][2][1]])              
        rdf = pd.DataFrame(res, columns=res_label)
        for c in range(3): #numclasses
            cResults.append([model, idx2moc[c], max(rdf[f'{c}-CP']), max(rdf[f'{c}-CR']), statistics.mean(rdf[f'{c}-CP']), statistics.mean(rdf[f'{c}-CR']), statistics.stdev(rdf[f'{c}-CP']), statistics.stdev(rdf[f'{c}-CR'])])
    df = pd.DataFrame(cResults, columns=columns)
    result_df = pd.DataFrame()
    gdf = df.groupby('Class')
    for name, group in gdf:
        result_df = pd.concat([result_df, group])
                
    return result_df

def get_coverage_results_all_folds(results_models, idx2moc):
    columns = ['Model', 'Class', 'Fold', 'Coverage P','Coverage R']
    cResults = []
    for model in results_models:
        folds = results_models[model]
        for fold in folds:           
            for c in range(3): #num_classes
                cp, cr = folds[fold][c]
                cResults.append([model, idx2moc[c], fold, cp, cr])
    df = pd.DataFrame(cResults, columns=columns)
    result_df = pd.DataFrame()
    gdf = df.groupby('Class')
    for name, group in gdf:
        result_df = pd.concat([result_df, group])
                
    return result_df