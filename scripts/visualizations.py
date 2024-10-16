import seaborn as sns
import matplotlib.pyplot as plt
import os
import logging

# Logging configuration
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'logs')

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_file_info = os.path.join(log_dir, 'info.log')
log_file_error = os.path.join(log_dir, 'error.log')

info_handler = logging.FileHandler(log_file_info)
info_handler.setLevel(logging.INFO)

error_handler = logging.FileHandler(log_file_error)
error_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
info_handler.setFormatter(formatter)
error_handler.setFormatter(formatter)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(info_handler)
logger.addHandler(error_handler)

def plot_univariate_analysis(df, columns):
    for column in columns:
        plt.figure(figsize=(10, 5))
        
        try:
            if df[column].dtype == 'object':
                sns.countplot(data=df, x=column)
                plt.title(f'Count Plot for {column}')
                logger.info(f'Successfully plotted count plot for {column}')
            else:
                sns.histplot(data=df, x=column, kde=True)
                plt.title(f'Histogram for {column}')
                logger.info(f'Successfully plotted histogram for {column}')
            
            plt.xlabel(column)
            plt.ylabel('Frequency')
            plt.show()
        except Exception as e:
            logger.error(f'Error plotting {column}: {e}')


def plot_bivariate_analysis(df, columns):
    plots = []
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            plt.figure(figsize=(10, 5))
            
            try:
                if df[columns[i]].dtype == 'object' or df[columns[j]].dtype == 'object':
                    plot = sns.countplot(data=df, x=columns[i], hue=columns[j])
                    plt.title(f'Count Plot for {columns[i]} vs {columns[j]}')
                    logger.info(f'Successfully plotted count plot for {columns[i]} vs {columns[j]}')
                else:
                    plot = sns.scatterplot(data=df, x=columns[i], y=columns[j])
                    plt.title(f'Scatter Plot for {columns[i]} vs {columns[j]}')
                    logger.info(f'Successfully plotted scatter plot for {columns[i]} vs {columns[j]}')
                
                plt.xlabel(columns[i])
                plt.ylabel(columns[j])
                plt.show()
                plots.append(plot)
            except Exception as e:
                logger.error(f'Error plotting {columns[i]} vs {columns[j]}: {e}')
    return plots