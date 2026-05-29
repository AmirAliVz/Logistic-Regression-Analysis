"""
Task 2: Logistic Regression

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import skew, mode
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from scipy.stats import chi2_contingency
from statsmodels.api import Logit, add_constant
from sklearn.metrics import confusion_matrix, accuracy_score
from statsmodels.stats.outliers_influence import variance_inflation_factor



def plot_histogram_boxplot(data, filename, kde=True, display=True, figsize=(10, 6), fontsize=12, fontcolor='black'):

    """
    Plots a combined figure of a box plot and a histogram for a specified column of data in a Series.

    Parameters:
        data (pd.Series): The input column of data. Must be provided.
        filename (str): The name of the column to visualize. Must be provided.
        kde (bool, optional): Whether to use a kernel density estimator or not.
        display (bool): Whether or not to display the figure.
        ylog (bool, optional): If True, sets the histogram's y-axis to a logarithmic scale. Defaults to False.
        figsize (tuple, optional): The size of the figure shown. Defaults to (10, 6).
        fontsize (int, optional): Font size for annotations within the box plot. Defaults to 8.
        fontcolor (str, optional): Font color for annotations within the box plot. Defaults to 'black'.



    Returns:
        lower_bound (float): The lower bound used in the box plot for outlier detection.
        upper_bound (float): The upper bound used in the box plot for outlier detection.

    Note:
        The plot is saved as a .jpg image in the 'Figures' folder, named after the column with the specified suffix.
    """

    # filename = filename.capitalize()
    # Calculate boxplot stats
    q1 = np.percentile(data, 25)
    median = np.median(data)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_whisker = max(data.min(), q1 - 1.5 * iqr)
    upper_whisker = min(data.max(), q3 + 1.5 * iqr)

    if not display:
        return lower_whisker, upper_whisker

    # Setup the figure
    fig, (ax_box, ax_hist) = plt.subplots(
        2, 1, figsize=figsize, gridspec_kw={"height_ratios": (0.25, 0.75)}, sharex=True
    )

    cmap = plt.get_cmap('inferno')

    # --- Boxplot
    sns.boxplot(x=data, ax=ax_box, color='skyblue')

    # Annotate stats on boxplot
    box_stats = {
        'Min': lower_whisker,
        'Q1': q1,
        'Median': median,
        'Q3': q3,
        'Max': upper_whisker
    }

    # ------------------------
    for label, val in box_stats.items():
        ax_box.text(val, 0.02, f'{label}\n{val:.2f}',
                    ha='center', va='bottom', fontsize=fontsize,
                    color=fontcolor, rotation=45, weight='bold')

    ax_box.set(xlabel='')

    # --- Histogram
    hist_plot  = sns.histplot(data, bins='auto', kde=kde, color='steelblue', edgecolor='black', ax=ax_hist)

    # Annotate stats on boxplot
    hist_stats = {
        'Skew': skew(data),
        'STD': np.std(data),
        'Mode': mode(data, keepdims=True).mode[0],
        'Mean': np.mean(data)
    }

    # Example stats text
    stats_text = '\n'.join([
        f'Mean: {data.mean():.2f}',
        f'Mode: {data.mode().values[0]:.2f}',
        f'Std: {data.std():.2f}',
        f'Skew: {data.skew():.2f}'
    ])

    # Draw rectangle (manual position and size in Axes coordinates: 0–1 range)
    bbox_props = dict(boxstyle="round4,pad=1.2", fc="seashell", ec="black", alpha=0.5)
    ax_hist.text(
        0.85, 0.6, stats_text,
        transform=ax_hist.transAxes,
        verticalalignment='top', horizontalalignment='right',
        bbox=bbox_props, ha='center',
        va='bottom', fontsize=fontsize+2, color=fontcolor, weight='bold'
    )

    # Accessing the patches (bars) and bins from the plot
    patches = hist_plot.patches  # These are the bars of the histogram

    # To get the counts, you can use `patches` to calculate the heights (counts)
    n = [patch.get_height() for patch in patches]

    # Gradient fill on histogram bars
    for patch, count in zip(patches, n):
        color = cmap(0.3 + 0.7 * count / max(n))
        patch.set_facecolor(color)

    # Annotate count values on bars
    for patch, count in zip(patches, n):
        if count > 0:
            ax_hist.text(patch.get_x() + patch.get_width() / 2,
                         count,
                         f'{int(count)}',
                         ha='center', va='bottom', fontsize=8)

    # Titles and labels
    ax_hist.set_title(f'Distribution of {filename}', fontweight='bold')
    ax_hist.set_xlabel(filename, fontweight='bold')
    ax_hist.set_ylabel('Count', fontweight='bold')

    plt.tight_layout()

    # Check if 'Figures' folder exists, and if not, create it
    if not os.path.exists("Figures"):
        os.makedirs("Figures")

    plt.savefig('Figures/' + filename + '.jpg', dpi=300)
    if display:
        plt.show()
    else:
        plt.close()

    return lower_whisker, upper_whisker

def barchart(data, filename, fontsize=14, verbose=True):
    """
    Generates and saves a bar chart based on the frequency of categorical data.

    Parameters:
    -----------
    data : pandas.Series or list-like
        The data column containing categorical values to be plotted in the bar chart.

    filename : str
        The name of the output image file to be saved in the 'Figures' directory (without path).

    fontsize : int, optional (default=14)
        Font size for axis labels and tick marks.

    verbose : bool, optional (default=True)
        If True, displays the plot after saving. If False, suppresses plot display.
    """

    # filename = filename.capitalize()
    # Get frequency counts
    counts = data.value_counts()
    # counts.index = counts.index.str.capitalize()

    # Set figure size
    plt.figure(figsize=(8, 6))

    # Define custom colors using Seaborn's palette
    colors = sns.color_palette("pastel")[:len(counts)]

    # Create bar chart with custom aesthetics
    bars = plt.bar(counts.index, counts, color=colors,
                   edgecolor='black', linewidth=1.5)

    # Add labels on top of bars
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width()/2, 1.03*bar.get_height(), f'{int(bar.get_height())}',
                 ha='center', fontsize=fontsize, fontweight='bold', color='black')

        prc = round(bar.get_height()/len(data)*100, 1)
        plt.text(bar.get_x() + bar.get_width()/2, 0.5*bar.get_height(), f'{float(prc)}' + '%',
                 ha='center', fontsize=fontsize, fontweight='bold', color='black')

    # Style the chart
    plt.title('Distribution of ' + filename, fontsize=fontsize+4, fontweight='bold', color='#333333')
    plt.xlabel(filename, fontsize=fontsize+2, fontweight='bold', color='#555555')
    plt.ylabel('Frequency Count', fontsize=fontsize+2, fontweight='bold', color='#555555')

    # Adjust grid aesthetics
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    # Remove top and right borders for a cleaner look
    plt.box(False)

    # Check if 'Figures' folder exists, and if not, create it
    if not os.path.exists("Figures"):
        os.makedirs("Figures")

    # Save & show the plot
    plt.savefig('Figures/' + filename + '.jpg', dpi=300)
    if verbose:
        plt.show()
    else:
        plt.close()

def catNum(df, numeric_col, categorical_col, verbose=True):
    """
    Creates a bivariate visualization for one continuous numeric column and one categorical column.

    Parameters:
    df (pd.DataFrame): The dataframe containing the data.
    numeric_col (str): The name of the continuous numeric column.
    categorical_col (str): The name of the categorical column.

    This function creates a boxplot to show the distribution of the numeric variable across categories.
    """
    # Set figure size and style
    plt.figure(figsize=(14, 6))

    # === Violin Plot ===
    plt.subplot(1, 2, 1)
    sns.violinplot(x=categorical_col, y=numeric_col, hue=categorical_col,
                   data=df, inner='quartile',
                   palette='Set2', dodge=False, legend=False)
    plt.title(numeric_col + ' Distribution by ' + categorical_col + ' (Violin Plot)', fontsize=14, fontweight='bold')
    plt.xlabel(categorical_col, fontsize=14, fontweight='bold')
    plt.ylabel(numeric_col, fontsize=14, fontweight='bold')

    # === Histogram + KDE Overlay ===
    plt.subplot(1, 2, 2)
    sns.histplot(data=df, x=numeric_col, hue=categorical_col, kde=True,
                 element='step', stat='density', common_norm=False,palette='Set1')
    plt.title(numeric_col + ' Distribution by ' + categorical_col + ' (Histogram + KDE)', fontsize=14, fontweight='bold')
    plt.xlabel(numeric_col, fontsize=14, fontweight='bold')
    plt.ylabel('Density', fontsize=14, fontweight='bold')

    # Display both plots
    plt.tight_layout()
    plt.savefig('Figures/' + numeric_col + 'by' + categorical_col + '.jpg', dpi=300)
    if verbose:
        plt.show()
    else:
        plt.close()

def forward_stepwise_selection(X, y, threshold_in=0.05, verbose=True):
    """
    Perform forward stepwise selection for linear regression.
    Args:
        X (pd.DataFrame): Candidate predictor variables (already preprocessed, e.g., dummies for categoricals).
        y (pd.Series): Target variable.
        threshold_in (float): p-value threshold for adding a variable.
        verbose (bool): Whether to print progress.
    Returns:
        List of selected features.
    """
    included = []
    while True:
        changed = False
        # Find remaining features not yet included
        excluded = list(set(X.columns) - set(included))
        new_pval = pd.Series(index=excluded, dtype=float)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(X[included + [new_column]])).fit()
            new_pval[new_column] = model.pvalues[new_column]
        # Find the best candidate
        if not new_pval.empty:
            best_pval = new_pval.min()
            if best_pval < threshold_in:
                best_feature = new_pval.idxmin()
                included.append(best_feature)
                changed = True
                if verbose:
                    print(f'Add {best_feature:30} with p-value {best_pval:.6f}')
        if not changed:
            break
    return included

def backward_stepwise_elimination(X, y, threshold_out=0.05, verbose=True):
    """
    Perform backward stepwise elimination for linear regression.
    Removes features with p-value above threshold_out.
    """
    features = list(X.columns)
    while True:
        X_with_const = sm.add_constant(X[features])
        model = sm.OLS(y, X_with_const).fit()
        pvalues = model.pvalues.iloc[1:]  # exclude intercept
        max_pval = pvalues.max()
        if max_pval > threshold_out:
            excluded_feature = pvalues.idxmax()
            features.remove(excluded_feature)
            if verbose:
                print(f"Remove {excluded_feature:30} with p-value {max_pval:.6f}")
        else:
            break
    return features

def rfe_statsmodels(X, y, n_features_to_select=5, verbose=True):
    """
    Recursive Feature Elimination using statsmodels OLS.
    Removes the least significant feature (highest p-value) at each iteration.
    Stops when n_features_to_select features remain.
    """
    features = list(X.columns)
    while len(features) > n_features_to_select:
        X_with_const = sm.add_constant(X[features])
        model = sm.OLS(y, X_with_const).fit()
        # Exclude intercept from p-values
        pvalues = model.pvalues.iloc[1:]
        worst_feature = pvalues.idxmax()
        if verbose:
            print(f"Removing {worst_feature} (p-value: {pvalues[worst_feature]:.4f})")
        features.remove(worst_feature)
    return features

def plot_residuals_analysis(model):
    """
    Plots residuals vs fitted values and autocorrelation of residuals using provided fig and ax.
    If fig and ax are None, creates new ones.
    """
    # Get fitted values and residuals
    fitted_vals = model.fittedvalues
    residuals = model.resid

    # Create figure and axes
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Residuals vs Fitted plot on the first axis
    ax[0].scatter(fitted_vals, residuals, alpha=0.5)
    ax[0].axhline(0, color='red', linestyle='--')
    ax[0].tick_params(axis='x', rotation=30)
    ax[0].set_xlabel('Fitted values')
    ax[0].set_ylabel('Residuals')
    ax[0].set_title('Residuals vs Fitted Values', weight='bold')

    # Autocorrelation plot of residuals on the second axis
    sm.graphics.tsa.plot_acf(residuals, lags=40, alpha=0.05, zero=False, ax=ax[1])
    ax[1].set_title('Autocorrelation of Residuals', weight='bold')
    ax[1].set_xlabel('Lag')           # X-axis label
    ax[1].set_ylabel('Autocorrelation') # Y-axis label

    plt.tight_layout()
    plt.savefig('Figures/' + 'residuals_analysis' + '.jpg', dpi=300)
    plt.show()

def two_categorical(x, y, col1, col2, verbose=True):
    """
    Performs a Chi-Square test of independence between two categorical variables,
    and generates visualizations for interpretation.

    Parameters:
    -----------
    x : pandas.Series
        The first categorical variable.

    y : pandas.Series
        The second categorical variable.

    col1 : str
        Name or label for the first variable (used in plots and output titles).

    col2 : str
        Name or label for the second variable.

    verbose : bool, optional (default=True)
        If True, displays:
            - A heatmap of the contingency table
            - Side-by-side bar charts for distribution comparison
            - Test results summary

    Returns:
    --------
    chi2 : float
        The Chi-Square test statistic.

    p : float
        The p-value of the test.
    """


    col1 = col1.capitalize()
    col2 = col2.capitalize()

    # Assuming your dataset is in a pandas Series
    cross_tab = pd.crosstab(x, y)
    # if verbose:
    #     print(cross_tab)


    # Perform Chi-squared test
    chi2, p, dof, expected = chi2_contingency(cross_tab)
    # if verbose:
    #     print(f"Chi-squared value of {col1} & {col2}: {chi2}")
    #     print(f"P-value: {p}")
    #     print(50*'=')

    # Setup the figure
    fig, (ax_heat, ax_bar) = plt.subplots(
        2, 1, figsize=(10,8))

    # Barplot
    sns.countplot(x=x, hue=y, ax=ax_bar, alpha=0.7)
    ax_heat.set(ylabel='Count')
    ax_bar.set_xlabel(col1, fontsize=14, fontweight='bold')  # X-axis label
    ax_bar.set_ylabel(col2 + ' Frequency', fontsize=14, fontweight='bold')  # Y-axis label

    # Add counts inside bars, just below the top edge
    for patch in ax_bar.patches:
        height = patch.get_height()
        if height > 0:
            ax_bar.text(
                patch.get_x() + patch.get_width() / 2.,
                height - (0.05 * height),  # 5% below the top of the bar
                int(height),
                ha='center',
                va='top',  # Align text to the top so it stays inside
                fontsize=12,
                color='black'  # Use white for dark bars, or black for light
            )

    # Heatmap of the contingency table
    sns.heatmap(cross_tab, annot=True, cmap='Blues', fmt='d', ax=ax_heat)
    ax_heat.set_title('Heatmap & Grouped BarChart of '+col1+' & '+col2 +
                      '\n \n chi2: ' + str(format(chi2, ".2e")) + '    -    p-value: ' + str(format(p, ".2e")), fontsize=16, fontweight='bold')  # Title
    ax_heat.set_xlabel(col2, fontsize=14, fontweight='bold')  # X-axis label
    ax_heat.set_ylabel(col1, fontsize=14, fontweight='bold')  # Y-axis label

    # Check if 'Figures' folder exists, and if not, create it
    if not os.path.exists("Figures"):
        os.makedirs("Figures")

    plt.savefig('Figures/' + col1 + ' vs ' + col2 + '.jpg', dpi=300)
    if verbose:
        plt.show()
    else:
        plt.close()

    return chi2, p

def forward_stepwise_logistic(X, y, initial_features=None, threshold_in=0.05, verbose=True):
    """
    Perform forward stepwise selection for logistic regression.

    Parameters:
        X (DataFrame): Predictor variables.
        y (Series): Binary outcome variable.
        initial_features (list): Features to start with (default: None).
        threshold_in (float): p-value threshold for including a feature.
        verbose (bool): Print progress if True.

    Returns:
        selected_features (list): List of selected features.
        final_model (statsmodels LogitResults): Fitted logistic regression model.
    """
    included = list(initial_features) if initial_features else []
    available = list(X.columns.difference(included))
    while True:
        changed = False
        # Try adding each available feature
        new_pvals = pd.Series(index=available)
        for col in available:
            try:
                model = Logit(y, add_constant(X[included + [col]])).fit(disp=0)
                pval = model.pvalues[col]
                new_pvals[col] = pval
            except Exception:
                new_pvals[col] = np.nan
        min_pval = new_pvals.min()
        if min_pval is not np.nan and min_pval < threshold_in:
            best_feature = new_pvals.idxmin()
            included.append(best_feature)
            available.remove(best_feature)
            changed = True
            if verbose:
                print(f"Add {best_feature} with p-value {min_pval:.4f}")
        if not changed:
            break
    # Fit final model
    final_model = Logit(y, add_constant(X[included])).fit(disp=0)
    return included, final_model

def backward_stepwise_logistic(X, y, threshold_out=0.05, verbose=True):
    """
    Perform backward stepwise selection for logistic regression.

    Parameters:
        X (DataFrame): Predictor variables.
        y (Series): Binary outcome variable.
        threshold_out (float): p-value threshold for removing a feature.
        verbose (bool): Print progress if True.

    Returns:
        selected_features (list): List of selected features.
        final_model (statsmodels LogitResults): Fitted logistic regression model.
    """
    included = list(X.columns)
    while True:
        changed = False
        model = Logit(y, add_constant(X[included])).fit(disp=0)
        pvals = model.pvalues.iloc[1:]  # exclude intercept
        max_pval = pvals.max()
        if max_pval > threshold_out:
            worst_feature = pvals.idxmax()
            included.remove(worst_feature)
            changed = True
            if verbose:
                print(f"Remove {worst_feature} with p-value {max_pval:.4f}")
        if not changed:
            break
    final_model = Logit(y, add_constant(X[included])).fit(disp=0)
    return included, final_model

def recursive_feature_elimination_logistic(X, y, num_features_to_select=5, verbose=True):
    """
    Perform Recursive Feature Elimination (RFE) for logistic regression.

    Parameters:
        X (DataFrame): Predictor variables.
        y (Series): Binary outcome variable.
        num_features_to_select (int): Number of features to keep.
        verbose (bool): Print progress if True.

    Returns:
        selected_features (list): List of selected features.
        final_model (statsmodels LogitResults): Fitted logistic regression model.
    """
    included = list(X.columns)

    while len(included) > num_features_to_select:
        model = Logit(y, add_constant(X[included])).fit(disp=0)
        pvals = model.pvalues.iloc[1:]  # exclude intercept
        worst_feature = pvals.idxmax()
        included.remove(worst_feature)
        if verbose:
            print(f"Remove {worst_feature} with p-value {pvals[worst_feature]:.4f}")

    final_model = Logit(y, add_constant(X[included])).fit(disp=0)
    return included, final_model

def plot_residuals_analysis(model):
    """
    Plots residuals vs fitted values and autocorrelation of residuals using provided fig and ax.
    If fig and ax are None, creates new ones.
    """
    # Get fitted values and residuals
    fitted_vals = model.fittedvalues
    residuals = model.resid

    # Create figure and axes
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Residuals vs Fitted plot on the first axis
    ax[0].scatter(fitted_vals, residuals, alpha=0.5)
    ax[0].axhline(0, color='red', linestyle='--')
    ax[0].tick_params(axis='x', rotation=30)
    ax[0].set_xlabel('Fitted values')
    ax[0].set_ylabel('Residuals')
    ax[0].set_title('Residuals vs Fitted Values', weight='bold')

    # Autocorrelation plot of residuals on the second axis
    sm.graphics.tsa.plot_acf(residuals, lags=40, alpha=0.05, zero=False, ax=ax[1])
    ax[1].set_title('Autocorrelation of Residuals', weight='bold')
    ax[1].set_xlabel('Lag')           # X-axis label
    ax[1].set_ylabel('Autocorrelation') # Y-axis label

    plt.tight_layout()
    plt.savefig('Figures/' + 'residuals_analysis' + '.jpg', dpi=300)
    plt.show()

def conf_matrix_plot(conf_matrix, accuracy, dataset=''):
    # Plot the confusion matrix
    # Custom labels for each cell
    labels = np.array([
        ["True Negative", "False Positive"],
        ["False Negative", "True Positive"]
    ])

    fig, ax = plt.subplots(figsize=(6, 5))
    cmap = plt.cm.Oranges
    # Set vmin lower and vmax higher than our data for a lighter overall map
    vmin = 0
    vmax = conf_matrix.max() + 500

    # Heatmap
    im = ax.imshow(conf_matrix, interpolation='nearest', cmap=cmap, vmin=vmin, vmax=vmax)

    # Add text annotations
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, f"{labels[i, j]}\n {conf_matrix[i, j]}",
                ha="center", va="center", color="black", fontsize=10, fontweight='bold'
            )

    # Axes setup
    ax.set(
        xticks=[0, 1], yticks=[0, 1],
        xticklabels=["Not Luxury", "Luxury"],
        yticklabels=["Not Luxury", "Luxury"],
        ylabel='Actual Label',
        xlabel='Predicted Label',
        title= f"Confusion Matrix ({dataset}) \n Accuracy: {round(accuracy, 2)}"
    )

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig('Figures/ConfusionMatrix_' + dataset + '.jpg', dpi=300)
    if verbose:
        plt.show()
    else:
        plt.close()


verbose = 1
# Change global font to 'Arial', size 14
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 14

# Read Dataset file into a DataFrame
df = pd.read_csv("data/D600 Task 2 Dataset 1 Housing Information.csv")

# Preparing the dataset for our analysis
df_clean = df.copy()
df_clean = df_clean.drop(['ID', 'PreviousSalePrice', 'Price'], axis=1)
df_clean['Windows'] = df_clean['Windows'].abs()

# ==============================================================================
# part C. Descriptive Statistics and Visualizations
# ==============================================================================

# Visualization of the Numerical Variables Distributions
numeric = df_clean.select_dtypes(include=np.number)
for col in numeric:
    if col not in ['NumBedrooms', 'Floors', 'IsLuxury']:
        # Univariate Visualizations (Histogram & BoxPlot)
        plot_histogram_boxplot(df_clean[col], col, display=verbose)
        # Bivariate Visualizations (Violin Plot & Histogram)
        catNum(df_clean, col, 'IsLuxury', verbose=verbose)


# Visualization of the Categorical Variables
for col in ['Fireplace', 'Garage', 'IsLuxury', 'HouseColor', 'NumBedrooms', 'Floors']:
    # Univariate Visualizations (barchart)
    barchart(df_clean[col], col, verbose=verbose)
    if col!='IsLuxury':
        # Bivariate Visualizations (Violin Plot & Histogram)
        two_categorical(df_clean[col], df_clean['IsLuxury'], col, 'IsLuxury', verbose=True)


# ==============================================================================
# part D. Data Analysis and Report
# ==============================================================================

# Split the dataset: 80% for training, 20% for testing
train_df, test_df = train_test_split(df_clean, test_size=0.2, random_state=42)

# Save the splits to CSV files
train_df.to_csv('data/train_dataset.csv', index=False)
test_df.to_csv('data/test_dataset.csv', index=False)

print("Training and test datasets created and saved as 'train_dataset.csv' and 'test_dataset.csv'.")

# Logistic Regression with Step Forward Optimization
X_train = train_df.drop(columns=['IsLuxury'])
y_train = train_df['IsLuxury']
X_train = pd.get_dummies(X_train, drop_first=True)  # Encode categoricals if needed
# This will find all boolean columns and convert them to int
bool_cols = X_train.select_dtypes(include='bool').columns
X_train[bool_cols] = X_train[bool_cols].astype(int)

# Feature Selection --------------------------------
selected_features, final_model = forward_stepwise_logistic(X_train, y_train)
# selected_features, model = backward_stepwise_logistic(X_train_scaled, y_train)
# selected_features, model = recursive_feature_elimination_logistic(X_train_scaled, y_train)

print('Forward_Stepwise_Logistic : ')
print("Selected features:", selected_features)
print(final_model.summary())
print("AIC:", final_model.aic)
print("BIC:", final_model.bic)

# part D3. Confusion Matrix and Accuracy on training set ----------------------
# Predict probabilities on the training set
pred_probs_train = final_model.predict(add_constant(X_train[selected_features]))
# Convert probabilities to class labels (threshold = 0.5)
pred_labels_train = (pred_probs_train >= 0.5).astype(int)

# True labels: y_train
conf_matrix_train = confusion_matrix(y_train, pred_labels_train)
accuracy_train = accuracy_score(y_train, pred_labels_train)
conf_matrix_plot(conf_matrix_train, accuracy_train, 'Trainset')

# part D4. Confusion Matrix and Accuracy on test set ----------------------
X_test = test_df.drop(columns=['IsLuxury'])
y_test = test_df['IsLuxury']
X_test = pd.get_dummies(X_test, drop_first=True)  # Encode categoricals if needed
# This will find all boolean columns and convert them to int
bool_cols = X_test.select_dtypes(include='bool').columns
X_test[bool_cols] = X_test[bool_cols].astype(int)

Xtest_selected = sm.add_constant(X_test[selected_features])

# Predict probabilities on the training set
pred_probs_test = final_model.predict(Xtest_selected)
# Convert probabilities to class labels (threshold = 0.5)
pred_labels_test = (pred_probs_test >= 0.5).astype(int)

# True labels: y_train
conf_matrix_test = confusion_matrix(y_test, pred_labels_test)
accuracy_test = accuracy_score(y_test, pred_labels_test)
conf_matrix_plot(conf_matrix_test, accuracy_test, 'Testset')


# E5. Logistic Regression Assumptions Verification
# Plot logit vs individual continuous predictors
for col in X_train[selected_features].columns:

    print('Checking Linearity assumption For column (Box-Tidwell): ', col)

    # Example for one continuous variable "X_var"
    X_BoxTidwell = X_train[selected_features].copy()

    # Add small constant if any value <=0
    if (X_BoxTidwell[col] <= 0).any():
        offset = abs(X_BoxTidwell[col].min()) + 1e-6
        X_BoxTidwell[col] = X_BoxTidwell[col] + offset

    # Box-Tidwell transformation
    X_BoxTidwell[col+' Box-Tidwell'] = X_BoxTidwell[col] * np.log(X_BoxTidwell[col])
    X_bt = add_constant(X_BoxTidwell)

    # Fit logistic regression
    model_BT = Logit(y_train, X_bt).fit()
    pval_bt = model_BT.pvalues[col+' Box-Tidwell']
    print(col+' Interaction term (X * ln(X)) p-value: ', pval_bt)

    plt.figure()
    ax = sns.regplot(x=X_train[selected_features][col], y=np.log(pred_probs_train / (1 - pred_probs_train)),
                     lowess=True, line_kws={'color': 'red'})

    linearity_label = 'non-Linear' if pval_bt < 0.05 else 'Linear'
    status_text = f'Box-Tidwell (X * ln(X)) term p-value = {pval_bt:.3g}\n Linearity = ✓ {linearity_label}' if pval_bt >= 0.05 else f'Box-Tidwell (X * ln(X)) term p-value = {pval_bt:.3g}\n Linearity = ✖ {linearity_label}'  # check mark if p >= 0.05, red X if not
    status_color = 'green' if pval_bt >= 0.05 else 'red'
    # Annotate the Box-Tidwell p-value on the plot (as a floating text box)
    ax.text(
        0.05, 0.95,
        status_text,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment='top',
        bbox=dict(alpha=0.6, edgecolor='gray', facecolor=status_color)
    )

    plt.title(f'Linearity Check: Logit vs {col}', weight='bold')
    plt.xlabel(col, weight='bold')
    plt.ylabel("Logit(P)", weight='bold')
    plt.savefig('Figures/' + f'LinearCheck{col}' + '.jpg', dpi=300)
    plt.show()


# Check Multicollinearity ---------------
print('Checking Multicollinearity')
# Assume X is your DataFrame of independent variables
X_vif = add_constant(X_train[selected_features])  # Adds intercept term

# Create a DataFrame for VIF values
vif_df = pd.DataFrame()
vif_df["Variable"] = X_vif.columns
vif_df["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]

# Display the VIFs (ignore the 'const' intercept row)
print(vif_df)

