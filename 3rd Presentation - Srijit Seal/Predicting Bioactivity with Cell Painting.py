# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy==2.2.6",
#     "pandas==2.3.3",
#     "pauc==0.2.1",
#     "plotly==6.3.1",
#     "scikit-learn==1.7.2",
#     "scipy==1.16.2",
#     "tqdm==4.67.1",
# ]
# ///

import marimo

__generated_with = "0.18.4"
app = marimo.App(
    css_file="/usr/local/_marimo/custom.css",
    auto_download=["html"],
)


@app.cell
def _():
    import pandas as pd
    import marimo as mo
    data = pd.read_csv("EveBio_CP_data.csv")
    return data, mo, pd


@app.cell
def _(data):
    data
    return


@app.cell
def _(mo):
    mo.md("""
    # Predicting Biological Activity from Cell Painting (CellProfiler-analysed) Features
    This interactive tutorial shows how to use **machine learning** to predict biological assay outcomes from **Cell Painting imaging features**.
    You’ll:
    1. Explore the dataset
    2. Choose a biological endpoint (data from Eve Bio https://data.evebio.org/ Data Release #4  )
    3. Train a Random Forest model (manual or optimized)
    4. Evaluate its performance with various metrics and plots
    """)
    return


@app.cell
def _(data):
    CellProfiler = data.columns.to_list()[11:]
    return (CellProfiler,)


@app.cell
def _(data):
    assays_list = data.columns.to_list()[1:11]
    #len(assays_list)
    return (assays_list,)


@app.cell
def _():
    #data.columns
    return


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md("""
    ### Step 1: Choose a Biological Endpoint

    Here, you can select which assay you want to predict.
    Each assay represents a specific biological response, and our goal is to predict it from the Cell Painting features.
    """)
    return


@app.cell
def _(assays_list, mo):
    selected_endpoint = mo.ui.dropdown(
            options=assays_list,
            label="Select a biological endpoint:",
            value="PR_Agonist"
    )
    return (selected_endpoint,)


@app.cell
def _(selected_endpoint):
    selected_endpoint
    return


@app.cell
def _(mo, selected_endpoint):
    # show result
    mo.md(f"**You selected the endpoint:** {selected_endpoint.value}")
    return


@app.cell
def _(CellProfiler, data, mo):
    mo.md(f"""
    ### Dataset Overview: {data.shape[0]} compounds × {len(CellProfiler)} Cell Painting features
    """)
    return


@app.cell
def _(CellProfiler, data, mo, pd, px, selected_endpoint):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    # Filter out rows with missing endpoint values
    df = data[~data[selected_endpoint.value].isna()].reset_index(drop=True)

    # Extract features and endpoint
    X_PCA = df[CellProfiler].values
    scaler = StandardScaler()
    X_PCA = scaler.fit_transform(X_PCA)

    y_PCA = df[selected_endpoint.value].astype(int).values

    # Perform PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_PCA)

    # Variance explained
    explained_var = pca.explained_variance_ratio_
    mo.md(f"**PCA - variance explained:** PC1: {explained_var[0]:.2f}, PC2: {explained_var[1]:.2f}")

    # Create DataFrame for plotting
    pca_df = pd.DataFrame({
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1],
        'Endpoint': y_PCA.astype(str)  # convert to string for categorical coloring
    })

    # Plot PCA scatter
    fig_pca = px.scatter(
        pca_df,
        x='PC1',
        y='PC2',
        color='Endpoint',
        title=f"PCA of CellProfiler Features Colored by {selected_endpoint.value}",
        labels={'PC1': f"PC1 ({explained_var[0]*100:.1f}% var)", 
                'PC2': f"PC2 ({explained_var[1]*100:.1f}% var)"}
    )
    fig_pca.update_traces(marker=dict(size=6, opacity=0.8))

    mo.ui.plotly(fig_pca)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Step 2: Explore Class Distribution

    It’s important to check the balance of your classes (active vs inactive compounds).
    A very imbalanced dataset may require special considerations for modeling.
    """)
    return


@app.cell
def _(data, mo, selected_endpoint):
    import plotly.express as px

    # Count occurrences of each class (0 and 1)
    endpoint_counts = data[selected_endpoint.value].value_counts().reset_index()
    endpoint_counts.columns = ['Class', 'Count']

    # Make the bar chart
    fig_dist = px.bar(
        endpoint_counts,
        x='Class',
        y='Count',
        text='Count',
        title=f"Distribution of Classes for {selected_endpoint.value}",
        height= 500,
        width= 500
    )
    fig_dist.update_traces(textposition='outside')

    # Display interactively in Marimo
    mo.ui.plotly(fig_dist)
    return (px,)


@app.cell
def _():
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from tqdm import tqdm
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.model_selection import RandomizedSearchCV
    import os
    return RandomForestClassifier, RandomizedSearchCV, roc_auc_score, tqdm


@app.cell
def _(CellProfiler):
    import numpy as np
    from sklearn.metrics import matthews_corrcoef, confusion_matrix, roc_curve, precision_recall_curve, auc, cohen_kappa_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.utils import class_weight
    from sklearn.utils import shuffle

    def calculate_metrics(y_true, y_pred_proba, threshold=0.5):
        y_pred_opt = (y_pred_proba >= threshold).astype(int)
        (tn, fp, fn, tp) = confusion_matrix(y_true, y_pred_opt).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        balanced_accuracy = (sensitivity + specificity) / 2
        mcc = matthews_corrcoef(y_true, y_pred_opt)
        kappa = cohen_kappa_score(y_true, y_pred_opt)
        (precision, recall, _) = precision_recall_curve(y_true, y_pred_proba)
        aucpr = auc(recall, precision)
        kappa_at_50 = cohen_kappa_score(y_true, (y_pred_proba >= 0.5).astype(int))
    # Function to calculate metrics
        return {'Threshold': threshold, 'Balanced Accuracy': balanced_accuracy, 'Sensitivity': sensitivity, 'Specificity': specificity, 'MCC': mcc, 'AUCPR': aucpr, "Cohen's Kappa": kappa, "Cohen's Kappa @ 0.50": kappa_at_50, 'Confusion Matrix': {'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp}, 'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp}

    def find_best_threshold_for_kappa(y_true, y_pred_proba):
        thresholds = np.linspace(0, 1, 100)
        best_kappa = -1
        best_threshold = 0.5
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            kappa = cohen_kappa_score(y_true, y_pred)
            if kappa > best_kappa:  # AUCPR
                best_kappa = kappa
                best_threshold = threshold
        return (best_threshold, best_kappa)

    stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    feature_sets = {'CellCount': ['Cells_Number_Object_Number'], 'CellProfiler': CellProfiler}
    return (
        calculate_metrics,
        confusion_matrix,
        cross_val_predict,
        feature_sets,
        find_best_threshold_for_kappa,
        np,
        stratified_kfold,
    )


@app.cell
def _(mo):
    mo.md("""
    ### Step 3: Configure Random Forest Classifier

    You can either:
    - Let the app **automatically optimize** hyperparameters
    - **Manually select** hyperparameters using sliders below

    Random Forest is robust to overfitting and can handle many correlated features, making it ideal for Cell Painting data.
    """)
    return


@app.cell
def _(mo):
    mode_choice = mo.ui.dropdown(
        options=["Optimize Automatically", "Manual Parameters"],
        value="Manual Parameters",
        label="RandomForest Mode"
    )
    return (mode_choice,)


@app.cell
def _(mode_choice):
    mode_choice
    return


@app.cell
def _(mo):
    # Manual parameters (shown if Manual selected)
    n_estimators = mo.ui.slider(
        value=100,
        label="Number of trees (n_estimators)",
        start=1,
        stop=500,
        step=1
    )

    max_depth = mo.ui.slider(
        value=10,
        label="Max depth",
        start=0,
        stop=25,
        step=1
    )

    min_samples_split = mo.ui.slider(
        value=5,
        label="Min samples split",
        start=2,
        stop=20,
        step=1
    )

    min_samples_leaf = mo.ui.slider(
        value=5,
        label="Min samples leaf",
        start=1,
        stop=10,
        step=1
    )
    return max_depth, min_samples_leaf, min_samples_split, n_estimators


@app.cell
def _(
    max_depth,
    min_samples_leaf,
    min_samples_split,
    mo,
    mode_choice,
    n_estimators,
):
    manual_params_ui = (
        [n_estimators,
        max_depth,
        min_samples_split,
        min_samples_leaf]
        if mode_choice.value == 'Manual Parameters'
        else []
    )
    mo.vstack(manual_params_ui)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Step 4: Train the Model

    The model will be trained using stratified k-fold cross-validation to ensure that both classes are represented in each fold.
    For each fold, we will:
    1. Train the model using a 5-fold Cross Validation
    2. Train on the 80% data and evaluate on 20% data
    3. Repeat this in a 5-fold Cross Validation
    4. Calculate metrics like AUC, Balanced Accuracy, AUCPR, and Cohen’s Kappa
    """)
    return


@app.cell
def _(
    RandomForestClassifier,
    RandomizedSearchCV,
    calculate_metrics,
    cross_val_predict,
    data,
    feature_sets,
    find_best_threshold_for_kappa,
    max_depth,
    min_samples_leaf,
    min_samples_split,
    mode_choice,
    n_estimators,
    np,
    pd,
    roc_auc_score,
    selected_endpoint,
    stratified_kfold,
    tqdm,
):
    all_auc_scores = []
    all_predictions = [] 

    for assay in tqdm([selected_endpoint.value]):

        assay_data = data[~data[assay].isna()].reset_index(drop=True)
        print(assay_data[assay].value_counts())
        y = assay_data[assay].astype(int)

        if len(np.unique(y)) < 2:
            print(f'Skipping assay {assay} due to lack of positive/negative classes')
            continue

        rf_param_grid = {'n_estimators': [100, 200, 300], 'max_depth': [5, 10, None], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4], 'class_weight': ['balanced']}

        for (feature_set_name, feature_columns) in feature_sets.items():
            X = assay_data[feature_columns]

            for (fold, (train_idx, test_idx)) in enumerate(stratified_kfold.split(X, y)):
                (X_train, X_test) = (X.iloc[train_idx], X.iloc[test_idx])
                (y_train, y_test) = (y.iloc[train_idx], y.iloc[test_idx])
    # Function to find the best threshold based on Cohen's Kappa

                if mode_choice.value == "Optimize Automatically":
                    model = RandomForestClassifier(random_state=42, n_jobs=-1)
                    random_search = RandomizedSearchCV(estimator=model, param_distributions=rf_param_grid, n_iter=10, scoring='roc_auc', cv=3, n_jobs=-1, verbose=2, random_state=42)
                    random_search.fit(X_train, y_train)
                    best_model = random_search.best_estimator_

                else:
                    # Manual parameters
                    best_model = RandomForestClassifier(
                        n_estimators=int(n_estimators.value),
                        max_depth=None if int(max_depth.value) == 0 else int(max_depth.value),
                        min_samples_split=int(min_samples_split.value),
                        min_samples_leaf=int(min_samples_leaf.value),
                        random_state=42,
                        n_jobs=-1
                    )
                    best_model.fit(X_train, y_train)


                oob_pred_proba = cross_val_predict(best_model, X_train, y_train, cv=stratified_kfold, method='predict_proba')[:, 1]

                (opt_threshold, best_kappa) = find_best_threshold_for_kappa(y_train, oob_pred_proba)
                y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    # Initialize results
                test_metrics = calculate_metrics(y_test, y_pred_proba, threshold=opt_threshold)

                y_pred_opt = (y_pred_proba >= opt_threshold).astype(int)

                all_predictions.append({'Assay': assay,
                                        'Feature Set': feature_set_name,
                                        'Fold': fold + 1,
                                        'y_true': y_test.values.tolist(),
                                        'y_pred': y_pred_opt.tolist(),
                                        'y_proba': y_pred_proba.tolist(),
                                        'best_model': best_model
                                    })

                print(f'Assay: {assay}, Fold: {fold + 1}, AUC: {roc_auc_score(y_test, y_pred_proba):.3f}')

                all_auc_scores.append({'Feature Set': feature_set_name, 'Fold': fold + 1, 'Task': assay, 'AUC': roc_auc_score(y_test, y_pred_proba), 'Optimal Threshold': opt_threshold, 'Optimal Kappa': best_kappa, 'Kappa': test_metrics["Cohen's Kappa"], 'Kappa @ 0.50': test_metrics["Cohen's Kappa @ 0.50"], 'Balanced Accuracy': test_metrics['Balanced Accuracy'], 'AUCPR': test_metrics['AUCPR'], 'Sensitivity': test_metrics['Sensitivity'], 'Specificity': test_metrics['Specificity'], 'MCC': test_metrics['MCC'], 'TP': test_metrics['TP'], 'TN': test_metrics['TN'], 'FP': test_metrics['FP'], 'FN': test_metrics['FN'], 'Total Actives Train': sum(y_train), 'Total Inactives Train': len(y_train) - sum(y_train), 'Total Compounds Train': len(y_train), 'Total Actives Test': sum(y_test), 'Total Inactives Test': len(y_test) - sum(y_test), 'Total Compounds Test': len(y_test)})  # Feature set cell count

                auc_df = pd.DataFrame(all_auc_scores)

    # Loop through each assay to create individual models
    auc_df = pd.DataFrame(all_auc_scores)
    pred_df = pd.DataFrame(all_predictions)
    return auc_df, pred_df


@app.cell
def _(mo):
    mo.md("""
    ### Step 5: Review Model Performance

    Here we summarize the mean model performance across feature sets and 5 folds of the CV.
    - **AUC-ROC** measures overall ability to discriminate between classes.
    - **Balanced Accuracy** accounts for class imbalance.
    - **AUCPR** is useful when classes are imbalanced.
    """)
    return


@app.cell
def _(auc_df):
    auc_df.groupby(["Task", "Feature Set"]).mean().round(2)[['AUC', 'Balanced Accuracy', 'AUCPR','Sensitivity', 'Specificity', 'MCC']]
    return


@app.cell
def _(auc_df, mo, px, selected_endpoint):
    # Filter results for the currently selected endpoint
    subset_auc = auc_df

    # Aggregate metrics by feature set
    metrics_summary = (
        subset_auc.groupby("Feature Set", as_index=False)
        .agg({
            "AUC": "mean",
            "Balanced Accuracy": "mean",
            "AUCPR": "mean"
        })
        .round(3)
    )

    # --- Create separate figures ---
    fig_auc = px.bar(
        metrics_summary,
        x="Feature Set",
        y="AUC",
        color="Feature Set",
        title=f"AUC-ROC ({selected_endpoint.value})"
    )
    fig_auc.update_layout(width=600, height=400, yaxis_range=[0,1])

    fig_balacc = px.bar(
        metrics_summary,
        x="Feature Set",
        y="Balanced Accuracy",
        color="Feature Set",
        title=f"Balanced Accuracy ({selected_endpoint.value})"
    )
    fig_balacc.update_layout(width=600, height=400, yaxis_range=[0,1])

    fig_aucpr = px.bar(
        metrics_summary,
        x="Feature Set",
        y="AUCPR",
        color="Feature Set",
        title=f"Precision-Recall AUC ({selected_endpoint.value})"
    )
    fig_aucpr.update_layout(width=600, height=400, yaxis_range=[0,1])

    # --- Display in tabs ---
    mo.md("## 📊 Model Performance Summary")
    mo.ui.tabs({
        "AUC": mo.ui.plotly(fig_auc),
        "Balanced Accuracy": mo.ui.plotly(fig_balacc),
        "AUCPR": mo.ui.plotly(fig_aucpr)
    })
    return


@app.cell
def _(mo):
    mo.md("""
    ### Step 6: Confusion Matrices

    Confusion matrices allow you to see the distribution of true positives, false positives, true negatives, and false negatives for each feature set.
    """)
    return


@app.cell
def _(confusion_matrix, go, np, pred_df):
    from plotly.subplots import make_subplots

    # Create subplot: 1 row, 2 columns
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["CellProfiler", "CellCount"]
    )

    for i, fs in enumerate(["CellProfiler", "CellCount"], start=1):
        subset = pred_df[pred_df['Feature Set'] == fs]

        # Combine all folds
        y_true = np.concatenate(subset['y_true'].values)
        y_pred = np.concatenate(subset['y_pred'].values)

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Add heatmap to subplot
        fig.add_trace(
            go.Heatmap(
                z=cm,
                x=['Pred 0', 'Pred 1'],
                y=['True 0', 'True 1'],
                colorscale='Blues',
                text=cm,
                texttemplate="%{text}",
                showscale=False
            ),
            row=1, col=i
        )

        # Axes titles
        fig.update_xaxes(title_text="Predicted Label", row=1, col=i)
        fig.update_yaxes(title_text="True Label", row=1, col=i)

    # Layout
    fig.update_layout(
        title="Confusion Matrices",
        width = 900,
        height = 400
    )

    fig
    return


@app.cell
def _(mo):
    mo.md("""
    ### Step 7: ROC Curves

    ROC curves show the tradeoff between sensitivity (True Positive Rate) and specificity (1 - False Positive Rate).
    A model with an AUC close to 1 performs very well; close to 0.5 is no better than random guessing.
    """)
    return


@app.cell
def _(np, pred_df):
    import pauc

    roc_objects_combined = []

    for feat_set in pred_df['Feature Set'].unique():
        feat_subset = pred_df[pred_df['Feature Set'] == feat_set]

        # Combine all folds
        all_y_true = np.concatenate(feat_subset['y_true'].values)
        all_y_proba = np.concatenate(feat_subset['y_proba'].values)

        roc_obj = pauc.ROC(
            y_true=all_y_true,
            y_score=all_y_proba,
            name=f"{feat_set} (All Folds)"
        )
        roc_objects_combined.append(roc_obj)
    return (roc_objects_combined,)


@app.cell
def _(roc_objects_combined):
    import plotly.graph_objects as go

    fig2 = go.Figure()

    for roc in roc_objects_combined:
        # roc.fpr, roc.tpr are arrays in pauc.ROC
        fig2.add_trace(
            go.Scatter(
                x=roc.fpr,
                y=roc.tpr,
                mode='lines',
                name=f"{roc.name} (AUC={roc.auc:.3f})"
            )
        )

    # Diagonal line for random classifier
    fig2.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            line=dict(dash='dash', color='gray'),
            showlegend=False
        )
    )

    fig2.update_layout(
        title="Combined ROC Curves per Feature Set",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        width=700,
        height=500
    )

    # Display in Marimo/Plotly
    fig2
    return (go,)


@app.cell
def _():
    return


@app.cell
def _(feature_sets, np, pd, pred_df):
    # Dict to store feature importances
    feature_importances = {}

    for featureset_name, feature_spaces in feature_sets.items():
        if(featureset_name == "CellProfiler"):
            importances_list = []

            # Loop over folds
            df_subset = pred_df[pred_df['Feature Set'] == featureset_name]
            for fold_idx, row in df_subset.iterrows():
                # Only Random Forest has feature_importances_
                if len(feature_spaces) > 1:  
                    model_test = row['best_model']  # You need to store trained model in your predictions df
                    importances = model_test.feature_importances_
                    importances_list.append(importances)
                else:
                    # For LogisticRegression with single feature
                    model_test = row['best_model']
                    coef = np.abs(model_test.coef_[0])
                    importances_list.append(coef)

            # Average importances across folds
            mean_importances = np.mean(importances_list, axis=0)
            feat_imp_df = pd.DataFrame({
                'Feature': feature_spaces,
                'Importance': mean_importances
            }).sort_values(by='Importance', ascending=False)

            # Store top 5
            feature_importances[featureset_name] = feat_imp_df.sort_values("Importance", ascending=False).head(10)
    return (feature_importances,)


@app.cell
def _(mo):
    mo.md("""
    ### Step 8: Feature Importance

    Random Forest models provide feature importance scores.
    Here we show the **top 10 features** contributing most to the prediction for each feature set.
    This helps interpret which Cell Painting features are most informative for the selected endpoint.
    """)
    return


@app.cell
def _(feature_importances, go, mo):
    # Prepare traces
    traces = []
    for feat, df_featureimportances in feature_importances.items():
        traces.append(
            go.Bar(
                y=df_featureimportances['Feature'],
                x=df_featureimportances['Importance'],
                name=feat,
                orientation='h'  # Horizontal bars for readability
            )
        )

    # Create figure
    fig_fi = go.Figure(data=traces)
    fig_fi.update_layout(
        title='Top 10 Features Contributing to Model Performance',
        xaxis_title='Importance',
        yaxis_title='Feature',
        yaxis=dict(autorange='reversed'),  # ⬅️ Reverse order (top = most important)
        barmode='group',
        width=800,
        height=500
    )

    # Display in Marimo
    mo.ui.plotly(fig_fi)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 🔬 Feature Distribution Analysis
    """)
    return


@app.cell
def _(feature_importances, mo):
    # Dropdown to select feature
    selected_feature = mo.ui.dropdown(
        options= feature_importances["CellProfiler"].Feature.to_list(),
        label="Select a feature to analyze:",
        value=feature_importances["CellProfiler"].Feature.to_list()[0]
    )
    return (selected_feature,)


@app.cell
def _(selected_feature):
    selected_feature
    return


@app.cell
def _(data, mo, px, selected_endpoint, selected_feature):
    from scipy import stats

    # Filter data for selected endpoint (remove NaN)
    analysis_data = data[~data[selected_endpoint.value].isna()].copy()

    # Get feature values for each class
    class_0 = analysis_data[analysis_data[selected_endpoint.value] == 0][selected_feature.value]
    class_1 = analysis_data[analysis_data[selected_endpoint.value] == 1][selected_feature.value]

    # Perform Mann-Whitney U test (non-parametric)
    statistic, p_value = stats.mannwhitneyu(class_0, class_1, alternative='two-sided')

    # Also perform t-test for comparison
    t_statistic, t_p_value = stats.ttest_ind(class_0, class_1)

    # Create violin plot with all points (swarm-like)
    fig_violin = px.violin(
        analysis_data,
        y=selected_feature.value,
        x=selected_endpoint.value,
        color=selected_endpoint.value,
        box=True,
        points="all",  # Changed from "outliers" to "all" to show swarm
        title=f"Distribution of {selected_feature.value} by {selected_endpoint.value}",
        labels={selected_endpoint.value: "Class (0=Inactive, 1=Active)"}
    )

    # Update layout
    fig_violin.update_layout(
        width=800,
        height=500,
        xaxis=dict(tickmode='array', tickvals=[0, 1], ticktext=['Inactive (0)', 'Active (1)'])
    )

    # Display plot
    plot_output = mo.ui.plotly(fig_violin)

    # Create statistical summary
    stats_summary = mo.md(f"""
    ### Statistical Test Results

    **Feature:** `{selected_feature.value}`  
    **Endpoint:** `{selected_endpoint.value}`

    | Metric | Value |
    |--------|-------|
    | **Mann-Whitney U statistic** | {statistic:.2f} |
    | **Mann-Whitney p-value** | {p_value:.4e} |
    | **T-test statistic** | {t_statistic:.2f} |
    | **T-test p-value** | {t_p_value:.4e} |
    | **Significance** | {'✅ Significant (p < 0.05)' if p_value < 0.05 else '❌ Not significant (p ≥ 0.05)'} |

    **Sample Sizes:**
    - Inactive (0): {len(class_0)} compounds (mean = {class_0.mean():.3f}, std = {class_0.std():.3f})
    - Active (1): {len(class_1)} compounds (mean = {class_1.mean():.3f}, std = {class_1.std():.3f})

    *Note: Mann-Whitney U test is used as the primary test (non-parametric, doesn't assume normal distribution)*
    """)

    mo.vstack([plot_output, stats_summary])
    return


@app.cell
def _(mo):
    mo.md("""
    ## ✅ Tutorial Complete

    You have now:
    - Explored Cell Painting data
    - Trained and tuned a Random Forest classifier
    - Evaluated it using multiple metrics
    - Visualized confusion matrices, ROC curves, and feature importances

    You can repeat this workflow for other assays or feature sets, and experiment with hyperparameters to improve performance.
    """)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
