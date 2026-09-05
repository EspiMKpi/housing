# Vietnam Housing: Processing and Model Rationale

## Processing

Each row is a property listing and `Price` is the regression target in billion VND. Duplicate listings are removed and rows without a target are excluded. The free-text `Address` is reduced to its final comma-separated component, `City`, because location is useful signal while the complete address would create extremely sparse, overly specific categories.

Numerical variables such as area, frontage, road access, floors, bedrooms, and bathrooms use median imputation and standardization. Categorical variables such as city, directions, legal status, and furniture state use most-frequent imputation and one-hot encoding. All transformations are fitted only on the training split inside a pipeline.

The notebook visualizes price distribution, area versus price, city coverage, error metrics, and a controlled with-City versus without-City experiment. Each visualization supports a decision about skew, signal, representation, or error.

## Algorithms

- Linear Regression is an interpretable baseline for an approximately additive relationship.
- Ridge Regression controls coefficient instability caused by correlated and one-hot encoded features.
- Gradient Boosting models nonlinear residual patterns through sequential trees.
- Random Forest averages diverse trees, making it robust to nonlinear feature interactions and mixed property effects.

MAE expresses typical error in billion VND, RMSE penalizes large pricing mistakes, and R2 summarizes explained variation. The selected model is the lowest-RMSE candidate, then persisted as `house_price_model.sav` with its preprocessing steps.
