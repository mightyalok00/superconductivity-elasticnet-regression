from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet


def build_models(random_state=42):
    """Create consistent regression pipelines for fair comparison."""
    return {
        "Linear": Pipeline([("scale", StandardScaler()), ("model", LinearRegression())]),
        "Ridge": Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=1.0))]),
        "Lasso": Pipeline([("scale", StandardScaler()), ("model", Lasso(alpha=0.01, max_iter=20000))]),
        "ElasticNet": Pipeline([("scale", StandardScaler()), ("model", ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=20000, random_state=random_state))]),
    }
