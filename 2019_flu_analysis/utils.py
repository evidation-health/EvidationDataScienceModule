import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numpy as np
import os
import itertools

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.exceptions import NotFittedError
from linearmodels import PanelOLS


np.random.seed(42)


class PanelEventPlotter:
    """Plot sample means surrounding events, with confidence.
    This estimator uses fixed effects regression (e.g. the "within"
    transform) to estimate how a feature changes in the neighborhood
    of an event. The model specification regresses a feature of
    interest on a set of dummy variables, one included for each
    time period, less the first time period (in order to avoid
    perfect multicollinearity). This setup allows the analyst
    to estimate how a feature is changing in the neighborhood of an
    event of interest, while controlling for individual level fixed effects
    such as race, gender, etc. Additionally, day-of-week dummies can
    be specified in order to account for variability in activity features
    arising from day of week calendar effects.
    Parameters
    ----------
    entity_col : str, default 'user_id'
        Column name corresponding to the "entity" index level of the panel
    time_col : str, default 'date'
        Column name corresponding to the "time" index level of the panel.
        Assumed to be of type datetime.
    event_col : str, default 'event'
        Column corresponding to the per-entity event date over which the
        relative index will be computed. Assumed to be of type datetime.
    Attributes
    ----------
    regression_results_ : linearmodels.panel.results.PanelEffectsResults, default None
        Houses the full set of regression results post fitting.
    Examples
    --------
    TODO
    Notes
    -----
    It is recommended that the fit procedure is called with entity_effects set
    to True. Estimating the model without entity effects is equivalent to
    pooled OLS and can introduce significant bias into the estimated parameter.
    Setting cov_type = 'clustered' during estimation is typically best practice
    in this seting due to autocorrelation within panel blocks. While the covariance
    matrix specified is not robust to large values of rho (there are specialized
    covariance structures for this) the current implementation is somewhat limited.
    """
    def __init__(self, entity_col='user_id', time_col='date', event_col='event_date'):
        self.entity_col = entity_col
        self.time_col = time_col
        self.event_col = event_col
        self.regression_results_ = None
        self._event_relative_idx = None
        self._idx_coefs = None
        self._idx_cia = None
        self._depvar_label = None

    def fit(self, X, y, entity_effects=True, weekday_effects=True, cov_type='clustered'):
        """
        Parameters
        ----------
        X : Pandas DataFrame
            Panel DataFrame of entities observed at multiple points in time.
        y : str
            Column to be used as regression target.
        entity_effects : bool, default True
            If True, include entity fixed effects into the model. If False,
            the estimation procedure is equivalent to pooled OLS.
        weekday_effects : bool, default True
            If True, include a dummy for each day of the week. Due to the
            large variance in activity features between weekdays, for certain
            situations this is highly recommended.
        cov_type : str, default 'clustered'
            Covariance matrix structure. Must be one of 'clustered', 'robust'.
            Note if entity_effects is set to True, robust standard errors are
            no longer robust.
        Returns
        -------
        self.regression_results_ : linearmodels.panel.results.PanelEffectsResults
            Summary of estimation results.
        """
        self._depvar_label = ' '.join([w.capitalize() for w in y.split('_')])
        idx_cols = [self.entity_col, self.time_col]
        relative_idx = ((X[self.time_col] - X[self.event_col]) / dt.timedelta(days=1)).astype(int)

        dummies = onehot_integer_series(relative_idx)
        # Add in dummy variables for observation distance to event
        X = pd.concat([X[[self.entity_col, self.time_col, y]], dummies], axis=1)

        # Set our estimation target
        indvars = list(dummies.columns)

        if weekday_effects:
            X['day_of_week'] = X[self.time_col].dt.strftime('%A')
            indvars = indvars + ['day_of_week']

        X.set_index(idx_cols, inplace=True)
        X.sort_index(inplace=True)

        depvar = X[y]

        model = PanelOLS(dependent=depvar, exog=X[indvars], entity_effects=entity_effects)
        self.regression_results_ = model.fit(cov_type='clustered')

        # Extract point estimates
        coefs = self.regression_results_.params.reset_index()
        coefs = coefs[coefs['index'].str.contains('relative_idx')]
        coefs['index'] = coefs['index'].apply(self.parse_dummies)
        coefs.sort_values('index', inplace=True)
        self._idx_coefs = coefs.rename(columns={'index': 'relative_idx'}).set_index('relative_idx')

        # Extract integer index, we can just use the coef index since cis are the same indexing
        self._event_relative_idx = coefs['index'].values

        # Extract confidence intervals
        cis = self.regression_results_.conf_int().reset_index()
        cis = cis[cis['index'].str.contains('relative_idx')]
        cis['index'] = cis['index'].apply(self.parse_dummies)
        cis.sort_values('index', inplace=True)
        self._idx_cis = cis.rename(columns={'index': 'relative_idx'}).set_index('relative_idx')

        return self.regression_results_

    def plot(self):
        """Plot the estimated coefficients for the time effects features.
        """
        # Sufficient to check a single existence condition
        if self._idx_coefs is None:
            raise NotFittedError('You must estimate the regression by calling fit prior to plotting.')

        plt.figure(figsize=(15,8))
        plt.plot(self._idx_coefs, color='black')
        plt.axhline(y=0, linestyle='--', color='black')
        plt.fill_between(
            self._idx_cis.index.values,
            self._idx_cis.lower.values.reshape(-1),
            self._idx_cis.upper.values.reshape(-1),
            color='cyan',
            alpha=0.3
        )
        plt.title(self._depvar_label)
        plt.ylabel('Estimated Coefficient')
        plt.xlabel('Days since Event')

    @staticmethod
    def parse_dummies(s):
        """
        """
        s = int(s[:-1].split('[')[1].split('.')[0])
        return s

    
def onehot_integer_series(s, drop_idx=None):
    """One-hot encode an integer series.
    Parameters
    ----------
    s : Pandas.Series
        Integer valued series
    drop_idx : int or None, default None
        The index date to use as the dummy drop level.
        Passing None drops the first level of the ordered index.
    """
    dummy_str = s.astype(str)
    dummies = pd.get_dummies(dummy_str)
    cols = sorted(list(dummies.columns), key=float)

    dummies = dummies[cols]
    dummies.columns = ['relative_idx' + '[' + i + ']' for i in dummies.columns]

    if drop_idx is None:
        idx_min = 'relative_idx' + '[' + str(s.min()) + ']'
        dummies.drop(idx_min, axis=1, inplace=True)
    else:
        idx_min = 'relative_idx' + '[' + str(drop_idx) + ']'
        dummies.drop(idx_min, axis=1, inplace=True)
    return dummies


def event_window(event_date, window_size):
    """
    """
    event_dt = pd.to_datetime(event_date)
    begin_dt = event_dt - pd.Timedelta(days=window_size)
    end_dt = event_dt + pd.Timedelta(days=window_size)
    
    pre_window = pd.Series(pd.date_range(start=begin_dt, end=end_dt))
    post_window = pd.Series(pd.date_range(start=event_dt+pd.Timedelta(days=1), end=event_date))
    return pd.concat([pre_window, post_window])


def generate_shock_distribution(df, feature_mean, shock_mean, feature_std, shock_std, feature_name):
    """
    """
    f_i_mean = max(np.random.normal(feature_mean, feature_std, 1)[0], feature_mean*0.2)
    f_i_std = abs(np.random.normal(feature_std, max(0, 0.1*f_i_mean), 1)[0])
    df = df.copy()
    df['baseline_dist'] = np.random.normal(f_i_mean, f_i_std, 29)
    df['shock'] = np.random.normal(shock_mean, shock_std, 29)
    df['attenuation_weight'] = df.relative_idx.apply(lambda x: 1 / (1+abs(x**2)))+np.random.normal(0, 0.5, 29)
    df['attenuation_weight'] = df.attenuation_weight.rolling(5, win_type='triang', center=True, min_periods=1).mean()
    df[feature_name] = df.baseline_dist + (df.shock)*df.attenuation_weight
    df[feature_name] = df[feature_name].ffill().bfill().astype(int)

    return df[['date', feature_name]]


def generate_event_data():
    """Generates mock flu event data. Somewhat hacky, user be warned!
    """
    # Generate the date of event, distibuted normally around an "anchor date"
    anchor_date = pd.to_datetime('2019-02-13')
    event_dates = [anchor_date + pd.Timedelta(days=np.random.normal(0, 30)) for i in range(3000)]

    # Create a table that contains the reported ILI peak symptom date
    user_events = pd.DataFrame({'user_id': list(range(3000)), 'event_date': event_dates})
    user_events['event_date'] = user_events.event_date.dt.date
    user_events['event_date'] = pd.to_datetime(user_events.event_date)

    # Create a 29 day window indexed on the event date, for each user
    user_windows = pd.DataFrame(
        {'date': list(itertools.chain.from_iterable([event_window(d, 14) for d in user_events.event_date.values]))}
    )
    user_windows['user_id'] = list(itertools.chain.from_iterable([np.repeat(i, 29) for i in range(3000)]))
    user_windows = user_windows.merge(user_events, on='user_id')
    user_windows['event_date'] = pd.to_datetime(user_windows.event_date)
    user_windows['relative_idx'] = (user_windows.date - user_windows.event_date) / pd.Timedelta(days=1)

    # Generate user features
    walk_steps = (
        user_windows
        .groupby('user_id')
        .apply(lambda df: generate_shock_distribution(df, 10000, -2000, 2000, 1500, 'steps_sum'))
        .reset_index()
        .drop('level_1', axis=1)
    )
    walk_steps.loc[walk_steps.steps_sum<0, 'steps_sum'] = 0

    sleep_disturbances = (
        user_windows
        .groupby('user_id')
        .apply(lambda df: generate_shock_distribution(df, 0, 4, 1, 2, 'sleep_disturbances'))
        .reset_index()
        .drop('level_1', axis=1)
    )
    sleep_disturbances.loc[sleep_disturbances.sleep_disturbances<0, 'sleep_disturbances'] = 0

    resting_heart_rate = (
        user_windows
        .groupby('user_id')
        .apply(lambda df: generate_shock_distribution(df, 60, 5, 6, 2, 'resting_heart_rate'))
        .reset_index()
        .drop('level_1', axis=1)
    )
    resting_heart_rate.loc[resting_heart_rate.resting_heart_rate<38, 'resting_heart_rate'] = 38
    resting_heart_rate.loc[resting_heart_rate.resting_heart_rate>110, 'resting_heart_rate'] = 110
    
    features = (
        walk_steps
        .merge(sleep_disturbances, on=['user_id', 'date'])
        .merge(resting_heart_rate, on=['user_id', 'date'])
    )
    
    return features, user_events
