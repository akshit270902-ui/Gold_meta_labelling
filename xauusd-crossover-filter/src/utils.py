import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    data.columns = [c.lower() for c in data.columns]
    data['time'] = pd.to_datetime(data['time'])
    data = data.sort_values('time').reset_index(drop=True)
    return data
