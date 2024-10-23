import pytz
from time import mktime
from datetime import datetime
import time
import psycopg2
from psycopg2 import extras as ex
import os
import torch
import numpy as np
from torch_geometric.nn import TGNMemory, TransformerConv
from torch_geometric.nn.models.tgn import LastNeighborLoader, IdentityMessage, LastAggregator
from tqdm import tqdm
import xxhash
from config import *

def ns_time_to_datetime(ns):
    """
    Converts nanosecond timestamp to datetime string.

    :param ns: int nano timestamp
    :return: datetime string in format: 2013-10-10 23:40:00.000000000
    """
    try:
        dt = datetime.fromtimestamp(int(ns) // 1000000000)
        s = dt.strftime('%Y-%m-%d %H:%M:%S')
        s += '.' + str(int(int(ns) % 1000000000)).zfill(9)
        return s
    except Exception as e:
        print(f"Error in ns_time_to_datetime: {e}")
        return None

def ns_time_to_datetime_US(ns):
    """
    Converts nanosecond timestamp to datetime string in US Eastern timezone.

    :param ns: int nano timestamp
    :return: datetime string in format: 2013-10-10 23:40:00.000000000
    """
    try:
        tz = pytz.timezone('US/Eastern')
        dt = datetime.fromtimestamp(int(ns) // 1000000000, tz)
        s = dt.strftime('%Y-%m-%d %H:%M:%S')
        s += '.' + str(int(int(ns) % 1000000000)).zfill(9)
        return s
    except Exception as e:
        print(f"Error in ns_time_to_datetime_US: {e}")
        return None

def datetime_to_ns_time(date):
    """
    Converts a datetime string to nanosecond timestamp.

    :param date: str in format: %Y-%m-%d %H:%M:%S
    :return: int nano timestamp
    """
    try:
        time_array = time.strptime(date, "%Y-%m-%d %H:%M:%S")
        time_stamp = int(time.mktime(time_array)) * 1000000000
        return time_stamp
    except Exception as e:
        print(f"Error in datetime_to_ns_time: {e}")
        return None

def datetime_to_ns_time_US(date):
    """
    Converts a datetime string in US Eastern timezone to nanosecond timestamp.

    :param date: str in format: %Y-%m-%d %H:%M:%S
    :return: int nano timestamp
    """
    try:
        tz = pytz.timezone('US/Eastern')
        time_array = time.strptime(date, "%Y-%m-%d %H:%M:%S")
        dt = datetime.fromtimestamp(mktime(time_array))
        timestamp = tz.localize(dt).timestamp()
        return int(timestamp * 1000000000)
    except Exception as e:
        print(f"Error in datetime_to_ns_time_US: {e}")
        return None

def init_database_connection():
    """
    Initializes a connection to the database.

    :return: cursor, connection
    """
    try:
        if host is not None:
            connect = psycopg2.connect(database=database, host=host, user=user, password=password, port=port)
        else:
            connect = psycopg2.connect(database=database, user=user, password=password, port=port)
        cur = connect.cursor()
        return cur, connect
    except Exception as e:
        print(f"Error in init_database_connection: {e}")
        return None, None

def gen_nodeid2msg(cur):
    """
    Generates a mapping between node IDs and messages from the database.

    :param cur: Database cursor
    :return: Dictionary mapping node IDs to messages
    """
    try:
        sql = "SELECT * FROM node2id ORDER BY index_id;"
        cur.execute(sql)
        rows = cur.fetchall()
        nodeid2msg = {}
        for i in rows:
            nodeid2msg[i[0]] = i[-1]
            nodeid2msg[i[-1]] = {i[1]: i[2]}
        return nodeid2msg
    except Exception as e:
        print(f"Error in gen_nodeid2msg: {e}")
        return {}

def tensor_find(t, x):
    """
    Finds the index of the first occurrence of a value in a tensor.

    :param t: Input tensor
    :param x: Value to find
    :return: Index of the value
    """
    try:
        t_np = t.cpu().numpy()
        idx = np.argwhere(t_np == x)
        return idx[0][0] + 1 if len(idx) > 0 else -1
    except Exception as e:
        print(f"Error in tensor_find: {e}")
        return -1

def std(t):
    """Calculates the standard deviation of a list or array."""
    try:
        return np.std(np.array(t))
    except Exception as e:
        print(f"Error in std: {e}")
        return None

def var(t):
    """Calculates the variance of a list or array."""
    try:
        return np.var(np.array(t))
    except Exception as e:
        print(f"Error in var: {e}")
        return None

def mean(t):
    """Calculates the mean of a list or array."""
    try:
        return np.mean(np.array(t))
    except Exception as e:
        print(f"Error in mean: {e}")
        return None

def hashgen(l):
    """
    Generate a single hash value from a list.

    :param l: List of string values (e.g., properties of a node/edge)
    :return: Single hashed integer value
    """
    try:
        hasher = xxhash.xxh64()
        for e in l:
            hasher.update(e)
        return hasher.intdigest()
    except Exception as e:
        print(f"Error in hashgen: {e}")
        return None
