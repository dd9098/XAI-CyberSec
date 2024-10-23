import os
import re
import torch
from tqdm import tqdm
import hashlib
import psycopg2.extras as ex  # Assuming ex is psycopg2.extras for execute_values

from config import *
from kairos_utils import *

# List of files to process
filelist = [
    'ta1-cadets-e3-official.json',
    'ta1-cadets-e3-official.json.1',
    'ta1-cadets-e3-official.json.2',
    'ta1-cadets-e3-official-1.json',
    'ta1-cadets-e3-official-1.json.1',
    'ta1-cadets-e3-official-1.json.2',
    'ta1-cadets-e3-official-1.json.3',
    'ta1-cadets-e3-official-1.json.4',
    'ta1-cadets-e3-official-2.json',
    'ta1-cadets-e3-official-2.json.1'
]

def stringtomd5(originstr):
    """
    Convert a string to its SHA256 hash representation.
    """
    originstr = originstr.encode("utf-8")
    signaturemd5 = hashlib.sha256()
    signaturemd5.update(originstr)
    return signaturemd5.hexdigest()

def store_netflow(file_path, cur, connect):
    """
    Parse NetFlowObject data from logs and store into the database.
    """
    netobjset = set()
    netobj2hash = {}
    for file in tqdm(filelist, desc="Processing NetFlow Files"):
        full_path = os.path.join(file_path, file)
        if not os.path.isfile(full_path):
            print(f"File {full_path} does not exist. Skipping...")
            continue
        try:
            with open(full_path, "r") as f:
                for line in f:
                    if "NetFlowObject" in line:
                        try:
                            res = re.findall(
                                r'NetFlowObject":{"uuid":"(.*?)"(.*?)"localAddress":"(.*?)","localPort":(.*?),"remoteAddress":"(.*?)","remotePort":(.*?),',
                                line)
                            if res:
                                res = res[0]
                                nodeid = res[0]
                                srcaddr = res[2]
                                srcport = res[3]
                                dstaddr = res[4]
                                dstport = res[5]

                                nodeproperty = srcaddr + "," + srcport + "," + dstaddr + "," + dstport
                                hashstr = stringtomd5(nodeproperty)
                                netobj2hash[nodeid] = [hashstr, nodeproperty]
                                netobj2hash[hashstr] = nodeid
                                netobjset.add(hashstr)
                        except Exception as e:
                            print(f"Error parsing line in {file}: {e}")
                            continue
        except Exception as e:
            print(f"Error reading file {file}: {e}")
            continue

    # Store data into database
    datalist = []
    for i in netobj2hash.keys():
        if len(i) != 64:  # Assuming UUIDs are not 64 characters
            datalist.append([i] + [netobj2hash[i][0]] + netobj2hash[i][1].split(","))
    if datalist:
        sql = '''INSERT INTO netflow_node_table
                 VALUES %s
              '''
        try:
            ex.execute_values(cur, sql, datalist, page_size=10000)
            connect.commit()
        except Exception as e:
            print(f"Error inserting netflow data into database: {e}")
            connect.rollback()
    else:
        print("No netflow data to insert.")

def store_subject(file_path, cur, connect):
    """
    Parse subject data from logs and store into the database.
    """
    scusess_count = 0
    fail_count = 0
    subject_obj2hash = {}
    for file in tqdm(filelist, desc="Processing Subject Files"):
        full_path = os.path.join(file_path, file)
        if not os.path.isfile(full_path):
            print(f"File {full_path} does not exist. Skipping...")
            continue
        try:
            with open(full_path, "r") as f:
                for line in f:
                    if "Event" in line:
                        try:
                            subject_uuid = re.findall(
                                r'"subject":{"com.bbn.tc.schema.avro.cdm18.UUID":"(.*?)"}(.*?)"exec":"(.*?)"', line)
                            if subject_uuid:
                                subject_obj2hash[subject_uuid[0][0]] = subject_uuid[0][-1]
                                scusess_count += 1
                            else:
                                # Handle cases where 'exec' is missing
                                subject_uuid = re.findall(
                                    r'"subject":{"com.bbn.tc.schema.avro.cdm18.UUID":"(.*?)"}', line)
                                if subject_uuid:
                                    subject_obj2hash[subject_uuid[0]] = "null"
                                    fail_count += 1
                        except Exception as e:
                            print(f"Error parsing subject in line: {e}")
                            continue
        except Exception as e:
            print(f"Error reading file {file}: {e}")
            continue

    # Store into database
    datalist = []
    for i in subject_obj2hash.keys():
        if len(i) != 64:
            datalist.append([i] + [stringtomd5(subject_obj2hash[i]), subject_obj2hash[i]])
    if datalist:
        sql = '''INSERT INTO subject_node_table
                 VALUES %s
              '''
        try:
            ex.execute_values(cur, sql, datalist, page_size=10000)
            connect.commit()
        except Exception as e:
            print(f"Error inserting subject data into database: {e}")
            connect.rollback()
    else:
        print("No subject data to insert.")

def store_file(file_path, cur, connect):
    """
    Parse file data from logs and store into the database.
    """
    file_node = set()
    for file in tqdm(filelist, desc="Processing File Nodes"):
        full_path = os.path.join(file_path, file)
        if not os.path.isfile(full_path):
            print(f"File {full_path} does not exist. Skipping...")
            continue
        try:
            with open(full_path, "r") as f:
                for line in f:
                    if "com.bbn.tc.schema.avro.cdm18.FileObject" in line:
                        try:
                            Object_uuid = re.findall(r'FileObject":{"uuid":"(.*?)",', line)
                            if Object_uuid:
                                file_node.add(Object_uuid[0])
                        except Exception as e:
                            print(f"Error parsing file node in line: {e}")
                            continue
        except Exception as e:
            print(f"Error reading file {file}: {e}")
            continue

    file_obj2hash = {}
    for file in tqdm(filelist, desc="Processing File Objects"):
        full_path = os.path.join(file_path, file)
        if not os.path.isfile(full_path):
            print(f"File {full_path} does not exist. Skipping...")
            continue
        try:
            with open(full_path, "r") as f:
                for line in f:
                    if '{"datum":{"com.bbn.tc.schema.avro.cdm18.Event"' in line:
                        try:
                            predicateObject_uuid = re.findall(r'"predicateObject":{"com.bbn.tc.schema.avro.cdm18.UUID":"(.*?)"}',
                                                              line)
                            if predicateObject_uuid and predicateObject_uuid[0] in file_node:
                                if '"predicateObjectPath":null,' not in line and '<unknown>' not in line:
                                    path_name = re.findall(r'"predicateObjectPath":{"string":"(.*?)"', line)
                                    if path_name:
                                        file_obj2hash[predicateObject_uuid[0]] = path_name
                        except Exception as e:
                            print(f"Error parsing file object in line: {e}")
                            continue
        except Exception as e:
            print(f"Error reading file {file}: {e}")
            continue

    datalist = []
    for i in file_obj2hash.keys():
        if len(i) != 64:
            datalist.append([i] + [stringtomd5(file_obj2hash[i][0]), file_obj2hash[i][0]])
    if datalist:
        sql = '''INSERT INTO file_node_table
                 VALUES %s
              '''
        try:
            ex.execute_values(cur, sql, datalist, page_size=10000)
            connect.commit()
        except Exception as e:
            print(f"Error inserting file data into database: {e}")
            connect.rollback()
    else:
        print("No file data to insert.")

def create_node_list(cur, connect):
    """
    Create a node list combining data from file, subject, and netflow nodes, and store into the database.
    """
    node_list = {}

    try:
        # File nodes
        sql = "SELECT * FROM file_node_table;"
        cur.execute(sql)
        records = cur.fetchall()
        for i in records:
            node_list[i[1]] = ["file", i[-1]]
        file_uuid2hash = {i[0]: i[1] for i in records}

        # Subject nodes
        sql = "SELECT * FROM subject_node_table;"
        cur.execute(sql)
        records = cur.fetchall()
        for i in records:
            node_list[i[1]] = ["subject", i[-1]]
        subject_uuid2hash = {i[0]: i[1] for i in records}

        # Netflow nodes
        sql = "SELECT * FROM netflow_node_table;"
        cur.execute(sql)
        records = cur.fetchall()
        for i in records:
            node_list[i[1]] = ["netflow", i[-2] + ":" + i[-1]]
        net_uuid2hash = {i[0]: i[1] for i in records}

        # Insert node list into database
        node_list_database = []
        node_index = 0
        for i in node_list:
            node_list_database.append([i] + node_list[i] + [node_index])
            node_index += 1

        if node_list_database:
            sql = '''INSERT INTO node2id
                     VALUES %s
                  '''
            ex.execute_values(cur, sql, node_list_database, page_size=10000)
            connect.commit()
        else:
            print("No nodes to insert into node2id.")

        # Create nodeid2msg mapping
        sql = "SELECT * FROM node2id ORDER BY index_id;"
        cur.execute(sql)
        rows = cur.fetchall()
        nodeid2msg = {}
        for i in rows:
            nodeid2msg[i[0]] = i[-1]
            nodeid2msg[i[-1]] = {i[1]: i[2]}

        return nodeid2msg, subject_uuid2hash, file_uuid2hash, net_uuid2hash

    except Exception as e:
        print(f"Error creating node list: {e}")
        connect.rollback()
        return {}, {}, {}, {}

def store_event(file_path, cur, connect, reverse, nodeid2msg, subject_uuid2hash, file_uuid2hash, net_uuid2hash):
    """
    Process events from logs and store into the database.
    """
    datalist = []
    for file in tqdm(filelist, desc="Processing Events"):
        full_path = os.path.join(file_path, file)
        if not os.path.isfile(full_path):
            print(f"File {full_path} does not exist. Skipping...")
            continue
        try:
            with open(full_path, "r") as f:
                for line in f:
                    if '{"datum":{"com.bbn.tc.schema.avro.cdm18.Event"' in line and "EVENT_FLOWS_TO" not in line:
                        try:
                            subject_uuid = re.findall(r'"subject":{"com.bbn.tc.schema.avro.cdm18.UUID":"(.*?)"}', line)
                            predicateObject_uuid = re.findall(r'"predicateObject":{"com.bbn.tc.schema.avro.cdm18.UUID":"(.*?)"}', line)
                            if subject_uuid and predicateObject_uuid:
                                subject_uuid = subject_uuid[0]
                                predicateObject_uuid = predicateObject_uuid[0]
                                if subject_uuid in subject_uuid2hash and (predicateObject_uuid in file_uuid2hash or predicateObject_uuid in net_uuid2hash):
                                    relation_type = re.findall(r'"type":"(.*?)"', line)[0]
                                    time_rec = re.findall(r'"timestampNanos":(.*?),', line)[0]
                                    time_rec = int(time_rec)
                                    subjectId = subject_uuid2hash[subject_uuid]
                                    if predicateObject_uuid in file_uuid2hash:
                                        objectId = file_uuid2hash[predicateObject_uuid]
                                    else:
                                        objectId = net_uuid2hash[predicateObject_uuid]
                                    if relation_type in reverse:
                                        datalist.append(
                                            [objectId, nodeid2msg.get(objectId, ''), relation_type, subjectId, nodeid2msg.get(subjectId, ''),
                                             time_rec])
                                    else:
                                        datalist.append(
                                            [subjectId, nodeid2msg.get(subjectId, ''), relation_type, objectId, nodeid2msg.get(objectId, ''),
                                             time_rec])
                        except Exception as e:
                            print(f"Error parsing event in line: {e}")
                            continue
        except Exception as e:
            print(f"Error reading file {file}: {e}")
            continue

    if datalist:
        sql = '''INSERT INTO event_table
                 VALUES %s
              '''
        try:
            ex.execute_values(cur, sql, datalist, page_size=10000)
            connect.commit()
        except Exception as e:
            print(f"Error inserting event data into database: {e}")
            connect.rollback()
    else:
        print("No event data to insert.")

if __name__ == "__main__":
    try:
        cur, connect = init_database_connection()
    except Exception as e:
        print(f"Error initializing database connection: {e}")
        exit(1)

    # Process netflow data
    print("Processing netflow data")
    try:
        store_netflow(file_path=RAW_DIR, cur=cur, connect=connect)
    except Exception as e:
        print(f"Error in processing netflow data: {e}")

    # Process subject data
    print("Processing subject data")
    try:
        store_subject(file_path=RAW_DIR, cur=cur, connect=connect)
    except Exception as e:
        print(f"Error in processing subject data: {e}")

    # Process file data
    print("Processing file data")
    try:
        store_file(file_path=RAW_DIR, cur=cur, connect=connect)
    except Exception as e:
        print(f"Error in processing file data: {e}")

    # Extract node list
    print("Extracting the node list")
    try:
        nodeid2msg, subject_uuid2hash, file_uuid2hash, net_uuid2hash = create_node_list(cur=cur, connect=connect)
    except Exception as e:
        print(f"Error in creating node list: {e}")
        nodeid2msg, subject_uuid2hash, file_uuid2hash, net_uuid2hash = {}, {}, {}, {}

    # Process events
    print("Processing the events")
    try:
        store_event(
            file_path=RAW_DIR,
            cur=cur,
            connect=connect,
            reverse=EDGE_REVERSED,
            nodeid2msg=nodeid2msg,
            subject_uuid2hash=subject_uuid2hash,
            file_uuid2hash=file_uuid2hash,
            net_uuid2hash=net_uuid2hash
        )
    except Exception as e:
        print(f"Error in processing events: {e}")
