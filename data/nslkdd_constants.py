"""
Constants for the NSL-KDD dataset.

This module defines constants used by the NSL-KDD dataset loader, including feature names,
attack types, and feature categories.
"""

# Constants for the dataset
BENIGN_LABEL = 0
MALICIOUS_LABEL = 1

# Main attack classes in the dataset
ATTACK_CLASSES = {
    'normal': 'normal',
    'dos': 'DoS',
    'probe': 'Probe',
    'r2l': 'R2L',
    'u2r': 'U2R'
}

# Specific attack types grouped by class
ATTACK_TYPES = {
    'normal': ['normal'],
    'dos': ['back', 'land', 'neptune', 'pod', 'smurf', 'teardrop'],
    'probe': ['ipsweep', 'nmap', 'portsweep', 'satan'],
    'r2l': ['ftp_write', 'guess_passwd', 'imap', 'multihop', 'phf', 'spy', 'warezclient', 'warezmaster'],
    'u2r': ['buffer_overflow', 'loadmodule', 'perl', 'rootkit']
}

# Protocol types
PROTOCOL_TYPES = ['tcp', 'udp', 'icmp']

# Flag values 
FLAG_TYPES = ['SF', 'S0', 'REJ', 'RSTR', 'RSTO', 'SH', 'S1', 'S2', 'S3', 'OTH', 'RSTOS0']

# Feature types by index
FEATURE_TYPES = {
    'categorical': [1, 2, 3, 41],  # 0-indexed (protocol_type, service, flag, class)
    'binary': [6, 11, 13, 19, 20, 21],  # 0-indexed (land, logged_in, etc.)
    'discrete': [7, 8, 14, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 42],  # 0-indexed
    'continuous': [0, 4, 5, 9, 10, 12, 15, 16, 17, 18]  # 0-indexed
}

# Complete feature names for the NSL-KDD dataset
FEATURE_NAMES = [
    'duration',
    'protocol_type',
    'service',
    'flag',
    'src_bytes',
    'dst_bytes',
    'land',
    'wrong_fragment',
    'urgent',
    'hot',
    'num_failed_logins',
    'logged_in',
    'num_compromised',
    'root_shell',
    'su_attempted',
    'num_root',
    'num_file_creations',
    'num_shells',
    'num_access_files',
    'num_outbound_cmds',
    'is_host_login',
    'is_guest_login',
    'count',
    'srv_count',
    'serror_rate',
    'srv_serror_rate',
    'rerror_rate',
    'srv_rerror_rate',
    'same_srv_rate',
    'diff_srv_rate',
    'srv_diff_host_rate',
    'dst_host_count',
    'dst_host_srv_count',
    'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate',
    'dst_host_srv_serror_rate',
    'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate',
    'class',
    'difficulty'
]

# Categorize features for easier understanding
FEATURE_CATEGORIES = {
    'intrinsic': list(range(9)),  # Features 0-8
    'content': list(range(9, 22)),  # Features 9-21
    'time_based': list(range(22, 31)),  # Features 22-30 
    'host_based': list(range(31, 41))  # Features 31-40
}

# Service types (there are many service types in the dataset)
SERVICE_TYPES = [
    'aol', 'auth', 'bgp', 'courier', 'csnet_ns', 'ctf', 'daytime', 'discard',
    'domain', 'domain_u', 'echo', 'eco_i', 'ecr_i', 'efs', 'exec', 'finger',
    'ftp', 'ftp_data', 'gopher', 'harvest', 'hostnames', 'http', 'http_2784',
    'http_443', 'http_8001', 'imap4', 'IRC', 'iso_tsap', 'klogin', 'kshell',
    'ldap', 'link', 'login', 'mtp', 'name', 'netbios_dgm', 'netbios_ns',
    'netbios_ssn', 'netstat', 'nnsp', 'nntp', 'ntp_u', 'other', 'pm_dump', 
    'pop_2', 'pop_3', 'printer', 'private', 'red_i', 'remote_job', 'rje',
    'shell', 'smtp', 'sql_net', 'ssh', 'sunrpc', 'supdup', 'systat',
    'telnet', 'tftp_u', 'tim_i', 'time', 'urh_i', 'urp_i', 'uucp',
    'uucp_path', 'vmnet', 'whois', 'X11', 'Z39_50'
] 