# context.py - Global mining context variables
"""
Global context module for Bitcoin Solo Miner
Contains shared variables used across mining threads
"""

# Shutdown control
fShutdown = False

# Thread management
listfThreadRunning = [False] * 10

# Performance tracking
total_hashes_computed = 0
mining_time_per_block = []
nHeightDiff = {}

# Mining work variables (set by pool connection)
sub_details = None
extranonce1 = None  
extranonce2_size = 4
job_id = None
prevhash = None
coinb1 = None
coinb2 = None
merkle_branch = []
version = None
nbits = None
ntime = None
clean_jobs = None
updatedPrevHash = None
solved = False

# Mining statistics
blocks_found = 0
shares_submitted = 0
shares_accepted = 0
start_time = None
