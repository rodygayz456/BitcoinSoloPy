# Enhanced Python Bitcoin Solo Miner
import requests
import socket
import threading
import json
import hashlib
import binascii
import logging
import random
import time
import traceback
import context as ctx
import psutil
import os
import sys
from datetime import datetime, timedelta
from signal import SIGINT, signal
from colorama import Back, Fore, Style, init
from tabulate import tabulate
from tqdm import tqdm
import statistics
import pickle

# Initialize colorama for cross-platform color support
init()

sock = None
best_difficulty = 0
best_hash = None
difficulty = 0
best_share_difficulty = float('inf')
best_share_hash = None
difficulty = 16

# Enhanced configuration class
class MinerConfig:
    def __init__(self):
        self.pools = [
            {'name': 'solo.ckpool.org', 'host': 'solo.ckpool.org', 'port': 3333},
            {'name': 'backup.pool.com', 'host': 'backup.pool.com', 'port': 4444}  # Add backup pools
        ]
        self.current_pool_index = 0
        self.auto_switch_pools = True
        self.reconnect_attempts = 5
        self.stats_update_interval = 30
        self.save_stats_interval = 300  # Save stats every 5 minutes
        self.target_temperature = 85  # CPU temperature threshold
        self.enable_temperature_throttling = True
        self.log_level = logging.INFO

# Enhanced statistics tracking
class MinerStats:
    def __init__(self):
        self.start_time = time.time()
        self.total_hashes = 0
        self.blocks_found = 0
        self.shares_submitted = 0
        self.shares_accepted = 0
        self.shares_rejected = 0
        self.best_difficulties = []
        self.hash_rates = []
        self.uptime_sessions = []
        self.hardware_stats = []
        self.pool_switches = 0
        self.reconnections = 0
        
    def get_average_hashrate(self, window_minutes=60):
        """Calculate average hash rate over specified time window"""
        current_time = time.time()
        cutoff_time = current_time - (window_minutes * 60)
        recent_rates = [rate for rate in self.hash_rates if rate[0] > cutoff_time]
        if recent_rates:
            return statistics.mean([rate[1] for rate in recent_rates])
        return 0
    
    def get_uptime(self):
        """Get total uptime in seconds"""
        return time.time() - self.start_time
    
    def save_to_file(self, filename="miner_stats.pkl"):
        """Save stats to file"""
        try:
            with open(filename, 'wb') as f:
                pickle.dump(self.__dict__, f)
        except Exception as e:
            print(f"Failed to save stats: {e}")
    
    def load_from_file(self, filename="miner_stats.pkl"):
        """Load stats from file"""
        try:
            if os.path.exists(filename):
                with open(filename, 'rb') as f:
                    data = pickle.load(f)
                    self.__dict__.update(data)
                return True
        except Exception as e:
            print(f"Failed to load stats: {e}")
        return False

# Global stats instance
miner_stats = MinerStats()
config = MinerConfig()

def show_enhanced_splash():
    """Enhanced splash screen with system info"""
    ascii_art = """
⠀⠀⠀⠀⠀⠀⠀⠀⣀⣤⣴⣶⣾⣿⣿⣿⣿⣷⣶⣦⣤⣀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⣠⣴⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣦⣄⠀⠀⠀⠀⠀
⠀⠀⠀⣠⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣄⠀⠀⠀
⠀⠀⣴⣿⣿⣿⣿⣿⣿⣿⠟⠿⠿⡿⠀⢰⣿⠁⢈⣿⣿⣿⣿⣿⣿⣿⣿⣦⠀⠀
⠀⣼⣿⣿⣿⣿⣿⣿⣿⣿⣤⣄⠀⠀⠀⠈⠉⠀⠸⠿⣿⣿⣿⣿⣿⣿⣿⣿⣧⠀
⢰⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡏⠀⠀⢠⣶⣶⣤⡀⠀⠈⢻⣿⣿⣿⣿⣿⣿⣿⡆
⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠃⠀⠀⠼⣿⣿⡿⠃⠀⠀⢸⣿⣿⣿⣿⣿⣿⣿⣷
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡟⠀⠀⢀⣀⣀⠀⠀⠀⠀⢴⣿⣿⣿⣿⣿⣿⣿⣿⣿
⢿⣿⣿⣿⣿⣿⣿⣿⢿⣿⠁⠀⠀⣼⣿⣿⣿⣦⠀⠀⠈⢻⣿⣿⣿⣿⣿⣿⣿⡿
⠸⣿⣿⣿⣿⣿⣿⣏⠀⠀⠀⠀⠀⠛⠛⠿⠟⠋⠀⠀⠀⣾⣿⣿⣿⣿⣿⣿⣿⠇
⠀⢻⣿⣿⣿⣿⣿⣿⣿⣿⠇⠀⣤⡄⠀⣀⣀⣀⣀⣠⣾⣿⣿⣿⣿⣿⣿⣿⡟⠀
⠀⠀⠻⣿⣿⣿⣿⣿⣿⣿⣄⣰⣿⠁⢀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠟⠀⠀
⠀⠀⠀⠙⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠋⠀⠀⠀
⠀⠀⠀⠀⠀⠙⠻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠟⠋⠀⠀⠀⠀⠀
       WE ARE ALL SATOSHI - ENHANCED MINER v2.0
              B I T C O I N
    """
    orange_text = '\033[38;5;202m'
    reset_color = '\033[0m'
    
    print(orange_text + ascii_art + reset_color)
    
    # System information
    cpu_count = psutil.cpu_count()
    memory = psutil.virtual_memory()
    print(f"{Fore.CYAN}System Info: {cpu_count} CPU cores, {memory.total // (1024**3)}GB RAM{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Python Version: {sys.version.split()[0]}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Enhanced Features: Auto-failover, Stats tracking, Temperature monitoring{Style.RESET_ALL}")

def get_system_info():
    """Get detailed system information"""
    info = {
        'cpu_percent': psutil.cpu_percent(interval=1),
        'memory_percent': psutil.virtual_memory().percent,
        'cpu_temp': get_cpu_temperature(),
        'cpu_freq': psutil.cpu_freq().current if psutil.cpu_freq() else 0
    }
    return info

def get_cpu_temperature():
    """Get CPU temperature (platform dependent)"""
    try:
        if hasattr(psutil, "sensors_temperatures"):
            temps = psutil.sensors_temperatures()
            if temps:
                for name, entries in temps.items():
                    for entry in entries:
                        if entry.current:
                            return entry.current
    except:
        pass
    return None

def should_throttle_mining():
    """Check if mining should be throttled due to high temperatures"""
    if not config.enable_temperature_throttling:
        return False
    
    temp = get_cpu_temperature()
    if temp and temp > config.target_temperature:
        return True
    return False

def display_enhanced_stats():
    """Display comprehensive mining statistics"""
    stats = miner_stats
    uptime = stats.get_uptime()
    avg_hashrate = stats.get_average_hashrate()
    system_info = get_system_info()
    
    # Create statistics table
    table_data = [
        ["Uptime", f"{uptime//3600:.0f}h {(uptime%3600)//60:.0f}m"],
        ["Total Hashes", f"{stats.total_hashes:,}"],
        ["Average Hash Rate", f"{avg_hashrate:.2f} H/s"],
        ["Blocks Found", stats.blocks_found],
        ["Shares Submitted", stats.shares_submitted],
        ["Share Accept Rate", f"{stats.shares_accepted/max(stats.shares_submitted,1)*100:.1f}%"],
        ["Best Difficulty", f"{max(stats.best_difficulties) if stats.best_difficulties else 0:.2f}"],
        ["Pool Switches", stats.pool_switches],
        ["Reconnections", stats.reconnections],
        ["CPU Usage", f"{system_info['cpu_percent']:.1f}%"],
        ["Memory Usage", f"{system_info['memory_percent']:.1f}%"],
        ["CPU Temperature", f"{system_info['cpu_temp']:.1f}°C" if system_info['cpu_temp'] else "N/A"],
    ]
    
    print(f"\n{Fore.CYAN}=== ENHANCED MINING STATISTICS ==={Style.RESET_ALL}")
    print(tabulate(table_data, headers=["Metric", "Value"], tablefmt="grid"))

def auto_save_stats():
    """Automatically save stats periodically"""
    def save_worker():
        while not ctx.fShutdown:
            time.sleep(config.save_stats_interval)
            miner_stats.save_to_file()
            
    thread = threading.Thread(target=save_worker, daemon=True)
    thread.start()

def get_pool_info():
    """Get current pool information"""
    current_pool = config.pools[config.current_pool_index]
    return current_pool

def switch_pool():
    """Switch to next available pool"""
    config.current_pool_index = (config.current_pool_index + 1) % len(config.pools)
    miner_stats.pool_switches += 1
    current_pool = get_pool_info()
    print(f"{Fore.YELLOW}Switching to pool: {current_pool['name']}{Style.RESET_ALL}")

def timer():
    return datetime.now().time()

# Mining Address **Change Me**
address = 'bc1qj5j0a39g52r9a5rah9uxa7wyhwh97s4v0t57s5'  # Example address - change this!
print(Back.BLUE, Fore.WHITE, 'SOLO ADDRESS:', Fore.GREEN, str(address), Style.RESET_ALL)

def handler(signal_received, frame):
    ctx.fShutdown = True
    miner_stats.save_to_file()  # Save stats on shutdown
    print(Fore.MAGENTA, '[', timer(), ']', Fore.YELLOW, 'Shutting down gracefully... Stats saved.')

# Enhanced logging setup
def setup_logging():
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=config.log_level,
        format=log_format,
        handlers=[
            logging.FileHandler('enhanced_miner.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("enhanced_miner")

logger = setup_logging()

def logg(msg):
    logger.info(msg)

def get_current_block_height():
    """Get current block height with fallback sources"""
    sources = [
        'https://blockchain.info/latestblock',
        'https://blockstream.info/api/blocks/tip/height',
        'https://mempool.space/api/blocks/tip/height'
    ]
    
    for source in sources:
        try:
            r = requests.get(source, timeout=10)
            if 'blockchain.info' in source:
                return int(r.json()['height'])
            else:
                return int(r.text)
        except Exception as e:
            logger.warning(f"Failed to get block height from {source}: {e}")
    
    logger.error("All block height sources failed")
    return None

def check_for_shutdown(t):
    n = t.n
    if ctx.fShutdown:
        if n != -1:
            ctx.listfThreadRunning[n] = False
            t.exit = True

class ExitedThread(threading.Thread):
    def __init__(self, arg, n):
        super(ExitedThread, self).__init__()
        self.exit = False
        self.arg = arg
        self.n = n

    def run(self):
        self.thread_handler(self.arg, self.n)

    def thread_handler(self, arg, n):
        while True:
            check_for_shutdown(self)
            if self.exit:
                break
            ctx.listfThreadRunning[n] = True
            try:
                self.thread_handler2(arg)
            except Exception as e:
                logg("ThreadHandler()")
                print(Fore.MAGENTA, '[', timer(), ']', Fore.WHITE, 'ThreadHandler()')
                logg(str(e))
                print(Fore.GREEN, str(e))
                time.sleep(5)  # Wait before retrying
            ctx.listfThreadRunning[n] = False
            time.sleep(2)

    def thread_handler2(self, arg):
        raise NotImplementedError("must implement this function")

    def check_self_shutdown(self):
        check_for_shutdown(self)

    def try_exit(self):
        self.exit = True
        ctx.listfThreadRunning[self.n] = False

def enhanced_bitcoin_miner(t, restarted=False):
    """Enhanced mining function with better metrics and error handling"""
    global best_share_difficulty, best_share_hash, sock
    
    start_time = time.time()
    total_hashes = 0
    last_stats_update = time.time()
    
    if restarted:
        logg('\n[*] Enhanced Bitcoin Miner restarted')
        print(Fore.MAGENTA, '[', timer(), ']', Fore.YELLOW, 'Enhanced Solo Miner Active')
        print(Fore.MAGENTA, '[', timer(), ']', Fore.BLUE, '[*] Enhanced Bitcoin Miner Restarted')

    # Initialize variables
    share_difficulty = 0
    difficulty = 0
    best_difficulty = 0

    try:
        target = (ctx.nbits[2:] + '00' * (int(ctx.nbits[:2], 16) - 3)).zfill(64)
        extranonce2 = hex(random.randint(0, 2**32 - 1))[2:].zfill(2 * ctx.extranonce2_size)
        
        print(Fore.YELLOW, '[*] Target:', Fore.GREEN, '[', target, ']')
        print(Fore.YELLOW, '[*] Extranonce2:', Fore.GREEN, '[', extranonce2, ']')

        coinbase = ctx.coinb1 + ctx.extranonce1 + extranonce2 + ctx.coinb2
        coinbase_hash_bin = hashlib.sha256(hashlib.sha256(binascii.unhexlify(coinbase)).digest()).digest()
        print(Fore.YELLOW, '[*] Coinbase Hash:', Fore.GREEN, '[', coinbase, ']')

        merkle_root = coinbase_hash_bin
        for h in ctx.merkle_branch:
            merkle_root = hashlib.sha256(hashlib.sha256(merkle_root + binascii.unhexlify(h)).digest()).digest()

        merkle_root = binascii.hexlify(merkle_root).decode()
        print(Fore.YELLOW, '[*] Merkle Root:', Fore.YELLOW, '[', merkle_root, ']')

        merkle_root = ''.join([merkle_root[i] + merkle_root[i + 1] for i in range(0, len(merkle_root), 2)][::-1])
        work_on = get_current_block_height()
        
        if work_on is None:
            print(Fore.RED, "Failed to get block height, using fallback")
            work_on = 800000  # Fallback value
            
        ctx.nHeightDiff[work_on + 1] = 0

        _diff = int("00000000FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF", 16)
        print(Fore.YELLOW, '[*] Diff:', Fore.YELLOW, '[', int(_diff), ']')
        
        logg('[*] Working to solve block at block height {}'.format(work_on + 1))
        print(Fore.MAGENTA, '[', timer(), ']', Fore.YELLOW, '[*] Working to solve block at ', Fore.GREEN, 'height {}'.format(work_on + 1))

        while True:
            t.check_self_shutdown()
            if t.exit:
                break

            # Temperature throttling
            if should_throttle_mining():
                print(f"{Fore.RED}High CPU temperature detected, throttling...{Style.RESET_ALL}")
                time.sleep(2)
                continue

            if ctx.prevhash != ctx.updatedPrevHash:
                logg('[*] NEW BLOCK {} DETECTED ON NETWORK'.format(ctx.prevhash))
                print(Fore.YELLOW, '[', timer(), ']', Fore.MAGENTA, '[*] New block {} detected on', Fore.BLUE, ' network '.format(ctx.prevhash))
                logg('[*] Best difficulty previous block {} was {}'.format(work_on + 1, ctx.nHeightDiff[work_on + 1]))
                print(Fore.MAGENTA, '[', timer(), ']', Fore.GREEN, '[*] Best Diff Trying Block', Fore.YELLOW, ' {} ', Fore.BLUE, 'was {}'.format(work_on + 1, ctx.nHeightDiff[work_on + 1]))
                ctx.updatedPrevHash = ctx.prevhash
                enhanced_bitcoin_miner(t, restarted=True)
                print(Back.YELLOW, Fore.MAGENTA, '[', timer(), ']', Fore.BLUE, 'NEW BLOCK DETECTED - RESTARTING MINER...', Style.RESET_ALL)
                continue

            nonce = hex(random.randint(0, 2**32 - 1))[2:].zfill(8)
            blockheader = ctx.version + ctx.prevhash + merkle_root + ctx.ntime + ctx.nbits + nonce + '000000800000000000000000000000000000000000000000000000000000000000000000000000000000000080020000'
            hash_result = hashlib.sha256(hashlib.sha256(binascii.unhexlify(blockheader)).digest()).digest()
            hash_result = binascii.hexlify(hash_result).decode()

            # Update counters
            total_hashes += 1
            miner_stats.total_hashes += 1

            target_difficulty = '0000000000000000FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF'
            this_hash = int(hash_result, 16)

            if this_hash <= int(target_difficulty, 16):
                logg(f'[*] New hash: {hash_result} for block {work_on + 1}')
                print(Fore.MAGENTA, '[', timer(), ']', Fore.GREEN, f'[*] New hash: {hash_result} for block', Fore.YELLOW, work_on + 1)

            difficulty = _diff / this_hash

            if difficulty > best_difficulty:
                best_difficulty = difficulty
                best_hash = hash_result
                miner_stats.best_difficulties.append(best_difficulty)
                logg(f'[BEST HASH UPDATE] New best hash: {best_hash} with difficulty: {best_difficulty}')
                print(f'[BEST HASH UPDATE] New best hash: {best_hash} with difficulty: {best_difficulty}')

            if ctx.nHeightDiff[work_on + 1] < difficulty:
                ctx.nHeightDiff[work_on + 1] = difficulty

            # Update hash rate tracking
            current_time = time.time()
            elapsed_time = current_time - start_time
            hash_rate = total_hashes / elapsed_time if elapsed_time > 0 else 0
            
            # Store hash rate for statistics
            miner_stats.hash_rates.append((current_time, hash_rate))
            
            # Clean old hash rate data (keep last 24 hours)
            cutoff_time = current_time - 86400
            miner_stats.hash_rates = [(t, r) for t, r in miner_stats.hash_rates if t > cutoff_time]

            # Display stats periodically
            if current_time - last_stats_update > config.stats_update_interval:
                display_enhanced_stats()
                last_stats_update = current_time

            print(f"\rH/s: {hash_rate:.2f} | Total: {total_hashes:,} | Best: {best_difficulty:.2f} | Temp: {get_cpu_temperature() or 'N/A'}°C", end='', flush=True)

            if hash_result < target:
                miner_stats.blocks_found += 1
                logg('[*] BLOCK FOUND for block {}.'.format(work_on + 1))
                print(f"\n{Fore.GREEN}{'='*60}")
                print(f"{Fore.GREEN}🎉 BLOCK FOUND! BLOCK {work_on + 1} 🎉")
                print(f"{Fore.GREEN}{'='*60}{Style.RESET_ALL}")
                print(Fore.YELLOW, '[*] Block hash: {}'.format(hash_result))
                print(Fore.YELLOW, '[*] Nonce Value: {}'.format(nonce))
                
                # Submit the block
                payload = bytes('{"params": ["' + address + '", "' + ctx.job_id + '", "' + extranonce2 + '", "' + ctx.ntime + '", "' + nonce + '"], "id": 1, "method": "mining.submit"}\n', 'utf-8')
                
                try:
                    sock.sendall(payload)
                    ret = sock.recv(1024)
                    logg('[*] Pool response: {}'.format(ret))
                    print(Fore.GREEN, '[*] Pool Response:', Fore.CYAN, ' {}'.format(ret))
                    
                    # Play celebration
                    show_block_found_celebration()
                    
                except Exception as e:
                    print(f"{Fore.RED}Error submitting block: {e}{Style.RESET_ALL}")
                
                return True

            if difficulty >= 16:
                share_difficulty = _diff / this_hash
                
                if share_difficulty < best_share_difficulty:
                    best_share_difficulty = share_difficulty
                    best_share_hash = hash_result
                    logg(f'[BEST SHARE UPDATE] New best share hash: {best_share_hash} with difficulty: {best_share_difficulty}')

                # Submit share
                try:
                    share_payload = {
                        "params": [address, ctx.job_id, extranonce2, ctx.ntime, nonce],
                        "id": 1,
                        "method": "mining.submit"
                    }
                    
                    share_payload = json.dumps(share_payload) + '\n'
                    sock.sendall(share_payload.encode())
                    response = sock.recv(1024).decode()
                    
                    miner_stats.shares_submitted += 1
                    
                    if '"result":true' in response.lower():
                        miner_stats.shares_accepted += 1
                    else:
                        miner_stats.shares_rejected += 1
                    
                    logg('[*] Share submitted - Response: {}'.format(response))
                    
                except Exception as e:
                    print(f"{Fore.RED}Error submitting share: {e}{Style.RESET_ALL}")

    except Exception as e:
        logg(f"Mining error: {e}")
        print(f"{Fore.RED}Mining error: {e}{Style.RESET_ALL}")
        traceback.print_exc()
        return False

def show_block_found_celebration():
    """Enhanced block found celebration"""
    celebration_art = """
    🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉
    🎉                                     🎉
    🎉         BLOCK FOUND!!!!            🎉
    🎉       CHECK YOUR WALLET!           🎉
    🎉                                     🎉
    🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉
    """
    
    print(f"{Fore.GREEN}{celebration_art}{Style.RESET_ALL}")
    
    # Flash the terminal (if supported)
    for _ in range(3):
        print(f"{Back.GREEN}{'='*60}{Style.RESET_ALL}")
        time.sleep(0.2)
        print(f"{Back.RED}{'='*60}{Style.RESET_ALL}")
        time.sleep(0.2)

def enhanced_block_listener(t):
    """Enhanced block listener with auto-reconnection and pool switching"""
    global sock
    
    reconnect_count = 0
    current_pool = get_pool_info()
    
    while reconnect_count < config.reconnect_attempts:
        try:
            # Connect to pool
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(30)  # 30 second timeout
            
            print(f"{Fore.CYAN}Connecting to {current_pool['name']} ({current_pool['host']}:{current_pool['port']}){Style.RESET_ALL}")
            sock.connect((current_pool['host'], current_pool['port']))
            
            # Send subscribe message
            sock.sendall(b'{"id": 1, "method": "mining.subscribe", "params": []}\n')
            lines = sock.recv(1024).decode().split('\n')
            response = json.loads(lines[0])
            ctx.sub_details, ctx.extranonce1, ctx.extranonce2_size = response['result']
            
            # Send authorize message
            sock.sendall(b'{"params": ["' + address.encode() + b'", "x"], "id": 2, "method": "mining.authorize"}\n')
            response = b''
            while response.count(b'\n') < 4 and not (b'mining.notify' in response):
                response += sock.recv(1024)
            
            print(f"{Fore.GREEN}Successfully connected to {current_pool['name']}{Style.RESET_ALL}")
            
            # Parse initial work
            responses = [json.loads(res) for res in response.decode().split('\n') if len(res.strip()) > 0 and 'mining.notify' in res]
            ctx.job_id, ctx.prevhash, ctx.coinb1, ctx.coinb2, ctx.merkle_branch, ctx.version, ctx.nbits, ctx.ntime, ctx.clean_jobs = responses[0]['params']
            ctx.updatedPrevHash = ctx.prevhash
            
            reconnect_count = 0  # Reset counter on successful connection
            
            # Main listening loop
            while True:
                t.check_self_shutdown()
                if t.exit:
                    break

                try:
                    response = b''
                    sock.settimeout(60)  # 60 second timeout for receiving data
                    while response.count(b'\n') < 4 and not (b'mining.notify' in response):
                        data = sock.recv(1024)
                        if not data:
                            raise ConnectionError("No data received")
                        response += data

                    responses = [json.loads(res) for res in response.decode().split('\n') if len(res.strip()) > 0 and 'mining.notify' in res]

                    if responses and responses[0]['params'][1] != ctx.prevhash:
                        # New block detected
                        ctx.job_id, ctx.prevhash, ctx.coinb1, ctx.coinb2, ctx.merkle_branch, ctx.version, ctx.nbits, ctx.ntime, ctx.clean_jobs = responses[0]['params']
                        
                        show_enhanced_splash()
                        
                        logger.info(f"New Work Received from Pool {current_pool['name']}:\n"
                                   f"Job ID: {ctx.job_id}\n"
                                   f"Previous Block Hash: {ctx.prevhash}\n"
                                   f"Coinbase 1: {ctx.coinb1}\n"
                                   f"Coinbase 2: {ctx.coinb2}\n"
                                   f"Merkle Branch: {ctx.merkle_branch}\n"
                                   f"Version: {ctx.version}\n"
                                   f"nBits: {ctx.nbits}\n"
                                   f"nTime: {ctx.ntime}\n"
                                   f"Clean Jobs: {ctx.clean_jobs}")

                except socket.timeout:
                    print(f"{Fore.YELLOW}Socket timeout, checking connection...{Style.RESET_ALL}")
                    continue
                except Exception as e:
                    print(f"{Fore.RED}Error receiving data: {e}{Style.RESET_ALL}")
                    raise

        except Exception as e:
            reconnect_count += 1
            miner_stats.reconnections += 1
            
            print(f"{Fore.RED}Connection failed: {e}{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}Reconnection attempt {reconnect_count}/{config.reconnect_attempts}{Style.RESET_ALL}")
            
            if sock:
                sock.close()
            
            if reconnect_count >= config.reconnect_attempts and config.auto_switch_pools:
                switch_pool()
                current_pool = get_pool_info()
                reconnect_count = 0
            
            time.sleep(5)  # Wait before retrying
    
    print(f"{Fore.RED}Max reconnection attempts reached. Exiting.{Style.RESET_ALL}")

class EnhancedCoinMinerThread(ExitedThread):
    def __init__(self, arg=None):
        super(EnhancedCoinMinerThread, self).__init__(arg, n=0)

    def thread_handler2(self, arg):
        self.thread_bitcoin_miner(arg)

    def thread_bitcoin_miner(self, arg):
        ctx.listfThreadRunning[self.n] = True
        check_for_shutdown(self)
        try:
            ret = enhanced_bitcoin_miner(self)
            logg(f"[*] Enhanced Miner returned {ret}")
            print(f"{Fore.LIGHTCYAN_EX}[*] Enhanced Miner returned {ret}")
        except Exception as e:
            logg("[*] Enhanced Miner()")
            print(f"{Fore.RED}[*] Enhanced Miner Error: {e}{Style.RESET_ALL}")
            traceback.print_exc()
        ctx.listfThreadRunning[self.n] = False

class EnhancedSubscribeThread(ExitedThread):
    def __init__(self, arg=None):
        super(EnhancedSubscribeThread, self).__init__(arg, n=1)

    def thread_handler2(self, arg):
        self.thread_new_block(arg)

    def thread_new_block(self, arg):
        ctx.listfThreadRunning[self.n] = True
        check_for_shutdown(self)
        try:
            ret = enhanced_block_listener(self)
        except Exception as e:
            logg("[*] Enhanced Subscribe thread()")
            print(f"{Fore.RED}[*] Enhanced Subscribe thread error: {e}{Style.RESET_ALL}")
            traceback.print_exc()
        ctx.listfThreadRunning[self.n] = False

class StatsDisplayThread(threading.Thread):
    """Dedicated thread for displaying statistics"""
    def __init__(self):
        super().__init__(daemon=True)
        self.running = True

    def run(self):
        while self.running and not ctx.fShutdown:
            time.sleep(config.stats_update_interval)
            if not ctx.fShutdown:
                # Clear screen for better display (optional)
                os.system('clear' if os.name == 'posix' else 'cls')
                
                # Show enhanced splash
                show_enhanced_splash()
                
                # Show mining address
                print(Back.BLUE, Fore.WHITE, 'SOLO ADDRESS:', Fore.GREEN, str(address), Style.RESET_ALL)
                
                print("\n" + "="*60)
                display_enhanced_stats()
                print("="*60)
                
                # Show current pool info
                current_pool = get_pool_info()
                print(f"{Fore.CYAN}Current Pool: {current_pool['name']} ({current_pool['host']}:{current_pool['port']}){Style.RESET_ALL}")
                
                # Show real-time mining status
                if hasattr(ctx, 'job_id') and ctx.job_id:
                    print(f"{Fore.GREEN}Status: ✓ MINING ACTIVE - Job ID: {ctx.job_id[:16]}...{Style.RESET_ALL}")
                else:
                    print(f"{Fore.YELLOW}Status: Connecting to pool...{Style.RESET_ALL}")

    def stop(self):
        self.running = False

class HealthMonitorThread(threading.Thread):
    """Monitor system health and mining performance"""
    def __init__(self):
        super().__init__(daemon=True)
        self.running = True

    def run(self):
        while self.running and not ctx.fShutdown:
            try:
                system_info = get_system_info()
                miner_stats.hardware_stats.append({
                    'timestamp': time.time(),
                    'cpu_percent': system_info['cpu_percent'],
                    'memory_percent': system_info['memory_percent'],
                    'cpu_temp': system_info['cpu_temp'],
                    'cpu_freq': system_info['cpu_freq']
                })
                
                # Keep only last 1000 entries
                if len(miner_stats.hardware_stats) > 1000:
                    miner_stats.hardware_stats = miner_stats.hardware_stats[-1000:]
                
                # Alert for high temperatures
                if system_info['cpu_temp'] and system_info['cpu_temp'] > config.target_temperature + 10:
                    print(f"\n{Fore.RED}⚠️  WARNING: High CPU temperature detected: {system_info['cpu_temp']:.1f}°C{Style.RESET_ALL}")
                
                # Alert for high CPU usage
                if system_info['cpu_percent'] > 95:
                    print(f"\n{Fore.YELLOW}⚠️  High CPU usage: {system_info['cpu_percent']:.1f}%{Style.RESET_ALL}")
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                time.sleep(60)

    def stop(self):
        self.running = False

def create_web_dashboard():
    """Create a simple web dashboard for monitoring (optional enhancement)"""
    try:
        from flask import Flask, render_template_string
        
        app = Flask(__name__)
        
        @app.route('/')
        def dashboard():
            stats = miner_stats
            uptime = stats.get_uptime()
            avg_hashrate = stats.get_average_hashrate()
            system_info = get_system_info()
            
            html_template = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Bitcoin Solo Miner Dashboard</title>
                <meta http-equiv="refresh" content="30">
                <style>
                    body { font-family: Arial; background: #1a1a1a; color: #fff; margin: 20px; }
                    .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
                    .card { background: #333; padding: 20px; border-radius: 10px; border: 1px solid #555; }
                    .value { font-size: 2em; color: #4CAF50; }
                    .label { color: #ccc; }
                    .warning { color: #ff9800; }
                    .danger { color: #f44336; }
                </style>
            </head>
            <body>
                <h1>🚀 Enhanced Bitcoin Solo Miner Dashboard</h1>
                <div class="stats">
                    <div class="card">
                        <div class="label">Uptime</div>
                        <div class="value">{{uptime_hours}}h {{uptime_mins}}m</div>
                    </div>
                    <div class="card">
                        <div class="label">Hash Rate</div>
                        <div class="value">{{hash_rate}} H/s</div>
                    </div>
                    <div class="card">
                        <div class="label">Total Hashes</div>
                        <div class="value">{{total_hashes:,}}</div>
                    </div>
                    <div class="card">
                        <div class="label">Blocks Found</div>
                        <div class="value">{{blocks_found}}</div>
                    </div>
                    <div class="card">
                        <div class="label">CPU Temperature</div>
                        <div class="value {{temp_class}}">{{cpu_temp}}°C</div>
                    </div>
                    <div class="card">
                        <div class="label">CPU Usage</div>
                        <div class="value">{{cpu_percent}}%</div>
                    </div>
                    <div class="card">
                        <div class="label">Pool</div>
                        <div class="value">{{current_pool}}</div>
                    </div>
                    <div class="card">
                        <div class="label">Share Accept Rate</div>
                        <div class="value">{{accept_rate}}%</div>
                    </div>
                </div>
            </body>
            </html>
            """
            
            temp_class = ""
            if system_info['cpu_temp']:
                if system_info['cpu_temp'] > config.target_temperature:
                    temp_class = "danger"
                elif system_info['cpu_temp'] > config.target_temperature - 10:
                    temp_class = "warning"
            
            return render_template_string(html_template,
                uptime_hours=int(uptime//3600),
                uptime_mins=int((uptime%3600)//60),
                hash_rate=f"{avg_hashrate:.2f}",
                total_hashes=stats.total_hashes,
                blocks_found=stats.blocks_found,
                cpu_temp=f"{system_info['cpu_temp']:.1f}" if system_info['cpu_temp'] else "N/A",
                temp_class=temp_class,
                cpu_percent=f"{system_info['cpu_percent']:.1f}",
                current_pool=get_pool_info()['name'],
                accept_rate=f"{stats.shares_accepted/max(stats.shares_submitted,1)*100:.1f}"
            )
        
        # Run in background thread
        def run_dashboard():
            app.run(host='0.0.0.0', port=8080, debug=False, use_reloader=False)
        
        dashboard_thread = threading.Thread(target=run_dashboard, daemon=True)
        dashboard_thread.start()
        
        print(f"{Fore.CYAN}Web dashboard started at http://localhost:8080{Style.RESET_ALL}")
        
    except ImportError:
        print(f"{Fore.YELLOW}Flask not installed. Web dashboard disabled.{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Install with: pip install flask{Style.RESET_ALL}")

def enhanced_start_mining():
    """Enhanced mining startup with all features"""
    print(f"{Fore.CYAN}🚀 Starting Enhanced Bitcoin Solo Miner v2.0{Style.RESET_ALL}")
    
    # Load previous stats
    if miner_stats.load_from_file():
        print(f"{Fore.GREEN}Previous mining stats loaded{Style.RESET_ALL}")
    
    # Start auto-save thread
    auto_save_stats()
    
    # Start health monitoring
    health_monitor = HealthMonitorThread()
    health_monitor.start()
    
    # Start stats display thread
    stats_display = StatsDisplayThread()
    stats_display.start()
    
    # Start web dashboard (optional) - DISABLED due to errors
    # create_web_dashboard()  # Uncomment if you want to try the web dashboard
    print(f"{Fore.YELLOW}Web dashboard disabled - using terminal display only{Style.RESET_ALL}")
    
    # Start mining threads
    subscribe_t = EnhancedSubscribeThread(None)
    subscribe_t.start()
    logg("[*] Enhanced Subscribe thread started.")
    print(f"{Fore.GREEN}[*] Enhanced Subscribe thread started.{Style.RESET_ALL}")

    time.sleep(2)

    miner_t = EnhancedCoinMinerThread(None)
    miner_t.start()
    logg("[*] Enhanced Bitcoin Solo Miner Started")
    print(f"{Fore.GREEN}🚀 ENHANCED SOLO MINER STARTED 🚀{Style.RESET_ALL}")
    print(f"{Fore.BLUE}{'='*60}")
    print(f"{Fore.YELLOW}IN SATOSHI WE TRUST - ENHANCED EDITION")
    print(f"{Fore.GREEN}DO NOT TRUST, VERIFY - WITH BETTER STATS")
    print(f"{Fore.BLUE}{'='*60}{Style.RESET_ALL}")
    
    # Display initial system info
    display_enhanced_stats()

def show_help():
    """Display help information"""
    help_text = f"""
{Fore.CYAN}Enhanced Bitcoin Solo Miner v2.0 - Help{Style.RESET_ALL}

{Fore.YELLOW}New Features:{Style.RESET_ALL}
• Advanced statistics tracking with persistent storage
• Automatic pool failover and reconnection
• CPU temperature monitoring with throttling
• Web dashboard (if Flask is installed)
• Enhanced error handling and logging
• Real-time performance metrics
• Hardware health monitoring

{Fore.YELLOW}Configuration:{Style.RESET_ALL}
• Edit the 'address' variable to set your Bitcoin address
• Modify MinerConfig class to adjust settings
• Add backup pools in the pools list
• Set temperature thresholds and monitoring options

{Fore.YELLOW}Files Created:{Style.RESET_ALL}
• enhanced_miner.log - Detailed logging
• miner_stats.pkl - Persistent statistics storage

{Fore.YELLOW}Web Dashboard:{Style.RESET_ALL}
• Install Flask: pip install flask
• Access at http://localhost:8080 when running
• Auto-refreshes every 30 seconds

{Fore.YELLOW}Hotkeys:{Style.RESET_ALL}
• Ctrl+C - Graceful shutdown (saves stats)
    """
    print(help_text)

if __name__ == '__main__':
    # Initialize context
    if not hasattr(ctx, 'total_hashes_computed'):
        ctx.total_hashes_computed = 0
    if not hasattr(ctx, 'mining_time_per_block'):
        ctx.mining_time_per_block = []
    if not hasattr(ctx, 'fShutdown'):
        ctx.fShutdown = False
    if not hasattr(ctx, 'listfThreadRunning'):
        ctx.listfThreadRunning = [False] * 10
    if not hasattr(ctx, 'nHeightDiff'):
        ctx.nHeightDiff = {}

    # Show splash and help
    show_enhanced_splash()
    
    if len(sys.argv) > 1 and sys.argv[1] in ['--help', '-h', 'help']:
        show_help()
        sys.exit(0)
    
    # Setup signal handler
    signal(SIGINT, handler)
    
    # Start the enhanced miner
    enhanced_start_mining()
    
    # Keep main thread alive
    try:
        while not ctx.fShutdown:
            time.sleep(1)
    except KeyboardInterrupt:
        handler(SIGINT, None)
