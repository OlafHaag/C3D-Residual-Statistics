"""Iterate over all c3d files in a given directory and save statistics as files."""
from __future__ import print_function
import glob
import os
import sys
import time
from multiprocessing import Pool, freeze_support

import c3dstats as c3d


def mute():
    sys.stdout = open(os.devnull, 'w')
    
    
if __name__ == '__main__':
    freeze_support()
    if len(sys.argv) > 1:
        os.chdir(sys.argv[1])
        t0 = time.time()
        files = list()
        for filename in glob.glob("*.c3d"):
            files.append(filename)
        print("Analyzing {} files:".format(len(files)))
        for file in files:
            print(file)
        n_processes = min(len(files), 8)
        print("\nCreating pool with {} processes".format(n_processes))
        p = Pool(processes=n_processes, initializer=mute)  # Disable print in subprocesses.
        try:
            p.imap_unordered(c3d.save_c3dstats, files)
            #res = [p.apply_async(c3d.save_c3dstats, (filename,)) for filename in files]
            p.close()
            #for item in res:
            #    item.wait(timeout=9999999)  # Without timeout you can't interrupt this.
        except KeyboardInterrupt:
            p.terminate()
        finally:
            p.join()
            print("Process took: {:.2f} seconds".format(time.time()-t0))
    else:
        print("Writes c3d file statistics on conditionals")
        print("Usage: c3dstats_directory [folderpath]")
