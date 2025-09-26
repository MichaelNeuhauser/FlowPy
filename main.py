#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  7 14:23:00 2018

    Copyright (C) <2020>  <Michael Neuhauser>
    Michael.Neuhauser@bfw.gv.at

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
# import standard libraries
import os
import sys
import numpy as np
from datetime import datetime
from multiprocessing import cpu_count
import multiprocessing as mp
import logging
import pickle

# Flow-Py Libraries
import raster_io as io
import flow_core as fc
import split_and_merge as SPAM
from gui import Flow_Py_EXEC

# import for TerminalUpdate
from dataclasses import dataclass
from typing import Dict
import multiprocessing as mp

# rich (pip install rich)
from rich.live import Live
from rich.table import Table
from rich import box
from rich.progress import Progress, BarColumn, TextColumn, TaskProgressColumn, TimeRemainingColumn, TimeElapsedColumn

# --- progress messaging between workers and printer ---

UPDATE_Q = None  # set in pool initializer
@dataclass
class RowUpdate:
    worker_name: str
    status: str
    message: str

# message sent when one tile finishes
# include the worker name so the printer can attribute the completion
TILE_DONE_TAG = "__tile_done__"

# message to stop printer
STOP_MSG = ("__stop__", None)

def _pool_initializer(q):
    """Runs once in each worker process; gives workers access to the queue."""
    global UPDATE_Q
    UPDATE_Q = q

def _calc_wrapper(opt_tuple):
    """
    Wrap fc.calculation so each task streams status to the printer.
    Expects opt_tuple = (i, j, alpha, exp, cellsize, nodata, flux_threshold, max_z, temp_dir, infra_bool)
    """
    i, j = opt_tuple[0], opt_tuple[1]
    name = mp.current_process().name  # e.g., 'SpawnPoolWorker-1' on Windows

    # tell printer we (this worker) are starting a tile
    if UPDATE_Q is not None:
        UPDATE_Q.put(RowUpdate(name, "running", f"tile ({i},{j}) start"))

    # run the real computation
    fc.calculation(opt_tuple)

    # notify tile done and advance overall progress
    if UPDATE_Q is not None:
        UPDATE_Q.put(RowUpdate(name, "running", f"tile ({i},{j}) ✅ done"))
        UPDATE_Q.put((TILE_DONE_TAG, name))   # attribute completion to this worker

        
from collections import defaultdict

def _render_table(state: Dict[str, RowUpdate], counts: Dict[str, int]) -> Table:
    t = Table(title="Workers", box=box.SIMPLE_HEAVY)
    t.add_column("Worker", style="bold")
    t.add_column("Status")
    t.add_column("Message", overflow="fold")
    for w in sorted(state.keys()):
        r = state[w]
        done = counts.get(w, 0)
        status = f"{r.status} • {done} done" if done else r.status
        t.add_row(w, status, r.message)
    return t


def progress_printer(n_workers: int, total_tiles: int, q: mp.Queue):
    state: Dict[str, RowUpdate] = {}
    counts: Dict[str, int] = defaultdict(int)

    columns = [
        TextColumn("{task.description}", justify="left"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ]

    with Progress(*columns, transient=False) as progress:
        overall = progress.add_task("overall", total=total_tiles, start=True)

        with Live(_render_table(state, counts), refresh_per_second=20, transient=False) as live:
            done_tiles = 0
            while True:
                msg = q.get()
                if msg == STOP_MSG:
                    break

                # handle per-tile completion tagged with the worker name
                if isinstance(msg, tuple) and len(msg) == 2 and msg[0] == TILE_DONE_TAG:
                    worker_name = msg[1]
                    counts[worker_name] += 1
                    done_tiles += 1
                    progress.update(overall, completed=done_tiles)
                    # make sure the row exists so counts show up immediately
                    state.setdefault(worker_name, RowUpdate(worker_name, "running", ""))
                    live.update(_render_table(state, counts))
                    continue

                # normal row update
                upd: RowUpdate = msg
                state.setdefault(upd.worker_name, RowUpdate(upd.worker_name, "—", ""))
                state[upd.worker_name] = upd
                live.update(_render_table(state, counts))

            
def remove_temp_dir(path):
    if not os.path.exists(path):
        return
    # walk through all subdirs and files
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            try:
                os.remove(os.path.join(root, name))
            except Exception as e:
                print(f"Could not delete file {name}: {e}")
        for name in dirs:
            try:
                os.rmdir(os.path.join(root, name))
            except Exception as e:
                print(f"Could not delete dir {name}: {e}")
    try:
        os.rmdir(path)
        print(f"Temporary folder {path} removed.")
    except Exception as e:
        print(f"Could not remove folder {path}: {e}")




def main(args, kwargs): 

    alpha = args[0]
    exp = args[1]
    directory = args[2]
    dem_path = args[3]
    release_path = args[4]
    if 'infra' in kwargs:
        infra_path = kwargs.get('infra')
    else:
        infra_path = None
        
    if 'flux' in kwargs:
        flux_threshold = float(kwargs.get('flux'))
    else:
        flux_threshold = 3 * 10 ** -4
        
    if 'max_z' in kwargs:
        max_z = kwargs.get('max_z')
    else:
        max_z = 8848
        # Recomendet values:
                # Avalanche = 270
                # Rockfall = 50
                # Soil Slide = 12

    print("Starting...")
    print("...")

    start = datetime.now().replace(microsecond=0)
    infra_bool = False
    # Create result directory
    time_string = datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        os.makedirs(directory + 'res_{}/'.format(time_string))
        res_dir = ('res_{}/'.format(time_string))
    except FileExistsError:
        res_dir = ('res_{}/'.format(time_string))
        
    try:
        os.makedirs(directory + res_dir + 'temp/')
        temp_dir = (directory + res_dir + 'temp/')
    except FileExistsError:
        temp_dir = (directory + res_dir + 'temp/')

    # Setup logger
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=(directory + res_dir + 'log_{}.txt').format(time_string),
                        filemode='w')

    # Start of Calculation
    logging.info('Start Calculation')
    logging.info('Alpha Angle: {}'.format(alpha))
    logging.info('Exponent: {}'.format(exp))
    logging.info('Flux Threshold: {}'.format(flux_threshold))
    logging.info('Max Z_delta: {}'.format(max_z))
    logging.info
    # Read in raster files
    try:
        dem, header = io.read_raster(dem_path)
        logging.info('DEM File: {}'.format(dem_path))
    except FileNotFoundError:
        print("DEM: Wrong filepath or filename")
        return

    try:
        release, release_header = io.read_raster(release_path)
        logging.info('Release File: {}'.format(release_path))
    except FileNotFoundError:
        print("Wrong filepath or filename")
        return

    # Check if Layers have same size!!!
    if header['ncols'] == release_header['ncols'] and header['nrows'] == release_header['nrows']:
        print("DEM and Release Layer ok!")
    else:
        print("Error: Release Layer doesn't match DEM!")
        return

    try:
        infra, infra_header = io.read_raster(infra_path)
        if header['ncols'] == infra_header['ncols'] and header['nrows'] == infra_header['nrows']:
            print("Infra Layer ok!")
            infra_bool = True
            logging.info('Infrastructure File: {}'.format(infra_path))
        else:
            print("Error: Infra Layer doesn't match DEM!")
            return
    except:
        infra = np.zeros_like(dem)

    del dem, release, infra
    logging.info('Files read in')
    
    cellsize = header["cellsize"]
    nodata = header["noDataValue"]
    
    tileCOLS = int(15000 / cellsize)
    tileROWS = int(15000 / cellsize)
    U = int(5000 / cellsize) # 5km overlap
    
    logging.info("Start Tiling.")
    print("Start Tiling...")
    
    SPAM.tileRaster(dem_path, "dem", temp_dir, tileCOLS, tileROWS, U)
    SPAM.tileRaster(release_path, "init", temp_dir, tileCOLS, tileROWS, U, isInit=True)
    if infra_bool:
        SPAM.tileRaster(infra_path, "infra", temp_dir, tileCOLS, tileROWS, U)
    
    print("Finished Tiling...")    
    nTiles = pickle.load(open(temp_dir + "nTiles", "rb"))

    optList = []
    # das hier ist die batch-liste, die von mulitprocessing
    # abgearbeitet werden muss - sieht so aus:
    # [(0,0,alpha,exp,cellsize,-9999.),
    # (0,1,alpha,exp,cellsize,-9999.),
    # etc.]
       
    for i in range(nTiles[0]+1):
        for j in range(nTiles[1]+1):
            optList.append((i, j, alpha, exp, cellsize, nodata, flux_threshold,
                            max_z, temp_dir, infra_bool))

    # Calculation
    logging.info('Multiprocessing starts, used cores: {}'.format(cpu_count() - 1))
    # Calculation
    n_workers = max(1, mp.cpu_count() - 1)
    total_tiles = len(optList)
    logging.info('Multiprocessing starts, used cores: {}'.format(n_workers))
    print(f"{n_workers} processes started and {total_tiles} calculations to perform.")
    
    # queue for status updates
    q = mp.Manager().Queue()
    
    # start the printer as a process (keeps terminal output serialized & pretty)
    printer = mp.Process(target=progress_printer, args=(n_workers, total_tiles, q), daemon=True)
    printer.start()
    
    # start the pool with initializer so each worker can access UPDATE_Q
    with mp.Pool(processes=n_workers, initializer=_pool_initializer, initargs=(q,)) as pool:
        # Use imap_unordered so tiles complete out-of-order but streaming stays live
        list(pool.imap_unordered(_calc_wrapper, optList, chunksize=1))
    
    # tell printer to stop and wait for it
    q.put(STOP_MSG)
    printer.join()


    logging.info('Calculation finished, merging results.')
    
    # Merge calculated tiles
    z_delta = SPAM.MergeRaster(temp_dir, "res_z_delta")
    flux = SPAM.MergeRaster(temp_dir, "res_flux")
    cell_counts = SPAM.MergeRaster(temp_dir, "res_count")
    z_delta_sum = SPAM.MergeRaster(temp_dir, "res_z_delta_sum")
    fp_ta = SPAM.MergeRaster(temp_dir, "res_fp")
    sl_ta = SPAM.MergeRaster(temp_dir, "res_sl")
    if infra_bool:
        backcalc = SPAM.MergeRaster(temp_dir, "res_backcalc")
    
    # time_string = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.info('Writing Output Files')
    output_format = '.tif'
    io.output_raster(dem_path,
                     directory + res_dir + "flux{}".format(output_format),
                     flux)
    io.output_raster(dem_path,
                     directory + res_dir + "z_delta{}".format(output_format),
                     z_delta)
    io.output_raster(dem_path,
                     directory + res_dir + "FP_travel_angle{}".format(output_format),
                     fp_ta)
    io.output_raster(dem_path,
                     directory + res_dir + "SL_travel_angle{}".format(output_format),
                     sl_ta)
    if not infra_bool:  # if no infra
        io.output_raster(dem_path,
                         directory + res_dir + "cell_counts{}".format(output_format),
                         cell_counts)
        io.output_raster(dem_path,
                         directory + res_dir + "z_delta_sum{}".format( output_format),
                         z_delta_sum)
    if infra_bool:  # if infra
        io.output_raster(dem_path,
                         directory + res_dir + "backcalculation{}".format(output_format),
                         backcalc)

    print("Calculation finished")
    print("...")
    end = datetime.now().replace(microsecond=0)
    logging.info('Calculation needed: ' + str(end - start) + ' seconds')
    remove_temp_dir(temp_dir)


if __name__ == '__main__':
    argv = sys.argv[1:]  
    if len(argv) < 1:
    	print("Too few input arguments!!!")
    	sys.exit(1)
    if len(argv) == 1 and argv[0] == '--gui':
        Flow_Py_EXEC()
    else:
        args=[arg for arg in argv if arg.find('=')<0]
        kwargs={kw[0]:kw[1] for kw in [ar.split('=') for ar in argv if ar.find('=')>0]}

        main(args, kwargs)
    
# example dam: python main.py 25 8 ./examples/dam/ ./examples/dam/dam_010m_standard_cr100_sw250_f2500.20.6_n0.asc ./examples/dam/release_dam.tif
# example dam: python main.py 25 8 ./examples/dam/ ./examples/dam/dam_010m_standard_cr100_sw250_f2500.20.6_n0.asc ./examples/dam/release_dam.tif infra=./examples/dam/infra.tif flux=0.0003 max_z=270
